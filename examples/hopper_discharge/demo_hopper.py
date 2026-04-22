"""
Demo: Hopper Discharge Simulation

Simulation:
  Phase 1 — SETTLE: Particles are packed inside the hopper with a plug
            mesh blocking the outlet.  The simulation runs until kinetic
            energy drops below a threshold, indicating steady state.
            Settled positions are saved to ``packed_positions.npy``.
  Phase 2 — DISCHARGE: A new simulation is created with the hopper mesh
            only (no plug).  Settled positions are loaded and the outlet
            opens, allowing particles to flow out under gravity.  Flow
            pattern (mass / funnel / arching) depends on material
            properties and geometry.

Usage:
    # Default 35 mm particles, auto-estimated count, with plug:
    python demo_hopper.py --hopper-stl Hopper2.stl --plug-stl plug2.stl

    # Custom material / friction for funnel flow study:
    python demo_hopper.py --hopper-stl Hopper2.stl --plug-stl plug2.stl \\
                          --radius 0.0175 --friction-rolling 1.5 \\
                          --cohesion 25 --cohesion-wall 80

    # Skip settling (reuse previously packed positions):
    python demo_hopper.py --hopper-stl Hopper2.stl \\
                          --packed packed_positions.npy

Built by VeloxSim Tech Pty Ltd and Sam Wong.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Parse CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hopper discharge DEM demo")

# Geometry
parser.add_argument("--hopper-stl", type=str, required=True,
                    help="Path to hopper STL file (mm units assumed)")
parser.add_argument("--plug-stl", type=str, default=None,
                    help="STL mesh that blocks the outlet during settling. "
                         "If omitted, a plane is generated at the outlet z.")
parser.add_argument("--floor-stl", type=str, default=None,
                    help="Optional floor / catch-bin STL below the outlet")
parser.add_argument("--stl-scale", type=float, default=0.001,
                    help="STL unit conversion (default: 0.001 = mm -> m)")

# Particles
parser.add_argument("--radius", type=float, default=0.0175,
                    help="Particle radius in metres (default: 17.5 mm = 35 mm dia)")
parser.add_argument("--bulk-density", type=float, default=2000.0,
                    help="Bulk material density kg/m^3 (default: 2000). "
                         "Converted to solid particle density via packing fraction.")
parser.add_argument("--n-particles", type=int, default=None,
                    help="Number of particles. If omitted, auto-estimated from "
                         "hopper volume and particle size.")
parser.add_argument("--packing-fraction", type=float, default=0.55,
                    help="Target packing fraction for particle count estimation "
                         "(default: 0.55 for random loose packing)")

# Material
parser.add_argument("--friction-static", type=float, default=0.5)
parser.add_argument("--friction-dynamic", type=float, default=0.4)
parser.add_argument("--friction-rolling", type=float, default=0.01)
parser.add_argument("--restitution", type=float, default=0.3)
parser.add_argument("--cohesion", type=float, default=0.0,
                    help="Particle-particle JKR cohesion energy J/m^2 (default: 0)")
parser.add_argument("--cohesion-wall", type=float, default=None,
                    help="Particle-wall JKR cohesion energy J/m^2 "
                         "(default: same as --cohesion)")

# Timing
parser.add_argument("--dt", type=float, default=None,
                    help="Timestep seconds (default: auto)")
parser.add_argument("--settle-ke", type=float, default=1e-4,
                    help="KE threshold (J) to consider settled (default: 1e-4)")
parser.add_argument("--settle-max-steps", type=int, default=3_000_000,
                    help="Max steps for settling phase (default: 3M)")
parser.add_argument("--sim-time", type=float, default=15.0,
                    help="Discharge simulation time in seconds")

# Pre-packed shortcut
parser.add_argument("--packed", type=str, default=None,
                    help="Path to .npy file of pre-packed positions. "
                         "Skips the settling phase entirely.")

# Output
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--no-viewer", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Convert bulk density to solid particle density
# ---------------------------------------------------------------------------
# Bulk density includes void space between particles:
#   bulk_density = packing_fraction * solid_density
# DEM needs the solid (particle) density for mass calculation.
solid_density = args.bulk_density / args.packing_fraction
print(f"\nBulk density:   {args.bulk_density:.0f} kg/m^3")
print(f"Packing frac:   {args.packing_fraction}")
print(f"Solid density:  {solid_density:.0f} kg/m^3  "
      f"(= {args.bulk_density:.0f} / {args.packing_fraction})")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if args.output_dir:
    OUT_DIR = pathlib.Path(args.output_dir)
else:
    OUT_DIR = pathlib.Path.home() / "Desktop" / "veloxsim_hopper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import warp as wp
wp.init()

import trimesh

# Make the engine (veloxsim_dem.py) at the repo root importable from this subdir
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from veloxsim_dem import Simulation, SimConfig, load_mesh, create_plane_mesh

DEVICE = args.device

# ---------------------------------------------------------------------------
# Shared config helper
# ---------------------------------------------------------------------------

def make_config(global_damping: float = 0.0,
                cohesion_override: float | None = None,
                cohesion_wall_override: float | None = None) -> SimConfig:
    coh = cohesion_override if cohesion_override is not None else args.cohesion
    coh_w = cohesion_wall_override if cohesion_wall_override is not None else args.cohesion_wall
    return SimConfig(
        num_particles=args.n_particles,
        particle_radius=args.radius,
        particle_density=solid_density,
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        restitution=args.restitution,
        friction_static=args.friction_static,
        friction_dynamic=args.friction_dynamic,
        friction_rolling=args.friction_rolling,
        cohesion_energy=coh,
        cohesion_energy_wall=coh_w,
        dt=args.dt,
        gravity=(0.0, 0.0, -9.81),
        global_damping=global_damping,
        device=DEVICE,
    )

# ---------------------------------------------------------------------------
# Hopper geometry analysis & volume estimation
# ---------------------------------------------------------------------------
print(f"\nLoading hopper: {args.hopper_stl}", flush=True)
hopper_tri = trimesh.load(args.hopper_stl, force="mesh")
hopper_verts = np.array(hopper_tri.vertices, dtype=np.float32) * args.stl_scale
h_lo = hopper_verts.min(axis=0)
h_hi = hopper_verts.max(axis=0)
h_size = h_hi - h_lo
z_range = h_size[2]

print(f"Hopper bounds (m): lo=[{h_lo[0]:.3f}, {h_lo[1]:.3f}, {h_lo[2]:.3f}]  "
      f"hi=[{h_hi[0]:.3f}, {h_hi[1]:.3f}, {h_hi[2]:.3f}]")
print(f"Hopper size (mm):  {h_size[0]*1000:.0f} x {h_size[1]*1000:.0f} x {h_size[2]*1000:.0f}")

if hopper_tri.is_volume:
    hopper_vol_m3 = abs(hopper_tri.volume) * (args.stl_scale ** 3)
    vol_source = "mesh"
else:
    bb_vol = float(h_size[0] * h_size[1] * h_size[2])
    hopper_vol_m3 = bb_vol * 0.45
    vol_source = "estimated (45% of bounding box — mesh not watertight)"

print(f"Internal volume:   {hopper_vol_m3:.4f} m^3  ({hopper_vol_m3*1000:.1f} litres)  "
      f"[{vol_source}]")

# Auto-estimate particle count if not provided
R = args.radius
D_particle = 2.0 * R
V_sphere = (4.0 / 3.0) * np.pi * R ** 3

if args.n_particles is None:
    n_est = args.packing_fraction * hopper_vol_m3 / V_sphere
    args.n_particles = max(100, int(n_est // 100) * 100)
    print(f"\nAuto particle estimate:")
    print(f"  Particle diameter:  {D_particle*1000:.0f} mm")
    print(f"  V_sphere:           {V_sphere*1e9:.0f} mm^3")
    print(f"  Packing fraction:   {args.packing_fraction}")
    print(f"  Estimated count:    {n_est:.0f}")
    print(f"  Using:              {args.n_particles}")
else:
    print(f"\nUsing user-specified particle count: {args.n_particles}")

min_dim = min(h_size[0], h_size[1])
print(f"Narrowest XY span:  {min_dim*1000:.0f} mm = "
      f"{min_dim / D_particle:.1f} particle diameters")

# Outlet z — slightly above the very bottom
outlet_z = h_lo[2] + z_range * 0.1

# Domain bounds
domain_lo = np.array([h_lo[0] - 0.1, h_lo[1] - 0.1, h_lo[2] - 0.5])
domain_hi = np.array([h_hi[0] + 0.1, h_hi[1] + 0.1, h_hi[2] + 0.1])


# ---------------------------------------------------------------------------
# Generate initial grid positions inside the hopper
# ---------------------------------------------------------------------------

def generate_grid_positions(n: int) -> tuple[np.ndarray, int]:
    """Create a grid of positions inside the hopper.

    Particles are placed on a regular lattice inside the bounding box,
    then filtered against the hopper mesh to keep only those that are
    actually inside the geometry.  This handles conical / tapered
    hoppers correctly — particles outside the walls are discarded.
    """
    spacing = 2.01 * args.radius  # very tight packing grid

    margin = 1 * args.radius
    fill_x_lo = h_lo[0] + margin
    fill_x_hi = h_hi[0] - margin
    fill_y_lo = h_lo[1] + margin
    fill_y_hi = h_hi[1] - margin
    fill_z_lo = h_lo[2] + 4 * args.radius
    # Stack above the rim so particles fall in during settling.
    # With an open-top hopper, particles above the rim fall into the
    # hopper and compact.  0.5× hopper height above the rim is enough
    # to fill completely after compaction.
    fill_z_hi = h_hi[2] + z_range * 0.5

    nx = max(1, int((fill_x_hi - fill_x_lo) / spacing))
    ny = max(1, int((fill_y_hi - fill_y_lo) / spacing))
    nz = max(1, int((fill_z_hi - fill_z_lo) / spacing))
    print(f"Fill grid: {nx}x{ny}x{nz} = {nx*ny*nz} slots")
    print(f"Fill zone: x=[{fill_x_lo:.3f},{fill_x_hi:.3f}] "
          f"y=[{fill_y_lo:.3f},{fill_y_hi:.3f}] "
          f"z=[{fill_z_lo:.3f},{fill_z_hi:.3f}]")

    grid_points = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = fill_x_lo + (ix + 0.5) * spacing
                y = fill_y_lo + (iy + 0.5) * spacing
                z = fill_z_lo + (iz + 0.5) * spacing
                x += np.random.uniform(-0.1, 0.1) * args.radius
                y += np.random.uniform(-0.1, 0.1) * args.radius
                grid_points.append([x, y, z])
    grid_points = np.array(grid_points, dtype=np.float32)
    print(f"Grid points generated: {len(grid_points)}")

    # Filter: keep only points inside the hopper mesh
    inside_mask = np.ones(len(grid_points), dtype=bool)
    query_pts = grid_points / args.stl_scale

    if hopper_tri.is_volume:
        inside_mask = hopper_tri.contains(query_pts)
        print(f"Mesh contains() filter: {int(inside_mask.sum())} inside "
              f"/ {len(grid_points)} total")
    else:
        cx = (h_lo[0] + h_hi[0]) / 2.0
        cy = (h_lo[1] + h_hi[1]) / 2.0
        half_w_top = min(h_size[0], h_size[1]) / 2.0 - margin

        # Sample the hopper mesh's XY extent at a few z-levels
        z_levels = np.linspace(h_lo[2], h_hi[2], 20)
        max_r_at_z = []
        for zl in z_levels:
            band = np.abs(hopper_verts[:, 2] - zl) < z_range * 0.1
            if band.any():
                verts_band = hopper_verts[band]
                dists = np.sqrt((verts_band[:, 0] - cx)**2 +
                                (verts_band[:, 1] - cy)**2)
                max_r_at_z.append(float(dists.max()) - margin)
            else:
                max_r_at_z.append(half_w_top)
        max_r_at_z = np.array(max_r_at_z)

        top_r = max_r_at_z[-1]
        z_levels = np.append(z_levels, [h_hi[2] + z_range * 0.5])
        max_r_at_z = np.append(max_r_at_z, [top_r])

        for i, (px, py, pz) in enumerate(grid_points):
            r_xy = math.sqrt((px - cx)**2 + (py - cy)**2)
            allowed_r = float(np.interp(pz, z_levels, max_r_at_z))
            if r_xy > allowed_r:
                inside_mask[i] = False

        n_inside = int(inside_mask.sum())
        print(f"Geometry filter: {n_inside} inside / {len(grid_points)} total")

    inside_points = grid_points[inside_mask]

    if len(inside_points) > n:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(inside_points), size=n, replace=False)
        inside_points = inside_points[indices]

    n_placed = min(len(inside_points), n)
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:n_placed] = inside_points[:n_placed]
    if n_placed < n:
        pos[n_placed:, 2] = 10000.0
        if n_placed < n * 0.5:
            print(f"WARNING: Only placed {n_placed}/{n} particles — "
                  f"hopper may be too small for this particle count.")
        else:
            print(f"Placed {n_placed}/{n} particles inside hopper geometry.")

    return pos, n_placed


# =========================================================================
# PHASE 1 — SETTLE  (with outlet plugged)
# =========================================================================

packed_path = OUT_DIR / "packed_positions.npy"

if args.packed:
    packed_path = pathlib.Path(args.packed)
    print(f"\nLoading pre-packed positions: {packed_path}")
    settled_pos = np.load(packed_path)
    n_placed = len(settled_pos)
    args.n_particles = n_placed
    print(f"  {n_placed} particles loaded")

else:
    print(f"\n{'='*70}")
    print("PHASE 1: SETTLING (outlet plugged)")
    print(f"{'='*70}\n")

    # Use global damping during settling to dissipate energy faster.
    SETTLE_DAMPING = 5.0  # 1/s
    # Settle with ZERO cohesion — pure gravity packing.  This avoids
    # extreme contact forces from JKR adhesion during the initial
    # grid-to-packed transition.  Cohesion is applied in Phase 2 only.
    config_settle = make_config(global_damping=SETTLE_DAMPING,
                                cohesion_override=0.0,
                                cohesion_wall_override=0.0)
    sim_settle = Simulation(config_settle)
    print(f"Rayleigh dt:    {sim_settle.rayleigh_dt:.2e} s")
    print(f"Using dt:       {config_settle.dt:.2e} s")
    print(f"Global damping: {SETTLE_DAMPING} /s (settling only)\n")

    # Add hopper mesh
    sim_settle.add_mesh_from_file(args.hopper_stl, scale=args.stl_scale)

    # Add plug mesh to block the outlet
    if args.plug_stl:
        print(f"Loading plug: {args.plug_stl}")
        sim_settle.add_mesh_from_file(args.plug_stl, scale=args.stl_scale)
    else:
        plug_origin = np.array([
            (h_lo[0] + h_hi[0]) / 2,
            (h_lo[1] + h_hi[1]) / 2,
            h_lo[2] + z_range * 0.05,
        ], dtype=np.float32)
        plug_size = max(h_hi[0] - h_lo[0], h_hi[1] - h_lo[1]) * 1.5
        plug_mesh = create_plane_mesh(
            origin=tuple(plug_origin),
            normal=(0.0, 0.0, 1.0),
            size=plug_size,
            device=DEVICE,
        )
        sim_settle.add_mesh(plug_mesh)
        print(f"Generated plug plane at z={plug_origin[2]:.3f} m "
              f"(size={plug_size:.3f} m)")

    # Place particles on grid
    init_pos, n_placed = generate_grid_positions(args.n_particles)
    sim_settle.initialize_particles(init_pos)
    if n_placed < args.n_particles:
        sim_settle.active_count = n_placed
    print(f"Placed {n_placed} particles\n")

    # ------------------------------------------------------------------
    # Settling loop with multi-criteria convergence
    # ------------------------------------------------------------------
    particle_mass = sim_settle.particle_mass
    auto_ke_thresh = 0.5 * n_placed * particle_mass * (0.05 ** 2)
    KE_THRESHOLD = max(args.settle_ke, auto_ke_thresh)

    MAX_STEPS = args.settle_max_steps
    CHECK_EVERY = max(100, int(0.01 / config_settle.dt))
    report_every = max(1, MAX_STEPS // 20)

    # Convergence requires ALL of these sustained for CONVERGE_COUNT checks
    SPEED_THRESHOLD = 0.1     # m/s
    DISP_THRESHOLD = 0.05 * args.radius   # 5% of radius between checks
    CONVERGE_COUNT = 5

    print(f"KE threshold:   {KE_THRESHOLD:.4f} J  "
          f"(= avg speed < {SPEED_THRESHOLD} m/s for {n_placed} particles)")
    print(f"Max steps:      {MAX_STEPS:,}")
    print()

    t0 = time.perf_counter()
    settled = False
    converge_streak = 0
    prev_pos = None
    n_live = 0
    n_leaked = 0
    pos_live = np.zeros((0, 3))
    ke = 0.0
    speed_max = 0.0

    for step_i in range(1, MAX_STEPS + 1):
        sim_settle.step()

        if step_i % CHECK_EVERY == 0:
            wp.synchronize()
            N_active = sim_settle.active_count
            pos_np = sim_settle.get_positions()[:N_active]
            vel_np = sim_settle.get_velocities()[:N_active]

            # Only check particles inside the walls — particles above the
            # rim slide slowly (no walls) and would prevent convergence.
            in_hopper = ((pos_np[:, 2] > h_lo[2] - 0.05)
                         & (pos_np[:, 2] < h_hi[2])
                         & (pos_np[:, 2] < 500.0))
            pos_live = pos_np[in_hopper]
            vel_live = vel_np[in_hopper]
            n_live = len(pos_live)

            speeds = np.linalg.norm(vel_live, axis=1) if n_live > 0 else np.array([0.0])
            speed_max = float(speeds.max())
            ke = float(0.5 * particle_mass * np.sum(speeds ** 2)) if n_live > 0 else 0.0
            n_leaked = int(np.sum(pos_np[:N_active, 2] < h_lo[2] - 0.05))

            max_disp = float("inf")
            if prev_pos is not None and n_live == len(prev_pos):
                displacements = np.linalg.norm(pos_live - prev_pos, axis=1)
                max_disp = float(displacements.max())
            prev_pos = pos_live.copy()

            if step_i > CHECK_EVERY * 10:
                if (ke < KE_THRESHOLD
                        and speed_max < SPEED_THRESHOLD
                        and max_disp < DISP_THRESHOLD):
                    converge_streak += 1
                else:
                    converge_streak = 0

                if converge_streak >= CONVERGE_COUNT:
                    settled = True
                    elapsed = time.perf_counter() - t0
                    print(f"\n  Settled at step {step_i:,} "
                          f"(t={sim_settle.sim_time:.4f}s, KE={ke:.2e} J, "
                          f"max_speed={speed_max:.4f} m/s, "
                          f"max_disp={max_disp:.2e} m) "
                          f"in {elapsed:.0f}s wall")
                    break

        if step_i % report_every == 0:
            elapsed = time.perf_counter() - t0
            pct = step_i / MAX_STEPS * 100
            z_min = float(pos_live[:, 2].min()) if n_live > 0 else 0
            z_max = float(pos_live[:, 2].max()) if n_live > 0 else 0
            print(f"  {pct:5.0f}%  step={step_i:,}  t={sim_settle.sim_time:.3f}s  "
                  f"KE={ke:.2e}  speed_max={speed_max:.3f}  "
                  f"in_hopper={n_live}  z=[{z_min:.3f},{z_max:.3f}]  "
                  f"leaked={n_leaked}  streak={converge_streak}/{CONVERGE_COUNT}  "
                  f"({elapsed:.0f}s)", flush=True)

    if not settled:
        wp.synchronize()
        print(f"\n  WARNING: Did not fully settle after {MAX_STEPS:,} steps "
              f"(KE={ke:.2e} J, threshold={KE_THRESHOLD:.4f} J)")
        print(f"  Proceeding with current positions anyway.")

    # ------------------------------------------------------------------
    # Post-settle validation
    # ------------------------------------------------------------------
    wp.synchronize()
    settled_pos = sim_settle.get_positions()
    settled_contacts = sim_settle.contact_counts.numpy()
    settled_vel = sim_settle.get_velocities()
    n_placed = sim_settle.active_count

    live_mask = settled_pos[:n_placed, 2] < 500.0
    pos_live = settled_pos[:n_placed][live_mask]
    contacts_live = settled_contacts[:n_placed][live_mask]
    vel_live = settled_vel[:n_placed][live_mask]
    n_live = len(pos_live)

    n_with_contacts = int(np.sum(contacts_live > 0))
    in_walls = (pos_live[:, 2] > h_lo[2]) & (pos_live[:, 2] < h_hi[2])
    above_rim_with_contacts = (pos_live[:, 2] >= h_hi[2]) & (contacts_live > 0)
    in_hopper_mask = in_walls | above_rim_with_contacts
    n_in_hopper = int(np.sum(in_hopper_mask))
    n_escaped = n_live - n_in_hopper

    R = args.radius
    V_particle = (4.0 / 3.0) * np.pi * R ** 3
    if n_live > 0:
        bed_lo = pos_live.min(axis=0)
        bed_hi = pos_live.max(axis=0)
        bed_size = np.maximum(bed_hi - bed_lo, 2.0 * R)
        bed_vol = float(bed_size[0] * bed_size[1] * bed_size[2])
        packing = (n_live * V_particle) / bed_vol if bed_vol > 0 else 0.0
        mean_contacts = float(contacts_live.mean())
        max_contacts = int(contacts_live.max())
        speeds = np.linalg.norm(vel_live, axis=1)
        speed_mean = float(speeds.mean())
        speed_max = float(speeds.max())
        bed_height = float(bed_hi[2] - bed_lo[2])
    else:
        packing = mean_contacts = speed_mean = speed_max = bed_height = 0.0
        max_contacts = 0

    print(f"\n  {'-'*50}")
    print(f"  PACKED BED VALIDATION")
    print(f"  {'-'*50}")
    print(f"  Particles placed:       {n_placed:,}")
    print(f"  Particles in hopper:    {n_in_hopper:,}  "
          f"({n_in_hopper/n_placed*100:.1f}%)" if n_placed > 0 else "")
    print(f"  Particles escaped:      {n_escaped:,}")
    print(f"  Mean contacts/particle: {mean_contacts:.1f}")
    print(f"  Max contacts:           {max_contacts}")
    print(f"  Bed height:             {bed_height*1000:.0f} mm")
    print(f"  Max residual speed:     {speed_max:.4f} m/s")
    print(f"  Mean residual speed:    {speed_mean:.5f} m/s")
    print(f"  {'-'*50}")

    # Save only particles that are part of the packed bed
    packed_pos_to_save = pos_live[in_hopper_mask]
    np.save(packed_path, packed_pos_to_save)
    n_placed = len(packed_pos_to_save)
    print(f"  Saved {n_placed} in-hopper positions: {packed_path}")

    # Update settled_pos for Phase 2 — ensure allocation is large enough
    args.n_particles = max(args.n_particles, n_placed)
    settled_pos = np.zeros((args.n_particles, 3), dtype=np.float32)
    settled_pos[:n_placed] = packed_pos_to_save
    settled_pos[n_placed:, 2] = 10000.0

    del sim_settle


# =========================================================================
# PHASE 2 — DISCHARGE  (outlet open)
# =========================================================================

print(f"\n{'='*70}")
print("PHASE 2: DISCHARGE (outlet open)")
print(f"{'='*70}\n")

config_discharge = make_config()
sim = Simulation(config_discharge)
print(f"Using dt: {config_discharge.dt:.2e} s")

sim.add_mesh_from_file(args.hopper_stl, scale=args.stl_scale)

if args.floor_stl:
    print(f"Loading floor: {args.floor_stl}")
    sim.add_mesh_from_file(args.floor_stl, scale=args.stl_scale)

full_pos = np.zeros((args.n_particles, 3), dtype=np.float32)
full_pos[:n_placed] = settled_pos[:n_placed]
full_pos[n_placed:, 2] = 10000.0
sim.initialize_particles(full_pos)
sim.active_count = n_placed
print(f"Loaded {n_placed} settled particles (outlet now open)")

# Delete particles that fall 0.5 m below the hopper bottom
delete_z = h_lo[2] - 0.5


def delete_below(sim: Simulation, z_threshold: float) -> int:
    """Park particles below z_threshold at the dormant position (z=10000).

    Returns the number of particles deleted on this call.
    """
    N = sim.active_count
    if N == 0:
        return 0

    pos_np = sim.get_positions()[:N]
    below = pos_np[:, 2] < z_threshold
    below &= pos_np[:, 2] < 500.0
    dead_indices = np.where(below)[0]

    if len(dead_indices) == 0:
        return 0

    vel_np = sim.get_velocities()[:N]
    angvel_np = sim.get_angular_velocities()[:N]

    for idx in dead_indices:
        pos_np[idx] = [0.0, 0.0, 10000.0]
        vel_np[idx] = [0.0, 0.0, 0.0]
        angvel_np[idx] = [0.0, 0.0, 0.0]

    # Pad arrays if smaller than full capacity
    def _pad(arr):
        if N < sim.num_particles:
            return np.pad(arr, ((0, sim.num_particles - N), (0, 0)),
                          constant_values=(10000.0 if arr is pos_np else 0.0))
        return arr

    sim.positions = wp.array(_pad(pos_np), dtype=wp.vec3, device=sim.device)
    sim.velocities = wp.array(_pad(vel_np), dtype=wp.vec3, device=sim.device)
    sim.angular_velocities = wp.array(_pad(angvel_np), dtype=wp.vec3, device=sim.device)

    # Clear contact history for deleted particles
    cc = sim.contact_counts.numpy()
    for idx in dead_indices:
        cc[idx] = 0
    sim.contact_counts = wp.array(cc, dtype=wp.int32, device=sim.device)

    return len(dead_indices)


# ---------------------------------------------------------------------------
# Run discharge simulation
# ---------------------------------------------------------------------------
RECORD_EVERY = max(100, int(0.01 / config_discharge.dt))
DELETE_EVERY = max(100, int(0.02 / config_discharge.dt))

total_steps = int(args.sim_time / config_discharge.dt)
print(f"\nRunning {total_steps:,} steps ({args.sim_time:.1f}s sim time)...")
print(f"Frame record every {RECORD_EVERY} steps "
      f"({RECORD_EVERY * config_discharge.dt * 1000:.0f} ms sim)\n")

frames = []
n_deleted = 0
t_start = time.perf_counter()
report_interval = max(1, total_steps // 20)

for step_i in range(1, total_steps + 1):
    sim.step()

    if step_i % DELETE_EVERY == 0:
        n_deleted += delete_below(sim, delete_z)

    if step_i % RECORD_EVERY == 0:
        wp.synchronize()
        pos_np = sim.get_positions()[:sim.active_count]
        vel_np = sim.get_velocities()[:sim.active_count]

        live = pos_np[:, 2] < 500.0
        pos_live = pos_np[live]
        vel_live = vel_np[live]
        frames.append({
            "t": round(sim.sim_time, 6),
            "n": int(live.sum()),
            "pos": pos_live.tolist(),
            "vel": vel_live.tolist(),
        })

    if step_i % report_interval == 0:
        elapsed = time.perf_counter() - t_start
        pct = step_i / total_steps * 100
        wp.synchronize()
        vel_live_rpt = sim.get_velocities()[:sim.active_count]
        live_rpt_mask = sim.get_positions()[:sim.active_count, 2] < 500.0
        n_live_rpt = int(live_rpt_mask.sum())
        ke_live = (float(0.5 * sim.particle_mass *
                         np.sum(vel_live_rpt[live_rpt_mask] ** 2))
                   if n_live_rpt > 0 else 0.0)
        print(f"  {pct:5.0f}%  t={sim.sim_time:.3f}s  "
              f"live={n_live_rpt}  deleted={n_deleted}  "
              f"KE={ke_live:.4e}  ({elapsed:.0f}s wall)", flush=True)

# Flush any remaining dormant particles
n_deleted += delete_below(sim, delete_z)

total_elapsed = time.perf_counter() - t_start
print(f"\n{'='*70}")
print(f"Done: {len(frames)} frames in {total_elapsed:.0f}s wall")
if n_placed > 0:
    print(f"Discharged (deleted): {n_deleted}/{n_placed} particles "
          f"({n_deleted/n_placed*100:.1f}%)")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
stl_data = {}
for name, path in [("hopper", args.hopper_stl), ("floor", args.floor_stl)]:
    if path is None:
        continue
    m = trimesh.load(path, force="mesh")
    verts = np.array(m.vertices, dtype=np.float32) * args.stl_scale
    faces_flat = np.array(m.faces, dtype=np.int32).flatten()
    stl_data[name] = {
        "v": [[round(float(v[0]), 4), round(float(v[1]), 4),
               round(float(v[2]), 4)] for v in verts],
        "f": faces_flat.tolist(),
    }

results_path = OUT_DIR / "hopper_results.json"
export = {
    "config": {
        "radius": args.radius,
        "n_particles": n_placed,
        "solid_density": solid_density,
        "bulk_density": args.bulk_density,
        "sim_time": args.sim_time,
        "friction_static": args.friction_static,
        "friction_rolling": args.friction_rolling,
        "cohesion": args.cohesion,
        "cohesion_wall": args.cohesion_wall,
    },
    "stl": stl_data,
    "frames": frames,
}
with open(results_path, "w") as f:
    json.dump(export, f, separators=(",", ":"))
print(f"Results: {results_path} ({results_path.stat().st_size / 1024 / 1024:.1f} MB)")

# ---------------------------------------------------------------------------
# Generate interactive viewer
# ---------------------------------------------------------------------------
if not args.no_viewer:
    from hopper_viewer import generate_hopper_html

    viewer_path = OUT_DIR / "hopper_viewer.html"
    generate_hopper_html(
        results_path=results_path,
        output_path=viewer_path,
        title=f"VeloxSim-DEM Hopper Discharge ({n_placed} particles)",
    )
    print(f"Viewer: {viewer_path}")

    try:
        import webbrowser
        webbrowser.open(str(viewer_path))
    except Exception:
        pass

print("\nDone.")
