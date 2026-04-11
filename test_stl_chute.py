"""
Test: STL Transfer Chute with dynamic particle insertion.

Built by VeloxSim Tech Pty Ltd and Sam Wong.

Geometry from imported STL files (mm units, scaled to m):
  - feed.stl:         Inclined (~6 deg) feed conveyor, 3 m/s
  - impact plate.stl: Upper chute impact plate (static)
  - lower chute.stl:  Lower chute section (static)
  - receive.stl:      Receiving conveyor, 3 m/s in +Y
  - skirts.stl:       Containment skirts at chute exit (static)
  - top_skirts.stl:   Containment skirts on feed conveyor (static)
  - inlet.stl:        *Virtual* insertion region — defines where particles spawn

Particles are dynamically inserted at the inlet in overlap-safe grid layers.
Insertion rate is automatically derived from tonnage (tph), particle size,
bulk density, and the inlet rectangle dimensions.

Outputs to Desktop/veloxsim_conveyor_v2/:
  - stl_results.json       — raw frame data
  - stl_trajectory.png     — x-z trajectory plot
  - stl_animation.html     — Three.js interactive animation
"""

from __future__ import annotations

import sys
import os
import json
import math
import time
import pathlib
import argparse

import numpy as np
import trimesh
import warp as wp

sys.stdout.reconfigure(encoding="utf-8")

from veloxsim_dem import Simulation, SimConfig

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VeloxSim-DEM Transfer Chute Simulation")
parser.add_argument("--radius", type=float, default=0.0225, help="Particle radius in m (default: 0.0225 = 45mm dia)")
parser.add_argument("--belt-speed", type=float, default=3.0, help="Belt speed in m/s (default: 3.0)")
parser.add_argument("--tonnage", type=float, default=3000.0, help="Tonnage in tph (default: 3000)")
parser.add_argument("--sim-time", type=float, default=10.0, help="Simulation time in s (default: 10)")
parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: auto)")
parser.add_argument("--max-particles", type=int, default=250000, help="Max particle pool size")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STL_DIR = pathlib.Path("C:/Users/anthr/OneDrive/Desktop/VeloxSim Demo/Generic")
DESKTOP = pathlib.Path(os.path.expanduser("~")) / "Desktop"
if args.out_dir:
    OUT_DIR = pathlib.Path(args.out_dir)
else:
    dia_mm = int(args.radius * 2000)
    speed_str = f"{args.belt_speed:.0f}ms"
    tph_str = f"{args.tonnage:.0f}tph"
    OUT_DIR = DESKTOP / f"veloxsim_{dia_mm}mm_{speed_str}_{tph_str}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUT_DIR}")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R = args.radius
BULK_DENSITY = 2000.0  # kg/m3 (bulk density — includes voids)
PACKING_FRACTION = 0.6
SOLID_DENSITY = BULK_DENSITY / PACKING_FRACTION  # ~3333 kg/m3 (particle solid density)
BELT_SPEED = args.belt_speed
TONNAGE_TPH = args.tonnage
MAX_PARTICLES = args.max_particles

# Feed belt incline: ~5.8 deg from surface data
FEED_SLOPE = 0.1023    # dZ/dX
FEED_ANGLE = math.atan(FEED_SLOPE)
FEED_VX = BELT_SPEED * math.cos(FEED_ANGLE)
FEED_VZ = BELT_SPEED * math.sin(FEED_ANGLE)

G = 9.81
DT = 1.0e-5
SIM_TIME = args.sim_time
RECORD_EVERY = 2000    # every 20 ms
DEVICE = "cuda:0"

DELETE_EVERY = 5000     # check every 50ms
DOMAIN_MARGIN = 1.0    # m — margin around geometry extents for domain box

# ---------------------------------------------------------------------------
# Insertion algorithm — overlap-safe grid layers
# ---------------------------------------------------------------------------
print("=" * 65)
print("VeloxSim-DEM  |  STL Transfer Chute — Dynamic Insertion")
print("=" * 65)

# Load inlet.stl for bounding box (virtual — NOT added to sim)
inlet_mesh_tri = trimesh.load(str(STL_DIR / "inlet.stl"), force="mesh")
inlet_verts_m = inlet_mesh_tri.vertices / 1000.0  # mm -> m
inlet_lo = inlet_verts_m.min(axis=0)
inlet_hi = inlet_verts_m.max(axis=0)
inlet_size = inlet_hi - inlet_lo

print(f"  Inlet rectangle (from inlet.stl):")
print(f"    X: [{inlet_lo[0]:.3f}, {inlet_hi[0]:.3f}]  ({inlet_size[0]:.3f} m)")
print(f"    Y: [{inlet_lo[1]:.3f}, {inlet_hi[1]:.3f}]  ({inlet_size[1]:.3f} m)")
print(f"    Z: [{inlet_lo[2]:.3f}, {inlet_hi[2]:.3f}]  ({inlet_size[2]:.3f} m)")

# Rate calculation
SPACING = 2 * R * 1.05
# Tonnage → bulk volume flow → particle insertion rate
mass_flow_rate = TONNAGE_TPH * 1000.0 / 3600.0              # kg/s
bulk_volume_rate = mass_flow_rate / BULK_DENSITY             # m³/s of bulk material
V_particle = (4.0 / 3.0) * math.pi * R**3                   # m³ per solid sphere
particle_mass = SOLID_DENSITY * V_particle                   # kg per particle
insertion_rate = bulk_volume_rate * PACKING_FRACTION / V_particle  # particles/s

# Grid packing: how many particles fit in one layer of the inlet
nx_insert = max(1, int(inlet_size[0] / SPACING))
ny_insert = max(1, int(inlet_size[1] / SPACING))
n_per_layer = nx_insert * ny_insert

# Layer insertion interval (seconds between successive layers)
layer_interval = n_per_layer / insertion_rate
INSERT_EVERY = max(1, round(layer_interval / DT))  # steps between layer insertions

# Clearance velocity: particles must move down by ≥ SPACING before next layer
gravity_drop = 0.5 * G * layer_interval**2
if gravity_drop >= SPACING:
    v_insert = 0.0  # gravity alone provides clearance
else:
    v_insert = (SPACING - gravity_drop) / layer_interval

# Pre-compute one insertion layer grid (reused every insertion event)
layer_grid = np.zeros((n_per_layer, 3), dtype=np.float32)
idx = 0
for ix in range(nx_insert):
    for iy in range(ny_insert):
        layer_grid[idx, 0] = inlet_lo[0] + (ix + 0.5) * SPACING
        layer_grid[idx, 1] = inlet_lo[1] + (iy + 0.5) * SPACING
        layer_grid[idx, 2] = inlet_hi[2]  # top of inlet
        idx += 1

# How many layers before pool is exhausted?
max_layers = MAX_PARTICLES // n_per_layer
insertion_duration = max_layers * layer_interval

print(f"\n  Insertion parameters:")
print(f"    Tonnage: {TONNAGE_TPH:.0f} tph  ({mass_flow_rate:.1f} kg/s)")
print(f"    Particle: R={R*1000:.0f}mm, m={particle_mass*1000:.1f}g")
print(f"    Insertion rate: {insertion_rate:.0f} particles/s")
print(f"    Grid per layer: {nx_insert} x {ny_insert} = {n_per_layer} particles")
print(f"    Layer interval: {layer_interval*1000:.1f} ms ({INSERT_EVERY} steps)")
print(f"    Clearance velocity: {v_insert:.3f} m/s downward")
print(f"    Gravity drop in interval: {gravity_drop*1000:.1f} mm (need {SPACING*1000:.1f} mm)")
print(f"    Max particles: {MAX_PARTICLES}, max layers: {max_layers}")
print(f"    Insertion exhausted at t={insertion_duration:.2f}s")
print(f"\n  Feed belt: {BELT_SPEED} m/s along {math.degrees(FEED_ANGLE):.1f} deg incline")
print(f"  Sim time: {SIM_TIME} s ({int(SIM_TIME/DT):,} steps)")
print()


# ---------------------------------------------------------------------------
# Build simulation — all particles start dormant (parked at z=+100)
# ---------------------------------------------------------------------------
print("Building simulation...", flush=True)

config = SimConfig(
    num_particles=MAX_PARTICLES,
    particle_radius=R,
    particle_density=SOLID_DENSITY,
    young_modulus=5.0e6,
    poisson_ratio=0.3,
    restitution=0.4,
    friction_static=0.95,
    friction_dynamic=0.80,
    friction_rolling=0.25,
    cohesion_energy=8.0,
    dt=DT,
    gravity=(0.0, 0.0, -G),
    max_contacts_per_particle=32,
    # hash_grid_dim auto-computed from particle count and radius
    global_damping=1.0,
    device=DEVICE,
)

sim = Simulation(config)

# Park all particles far above the scene (dormant pool)
dormant_pos = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
dormant_pos[:, 2] = 10000.0  # z = +10km — far from everything, won't drift into domain
sim.initialize_particles(dormant_pos)
sim.active_count = 0  # start with zero active particles
print(f"  Dormant pool: {MAX_PARTICLES} particles allocated, active=0")

# Load physical STL meshes (mm -> m via scale=0.001)
feed_wp = sim.add_mesh_from_file(
    str(STL_DIR / "feed.stl"),
    scale=0.001,
    surface_velocity=(FEED_VX, 0.0, FEED_VZ),
)
print(f"  feed.stl loaded (surface_vel=({FEED_VX:.3f}, 0, {FEED_VZ:.3f}))")

impact_wp = sim.add_mesh_from_file(
    str(STL_DIR / "impact plate.stl"),
    scale=0.001,
    surface_velocity=(0.0, 0.0, 0.0),
)
print(f"  impact plate.stl loaded (static)")

lower_chute_wp = sim.add_mesh_from_file(
    str(STL_DIR / "lower chute.stl"),
    scale=0.001,
    surface_velocity=(0.0, 0.0, 0.0),
)
print(f"  lower chute.stl loaded (static)")

receive_wp = sim.add_mesh_from_file(
    str(STL_DIR / "receive.stl"),
    scale=0.001,
    surface_velocity=(0.0, BELT_SPEED, 0.0),
)
print(f"  receive.stl loaded (surface_vel=(0, {BELT_SPEED}, 0))")

skirts_wp = sim.add_mesh_from_file(
    str(STL_DIR / "skirts.stl"),
    scale=0.001,
    surface_velocity=(0.0, 0.0, 0.0),
)
print(f"  skirts.stl loaded (static)")

top_skirts_wp = sim.add_mesh_from_file(
    str(STL_DIR / "top_skirts.stl"),
    scale=0.001,
    surface_velocity=(0.0, 0.0, 0.0),
)
print(f"  top_skirts.stl loaded (static)")

# inlet.stl is VIRTUAL — loaded for viewer only, NOT added to sim.meshes
print(f"  inlet.stl loaded (virtual — no collision)")
print(f"  Total collision meshes: {len(sim.meshes)}")

# Compute domain box from combined extents of ALL geometry + inlet
all_stl_files = ["feed.stl", "impact plate.stl", "lower chute.stl",
                 "receive.stl", "skirts.stl", "top_skirts.stl", "inlet.stl"]
domain_lo = np.array([np.inf, np.inf, np.inf])
domain_hi = np.array([-np.inf, -np.inf, -np.inf])
for stl_file in all_stl_files:
    m = trimesh.load(str(STL_DIR / stl_file), force="mesh")
    v = m.vertices / 1000.0
    domain_lo = np.minimum(domain_lo, v.min(axis=0))
    domain_hi = np.maximum(domain_hi, v.max(axis=0))
domain_lo -= DOMAIN_MARGIN
domain_hi += DOMAIN_MARGIN

print(f"\n  Domain box (auto from geometry + {DOMAIN_MARGIN}m margin):")
print(f"    X: [{domain_lo[0]:.2f}, {domain_hi[0]:.2f}]")
print(f"    Y: [{domain_lo[1]:.2f}, {domain_hi[1]:.2f}]")
print(f"    Z: [{domain_lo[2]:.2f}, {domain_hi[2]:.2f}]")
print()


# ---------------------------------------------------------------------------
# Run simulation with dynamic particle insertion
# ---------------------------------------------------------------------------
num_steps = int(SIM_TIME / DT)
print(f"Running {num_steps:,} steps with dynamic insertion...", flush=True)

# Compile kernels
sim.step()
wp.synchronize()
print("  Kernels compiled.", flush=True)

frames = []
t_start = time.perf_counter()
report_interval = num_steps // 20  # report every 5%

active_count = 0       # highest used particle index (kernel dim)
total_inserted = 0     # cumulative count of particles ever inserted
total_deleted = 0      # cumulative count of particles recycled
free_indices = []      # recycled slot indices available for reuse

insert_vel_np = np.zeros((n_per_layer, 3), dtype=np.float32)
insert_vel_np[:, 2] = -v_insert  # downward velocity

# Dormant state for recycled particles
DORMANT_POS = np.array([0.0, 0.0, 10000.0], dtype=np.float32)
DORMANT_VEL = np.array([0.0, 0.0, 0.0], dtype=np.float32)

for step_i in range(1, num_steps + 1):
    sim.step()

    # --- Particle insertion: one grid layer every INSERT_EVERY steps ---
    if step_i % INSERT_EVERY == 0:
        # Determine target indices for this batch
        insert_indices = []
        new_active = active_count
        for _ in range(n_per_layer):
            if free_indices:
                insert_indices.append(free_indices.pop())
            elif new_active < MAX_PARTICLES:
                insert_indices.append(new_active)
                new_active += 1
            else:
                break

        if insert_indices:
            wp.synchronize()
            pos_np = sim.get_positions()
            vel_np = sim.get_velocities()
            angvel_np = sim.get_angular_velocities()
            for gi, idx in enumerate(insert_indices):
                pos_np[idx] = layer_grid[gi]
                vel_np[idx] = insert_vel_np[gi]
                angvel_np[idx] = [0.0, 0.0, 0.0]  # clear spin from previous life
            sim.positions = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
            sim.velocities = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)
            sim.angular_velocities = wp.array(angvel_np, dtype=wp.vec3, device=DEVICE)

            # Clear contact history for recycled slots to prevent force spikes
            contact_counts_np = sim.contact_counts.numpy()
            for idx in insert_indices:
                contact_counts_np[idx] = 0
            sim.contact_counts = wp.array(contact_counts_np, dtype=wp.int32, device=DEVICE)

            active_count = new_active
            sim.active_count = active_count
            total_inserted += len(insert_indices)

    # --- Particle deletion: park out-of-domain particles, add to free list ---
    if step_i % DELETE_EVERY == 0 and active_count > 0:
        wp.synchronize()
        pos_np = sim.get_positions()
        vel_np = sim.get_velocities()

        # Find particles outside domain box (vectorized)
        active_pos = pos_np[:active_count]
        out_mask = (
            (active_pos[:, 0] < domain_lo[0]) | (active_pos[:, 0] > domain_hi[0]) |
            (active_pos[:, 1] < domain_lo[1]) | (active_pos[:, 1] > domain_hi[1]) |
            (active_pos[:, 2] < domain_lo[2]) | (active_pos[:, 2] > domain_hi[2])
        )
        # Exclude dormant particles (z > 1000) from deletion
        out_mask &= (active_pos[:, 2] < 1000.0)
        dead_indices = np.where(out_mask)[0]

        if len(dead_indices) > 0:
            angvel_np = sim.get_angular_velocities()
            # Park dead particles and clear all state (no reordering!)
            for idx in dead_indices:
                pos_np[idx] = DORMANT_POS
                vel_np[idx] = DORMANT_VEL
                angvel_np[idx] = [0.0, 0.0, 0.0]
                free_indices.append(int(idx))

            sim.positions = wp.array(pos_np, dtype=wp.vec3, device=DEVICE)
            sim.velocities = wp.array(vel_np, dtype=wp.vec3, device=DEVICE)
            sim.angular_velocities = wp.array(angvel_np, dtype=wp.vec3, device=DEVICE)
            total_deleted += len(dead_indices)

    # --- Record frame (filter out dormant particles at z=100) ---
    if step_i % RECORD_EVERY == 0:
        wp.synchronize()
        pos_all = sim.get_positions()[:active_count]
        vel_all = sim.get_velocities()[:active_count]
        live = pos_all[:, 2] < 500.0  # anything below z=50 is live
        pos_live = pos_all[live]
        vel_live = vel_all[live]
        n_live = len(pos_live)
        frames.append({
            "t": round(sim.sim_time, 6),
            "n": n_live,
            "pos": pos_live.tolist(),
            "vel": vel_live.tolist(),
        })

    # --- Progress report ---
    if step_i % report_interval == 0:
        elapsed = time.perf_counter() - t_start
        pct = step_i / num_steps * 100
        wp.synchronize()
        pos_all = sim.get_positions()[:active_count]
        live = pos_all[:, 2] < 500.0
        n_live = int(live.sum())
        pos_live = pos_all[live]
        vel_live = sim.get_velocities()[:active_count][live]
        if n_live > 0:
            z_min = float(pos_live[:, 2].min())
            z_max = float(pos_live[:, 2].max())
            vx_mean = float(vel_live[:, 0].mean())
        else:
            z_min = z_max = vx_mean = 0.0
        print(f"  {pct:5.0f}%  t={sim.sim_time:.2f}s  "
              f"live={n_live:,}  pool={active_count:,}  "
              f"ins={total_inserted:,}  del={total_deleted:,}  free={len(free_indices):,}  "
              f"z=[{z_min*1000:.0f}, {z_max*1000:.0f}] mm  "
              f"vx={vx_mean:.2f}  "
              f"({elapsed:.0f}s)", flush=True)

wp.synchronize()
total_elapsed = time.perf_counter() - t_start
print(f"  Done: {len(frames)} frames in {total_elapsed:.0f}s")
print(f"  Total inserted: {total_inserted:,}, deleted: {total_deleted:,}, final active: {active_count:,}")
print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
# Load STL vertex data for the animation
stl_anim = {}
stl_name_map = {
    "feed": "feed.stl",
    "impact_plate": "impact plate.stl",
    "lower_chute": "lower chute.stl",
    "receive": "receive.stl",
    "skirts": "skirts.stl",
    "top_skirts": "top_skirts.stl",
    "inlet": "inlet.stl",
}
for name, stl_file in stl_name_map.items():
    m = trimesh.load(str(STL_DIR / stl_file), force="mesh")
    verts = (np.array(m.vertices, dtype=np.float32) / 1000.0)
    faces = np.array(m.faces, dtype=np.int32).flatten()
    stl_anim[name] = {
        "v": [[round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
              for v in verts],
        "f": faces.tolist(),
    }

results_path = OUT_DIR / "stl_results.json"
export = {
    "config": {
        "radius": R, "n_particles": MAX_PARTICLES,
        "feed_speed": BELT_SPEED, "receive_speed": BELT_SPEED,
        "feed_angle_deg": math.degrees(FEED_ANGLE),
        "tonnage_tph": TONNAGE_TPH,
    },
    "frames": frames,
}
with open(results_path, "w") as f:
    json.dump(export, f)
print(f"Results saved: {results_path} ({results_path.stat().st_size/1024/1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("Sanity Checks")
print("=" * 65)

last = frames[-1]
n_live_final = last["n"]
all_z = [last["pos"][p][2] for p in range(n_live_final)]
z_min_final = min(all_z) if all_z else 0
print(f"  Final live: {n_live_final:,} (pool slots: {active_count:,}, free: {len(free_indices):,})")
print(f"  Total inserted: {total_inserted:,}, recycled: {total_deleted:,}")
print(f"  Final z_min: {z_min_final*1000:.1f} mm")

n_on_recv = sum(1 for p in range(n_live_final)
                if last["pos"][p][2] < -2.5 and last["pos"][p][1] > -3.5)
print(f"  Particles on receiving belt: {n_on_recv}/{n_live_final}")

discharge_vx = []
for fr in frames[len(frames)//3:]:
    n_fr = fr["n"]
    for p in range(min(n_fr, len(fr["pos"]))):
        x = fr["pos"][p][0]
        z = fr["pos"][p][2]
        vx = fr["vel"][p][0]
        if -1.0 < x < -0.1 and z > 0.3:
            discharge_vx.append(vx)
if discharge_vx:
    print(f"  Mean vx near discharge: {np.mean(discharge_vx):.2f} m/s (target: {FEED_VX:.2f})")
print()


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------
print("Generating trajectory plot...", end="", flush=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot trajectories for a sample of particles
    # Use particles from the first few batches (they travel the full path)
    n_sample = min(n_per_layer * 3, total_inserted)
    stride = max(1, n_sample // 100)
    for p in range(0, n_sample, stride):
        xs, zs = [], []
        for fr in frames:
            if p < fr["n"]:
                xs.append(fr["pos"][p][0])
                zs.append(fr["pos"][p][2])
        if xs:
            ax.plot(xs, zs, lw=0.3, alpha=0.4, color="#3b82f6")

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("z (m)", fontsize=12)
    ax.set_title("STL Transfer Chute — Particle Trajectories (x-z)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    traj_path = OUT_DIR / "stl_trajectory.png"
    fig.tight_layout()
    fig.savefig(traj_path, dpi=150)
    plt.close(fig)
    print(f" saved: {traj_path}")
except ImportError:
    print(" [SKIPPED]")


# ---------------------------------------------------------------------------
# Three.js animation
# ---------------------------------------------------------------------------
print("Generating Three.js animation...", end="", flush=True)

# Compact frames — variable particle count per frame
MAX_ANIM_FRAMES = 200
step = max(1, len(frames) // MAX_ANIM_FRAMES)
anim_frames = []
for fr in frames[::step]:
    n_fr = fr["n"]
    compact = {"t": round(fr["t"], 3), "n": n_fr, "p": [], "s": []}
    for p in range(n_fr):
        px, py, pz = fr["pos"][p]
        vx, vy, vz = fr["vel"][p]
        compact["p"].append([round(px, 4), round(py, 4), round(pz, 4)])
        compact["s"].append(round(math.sqrt(vx*vx + vy*vy + vz*vz), 3))
    anim_frames.append(compact)

payload = json.dumps({
    "config": export["config"],
    "stl": stl_anim,
    "frames": anim_frames,
}, separators=(",", ":"))
print(f" ({len(payload)/1024:.0f} KB, {len(anim_frames)} frames)...", end="", flush=True)

HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>VeloxSim-DEM STL Transfer Chute</title>
<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js" } }
</script>
<style>
  *{margin:0;padding:0} body{background:#1e293b;overflow:hidden}
  #controls{position:absolute;bottom:0;left:0;right:0;background:rgba(15,23,42,0.90);
    backdrop-filter:blur(6px);padding:12px 20px;display:flex;align-items:center;gap:14px;
    border-top:1px solid #334155;z-index:10}
  button{background:#334155;border:none;color:#e2e8f0;padding:6px 14px;border-radius:6px;
    cursor:pointer;font-size:13px;transition:background 0.15s}
  button:hover{background:#475569}
  button.active{background:#3b82f6;color:#fff}
  input[type=range]{flex:1;accent-color:#3b82f6;cursor:pointer}
  .label{font-size:12px;color:#94a3b8;white-space:nowrap}
  #info{position:absolute;top:12px;left:16px;background:rgba(15,23,42,0.9);
    padding:12px 16px;border-radius:8px;color:#e2e8f0;font:13px/1.8 'Segoe UI',sans-serif;
    border:1px solid #334155;z-index:10}
  .c{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:middle}
  #time-label{font-size:13px;color:#cbd5e1;min-width:80px;text-align:right}
  #particle-count{font-size:12px;color:#94a3b8;min-width:100px}
  #colorbar{position:absolute;bottom:70px;left:16px;background:rgba(15,23,42,0.9);
    border:1px solid #334155;border-radius:8px;padding:10px 14px;z-index:10;display:none}
  #colorbar h3{margin:0 0 6px;font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em}
  #cb-canvas{display:block;border-radius:3px}
  .cb-labels{display:flex;justify-content:space-between;font-size:10px;color:#6e7681;margin-top:3px;width:160px}
  .btn-group{display:flex;gap:2px}
  .btn-group button{border-radius:4px;padding:4px 10px;font-size:11px}
</style>
</head><body>
<div id="info">
  <div><b>VeloxSim-DEM &mdash; STL Transfer Chute</b></div>
  <div><span class="c" style="background:#3b82f6"></span>Feed belt (3 m/s)</div>
  <div><span class="c" style="background:#f59e0b"></span>Impact plate (static)</div>
  <div><span class="c" style="background:#d97706"></span>Lower chute (static)</div>
  <div><span class="c" style="background:#22c55e"></span>Receiving belt (3 m/s)</div>
  <div><span class="c" style="background:#94a3b8"></span>Skirts (static)</div>
  <div><span class="c" style="background:#78716c"></span>Top skirts (static)</div>
  <div><span class="c" style="background:#06b6d4;border:1px dashed #67e8f9"></span>Inlet (virtual)</div>
  <div><span class="c" style="background:#f87171"></span>__N__ particles, __D__mm, __TPH__ tph</div>
  <div style="margin-top:6px;color:#64748b;font-size:11px">Orbit: drag | Zoom: scroll | Pan: right-drag</div>
</div>
<div id="colorbar">
  <h3>Velocity (m/s)</h3>
  <canvas id="cb-canvas" width="160" height="14"></canvas>
  <div class="cb-labels"><span>0</span><span id="cb-mid"></span><span id="cb-max"></span></div>
</div>
<div id="controls">
  <button id="btn-play">&#9654; Play</button>
  <span class="label">Frame:</span>
  <input type="range" id="scrubber" min="0" value="0"/>
  <span id="time-label">t = 0.000 s</span>
  <span id="particle-count"></span>
  <div class="btn-group">
    <span class="label">Color:</span>
    <button id="btn-solid" class="active">Solid</button>
    <button id="btn-vel">Velocity</button>
  </div>
  <span class="label">Speed:</span>
  <input type="range" id="pb-speed" min="1" max="30" value="5" style="max-width:80px"/>
  <span class="label">&times;<span id="pb-val">5</span></span>
</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js";

const SIM = __PAYLOAD__;
const FRAMES = SIM.frames;
const STL = SIM.stl;
const N_MAX = SIM.config.n_particles;
const R = SIM.config.radius;

let globalMaxSpeed = 0;
for (const fr of FRAMES) {
    for (const s of fr.s) { if (s > globalMaxSpeed) globalMaxSpeed = s; }
}
globalMaxSpeed = Math.ceil(globalMaxSpeed) || 1;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e293b);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 200);
camera.up.set(0, 0, 1);
camera.position.set(-4, -8, 4);

scene.add(new THREE.AmbientLight(0x94a3b8, 2.0));
const dl = new THREE.DirectionalLight(0xffffff, 2.5);
dl.position.set(5, -10, 8); dl.castShadow = true; scene.add(dl);
const dl2 = new THREE.DirectionalLight(0xffffff, 1.2);
dl2.position.set(-5, 10, 3); scene.add(dl2);

// STL meshes — physical + virtual
const stlColors = {
    feed: 0x3b82f6, impact_plate: 0xf59e0b, lower_chute: 0xd97706,
    receive: 0x22c55e, skirts: 0x94a3b8, top_skirts: 0x78716c, inlet: 0x06b6d4
};
const virtualMeshes = new Set(["inlet"]);

for (const [name, data] of Object.entries(STL)) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(data.v.flat()), 3));
    geo.setIndex(data.f);
    geo.computeVertexNormals();

    const isVirtual = virtualMeshes.has(name);
    const color = stlColors[name] || 0xaaaaaa;

    if (isVirtual) {
        // Virtual mesh: wireframe only, dashed, no solid fill
        const edges = new THREE.EdgesGeometry(geo);
        scene.add(new THREE.LineSegments(edges,
            new THREE.LineDashedMaterial({ color, dashSize: 0.05, gapSize: 0.03, linewidth: 1, transparent: true, opacity: 0.7 })
        ));
    } else {
        scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
            color, roughness: 0.5, metalness: 0.2,
            side: THREE.DoubleSide, transparent: true, opacity: 0.5
        })));
        scene.add(new THREE.LineSegments(
            new THREE.WireframeGeometry(geo),
            new THREE.LineBasicMaterial({ color, opacity: 0.1, transparent: true })
        ));
    }
}

// Particles — allocate for max, render only active count
const pGeo = new THREE.SphereGeometry(R, 12, 8);
const pMat = new THREE.MeshStandardMaterial({
    roughness: 0.35, metalness: 0.15,
    emissive: 0xdc2626, emissiveIntensity: 0.25,
});
const inst = new THREE.InstancedMesh(pGeo, pMat, N_MAX);
inst.castShadow = true;
inst.frustumCulled = false;
inst.count = 0;  // start with zero visible
scene.add(inst);

let colorMode = 'solid';
const solidColor = new THREE.Color(0xf87171);

// Google Turbo colormap (Anton Mikhailov, 2019) — 256-entry LUT with linear interpolation
const TURBO=[[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.2086,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.225,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.1972,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.2483,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.693],[0.26074,0.29568,0.71162],[0.2628,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.7805],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.3697,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.2768,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.5304,0.98461],[0.27106,0.54015,0.9893],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.5895,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.9619],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.7168,0.93981],[0.16529,0.7262,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.8955],[0.13278,0.77165,0.8858],[0.12698,0.78037,0.8759],[0.12151,0.78896,0.86581],[0.11639,0.7974,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.0975,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.7724],[0.09267,0.86554,0.7623],[0.0932,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.8904,0.72393],[0.10342,0.896,0.715],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.6866],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.9268,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.2618,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.3614],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.9991,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.2528],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.2237],[0.67462,0.98246,0.2196],[0.68494,0.97941,0.21602],[0.69525,0.9761,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.9647,0.2064],[0.7361,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.2031],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.8141,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.8501,0.89328,0.21074],[0.85868,0.88655,0.2123],[0.86709,0.87968,0.21391],[0.8753,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.2188],[0.8987,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.2257],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.228],[0.95039,0.78823,0.22831],[0.9556,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98,0.73,0.22161],[0.98289,0.7214,0.21918],[0.98549,0.7125,0.2165],[0.98781,0.7033,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.1491],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.9859,0.50479,0.13367],[0.9836,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.4692,0.11817],[0.97545,0.4574,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.0931],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.3053,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.8676,0.2373,0.03082],[0.86079,0.22945,0.02875],[0.8538,0.2217,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.0166],[0.79971,0.17055,0.0152],[0.79125,0.16368,0.01387],[0.7826,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.7068,0.1068,0.00571],[0.6965,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.0898,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.0738,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.0041],[0.60746,0.05878,0.00427],[0.5955,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.0078],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.4796,0.01583,0.01055]];
function turboColor(t) {
    t = Math.max(0, Math.min(1, t));
    const x = t * 255;
    const a = Math.min(255, x | 0);
    const b = Math.min(255, a + 1);
    const f = x - a;
    const ca = TURBO[a], cb = TURBO[b];
    return new THREE.Color(
        ca[0] + (cb[0] - ca[0]) * f,
        ca[1] + (cb[1] - ca[1]) * f,
        ca[2] + (cb[2] - ca[2]) * f
    );
}

function drawColorBar() {
    const cb = document.getElementById('cb-canvas');
    const ctx = cb.getContext('2d');
    for (let i = 0; i < 160; i++) {
        const c = turboColor(i / 159);
        ctx.fillStyle = `rgb(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0})`;
        ctx.fillRect(i, 0, 1, 14);
    }
    document.getElementById('cb-mid').textContent = (globalMaxSpeed / 2).toFixed(1);
    document.getElementById('cb-max').textContent = globalMaxSpeed.toFixed(1);
}
drawColorBar();

const tc = new THREE.Color();
const d = new THREE.Object3D();

function setFrame(idx) {
    const fr = FRAMES[Math.min(idx, FRAMES.length - 1)];
    const activeN = fr.n || fr.p.length;

    // Only render active particles
    inst.count = activeN;

    for (let p = 0; p < activeN; p++) {
        const pos = fr.p[p];
        if (colorMode === 'velocity') {
            tc.copy(turboColor(Math.min(fr.s[p] / globalMaxSpeed, 1.0)));
        } else {
            tc.copy(solidColor);
        }
        inst.setColorAt(p, tc);
        d.position.set(pos[0], pos[1], pos[2]);
        d.updateMatrix();
        inst.setMatrixAt(p, d.matrix);
    }
    inst.instanceMatrix.needsUpdate = true;
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
    document.getElementById("time-label").textContent = `t = ${fr.t.toFixed(3)} s`;
    document.getElementById("particle-count").textContent = `${activeN.toLocaleString()} particles`;
    document.getElementById("scrubber").value = idx;
}

const orbitCtrl = new OrbitControls(camera, renderer.domElement);
orbitCtrl.target.set(-2, 0, -0.5);
orbitCtrl.enableDamping = true;

let playing = false, frameIdx = 0, lastTime = 0, playbackSpeed = 5;
const scrub = document.getElementById("scrubber");
scrub.max = FRAMES.length - 1;

function animate(ts) {
    requestAnimationFrame(animate);
    if (playing && ts - lastTime > 1000 / (30 * playbackSpeed / 5)) {
        lastTime = ts;
        frameIdx = (frameIdx + 1) % FRAMES.length;
        setFrame(frameIdx);
    }
    orbitCtrl.update();
    renderer.render(scene, camera);
}

document.getElementById("btn-play").addEventListener("click", () => {
    playing = !playing;
    document.getElementById("btn-play").innerHTML = playing ? "&#9646;&#9646; Pause" : "&#9654; Play";
});
scrub.addEventListener("input", () => { frameIdx = parseInt(scrub.value); setFrame(frameIdx); });
const pbs = document.getElementById("pb-speed");
pbs.addEventListener("input", () => {
    playbackSpeed = parseInt(pbs.value);
    document.getElementById("pb-val").textContent = playbackSpeed;
});

document.getElementById("btn-solid").addEventListener("click", () => {
    colorMode = 'solid';
    document.getElementById("btn-solid").classList.add("active");
    document.getElementById("btn-vel").classList.remove("active");
    document.getElementById("colorbar").style.display = "none";
    setFrame(frameIdx);
});
document.getElementById("btn-vel").addEventListener("click", () => {
    colorMode = 'velocity';
    document.getElementById("btn-vel").classList.add("active");
    document.getElementById("btn-solid").classList.remove("active");
    document.getElementById("colorbar").style.display = "block";
    setFrame(frameIdx);
});

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

setFrame(0);
animate(0);
</script>
</body></html>"""

html = HTML.replace("__PAYLOAD__", payload)
html = html.replace("__N__", str(MAX_PARTICLES))
html = html.replace("__D__", f"{R*2*1000:.0f}")
html = html.replace("__TPH__", f"{TONNAGE_TPH:.0f}")

anim_path = OUT_DIR / "stl_animation.html"
with open(anim_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f" saved: {anim_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("Output files:")
for f in sorted(OUT_DIR.iterdir()):
    if f.name.startswith("stl_"):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:8.1f} KB")
