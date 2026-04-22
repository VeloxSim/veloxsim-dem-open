"""PSD hopper simulation with correct particle counts.

Back-calculates particle counts from:
  - Hopper2 internal volume (cone + cylinder profile)
  - Bulk density of 2000 kg/m^3
  - PSD: 35 mm (40%), 60 mm (30%), 100 mm (30%) by volume
  - Particle solid density: 2500 kg/m^3

Uses both Hopper2.stl (funnel walls) and plug2.stl (closed bottom) so
particles accumulate inside the hopper.  Runs with global damping until
kinetic energy drops below a threshold, then renders a PyVista PNG.

Output: psd_hopper_test.png
"""

import math
import pathlib
import sys
import time

# Make the repo root (where veloxsim_dem.py and hopper_viewer.py live) importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pyvista as pv
import trimesh
import warp as wp

from veloxsim_dem import (
    ParticleSizeDistribution,
    SimConfig,
    Simulation,
    load_mesh,
)

HERE       = pathlib.Path(__file__).resolve().parent
STL_DIR    = HERE / "STL"
HOPPER_STL = STL_DIR / "Hopper2.stl"
PLUG_STL   = STL_DIR / "plug2.stl"
STL_SCALE  = 0.001                        # STL authored in mm -> m
OUTPUT_PNG = HERE / "psd_hopper_test.png"

# --- Physical parameters ---------------------------------------------------
BULK_DENSITY     = 2000.0                 # kg/m^3 target
PARTICLE_DENSITY = 2500.0                 # kg/m^3 solid
FILL_HEIGHT      = 0.80                   # fill hopper up to this z (metres)

# PSD by VOLUME fraction
PSD_RADII     = [0.035, 0.060, 0.100]    # metres
PSD_VOL_FRACS = [0.40,  0.30,  0.30]

# Hopper geometry (from mesh cross-section analysis)
R_OUTLET     = 0.250                      # m (bottom outlet radius)
R_CYLINDER   = 0.7317                     # m (cylinder section radius)
Z_CONE_TOP   = 0.343                      # m (where cone meets cylinder)
Z_HOPPER_TOP = 1.527                      # m

# Simulation
GLOBAL_DAMPING   = 5.0                    # 1/s — helps settling
DT               = 1.0e-4                 # s
MAX_SIM_TIME     = 6.0                    # s — safety cap
MEAN_SPEED_THRESHOLD = 0.01               # m/s — consider settled below this
CHECK_INTERVAL   = 500                    # steps between KE checks
REPORT_INTERVAL  = 2000                   # steps between printed reports

# Spawning
SPAWN_CYLINDER_R = 0.60                   # m — tighter than hopper opening
SPAWN_Z_START    = 1.70                   # m — just above hopper top
SPACING_FACTOR   = 2.2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def hopper_volume(fill_z: float) -> float:
    """Internal volume of the Hopper2 profile up to *fill_z* (metres)."""
    vol = 0.0
    z_end = min(fill_z, Z_CONE_TOP)
    if z_end > 0.0:
        a = R_OUTLET
        b = (R_CYLINDER - R_OUTLET) / Z_CONE_TOP
        h = z_end
        vol += math.pi * (a**2 * h + a * b * h**2 + b**2 * h**3 / 3.0)
    if fill_z > Z_CONE_TOP:
        vol += math.pi * R_CYLINDER**2 * (fill_z - Z_CONE_TOP)
    return vol


def compute_particle_counts():
    """Back-calculate per-class counts from bulk density and hopper volume.

    Returns (total_N, count_pcts, per_class_counts).
    count_pcts are the percentages to feed into ParticleSizeDistribution
    (by count, NOT by volume) so the resulting fill has the correct
    volume fractions.
    """
    V_fill = hopper_volume(FILL_HEIGHT)
    total_mass = BULK_DENSITY * V_fill
    V_solid = total_mass / PARTICLE_DENSITY

    # Per-particle volumes
    v_p = [(4.0 / 3.0) * math.pi * r**3 for r in PSD_RADII]

    # Convert volume fractions -> count fractions
    inv_sum = sum(vf / vp for vf, vp in zip(PSD_VOL_FRACS, v_p))
    S = 1.0 / inv_sum
    count_fracs = [vf * S / vp for vf, vp in zip(PSD_VOL_FRACS, v_p)]

    # Total particle count
    avg_vol = sum(cf * vp for cf, vp in zip(count_fracs, v_p))
    total_N = int(round(V_solid / avg_vol))

    # Per-class counts (round, last class absorbs remainder)
    counts = []
    remaining = total_N
    for cf in count_fracs[:-1]:
        k = int(round(cf * total_N))
        k = max(0, min(k, remaining))
        counts.append(k)
        remaining -= k
    counts.append(remaining)

    # Count percentages for PSD class
    count_pcts = [100.0 * c / total_N for c in counts]

    return total_N, count_pcts, counts, V_fill, V_solid


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def generate_cylinder_positions(
    n: int,
    particle_radius: float,
    cylinder_r: float,
    z_start: float,
    spacing_factor: float = 2.2,
) -> np.ndarray:
    """Place *n* particles on a grid inside a vertical cylinder."""
    spacing = particle_radius * spacing_factor
    positions: list[list[float]] = []
    z = z_start
    while len(positions) < n:
        nx = int(2.0 * cylinder_r / spacing) + 1
        for ix in range(nx):
            for iy in range(nx):
                x = (ix - nx / 2.0 + 0.5) * spacing
                y = (iy - nx / 2.0 + 0.5) * spacing
                if x * x + y * y <= cylinder_r * cylinder_r:
                    positions.append([x, y, z])
                    if len(positions) >= n:
                        return np.array(positions[:n], dtype=np.float32)
        z += spacing
    return np.array(positions[:n], dtype=np.float32)


def build_spawn_positions(
    radii_array: np.ndarray,
    class_radii: list[float],
) -> np.ndarray:
    """Generate initial positions for all particles, grouped by size class.

    Largest particles spawn lowest (closest to hopper), smallest on top.
    Each class gets its own cylindrical grid with class-appropriate spacing.
    """
    n_total = len(radii_array)
    positions = np.zeros((n_total, 3), dtype=np.float32)

    # Sort classes largest-first (they spawn closest to hopper opening)
    sorted_classes = sorted(class_radii, reverse=True)
    z_cursor = SPAWN_Z_START

    for r_class in sorted_classes:
        idx = np.where(np.isclose(radii_array, r_class, atol=1e-5))[0]
        if len(idx) == 0:
            continue

        class_pos = generate_cylinder_positions(
            len(idx), r_class, SPAWN_CYLINDER_R, z_cursor, SPACING_FACTOR,
        )
        positions[idx] = class_pos

        # Advance z cursor: top of this class + gap for next class
        z_top = class_pos[:, 2].max()
        next_r = r_class  # conservative gap
        z_cursor = z_top + r_class + next_r + 0.01

    return positions


# ---------------------------------------------------------------------------
# Settling loop
# ---------------------------------------------------------------------------

def run_settling(sim: Simulation, config: SimConfig) -> int:
    """Advance simulation until mean speed drops below threshold.

    Returns the number of steps taken.
    """
    max_steps = int(MAX_SIM_TIME / config.dt)
    step = 0

    print(f"\n{'step':>7s}  {'t (s)':>7s}  {'KE (J)':>12s}  {'<v> m/s':>9s}  "
          f"{'z_min':>7s}  {'z_max':>7s}  {'contacts':>9s}")
    print("-" * 72)

    t0 = time.perf_counter()
    while step < max_steps:
        sim.step()
        step += 1

        if step % CHECK_INTERVAL == 0:
            wp.synchronize()
            ke = sim.get_kinetic_energy()
            vel = sim.get_velocities()
            mean_speed = float(np.linalg.norm(vel, axis=1).mean())

            if step % REPORT_INTERVAL == 0 or mean_speed < MEAN_SPEED_THRESHOLD:
                pos = sim.get_positions()
                z_min = float(pos[:, 2].min())
                z_max = float(pos[:, 2].max())
                n_pairs = int(sim.contact_counts.numpy().sum()) // 2
                t_sim = step * config.dt
                print(f"{step:>7d}  {t_sim:>7.3f}  {ke:>12.4e}  {mean_speed:>9.5f}  "
                      f"{z_min:>7.3f}  {z_max:>7.3f}  {n_pairs:>9d}")

            if mean_speed < MEAN_SPEED_THRESHOLD:
                print(f"\n  ** Settled: mean speed = {mean_speed:.5f} m/s "
                      f"< {MEAN_SPEED_THRESHOLD} m/s **")
                break

    wall = time.perf_counter() - t0
    print(f"\nSettling: {step:,d} steps in {wall:.1f}s "
          f"({step / wall:,.0f} steps/s)")
    return step


# ---------------------------------------------------------------------------
# PyVista rendering
# ---------------------------------------------------------------------------

CLASS_LABELS = ["35 mm (40 vol%)", "60 mm (30 vol%)", "100 mm (30 vol%)"]
CLASS_COLORS = ["#2962ff", "#f57c00", "#d32f2f"]


def render_png(
    positions: np.ndarray,
    radii_array: np.ndarray,
    hopper_verts: np.ndarray,
    hopper_faces: np.ndarray,
    plug_verts: np.ndarray,
    plug_faces: np.ndarray,
    sim_time: float,
    output_path: pathlib.Path,
) -> None:
    """Render a single-panel PyVista snapshot of the settled bed."""
    # Build PyVista mesh for hopper + plug
    def _make_pv_mesh(verts, faces):
        pv_faces = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int32), faces,
        ]).flatten()
        return pv.PolyData(verts, pv_faces)

    hopper_poly = _make_pv_mesh(hopper_verts, hopper_faces)
    plug_poly   = _make_pv_mesh(plug_verts, plug_faces)

    # Camera
    centre = positions.mean(axis=0)
    all_pts = np.concatenate([positions, hopper_verts], axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    cam_pos = (
        (centre[0] + diag * 0.55, centre[1] - diag * 0.70, centre[2] + diag * 0.15),
        tuple(centre),
        (0.0, 0.0, 1.0),
    )

    sphere_geoms = [
        pv.Sphere(radius=r, theta_resolution=24, phi_resolution=16)
        for r in PSD_RADII
    ]

    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    plotter = pv.Plotter(
        window_size=(1200, 900),
        off_screen=True,
        border=False,
    )

    # Hopper: translucent surface + wireframe
    for mesh_poly in [hopper_poly, plug_poly]:
        plotter.add_mesh(
            mesh_poly, color="#60a5fa", opacity=0.12,
            smooth_shading=True, specular=0.3, specular_power=15,
            show_edges=False,
        )
        plotter.add_mesh(
            mesh_poly, style="wireframe",
            color="#1e3a8a", opacity=0.55, line_width=1.0,
        )

    # Particles by class
    for cls_i, (r_class, sphere, colour) in enumerate(
        zip(PSD_RADII, sphere_geoms, CLASS_COLORS)
    ):
        idx = np.where(np.isclose(radii_array, r_class, atol=1e-5))[0]
        if len(idx) == 0:
            continue
        pdata = pv.PolyData(positions[idx])
        glyphs = pdata.glyph(geom=sphere, scale=False, orient=False)
        plotter.add_mesh(
            glyphs, color=colour,
            smooth_shading=True, specular=0.45, specular_power=20,
            ambient=0.28, diffuse=0.85,
        )

    plotter.add_text(
        f"Settled state  t = {sim_time:.2f} s",
        position="upper_edge", font_size=14, color="black",
    )
    plotter.add_legend(
        labels=[[lbl, col] for lbl, col in zip(CLASS_LABELS, CLASS_COLORS)],
        bcolor="white", face="circle", size=(0.30, 0.18), loc="upper left",
    )
    plotter.add_text(
        "PSD hopper (bulk density 2000 kg/m^3)  -  35/60/100 mm at 40/30/30 vol%",
        position="lower_edge", font_size=10, color="#334155",
    )
    plotter.camera_position = cam_pos
    plotter.screenshot(str(output_path), transparent_background=False)
    plotter.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print(" PSD hopper  -  back-calculated from bulk density 2000 kg/m^3")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Back-calculate particle counts
    # ------------------------------------------------------------------
    total_N, count_pcts, counts, V_fill, V_solid = compute_particle_counts()

    print(f"\nHopper fill height : {FILL_HEIGHT:.2f} m")
    print(f"Fill volume        : {V_fill:.4f} m^3 ({V_fill * 1000:.0f} L)")
    print(f"Solid volume       : {V_solid:.4f} m^3")
    print(f"Target bulk density: {BULK_DENSITY:.0f} kg/m^3")
    print(f"Particle density   : {PARTICLE_DENSITY:.0f} kg/m^3")
    print(f"Total particles    : {total_N:,d}")
    print(f"\nPSD (by volume -> by count):")
    print(f"  {'radius':>8s}  {'vol %':>7s}  {'count %':>9s}  {'count':>7s}")
    for r, vf, cp, c in zip(PSD_RADII, PSD_VOL_FRACS, count_pcts, counts):
        print(f"  {r*1000:>6.0f}mm  {vf*100:>6.0f}%  {cp:>8.1f}%  {c:>7,d}")

    # ------------------------------------------------------------------
    # 2. Create PSD and simulation
    # ------------------------------------------------------------------
    psd = ParticleSizeDistribution(list(zip(PSD_RADII, count_pcts)))

    config = SimConfig(
        num_particles=total_N,
        psd=psd,
        particle_density=PARTICLE_DENSITY,
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        restitution=0.5,
        friction_static=0.5,
        friction_dynamic=0.4,
        friction_rolling=0.02,
        cohesion_energy=0.0,
        dt=DT,
        gravity=(0.0, 0.0, -9.81),
        max_contacts_per_particle=32,
        hash_grid_dim=128,
        global_damping=GLOBAL_DAMPING,
    )
    sim = Simulation(config)

    # Verify assigned radii
    radii = sim.get_radii()
    print(f"\nAssigned radii check:")
    for r_target, cp_target in zip(PSD_RADII, count_pcts):
        n_act = int(np.sum(np.isclose(radii, r_target, atol=1e-5)))
        print(f"  {r_target*1000:.0f}mm: target {cp_target:.1f}% -> "
              f"actual {100.0*n_act/total_N:.1f}% ({n_act:,d})")

    # ------------------------------------------------------------------
    # 3. Spawn positions (per-class cylindrical grids)
    # ------------------------------------------------------------------
    positions = build_spawn_positions(radii, PSD_RADII)
    sim.initialize_particles(positions)
    print(f"\nSpawn region: z = [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}] m")

    # ------------------------------------------------------------------
    # 4. Load meshes: hopper funnel + bottom plug
    # ------------------------------------------------------------------
    print(f"\nLoading {HOPPER_STL.name} + {PLUG_STL.name} (scale={STL_SCALE}) ...")
    hopper_wp = load_mesh(str(HOPPER_STL), scale=STL_SCALE, device=config.device)
    sim.add_mesh(hopper_wp)
    plug_wp = load_mesh(str(PLUG_STL), scale=STL_SCALE, device=config.device)
    sim.add_mesh(plug_wp)

    # Also load via trimesh for rendering
    hopper_tri = trimesh.load(str(HOPPER_STL), force="mesh")
    hopper_verts = (np.asarray(hopper_tri.vertices, dtype=np.float32) * STL_SCALE)
    hopper_faces = np.asarray(hopper_tri.faces, dtype=np.int32)

    plug_tri = trimesh.load(str(PLUG_STL), force="mesh")
    plug_verts = (np.asarray(plug_tri.vertices, dtype=np.float32) * STL_SCALE)
    plug_faces = np.asarray(plug_tri.faces, dtype=np.int32)

    print(f"\nSimulation parameters:")
    print(f"  E_eff         : {sim.E_eff:.3e} Pa")
    print(f"  G_eff         : {sim.G_eff:.3e} Pa")
    print(f"  beta          : {sim.beta:.4f}")
    print(f"  max_radius    : {sim.max_radius * 1000:.0f} mm")
    print(f"  cell_size     : {sim.cell_size * 1000:.0f} mm")
    print(f"  dt            : {config.dt:.1e} s")
    print(f"  global_damping: {config.global_damping:.1f} 1/s")

    # ------------------------------------------------------------------
    # 5. Settle
    # ------------------------------------------------------------------
    steps_taken = run_settling(sim, config)
    sim_time = steps_taken * config.dt

    # Final diagnostics
    wp.synchronize()
    vel = sim.get_velocities()
    v_mag = np.linalg.norm(vel, axis=1)
    if not np.isfinite(v_mag).all():
        raise RuntimeError("Simulation diverged: non-finite velocities")
    print(f"\nFinal velocity : mean {v_mag.mean():.4f}  max {v_mag.max():.4f} m/s")
    final_ke = sim.get_kinetic_energy()
    print(f"Final KE       : {final_ke:.4e} J")

    final_pos = sim.get_positions()
    print(f"Final z range  : [{final_pos[:,2].min():.3f}, {final_pos[:,2].max():.3f}] m")

    # ------------------------------------------------------------------
    # 6. Render PyVista PNG
    # ------------------------------------------------------------------
    print(f"\nRendering PNG with PyVista ...")
    render_png(
        final_pos, radii,
        hopper_verts, hopper_faces,
        plug_verts, plug_faces,
        sim_time, OUTPUT_PNG,
    )
    png_kb = OUTPUT_PNG.stat().st_size / 1024
    print(f"  wrote {png_kb:.0f} KB -> {OUTPUT_PNG}")

    print("\nDone.")


if __name__ == "__main__":
    main()
