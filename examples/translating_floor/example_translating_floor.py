"""Translating floor — Phase 1 geometry dynamics demo.

A horizontal floor rises at a constant velocity.  A particle rests on it.
Once the floor starts moving the particle is carried upward by contact forces;
in steady state the particle velocity matches the floor velocity exactly (zero
relative slip in the normal direction).

Physics validated
-----------------
* BVH refit: if the BVH is not updated after the vertices move, the particle
  falls through the rising floor.
* surface_velocity promotion: the contact friction kernel receives the mesh's
  linear_velocity as the wall velocity so that tangential friction is computed
  relative to the moving surface.

Usage
-----
Run normally (produces PNG + HTML viewer)::

    python example_translating_floor.py

CI acceptance test (quick assertion, no output files)::

    python example_translating_floor.py --ci

Viewer
------
Open ``translating_floor.html`` in a browser.  The blue semi-transparent floor
visibly rises frame-by-frame; the particle (red sphere) rides it upward
maintaining contact.  Scrub to any frame to confirm geometry and particle
positions agree.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp

from veloxsim_dem import (
    SimConfig,
    Simulation,
    create_rect_mesh,
)
from hopper_viewer import generate_hopper_html

HERE       = pathlib.Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "translating_floor.json"
OUTPUT_HTML = HERE / "translating_floor.html"
OUTPUT_PNG  = HERE / "translating_floor.png"

# ── Simulation parameters ────────────────────────────────────────────────────
PARTICLE_RADIUS  = 0.01       # m  (10 mm)
PARTICLE_DENSITY = 2500.0     # kg/m³
FLOOR_Z_INITIAL  = 0.0        # m  floor starts at z = 0
FLOOR_VELOCITY   = 0.10       # m/s  floor rises in +z
FLOOR_SIZE       = 0.5        # m half-width of square floor mesh
DT               = 5e-5       # s

SETTLE_STEPS     = 400        # let particle come to rest before floor moves
MOTION_STEPS     = 600        # steps with floor moving
FRAME_STRIDE     = 20         # steps between recorded frames

# CI acceptance: after MOTION_STEPS, particle z-velocity must match floor
CI_STEP_CHECK    = 400        # step (within MOTION_STEPS) to check convergence
CI_TOL           = 0.005      # m/s  |v_particle_z - v_floor| < this


def run(ci: bool = False):
    config = SimConfig(
        num_particles=1,
        particle_radius=PARTICLE_RADIUS,
        particle_density=PARTICLE_DENSITY,
        young_modulus=1e7,
        poisson_ratio=0.3,
        restitution=0.3,
        friction_static=0.5,
        friction_dynamic=0.4,
        friction_rolling=0.01,
        dt=DT,
        gravity=(0.0, 0.0, -9.81),
        global_damping=2.0,   # speed settling
        device="cuda:0",
    )
    sim = Simulation(config)

    # Particle starts just above the floor
    gap = PARTICLE_RADIUS * 1.05
    init_pos = np.array([[0.0, 0.0, FLOOR_Z_INITIAL + gap]], dtype=np.float32)
    sim.initialize_particles(init_pos)

    # Floor mesh (two triangles, horizontal, facing +z)
    floor = create_rect_mesh(
        -FLOOR_SIZE, FLOOR_SIZE,
        -FLOOR_SIZE, FLOOR_SIZE,
        z=FLOOR_Z_INITIAL,
        device=config.device,
    )
    floor_mesh_idx = 0
    sim.add_mesh(floor, linear_velocity=(0.0, 0.0, 0.0))  # static during settle

    frames = []

    def record():
        wp.synchronize()
        pos   = sim.get_positions()
        vel   = sim.get_velocities()
        speed = np.linalg.norm(vel, axis=1)
        frames.append({
            "t":          round(float(sim.sim_time), 5),
            "n":          1,
            "pos":        np.round(pos, 4).tolist(),
            "s":          np.round(speed, 4).tolist(),
            "mesh_poses": sim.get_mesh_poses(),
        })

    # ── Phase 1: settle ──────────────────────────────────────────────────────
    print(f"Settling {SETTLE_STEPS} steps …")
    record()
    for _ in range(SETTLE_STEPS):
        sim.step()
    wp.synchronize()
    print(f"  settled:  z={sim.get_positions()[0, 2]:.4f} m  "
          f"vz={sim.get_velocities()[0, 2]:.4f} m/s")

    # ── Phase 2: floor starts rising ─────────────────────────────────────────
    sim.set_mesh_velocity(floor_mesh_idx, (0.0, 0.0, FLOOR_VELOCITY))
    record()
    print(f"Floor moving at vz = {FLOOR_VELOCITY} m/s for {MOTION_STEPS} steps …")

    ci_passed = None
    t0 = time.perf_counter()
    for step in range(1, MOTION_STEPS + 1):
        sim.step()
        if step % FRAME_STRIDE == 0:
            record()

        # CI check: converged velocity
        if ci and step == CI_STEP_CHECK:
            wp.synchronize()
            vz_particle = float(sim.get_velocities()[0, 2])
            err = abs(vz_particle - FLOOR_VELOCITY)
            ci_passed = err < CI_TOL
            print(f"  CI check @ step {step}:  "
                  f"vz_particle={vz_particle:.4f}  vz_floor={FLOOR_VELOCITY}  "
                  f"err={err:.4f}  {'PASS' if ci_passed else 'FAIL'}")

    wp.synchronize()
    wall = time.perf_counter() - t0
    final_pos = sim.get_positions()[0]
    final_vel = sim.get_velocities()[0]
    floor_z   = sim.get_mesh_poses()[0]["pos"][2]
    print(f"  done in {wall:.1f}s  |  particle z={final_pos[2]:.4f}  "
          f"vz={final_vel[2]:.4f}  |  floor z={floor_z:.4f}")

    if ci:
        if ci_passed:
            print("CI: PASS")
            sys.exit(0)
        else:
            print("CI: FAIL — particle velocity did not converge to floor velocity")
            sys.exit(1)

    # ── Output ───────────────────────────────────────────────────────────────
    radii = sim.get_radii().tolist()
    payload = {
        "config": {
            "n_particles":   1,
            "radius":        PARTICLE_RADIUS,
            "radii":         [round(float(r), 6) for r in radii],
            "dt":            DT,
            "sim_time":      float(sim.sim_time),
            "floor_velocity": FLOOR_VELOCITY,
            "description":   "Particle riding a rising floor — Phase 1 geometry dynamics",
        },
        "stl": {
            "floor": _mesh_to_stl_dict(floor),
        },
        "frames": frames,
    }
    print(f"\nWriting {OUTPUT_JSON.name} …")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  {OUTPUT_JSON.stat().st_size / 1024:.0f} KB")

    print(f"Generating {OUTPUT_HTML.name} …")
    generate_hopper_html(
        str(OUTPUT_JSON),
        str(OUTPUT_HTML),
        title="VeloxSim-DEM — Translating Floor (Phase 1 dynamics)",
        max_anim_frames=300,
        max_particles_per_frame=1,
    )

    _render_png(frames, floor_z_initial=FLOOR_Z_INITIAL,
                floor_size=FLOOR_SIZE, floor_velocity=FLOOR_VELOCITY,
                settle_steps=SETTLE_STEPS)
    print(f"Done.  Open {OUTPUT_HTML} in a browser to see the floor rise.")


def _mesh_to_stl_dict(mesh: "wp.Mesh") -> dict:
    """Extract vertices and faces from a Warp Mesh for the viewer payload."""
    verts = mesh.points.numpy()
    # indices array is flat: 3 ints per triangle
    flat_idx = mesh.indices.numpy()
    return {
        "v": np.round(verts, 5).tolist(),
        "f": flat_idx.tolist(),
    }


def _render_png(frames, *, floor_z_initial, floor_size, floor_velocity, settle_steps):
    """Three-panel PyVista PNG: settled, mid-rise, final."""
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not available — skipping PNG render")
        return

    # Build floor quad at origin (mesh_poses displacement added per panel)
    hs = floor_size
    floor_verts = np.array([
        [-hs, -hs, floor_z_initial],
        [ hs, -hs, floor_z_initial],
        [ hs,  hs, floor_z_initial],
        [-hs,  hs, floor_z_initial],
    ], dtype=np.float32)
    floor_poly = pv.PolyData(floor_verts,
                              np.array([4, 0, 1, 2, 3], dtype=np.int32))

    r = PARTICLE_RADIUS
    sphere = pv.Sphere(radius=r, theta_resolution=24, phi_resolution=16)

    # Pick three representative frames
    n = len(frames)
    indices = [0, n // 2, n - 1]
    labels  = [
        "Settled (floor static)",
        f"Floor mid-rise (t = {frames[n // 2]['t']:.2f} s)",
        f"Final (t = {frames[-1]['t']:.2f} s)",
    ]

    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"
    plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 600), off_screen=True)

    cam_pos = (
        (0.6, -0.6, 0.5),
        (0.0, 0.0, 0.15),
        (0.0, 0.0, 1.0),
    )

    for col, (fi, label) in enumerate(zip(indices, labels)):
        plotter.subplot(0, col)
        fr = frames[fi]

        # Floor displaced by mesh_pose
        dz = fr["mesh_poses"][0]["pos"][2]
        shifted = floor_verts.copy()
        shifted[:, 2] += dz
        fp = pv.PolyData(shifted, np.array([4, 0, 1, 2, 3], dtype=np.int32))
        plotter.add_mesh(fp, color="#3b82f6", opacity=0.35,
                         show_edges=True, edge_color="#1e40af", line_width=1.5)

        # Particle
        pos = np.array(fr["pos"][0], dtype=np.float32)
        pdata = pv.PolyData(pos.reshape(1, 3))
        glyphs = pdata.glyph(geom=sphere, scale=False, orient=False)
        plotter.add_mesh(glyphs, color="#ef4444", smooth_shading=True,
                         specular=0.5, specular_power=20, ambient=0.3)

        plotter.add_text(label, position="upper_edge", font_size=10, color="black")
        plotter.camera_position = cam_pos

    plotter.add_text(
        f"Translating floor  |  v_floor = {floor_velocity} m/s  |  "
        f"particle rides floor in steady state",
        position="lower_edge", font_size=8, color="#334155",
        viewport=True,
    )
    plotter.screenshot(str(OUTPUT_PNG), transparent_background=False,
                       window_size=(1500, 600))
    plotter.close()
    print(f"  wrote {OUTPUT_PNG.stat().st_size / 1024:.0f} KB -> {OUTPUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ci", action="store_true",
                        help="Run CI acceptance test (no output files, exits 0/1)")
    args = parser.parse_args()
    run(ci=args.ci)
