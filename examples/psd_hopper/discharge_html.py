"""HTML animation of the hopper discharge.

Loads the settled state, runs the discharge, captures ~80 frames into JSON
in the format consumed by hopper_viewer.py, and generates a self-contained
interactive HTML viewer (Three.js — orbit, zoom, scrub, layer colouring).

Runs end-to-end in well under a minute (no per-frame rendering cost — the
browser renders).

Outputs:
    psd_hopper_discharge.json  (frame data + mesh)
    psd_hopper_discharge.html  (self-contained viewer)
"""

import json
import math
import pathlib
import shutil
import sys
import time

# Make the repo root (where veloxsim_dem.py and hopper_viewer.py live) importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import trimesh
import warp as wp

from veloxsim_dem import (
    ParticleSizeDistribution,
    SimConfig,
    Simulation,
    create_plane_mesh,
    load_mesh,
)
from hopper_viewer import generate_hopper_html

HERE         = pathlib.Path(__file__).resolve().parent
STL_DIR      = HERE / "STL"
HOPPER_STL   = STL_DIR / "Hopper2.stl"
SETTLED_NPZ  = HERE / "settled_state.npz"
OUTPUT_JSON  = HERE / "psd_hopper_discharge.json"
OUTPUT_HTML  = HERE / "psd_hopper_discharge.html"
STL_SCALE    = 0.001

PARTICLE_DENSITY = 2500.0
PSD_RADII        = [0.035, 0.060, 0.100]

DT             = 1.0e-4
DISCHARGE_TIME = 15.0
GLOBAL_DAMPING = 0.0
Z_FLOOR        = -2.5
FLOOR_SIZE     = 6.0

# Capture at 15 fps over 15 s -> ~225 frames (HTML viewer caps at 200)
FRAME_STRIDE   = 667                            # steps between frames (dt=1e-4 -> ~15 fps)


def count_pcts_from_radii(radii):
    total = len(radii)
    pcts = [100.0 * int(np.sum(np.isclose(radii, r, atol=1e-5))) / total
            for r in PSD_RADII]
    pcts[-1] = 100.0 - sum(pcts[:-1])
    return pcts


def remap_by_class(saved_pos, saved_rad, new_rad):
    out = np.zeros_like(saved_pos)
    for r in PSD_RADII:
        oi = np.where(np.isclose(saved_rad, r, atol=1e-5))[0]
        ni = np.where(np.isclose(new_rad,   r, atol=1e-5))[0]
        out[ni] = saved_pos[oi]
    return out


def main():
    print("=" * 70)
    print(" PSD hopper  -  DISCHARGE  -  HTML viewer")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load settled state
    # ------------------------------------------------------------------
    if not SETTLED_NPZ.exists():
        raise FileNotFoundError(SETTLED_NPZ)
    data = np.load(SETTLED_NPZ)
    saved_pos, saved_rad = data["positions"], data["radii"]
    total_N = len(saved_pos)

    psd_pcts = count_pcts_from_radii(saved_rad)
    psd = ParticleSizeDistribution(list(zip(PSD_RADII, psd_pcts)))

    config = SimConfig(
        num_particles=total_N, psd=psd,
        particle_density=PARTICLE_DENSITY, young_modulus=1.0e7,
        poisson_ratio=0.3, restitution=0.5,
        friction_static=0.5, friction_dynamic=0.4, friction_rolling=0.02,
        cohesion_energy=0.0, dt=DT, gravity=(0.0, 0.0, -9.81),
        max_contacts_per_particle=32, hash_grid_dim=128,
        global_damping=GLOBAL_DAMPING,
    )
    sim = Simulation(config)
    new_rad = sim.get_radii()
    positions = remap_by_class(saved_pos, saved_rad, new_rad)
    sim.initialize_particles(positions)

    # Hopper (open) + floor
    sim.add_mesh(load_mesh(str(HOPPER_STL), scale=STL_SCALE, device=config.device))
    sim.add_mesh(create_plane_mesh(
        origin=(0, 0, Z_FLOOR), normal=(0, 0, 1),
        size=FLOOR_SIZE, device=config.device,
    ))

    # Trimesh copy for the viewer payload
    tri_h = trimesh.load(str(HOPPER_STL), force="mesh")
    hopper_verts = (np.asarray(tri_h.vertices, dtype=np.float32) * STL_SCALE).tolist()
    hopper_faces = np.asarray(tri_h.faces, dtype=np.int32).flatten().tolist()

    # ------------------------------------------------------------------
    # Run discharge and record JSON frames
    # ------------------------------------------------------------------
    n_steps = int(DISCHARGE_TIME / DT)
    frames = []

    def _dump_frame():
        pos = sim.get_positions()
        vel = sim.get_velocities()
        speed = np.linalg.norm(vel, axis=1)
        frames.append({
            "t":   round(float(sim.sim_time), 4),
            "n":   total_N,
            "pos": np.round(pos, 3).tolist(),
            "s":   np.round(speed, 2).tolist(),
        })

    _dump_frame()   # t=0

    print(f"Running discharge ({DISCHARGE_TIME}s, dt={DT}, "
          f"frame every {FRAME_STRIDE} steps -> ~{n_steps // FRAME_STRIDE} frames)")
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step()
        if step % FRAME_STRIDE == 0:
            wp.synchronize()
            _dump_frame()
            if len(frames) % 10 == 0:
                last = np.asarray(frames[-1]["pos"])
                in_h = int(np.sum(last[:, 2] > 0.0))
                out  = int(np.sum(last[:, 2] < 0.0))
                print(f"  frame {len(frames):>3d}  t={sim.sim_time:.2f}s  "
                      f"in={in_h:,d}  out={out:,d}")
    wp.synchronize()
    wall = time.perf_counter() - t0
    print(f"  sim wall clock: {wall:.1f}s  ({len(frames)} frames)")

    # ------------------------------------------------------------------
    # Write JSON payload for hopper_viewer.py
    # ------------------------------------------------------------------
    payload = {
        "config": {
            "n_particles":    total_N,
            "radius":         float(psd.max_radius),
            "radii":          [round(float(r), 5) for r in new_rad],
            "psd":            [[float(r), float(p)] for r, p in psd.size_classes],
            "solid_density":  PARTICLE_DENSITY,
            "young_modulus":  config.young_modulus,
            "dt":             DT,
            "sim_time":       DISCHARGE_TIME,
            "friction_static":  config.friction_static,
            "friction_dynamic": config.friction_dynamic,
            "friction_rolling": config.friction_rolling,
            "restitution":    config.restitution,
            "phase":          "discharge",
        },
        "stl":   {"hopper": {"v": hopper_verts, "f": hopper_faces}},
        "frames": frames,
    }

    print(f"\nWriting JSON -> {OUTPUT_JSON.name}")
    t0 = time.perf_counter()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    mb = OUTPUT_JSON.stat().st_size / 1024 / 1024
    print(f"  wrote {mb:.1f} MB in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Generate self-contained HTML viewer
    # ------------------------------------------------------------------
    print(f"\nRendering HTML -> {OUTPUT_HTML.name}")
    generate_hopper_html(
        str(OUTPUT_JSON),
        str(OUTPUT_HTML),
        title=(f"VeloxSim-DEM - Hopper2 DISCHARGE - "
               f"PSD 35/60/100 mm at 40/30/30 vol%  ({total_N:,d} particles)"),
        max_anim_frames=200,
        max_particles_per_frame=total_N + 1,      # keep all particles
    )
    kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"  wrote {kb:.0f} KB -> {OUTPUT_HTML}")

    print("\nDone.")


if __name__ == "__main__":
    main()
