"""Render a static snapshot viewer from the phase-1 checkpoint.

Usage:
    python render_snapshot.py              # 100K particles (stride 11)
    python render_snapshot.py --stride N   # custom stride
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp

from veloxsim_dem import create_drum_with_lifters_mesh
from hopper_viewer import generate_hopper_html

HERE            = pathlib.Path(__file__).resolve().parent
CHECKPOINT_NPZ  = HERE / "big_drum_checkpoint.npz"

# Must match example_big_drum.py
DRUM_RADIUS     = 1.10
DRUM_LENGTH     = 3.00
DRUM_N_THETA    = 64
DRUM_N_LIFTERS  = 6
DRUM_LIFTER_H   = 0.12
PARTICLE_RADIUS = 0.010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=11,
                    help="Particle sub-sample stride (default 11 -> ~100K).")
    ap.add_argument("--out", type=str, default="big_drum_snapshot.html",
                    help="Output HTML filename.")
    args = ap.parse_args()

    print(f"Loading {CHECKPOINT_NPZ.name} ...")
    data = np.load(CHECKPOINT_NPZ)
    positions  = data["positions"]
    velocities = data["velocities"]
    quat       = data["drum_quaternion"]
    pos_mesh   = data["drum_position"]
    sim_time   = float(data["sim_time"][0])
    N          = positions.shape[0]

    idx = np.arange(0, N, args.stride, dtype=np.int64)
    pos_s   = positions[idx]
    speed_s = np.linalg.norm(velocities[idx], axis=1)
    print(f"  N_total = {N:,}   stride = {args.stride}   "
          f"N_sampled = {len(idx):,}")

    # Rebuild the drum mesh to extract rest-frame vertices
    drum = create_drum_with_lifters_mesh(
        radius=DRUM_RADIUS,
        length=DRUM_LENGTH,
        n_theta=DRUM_N_THETA,
        n_lifters=DRUM_N_LIFTERS,
        lifter_height=DRUM_LIFTER_H,
        end_caps=True,
        device="cuda:0",
    )
    verts  = drum.points.numpy()
    faces  = drum.indices.numpy()

    frame = {
        "t":   round(sim_time, 5),
        "n":   int(len(idx)),
        "pos": np.round(pos_s, 4).tolist(),
        "s":   np.round(speed_s, 3).tolist(),
        "mesh_poses": [{
            "pos":  pos_mesh.tolist(),
            "quat": quat.tolist(),
        }],
    }

    payload = {
        "config": {
            "n_particles":       int(len(idx)),
            "n_particles_total": int(N),
            "radius":        PARTICLE_RADIUS,
            "dt":            5e-5,
            "sim_time":      sim_time,
            "drum_radius":   DRUM_RADIUS,
            "drum_length":   DRUM_LENGTH,
            "n_lifters":     DRUM_N_LIFTERS,
            "lifter_height": DRUM_LIFTER_H,
            "description":
                f"Static snapshot at t={sim_time:.2f}s  "
                f"({N:,} total, {len(idx):,} rendered).",
        },
        "stl":    {"drum": {
            "v": np.round(verts, 5).tolist(),
            "f": faces.tolist(),
        }},
        "frames": [frame],
    }

    json_path = HERE / (pathlib.Path(args.out).stem + ".json")
    html_path = HERE / args.out

    print(f"  writing {json_path.name} ...")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_mb = json_path.stat().st_size / 1024**2
    print(f"    {size_mb:.1f} MB")

    print(f"  writing {html_path.name} ...")
    generate_hopper_html(
        str(json_path), str(html_path),
        title=f"VeloxSim-DEM — Big Drum snapshot  "
              f"({N:,} total, {len(idx):,} rendered)",
        max_anim_frames=1,
        max_particles_per_frame=len(idx),
    )
    print(f"\nOpen: {html_path}")


if __name__ == "__main__":
    main()
