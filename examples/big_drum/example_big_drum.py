"""Big rotating drum — 1M+ particle scaling benchmark.

Demonstrates the throughput of the VeloxSim-DEM engine on a large granular
system: a cylindrical drum (R = 1.1 m, L = 3.0 m) with six rectangular
lifters spinning at 30 rpm, filled with ~1.1 million 10 mm particles.

Two-phase run with checkpoint/resume so the simulation can be inspected
mid-way.  Total sim time 4 s (80,000 timesteps at dt = 5e-5 s).

Usage
-----
Phase 1 — run the first 2 s, write checkpoint + viewer, then stop::

    python example_big_drum.py

Inspect ``big_drum_phase1.html`` in a browser.  When happy, continue::

    python example_big_drum.py --resume

On completion the script reports total wallclock time and peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp

from veloxsim_dem import (
    SimConfig,
    Simulation,
    create_drum_with_lifters_mesh,
    transform_mesh_vertices_kernel,
)
from hopper_viewer import generate_hopper_html

HERE            = pathlib.Path(__file__).resolve().parent
CHECKPOINT_NPZ  = HERE / "big_drum_checkpoint.npz"
PHASE1_JSON     = HERE / "big_drum_phase1.json"
PHASE1_HTML     = HERE / "big_drum_phase1.html"
PHASE2_JSON     = HERE / "big_drum_phase2.json"
PHASE2_HTML     = HERE / "big_drum_phase2.html"

# ── Drum geometry (big) ──────────────────────────────────────────────────────
DRUM_RADIUS         = 1.50      # m
DRUM_LENGTH         = 4.00      # m  (axis along Y)
DRUM_N_THETA        = 64        # circumferential segments
DRUM_N_LIFTERS      = 6
DRUM_LIFTER_H       = 0.12      # m  lifter protrusion
DRUM_LIFTER_HALFANG = 0.06      # rad  lifter angular half-width

# ── Particles ────────────────────────────────────────────────────────────────
PARTICLE_RADIUS  = 0.010    # m  (10 mm)
PARTICLE_DENSITY = 2500.0
PARTICLE_SPACING = 0.0305   # m  grid spacing — sparse so ~1M fit in 28 m³

# ── Kinematics ───────────────────────────────────────────────────────────────
RPM             = 15.0
OMEGA           = RPM * 2.0 * math.pi / 60.0   # rad/s

DT              = 5e-5
PHASE_1_END     = 2.0       # sim-time at which phase 1 stops
PHASE_2_END     = 4.0       # default sim-time at which phase 2 stops
                            # (overridable via --end-time on --resume)

# ── Output sampling ──────────────────────────────────────────────────────────
FRAME_STRIDE            = 1000   # steps between recorded frames (every 0.05 s)
VIEWER_PARTICLE_LIMIT   = 100000 # sub-sample for the HTML viewer JSON

PROGRESS_EVERY  = 1000          # print every N steps


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory reporting
# ─────────────────────────────────────────────────────────────────────────────
def query_gpu_memory_mb() -> tuple[float, float] | None:
    """Return (used_MB, total_MB) for GPU 0, or None if unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024**2, info.total / 1024**2
    except Exception:
        pass

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "--id=0"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        used_str, total_str = [x.strip() for x in out.split(",")]
        return float(used_str), float(total_str)
    except Exception:
        return None


def fmt_mem(used_mb: float, total_mb: float) -> str:
    return f"{used_mb:7.0f} / {total_mb:7.0f} MB  ({100*used_mb/total_mb:5.1f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# Grid particle placement
# ─────────────────────────────────────────────────────────────────────────────
def build_grid_positions(R: float, L: float, r: float, spacing: float,
                         n_lifters: int = 0,
                         lifter_height: float = 0.0,
                         lifter_half_angle: float = 0.0) -> np.ndarray:
    """Cubic-grid particle positions clipped to a cylinder (axis = Y), with
    lifter-wedge regions excluded.

    Cylinder volume: x^2 + z^2 < (R - 1.5r)^2, -L/2 + r < y < L/2 - r.

    Lifter exclusion (when n_lifters > 0): a particle is rejected if its
    centre falls inside ANY of the n_lifters wedges, defined as
        radial in (R - lifter_height - r,  R)
        angular within ±(lifter_half_angle + r/r_eval) of theta_k
    where theta_k = 2*pi*k/n_lifters.  The angular buffer accounts for the
    particle's own size at its current radius.
    """
    safe_r   = R - 1.5 * r
    safe_hy  = 0.5 * L - r

    nx = int(2 * safe_r / spacing) + 1
    ny = int(2 * safe_hy / spacing) + 1
    nz = int(2 * safe_r / spacing) + 1

    xs = np.linspace(-safe_r, safe_r, nx, dtype=np.float32)
    ys = np.linspace(-safe_hy, safe_hy, ny, dtype=np.float32)
    zs = np.linspace(-safe_r, safe_r, nz, dtype=np.float32)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Clip to cylinder radius
    radial = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)
    keep = radial < safe_r
    pts    = pts[keep]
    radial = radial[keep]

    # Lifter wedge exclusion
    if n_lifters > 0 and lifter_height > 0.0:
        r_inner = R - lifter_height - r          # inner edge of lifter band
        # only points in the outer band can be inside a lifter
        in_band = radial > r_inner
        if np.any(in_band):
            # angle of every band point
            theta_pt = np.arctan2(pts[in_band, 2], pts[in_band, 0])
            r_pt     = radial[in_band]
            # angular buffer scales as r_part / r so particle of radius r at
            # current radius r_pt has tangential half-extent r/r_pt
            ang_buf = lifter_half_angle + r / np.maximum(r_pt, 1e-3)

            inside_any_lifter = np.zeros_like(theta_pt, dtype=bool)
            for k in range(n_lifters):
                theta_k = 2.0 * math.pi * k / n_lifters
                d = np.abs(((theta_pt - theta_k + math.pi) % (2 * math.pi))
                           - math.pi)
                inside_any_lifter |= d < ang_buf

            # Build full-length keep mask
            band_idx = np.where(in_band)[0]
            keep_band = np.ones_like(in_band, dtype=bool)
            keep_band[band_idx[inside_any_lifter]] = False
            pts = pts[keep_band]

    # Small per-axis jitter to avoid perfect vertical stacks on step 0
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.1 * r, 0.1 * r, size=pts.shape).astype(np.float32)
    pts += jitter

    return pts.astype(np.float32)


def build_topup_positions(R: float, L: float, r: float, spacing: float,
                          n_extra: int,
                          n_lifters: int,
                          lifter_height: float,
                          lifter_half_angle: float) -> np.ndarray:
    """Place ``n_extra`` new particles in the top portion of the drum, far
    above the settled bed.  Same cylinder + lifter clipping as
    :func:`build_grid_positions`, then we sort by z descending and take the
    top ``n_extra`` candidates.
    """
    pts = build_grid_positions(R, L, r, spacing,
                               n_lifters=n_lifters,
                               lifter_height=lifter_height,
                               lifter_half_angle=lifter_half_angle)
    if pts.shape[0] < n_extra:
        raise RuntimeError(
            f"only {pts.shape[0]:,} candidate slots available; "
            f"cannot place {n_extra:,}")
    order = np.argsort(-pts[:, 2])           # descending vertical (top first)
    return pts[order[:n_extra]].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation construction (shared by both phases)
# ─────────────────────────────────────────────────────────────────────────────
def make_simulation(num_particles: int) -> Simulation:
    config = SimConfig(
        num_particles=num_particles,
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
        max_contacts_per_particle=16,     # smaller contact slots to save memory
        hash_grid_dim=128,
        global_damping=0.0,
        device="cuda:0",
    )
    return Simulation(config)


def add_drum(sim: Simulation, omega_rad_s: float) -> int:
    drum = create_drum_with_lifters_mesh(
        radius=DRUM_RADIUS,
        length=DRUM_LENGTH,
        n_theta=DRUM_N_THETA,
        n_lifters=DRUM_N_LIFTERS,
        lifter_height=DRUM_LIFTER_H,
        lifter_half_angle=DRUM_LIFTER_HALFANG,
        end_caps=True,
        device=sim.config.device,
    )
    sim.add_mesh(drum, angular_velocity=(0.0, omega_rad_s, 0.0),
                 origin=(0.0, 0.0, 0.0))
    return 0   # drum_idx


# ─────────────────────────────────────────────────────────────────────────────
# Viewer output (sub-sampled)
# ─────────────────────────────────────────────────────────────────────────────
def _body_to_stl_dict(body) -> dict:
    verts = body.rest_points_wp.numpy()
    return {
        "v": np.round(verts, 5).tolist(),
        "f": body.mesh.indices.numpy().tolist(),
    }


def _subsample_indices(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    stride = max(1, n // limit)
    return np.arange(0, n, stride, dtype=np.int64)[:limit]


def write_viewer(frames: list, sample_idx: np.ndarray, radii: np.ndarray,
                 num_total: int, drum_body, sim_time: float, phase: str,
                 json_path: pathlib.Path, html_path: pathlib.Path):
    radii_sampled = [round(float(radii[i]), 6) for i in sample_idx]
    payload = {
        "config": {
            "n_particles":   int(len(sample_idx)),
            "n_particles_total": int(num_total),
            "radius":        PARTICLE_RADIUS,
            "radii":         radii_sampled,
            "dt":            DT,
            "sim_time":      float(sim_time),
            "drum_radius":   DRUM_RADIUS,
            "drum_length":   DRUM_LENGTH,
            "n_lifters":     DRUM_N_LIFTERS,
            "lifter_height": DRUM_LIFTER_H,
            "rpm":           RPM,
            "description":
                f"Big rotating drum — {phase} — "
                f"{num_total} total particles, viewer sub-sampled "
                f"to {len(sample_idx)}.",
        },
        "stl":    {"drum": _body_to_stl_dict(drum_body)},
        "frames": frames,
    }
    print(f"  writing {json_path.name} ...")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_mb = json_path.stat().st_size / 1024**2
    print(f"    {size_mb:.1f} MB")
    print(f"  writing {html_path.name} ...")
    generate_hopper_html(
        str(json_path), str(html_path),
        title=f"VeloxSim-DEM — Big Drum ({RPM:.0f} rpm, "
              f"{num_total:,} particles)  [{phase}]",
        max_anim_frames=2000,
        max_particles_per_frame=len(sample_idx),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint IO
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(sim: Simulation, *, num_particles: int,
                    phase1_wallclock: float, drum_idx: int):
    """Save minimal state: particle kinematics + drum pose.  Contact history
    is NOT saved — the first step of phase 2 rebuilds it (one-step transient
    is negligible at 30 rpm)."""
    wp.synchronize()
    body = sim.meshes[drum_idx]
    np.savez_compressed(
        CHECKPOINT_NPZ,
        positions          = sim.get_positions().astype(np.float32),
        velocities         = sim.get_velocities().astype(np.float32),
        angular_velocities = sim.angular_velocities.numpy().astype(np.float32),
        radii              = sim.get_radii().astype(np.float32),
        sim_time           = np.array([sim.sim_time], dtype=np.float64),
        step_count         = np.array([sim.step_count], dtype=np.int64),
        phase1_wallclock   = np.array([phase1_wallclock], dtype=np.float64),
        num_particles      = np.array([num_particles], dtype=np.int64),
        drum_position      = body.position.astype(np.float64),
        drum_quaternion    = body.quaternion.astype(np.float64),
        drum_linear_vel    = np.array(body.linear_velocity, dtype=np.float64),
        drum_angular_vel   = np.array(body.angular_velocity, dtype=np.float64),
        drum_origin        = body.origin.astype(np.float64),
    )
    size_mb = CHECKPOINT_NPZ.stat().st_size / 1024**2
    print(f"  checkpoint written: {CHECKPOINT_NPZ.name}  ({size_mb:.1f} MB)")


def load_checkpoint(n_extra: int = 0) -> tuple[Simulation, dict]:
    print(f"Loading checkpoint {CHECKPOINT_NPZ.name} ...")
    if not CHECKPOINT_NPZ.exists():
        print(f"  ERROR: checkpoint not found — run phase 1 first "
              f"(python {pathlib.Path(__file__).name})")
        sys.exit(1)
    data = np.load(CHECKPOINT_NPZ)
    num_old          = int(data["num_particles"][0])
    sim_time         = float(data["sim_time"][0])
    step_count       = int(data["step_count"][0])
    phase1_wallclock = float(data["phase1_wallclock"][0])

    positions   = data["positions"]
    velocities  = data["velocities"]
    ang_vels    = data["angular_velocities"]

    if n_extra > 0:
        print(f"  adding {n_extra:,} top-up particles to existing {num_old:,} ...")
        topup = build_topup_positions(
            DRUM_RADIUS, DRUM_LENGTH, PARTICLE_RADIUS, PARTICLE_SPACING,
            n_extra,
            n_lifters         = DRUM_N_LIFTERS,
            lifter_height     = DRUM_LIFTER_H,
            lifter_half_angle = DRUM_LIFTER_HALFANG,
        )
        zeros      = np.zeros((n_extra, 3), dtype=np.float32)
        positions  = np.concatenate([positions,  topup],  axis=0)
        velocities = np.concatenate([velocities, zeros],  axis=0)
        ang_vels   = np.concatenate([ang_vels,   zeros],  axis=0)
        z_min = float(topup[:, 2].min())
        z_max = float(topup[:, 2].max())
        print(f"    top-up band: z in [{z_min:+.3f}, {z_max:+.3f}] m  "
              f"(drum top z = {DRUM_RADIUS:+.3f})")

    num_particles = num_old + n_extra

    sim = make_simulation(num_particles)
    sim.initialize_particles(
        positions,
        velocities         = velocities,
        angular_velocities = ang_vels,
    )
    sim.sim_time   = sim_time
    sim.step_count = step_count

    # Radii — engine already generated uniform radii matching config.
    # Skip overriding for simplicity.

    # Re-add drum and restore pose.  Kernel transforms rest vertices
    # by the restored quaternion/translation so the BVH matches visually.
    drum_idx = add_drum(sim, OMEGA)
    body = sim.meshes[drum_idx]
    body.position         = data["drum_position"].copy()
    body.quaternion       = data["drum_quaternion"].copy()
    body.linear_velocity  = list(data["drum_linear_vel"])
    body.angular_velocity = list(data["drum_angular_vel"])
    body.origin           = data["drum_origin"].copy()

    wp.launch(
        kernel=transform_mesh_vertices_kernel,
        dim=body.rest_points_wp.shape[0],
        inputs=[
            body.rest_points_wp,
            body.mesh.points,
            wp.vec3(*[float(x) for x in body.origin]),
            wp.quat(float(body.quaternion[0]), float(body.quaternion[1]),
                    float(body.quaternion[2]), float(body.quaternion[3])),
            wp.vec3(*[float(x) for x in body.position]),
        ],
        device=sim.device,
    )
    body.mesh.refit()
    wp.synchronize()

    print(f"  restored: N={num_particles:,}  "
          f"sim_time={sim_time:.3f}s  step={step_count}")
    print(f"  phase-1 wallclock: {phase1_wallclock:.1f} s")

    meta = {
        "num_particles":     num_particles,
        "num_old":           num_old,
        "num_added":         n_extra,
        "sim_time":          sim_time,
        "step_count":        step_count,
        "phase1_wallclock":  phase1_wallclock,
        "drum_idx":          drum_idx,
    }
    return sim, meta


# ─────────────────────────────────────────────────────────────────────────────
# Frame recording (sub-sampled)
# ─────────────────────────────────────────────────────────────────────────────
def record_frame(sim: Simulation, sample_idx: np.ndarray, frames: list):
    wp.synchronize()
    pos_all = sim.get_positions()
    vel_all = sim.get_velocities()
    pos = pos_all[sample_idx]
    speed = np.linalg.norm(vel_all[sample_idx], axis=1)
    frames.append({
        "t":          round(float(sim.sim_time), 5),
        "n":          int(len(sample_idx)),
        "pos":        np.round(pos, 4).tolist(),
        "s":          np.round(speed, 3).tolist(),
        "mesh_poses": sim.get_mesh_poses(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 driver
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1():
    print("=" * 72)
    print(" Big Drum benchmark — Phase 1 (0 -> 2 s)")
    print("=" * 72)

    # Baseline GPU memory before anything
    mem0 = query_gpu_memory_mb()
    if mem0:
        print(f"  baseline GPU memory:   {fmt_mem(*mem0)}")

    t_build = time.perf_counter()
    positions = build_grid_positions(
        DRUM_RADIUS, DRUM_LENGTH, PARTICLE_RADIUS, PARTICLE_SPACING,
        n_lifters=DRUM_N_LIFTERS,
        lifter_height=DRUM_LIFTER_H,
        lifter_half_angle=DRUM_LIFTER_HALFANG,
    )
    N = positions.shape[0]
    print(f"  grid positions built: N = {N:,}  "
          f"({time.perf_counter() - t_build:.1f} s)")

    sim = make_simulation(N)
    sim.initialize_particles(positions)
    drum_idx = add_drum(sim, OMEGA)

    # Warm-up: first step JITs kernels — time it separately so it doesn't
    # distort the steady-state rate
    print("  warming up (compile + first step) ...")
    t_warm = time.perf_counter()
    sim.step()
    wp.synchronize()
    t_warm_elapsed = time.perf_counter() - t_warm
    print(f"    warm-up:  {t_warm_elapsed:.1f} s")

    mem_after_init = query_gpu_memory_mb()
    if mem_after_init:
        print(f"  GPU memory after init: {fmt_mem(*mem_after_init)}")

    peak_mem = mem_after_init[0] if mem_after_init else 0.0

    sample_idx = _subsample_indices(N, VIEWER_PARTICLE_LIMIT)
    print(f"  viewer sub-sample: {len(sample_idx):,} particles "
          f"(stride {N // len(sample_idx)})")

    frames = []
    record_frame(sim, sample_idx, frames)   # frame 0

    t0 = time.perf_counter()
    last_print = t0
    total_steps = 0

    while sim.sim_time < PHASE_1_END - 0.5 * DT:
        sim.step()
        total_steps += 1

        if total_steps % FRAME_STRIDE == 0:
            record_frame(sim, sample_idx, frames)

        if total_steps % PROGRESS_EVERY == 0:
            wp.synchronize()
            now = time.perf_counter()
            rate = PROGRESS_EVERY / (now - last_print)
            pst  = rate * N
            last_print = now
            pct = 100 * sim.sim_time / PHASE_1_END
            mem = query_gpu_memory_mb()
            mem_str = fmt_mem(*mem) if mem else "n/a"
            if mem:
                peak_mem = max(peak_mem, mem[0])
            print(f"  step {total_steps:6d}  t={sim.sim_time:.3f}s "
                  f"({pct:5.1f}%)  {rate:6.0f} step/s  "
                  f"{pst/1e6:5.2f} MPart-step/s  mem={mem_str}")

    wp.synchronize()
    wall_phase1 = time.perf_counter() - t0
    record_frame(sim, sample_idx, frames)   # final frame

    mem_end = query_gpu_memory_mb()
    if mem_end:
        peak_mem = max(peak_mem, mem_end[0])

    print()
    print(f"  Phase 1 complete:")
    print(f"    sim time:           {sim.sim_time:.3f} s")
    print(f"    total steps:        {total_steps:,}")
    print(f"    wallclock:          {wall_phase1:.1f} s")
    print(f"    throughput:         {total_steps / wall_phase1:.0f} step/s")
    print(f"    particle-steps/s:   "
          f"{(total_steps * N) / wall_phase1 / 1e6:.2f}  Mpart-step/s")
    if mem_end:
        print(f"    peak GPU memory:    {peak_mem:.0f} MB")
        print(f"    final GPU memory:   {fmt_mem(*mem_end)}")

    print()
    print("  writing viewer + checkpoint ...")
    write_viewer(frames, sample_idx, sim.get_radii(), N,
                 sim.meshes[drum_idx], sim.sim_time, "Phase 1 (0 -> 2 s)",
                 PHASE1_JSON, PHASE1_HTML)

    save_checkpoint(sim, num_particles=N, phase1_wallclock=wall_phase1,
                    drum_idx=drum_idx)

    print()
    print("Phase 1 done.  Open the viewer:")
    print(f"  {PHASE1_HTML}")
    print()
    print("When you're happy, continue phase 2 with:")
    print(f"  python {pathlib.Path(__file__).name} --resume")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 driver (resume)
# ─────────────────────────────────────────────────────────────────────────────
def run_phase2(end_time: float = PHASE_2_END, n_extra: int = 0):
    print("=" * 72)
    print(f" Big Drum benchmark — Phase 2 ({PHASE_1_END:.1f} -> {end_time:.1f} s)")
    print("=" * 72)

    mem0 = query_gpu_memory_mb()
    if mem0:
        print(f"  baseline GPU memory:   {fmt_mem(*mem0)}")

    sim, meta = load_checkpoint(n_extra=n_extra)
    N                = meta["num_particles"]
    num_old          = meta["num_old"]
    num_added        = meta["num_added"]
    phase1_wallclock = meta["phase1_wallclock"]
    drum_idx         = meta["drum_idx"]

    # Warm-up after reload
    print("  warming up (recompile + first step) ...")
    t_warm = time.perf_counter()
    sim.step()
    wp.synchronize()
    print(f"    warm-up:  {time.perf_counter() - t_warm:.1f} s")

    mem_after_init = query_gpu_memory_mb()
    if mem_after_init:
        print(f"  GPU memory after init: {fmt_mem(*mem_after_init)}")

    peak_mem = mem_after_init[0] if mem_after_init else 0.0

    sample_idx = _subsample_indices(N, VIEWER_PARTICLE_LIMIT)
    frames = []
    record_frame(sim, sample_idx, frames)

    t0 = time.perf_counter()
    last_print = t0
    total_steps = 0

    while sim.sim_time < end_time - 0.5 * DT:
        sim.step()
        total_steps += 1

        if total_steps % FRAME_STRIDE == 0:
            record_frame(sim, sample_idx, frames)

        if total_steps % PROGRESS_EVERY == 0:
            wp.synchronize()
            now = time.perf_counter()
            rate = PROGRESS_EVERY / (now - last_print)
            pst  = rate * N
            last_print = now
            pct = 100 * (sim.sim_time - PHASE_1_END) / (end_time - PHASE_1_END)
            mem = query_gpu_memory_mb()
            mem_str = fmt_mem(*mem) if mem else "n/a"
            if mem:
                peak_mem = max(peak_mem, mem[0])
            print(f"  step {total_steps:6d}  t={sim.sim_time:.3f}s "
                  f"({pct:5.1f}%)  {rate:6.0f} step/s  "
                  f"{pst/1e6:5.2f} MPart-step/s  mem={mem_str}")

    wp.synchronize()
    wall_phase2 = time.perf_counter() - t0
    record_frame(sim, sample_idx, frames)

    mem_end = query_gpu_memory_mb()
    if mem_end:
        peak_mem = max(peak_mem, mem_end[0])

    total_wall = phase1_wallclock + wall_phase2
    total_steps_total = int(round(sim.sim_time / DT))

    print()
    print("=" * 72)
    print(" FINAL REPORT — Big Drum benchmark")
    print("=" * 72)
    if num_added > 0:
        print(f"  particles:           {N:,}  "
              f"({num_old:,} initial + {num_added:,} added at t={PHASE_1_END:.1f}s)")
    else:
        print(f"  particles:           {N:,}")
    print(f"  total sim time:      {sim.sim_time:.3f} s")
    print(f"  total steps:         {total_steps_total:,}")
    print(f"  phase 1 wallclock:   {phase1_wallclock:.1f} s")
    print(f"  phase 2 wallclock:   {wall_phase2:.1f} s")
    print(f"  TOTAL wallclock:     {total_wall:.1f} s  "
          f"({total_wall/60:.1f} min)")
    print(f"  throughput (phase 2):{total_steps / wall_phase2:.0f} step/s  "
          f"({(total_steps * N) / wall_phase2 / 1e6:.2f} Mpart-step/s)")
    if mem_end:
        print(f"  peak GPU memory:     {peak_mem:.0f} MB")
        print(f"  final GPU memory:    {fmt_mem(*mem_end)}")
    print("=" * 72)

    print()
    print("  writing phase-2 viewer ...")
    write_viewer(frames, sample_idx, sim.get_radii(), N,
                 sim.meshes[drum_idx], sim.sim_time,
                 f"Phase 2 ({PHASE_1_END:.1f} -> {end_time:.1f} s)",
                 PHASE2_JSON, PHASE2_HTML)

    print()
    print(f"Phase 2 done.  Viewer: {PHASE2_HTML}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the phase-1 checkpoint.")
    parser.add_argument("--end-time", type=float, default=PHASE_2_END,
                        help=f"Sim-time at which phase 2 stops "
                             f"(default {PHASE_2_END}).")
    parser.add_argument("--add-particles", type=int, default=0,
                        help="On --resume, inject N additional particles "
                             "at the top of the drum (default 0).")
    args = parser.parse_args()

    if args.resume:
        run_phase2(end_time=args.end_time, n_extra=args.add_particles)
    else:
        run_phase1()
