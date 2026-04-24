"""Rotating drum — Phase 2 kinematic rotation validation.

A horizontal cylindrical drum (axis along Y) rotates at a prescribed angular
velocity about its central axis.  Particles are first settled inside the drum
under gravity; the drum then spins and the particles are tumbled.

Physics validated
-----------------
* Correct rotational contact arm: the wall velocity at the contact point is
  ``v_wall = omega x (p_contact - origin)``.  If the arm is wrong the
  particles do not accelerate tangentially (they just rattle against a
  "stationary" wall even though the BVH geometry is rotating).
* Contact history rotation: stored tangent/roll spring displacements are
  rotated by the step quaternion each step.  Without this, friction forces
  develop a sawtooth artefact once the drum has turned more than a few
  degrees from the frame in which the displacements were computed.
* BVH refit: the transform kernel updates all vertices from rest-frame
  positions each step, then refit() updates the BVH.  Particles must not
  fall through the drum wall.

Operating parameters
--------------------
Drum radius R = 0.15 m, length L = 0.30 m (axis along Y).
Default omega = 3 rad/s (~29 rpm) — well below the critical speed at which
particles centrifuge to the wall (~8 rad/s for r_particle = 0.01 m).
Tunnelling criterion: v_wall * dt < r_particle
    => 0.15 * 3 * 5e-5 = 2.25e-5 m  <<  0.01 m  (safe)

Usage
-----
Full run (produces JSON + HTML viewer + PNG summary)::

    python example_rotating_drum.py

Custom RPM::

    python example_rotating_drum.py --rpm 20

CI acceptance test::

    python example_rotating_drum.py --ci
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
    create_cylinder_mesh,
)
from hopper_viewer import generate_hopper_html

HERE        = pathlib.Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "rotating_drum.json"
OUTPUT_HTML = HERE / "rotating_drum.html"
OUTPUT_PNG  = HERE / "rotating_drum.png"

# ── Simulation parameters ────────────────────────────────────────────────────
DRUM_RADIUS      = 0.15      # m
DRUM_LENGTH      = 0.30      # m
DRUM_N_THETA     = 48        # circumferential segments

PARTICLE_RADIUS  = 0.010     # m  (10 mm)
PARTICLE_DENSITY = 2500.0    # kg/m³
N_PARTICLES      = 60

DEFAULT_RPM      = 30.0      # rev/min (≈ 3.14 rad/s)

DT               = 5e-5      # s

SETTLE_STEPS     = 2000      # steps with drum stationary
SPIN_STEPS       = 3000      # steps with drum rotating

FRAME_STRIDE     = 50        # steps between viewer frames

# CI tolerances
CI_SPIN_STEPS    = 2000      # shorter run for CI
CI_KE_RATIO_MIN  = 1.5       # KE after / KE before must exceed this


def _pack_positions_inside_drum(n, r_particle, drum_radius, drum_length, rng):
    """Randomly place n particles in a cylinder (axis = Y, centre = origin)."""
    drum_half = drum_length * 0.5 - r_particle
    safe_r    = drum_radius - r_particle * 1.5
    positions = []
    attempts  = 0
    while len(positions) < n and attempts < n * 500:
        attempts += 1
        # uniform in disk using rejection
        rx = rng.uniform(-safe_r, safe_r)
        rz = rng.uniform(-safe_r, safe_r)
        if math.sqrt(rx**2 + rz**2) > safe_r:
            continue
        ry = rng.uniform(-drum_half, drum_half)
        # overlap check
        ok = True
        for p in positions:
            if math.dist([rx, ry, rz], p) < r_particle * 2.2:
                ok = False
                break
        if ok:
            positions.append([rx, ry, rz])
    if len(positions) < n:
        raise RuntimeError(
            f"Only placed {len(positions)}/{n} particles — "
            "try fewer particles or a larger drum."
        )
    return np.array(positions, dtype=np.float32)


def run(rpm: float = DEFAULT_RPM, ci: bool = False):
    omega_rad_s = rpm * 2.0 * math.pi / 60.0   # rad/s, about Y axis

    spin_steps = CI_SPIN_STEPS if ci else SPIN_STEPS

    config = SimConfig(
        num_particles=N_PARTICLES,
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
        global_damping=2.0,   # speed settle; turned off during spin validation
        device="cuda:0",
    )
    sim = Simulation(config)

    rng = np.random.default_rng(42)
    init_pos = _pack_positions_inside_drum(
        N_PARTICLES, PARTICLE_RADIUS, DRUM_RADIUS, DRUM_LENGTH, rng
    )
    sim.initialize_particles(init_pos)

    # Drum mesh — created centred at world origin; rotation origin = (0,0,0)
    drum = create_cylinder_mesh(
        radius=DRUM_RADIUS,
        length=DRUM_LENGTH,
        n_theta=DRUM_N_THETA,
        end_caps=True,
        device=config.device,
    )
    drum_idx = 0
    sim.add_mesh(drum)   # stationary during settle

    # ── Phase 1: settle ───────────────────────────────────────────────────────
    print(f"Settling {SETTLE_STEPS} steps …")
    t0 = time.perf_counter()
    for _ in range(SETTLE_STEPS):
        sim.step()
    wp.synchronize()
    ke_settled = sim.get_kinetic_energy()
    settle_wall = time.perf_counter() - t0
    print(f"  done in {settle_wall:.1f}s  |  KE after settle = {ke_settled:.4e} J")

    # Turn off global damping before spin so we can measure proper KE increase
    sim.config.global_damping = 0.0

    # ── Phase 2: start drum rotation ─────────────────────────────────────────
    sim.set_mesh_angular_velocity(drum_idx, (0.0, omega_rad_s, 0.0), origin=(0.0, 0.0, 0.0))
    print(f"Spinning drum at {rpm:.1f} rpm  ({omega_rad_s:.3f} rad/s) "
          f"for {spin_steps} steps …")

    frames   = []
    ke_log   = []
    t_log    = []

    def record():
        wp.synchronize()
        pos   = sim.get_positions()
        vel   = sim.get_velocities()
        speed = np.linalg.norm(vel, axis=1)
        frames.append({
            "t":          round(float(sim.sim_time), 5),
            "n":          N_PARTICLES,
            "pos":        np.round(pos, 4).tolist(),
            "s":          np.round(speed, 4).tolist(),
            "mesh_poses": sim.get_mesh_poses(),
        })

    record()
    t0 = time.perf_counter()
    for step in range(1, spin_steps + 1):
        sim.step()
        if step % FRAME_STRIDE == 0:
            record()
            ke_log.append(sim.get_kinetic_energy())
            t_log.append(float(sim.sim_time))

    wp.synchronize()
    spin_wall = time.perf_counter() - t0
    ke_final  = sim.get_kinetic_energy()
    print(f"  done in {spin_wall:.1f}s  |  {len(frames)} frames  "
          f"|  KE final = {ke_final:.4e} J")

    # ── CI assertions ─────────────────────────────────────────────────────────
    ke_ratio = ke_final / (ke_settled + 1e-12)
    print(f"  KE ratio (final / settled) = {ke_ratio:.2f}  "
          f"(threshold {CI_KE_RATIO_MIN})")

    # Tunnelling check: no particle should be outside the drum radius
    wp.synchronize()
    pos = sim.get_positions()
    radial = np.sqrt(pos[:, 0]**2 + pos[:, 2]**2)
    max_radial = float(np.max(radial))
    escaped = int(np.sum(radial > DRUM_RADIUS + PARTICLE_RADIUS * 0.5))
    print(f"  Max radial position = {max_radial:.4f} m  "
          f"(drum radius = {DRUM_RADIUS} m)  |  escaped = {escaped}")

    if ci:
        ok = True
        if ke_ratio < CI_KE_RATIO_MIN:
            print(f"CI: FAIL — KE ratio {ke_ratio:.2f} < threshold {CI_KE_RATIO_MIN}")
            ok = False
        if escaped > 0:
            print(f"CI: FAIL — {escaped} particle(s) escaped drum (tunnelling)")
            ok = False
        if ok:
            print("CI: PASS")
            sys.exit(0)
        else:
            sys.exit(1)

    # ── Write JSON + HTML viewer ──────────────────────────────────────────────
    radii_np = sim.get_radii()
    payload = {
        "config": {
            "n_particles":  N_PARTICLES,
            "radius":       PARTICLE_RADIUS,
            "radii":        [round(float(r), 6) for r in radii_np],
            "dt":           DT,
            "sim_time":     float(sim.sim_time),
            "drum_radius":  DRUM_RADIUS,
            "drum_length":  DRUM_LENGTH,
            "rpm":          rpm,
            "omega_rad_s":  omega_rad_s,
            "description":  "Rotating drum — Phase 2 kinematic rotation",
        },
        "stl":    {"drum": _mesh_to_stl_dict(drum)},
        "frames": frames,
    }

    print(f"\nWriting {OUTPUT_JSON.name} …")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Generating {OUTPUT_HTML.name} …")
    generate_hopper_html(
        str(OUTPUT_JSON), str(OUTPUT_HTML),
        title=f"VeloxSim-DEM — Rotating Drum ({rpm:.0f} rpm)",
        max_anim_frames=400,
        max_particles_per_frame=N_PARTICLES,
    )

    _render_png(pos, radii_np, ke_log, t_log, ke_settled, rpm)
    print(f"Done.  Open {OUTPUT_HTML} in a browser to watch the drum spin.")


def _mesh_to_stl_dict(mesh) -> dict:
    verts = mesh.points.numpy()
    return {"v": np.round(verts, 5).tolist(), "f": mesh.indices.numpy().tolist()}


def _render_png(final_pos, radii, ke_log, t_log, ke_settled, rpm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
    except ImportError:
        print("matplotlib not available — skipping PNG")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: XZ scatter of final particle positions vs drum outline
    ax = axes[0]
    theta_plt = np.linspace(0, 2 * math.pi, 200)
    ax.plot(DRUM_RADIUS * np.cos(theta_plt), DRUM_RADIUS * np.sin(theta_plt),
            "b-", linewidth=2, label="Drum wall")
    colors = plt.cm.viridis(np.linspace(0, 1, len(final_pos)))
    for pos_i, c in zip(final_pos, colors):
        circle = mp.Circle((pos_i[0], pos_i[2]), radii[0], color=c, alpha=0.8)
        ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-DRUM_RADIUS * 1.2, DRUM_RADIUS * 1.2)
    ax.set_ylim(-DRUM_RADIUS * 1.2, DRUM_RADIUS * 1.2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Final particle positions (XZ cross-section)\n{rpm:.0f} rpm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: KE vs time during spin phase
    ax2 = axes[1]
    ax2.plot(t_log, ke_log, "r-", linewidth=1.5, label="Kinetic energy")
    ax2.axhline(ke_settled, color="gray", linestyle="--", linewidth=1,
                label=f"Settled KE = {ke_settled:.2e} J")
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_ylabel("Kinetic energy (J)")
    ax2.set_title("Particle KE during drum spin")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Rotating Drum  |  R = {DRUM_RADIUS*100:.0f} cm  |  "
        f"{rpm:.0f} rpm  |  {N_PARTICLES} particles",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PNG), dpi=150)
    plt.close(fig)
    print(f"  wrote {OUTPUT_PNG.stat().st_size / 1024:.0f} KB -> {OUTPUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rpm", type=float, default=DEFAULT_RPM,
                        help=f"Drum rotation speed in RPM (default {DEFAULT_RPM})")
    parser.add_argument("--ci", action="store_true",
                        help="CI acceptance test — exits 0 on pass, 1 on fail")
    args = parser.parse_args()
    run(rpm=args.rpm, ci=args.ci)
