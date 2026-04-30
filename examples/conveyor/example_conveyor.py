"""Conveyor belt — Coulomb friction ramp validation (Phase 1 dynamics).

A horizontal floor translates at a constant velocity in the X direction.  A
particle is dropped from rest above it.  Kinetic friction accelerates the
particle until its horizontal velocity equals the belt speed, then static
friction holds it at that plateau.

Physics validated
-----------------
* ``v_rel = v_particle - v_wall`` direction and magnitude: if the sign is wrong
  or surface_velocity is not updated, the ramp slope is wrong or the particle
  slides backward.
* Coulomb ramp slope: the measured acceleration during the kinetic phase must
  match ``mu_d * g`` within tolerance.
* Plateau at belt speed: once ``v_x ≈ v_belt``, acceleration drops to zero.

Analytical reference
--------------------
During kinetic slip::

    a_x = mu_d * g  (horizontal Coulomb friction)
    v_x(t) = mu_d * g * t        (from rest at first contact)

Expected slope with mu_d = 0.4:  3.924 m/s²

Usage
-----
Normal run (produces PNG plot + HTML viewer)::

    python example_conveyor.py

CI acceptance test::

    python example_conveyor.py --ci

Viewer
------
Open ``conveyor.html``.  The blue floor mesh translates rightward (+X).
The particle (red sphere) begins stationary, accelerates to match floor speed,
then rides the belt.  Switch to Velocity colour mode to watch speed increase.
"""

from __future__ import annotations

import argparse
import json
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

HERE        = pathlib.Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "conveyor.json"
OUTPUT_HTML = HERE / "conveyor.html"
OUTPUT_PNG  = HERE / "conveyor_friction_ramp.png"

# ── Parameters ───────────────────────────────────────────────────────────────
PARTICLE_RADIUS  = 0.01       # m
PARTICLE_DENSITY = 2500.0     # kg/m³
MU_S             = 0.5
MU_D             = 0.4
BELT_VELOCITY    = 2.0        # m/s  in +X
FLOOR_SIZE       = 6.0        # m  half-width — large enough that the belt never
                              #    slides out from under the particle during the run
                              #    (belt moves 2 m/s x 0.8 s = 1.6 m, well within 12 m)
DT               = 5e-5       # s
TOTAL_TIME       = 0.8        # s  (plateau at ~0.51 s, so 0.8 s gives clear plateau)
FRAME_STRIDE     = 40         # steps between viewer frames  (~15 fps)

# Expected ramp: a_x = mu_d * g
G                = 9.81
EXPECTED_SLOPE   = MU_D * G   # m/s²

# CI tolerances
CI_SLOPE_TOL     = 0.15       # m/s²  ramp slope error tolerance
CI_PLATEAU_TOL   = 0.05       # m/s  plateau error tolerance


def run(ci: bool = False):
    n_steps = int(TOTAL_TIME / DT)

    config = SimConfig(
        num_particles=1,
        particle_radius=PARTICLE_RADIUS,
        particle_density=PARTICLE_DENSITY,
        young_modulus=1e7,
        poisson_ratio=0.3,
        restitution=0.1,      # low restitution — particle settles quickly
        friction_static=MU_S,
        friction_dynamic=MU_D,
        friction_rolling=0.01,
        dt=DT,
        gravity=(0.0, 0.0, -9.81),
        global_damping=0.0,   # must be zero — damping adds drag that distorts slope
        device="cuda:0",
    )
    sim = Simulation(config)

    # Drop particle from just above the belt, zero initial velocity
    drop_height = PARTICLE_RADIUS * 1.5
    sim.initialize_particles(np.array([[0.0, 0.0, drop_height]], dtype=np.float32))

    # Belt (translates in +X at BELT_VELOCITY)
    belt = create_rect_mesh(
        -FLOOR_SIZE, FLOOR_SIZE,
        -FLOOR_SIZE, FLOOR_SIZE,
        z=0.0,
        device=config.device,
    )
    sim.add_mesh(belt, linear_velocity=(BELT_VELOCITY, 0.0, 0.0))

    # ── Run and record ───────────────────────────────────────────────────────
    ts        = [0.0]
    vx_log    = [0.0]
    pos_log   = [sim.get_positions()[0].tolist()]
    frames    = []
    contact_t = None   # sim time of first contact

    def record_frame():
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

    record_frame()
    t0 = time.perf_counter()

    print(f"Running {n_steps} steps  (belt vx = {BELT_VELOCITY} m/s, "
          f"dt = {DT}, expected slope = {EXPECTED_SLOPE:.3f} m/s²) …")

    for step in range(1, n_steps + 1):
        sim.step()

        if step % FRAME_STRIDE == 0:
            record_frame()

        # Light logging every 0.1 s sim time
        if step % max(1, int(0.1 / DT)) == 0:
            wp.synchronize()
            pos = sim.get_positions()[0]
            vel = sim.get_velocities()[0]
            vx  = float(vel[0])
            t   = float(sim.sim_time)
            ts.append(t)
            vx_log.append(vx)
            pos_log.append(pos.tolist())

            # Detect first contact (particle z drops to within 1.5r of floor)
            if contact_t is None and pos[2] < PARTICLE_RADIUS * 1.5:
                contact_t = t

    wp.synchronize()
    wall = time.perf_counter() - t0
    print(f"  done in {wall:.1f}s  |  {len(frames)} frames")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    ts_arr    = np.array(ts)
    vx_arr    = np.array(vx_log)

    # Find kinetic phase: from first contact to plateau (vx > 90% belt speed)
    if contact_t is not None:
        contact_mask = ts_arr >= contact_t
        plateau_mask = vx_arr >= 0.9 * BELT_VELOCITY
        kinetic_mask = contact_mask & ~plateau_mask

        measured_slope = None
        plateau_mean   = None

        if kinetic_mask.sum() >= 3:
            t_kin = ts_arr[kinetic_mask] - contact_t
            v_kin = vx_arr[kinetic_mask]
            coeffs = np.polyfit(t_kin, v_kin, 1)
            measured_slope = float(coeffs[0])
            print(f"  Ramp slope:  measured={measured_slope:.3f}  "
                  f"expected={EXPECTED_SLOPE:.3f}  "
                  f"err={abs(measured_slope - EXPECTED_SLOPE):.3f} m/s²")

        if plateau_mask.sum() >= 2:
            plateau_mean = float(np.mean(vx_arr[plateau_mask]))
            print(f"  Plateau vx:  measured={plateau_mean:.3f}  "
                  f"expected={BELT_VELOCITY}  "
                  f"err={abs(plateau_mean - BELT_VELOCITY):.3f} m/s")
    else:
        print("  WARNING: no contact detected")
        measured_slope = None
        plateau_mean   = None

    # ── CI assertions ────────────────────────────────────────────────────────
    if ci:
        ok = True
        if measured_slope is None:
            print("CI: FAIL — could not measure ramp slope (no contact?)")
            ok = False
        elif abs(measured_slope - EXPECTED_SLOPE) > CI_SLOPE_TOL:
            print(f"CI: FAIL — slope {measured_slope:.3f} vs expected "
                  f"{EXPECTED_SLOPE:.3f} (tol {CI_SLOPE_TOL})")
            ok = False

        if plateau_mean is None:
            print("CI: FAIL — particle never reached belt speed")
            ok = False
        elif abs(plateau_mean - BELT_VELOCITY) > CI_PLATEAU_TOL:
            print(f"CI: FAIL — plateau {plateau_mean:.3f} vs belt "
                  f"{BELT_VELOCITY} (tol {CI_PLATEAU_TOL})")
            ok = False

        if ok:
            print("CI: PASS")
            sys.exit(0)
        else:
            sys.exit(1)

    # ── Write JSON + HTML viewer ──────────────────────────────────────────────
    radii = sim.get_radii().tolist()
    payload = {
        "config": {
            "n_particles":    1,
            "radius":         PARTICLE_RADIUS,
            "radii":          [round(float(r), 6) for r in radii],
            "dt":             DT,
            "sim_time":       float(sim.sim_time),
            "belt_velocity":  BELT_VELOCITY,
            "mu_d":           MU_D,
            "mu_s":           MU_S,
            "expected_slope": EXPECTED_SLOPE,
            "description":    "Coulomb friction ramp on translating floor",
        },
        "stl":   {"belt": _mesh_to_stl_dict(belt)},
        "frames": frames,
    }

    print(f"\nWriting {OUTPUT_JSON.name} …")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Generating {OUTPUT_HTML.name} …")
    generate_hopper_html(
        str(OUTPUT_JSON), str(OUTPUT_HTML),
        title="VeloxSim-DEM — Conveyor Belt (Phase 1 dynamics)",
        max_anim_frames=300, max_particles_per_frame=1,
    )

    _render_png(ts_arr, vx_arr, contact_t, measured_slope, plateau_mean)
    print(f"Done.  Open {OUTPUT_HTML} in a browser to see the belt move.")


def _mesh_to_stl_dict(mesh) -> dict:
    verts = mesh.points.numpy()
    return {"v": np.round(verts, 5).tolist(), "f": mesh.indices.numpy().tolist()}


def _render_png(ts, vx, contact_t, measured_slope, plateau_mean):
    """v_x vs time with analytical curve overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PNG plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, vx, "b-", linewidth=1.8, label="Simulated $v_x$")
    ax.axhline(BELT_VELOCITY, color="k", linestyle="--", linewidth=1,
               label=f"Belt speed = {BELT_VELOCITY} m/s")

    # Analytical Coulomb ramp (starts at first contact)
    if contact_t is not None:
        t_ramp = np.linspace(contact_t, contact_t + BELT_VELOCITY / EXPECTED_SLOPE,
                             200)
        v_ramp = np.minimum(EXPECTED_SLOPE * (t_ramp - contact_t), BELT_VELOCITY)
        ax.plot(t_ramp, v_ramp, "r--", linewidth=1.5,
                label=f"Analytical  ($\\mu_d g$ = {EXPECTED_SLOPE:.3f} m/s²)")

    if measured_slope is not None:
        ax.text(0.02, 0.97,
                f"Measured slope: {measured_slope:.3f} m/s²\n"
                f"Expected slope: {EXPECTED_SLOPE:.3f} m/s²\n"
                f"Plateau $v_x$: {plateau_mean:.3f} m/s" if plateau_mean else "",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Simulation time (s)", fontsize=12)
    ax.set_ylabel("$v_x$ (m/s)", fontsize=12)
    ax.set_title("Conveyor Belt — Coulomb Friction Ramp", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, BELT_VELOCITY * 1.15)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PNG), dpi=150)
    plt.close(fig)
    print(f"  wrote {OUTPUT_PNG.stat().st_size / 1024:.0f} KB -> {OUTPUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ci", action="store_true",
                        help="CI acceptance test — exits 0 on pass, 1 on fail")
    args = parser.parse_args()
    run(ci=args.ci)
