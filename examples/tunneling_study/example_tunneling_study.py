"""Tunneling study — Phase 1 geometry dynamics safety envelope.

A horizontal floor translates upward toward a stationary particle.  If the
floor moves fast enough relative to the timestep, it can teleport past the
particle between steps without generating a contact — this is *tunneling*.

The safe-motion criterion is::

    v_wall * dt  <  r_particle

This script sweeps a grid of ``(dt, v_wall)`` pairs, detects tunneling
(particle z-coordinate drops below the floor by more than half a radius), and
produces a 2-D pass/fail table.  It documents the operating envelope for
Phase 1 kinematic geometry dynamics.

Usage
-----
Full 4×4 sweep (produces PNG heatmap)::

    python example_tunneling_study.py

CI mode (runs only the 4 safest cases — fast, exits 0/1)::

    python example_tunneling_study.py --ci

Viewer note
-----------
Each case is very short (a few hundred steps) so no interactive HTML viewer
is produced.  The PNG heatmap is the primary output — green = no tunneling,
red = tunneling detected.
"""

from __future__ import annotations

import argparse
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

HERE       = pathlib.Path(__file__).resolve().parent
OUTPUT_PNG = HERE / "tunneling_study.png"

# ── Sweep grid ────────────────────────────────────────────────────────────────
PARTICLE_RADIUS = 0.01   # m

# Full sweep: 4 dt values × 4 speeds = 16 cases
DT_VALUES    = [5e-5, 1e-4, 2e-4, 5e-4]       # seconds
SPEED_VALUES = [0.1, 0.5, 1.0, 5.0]           # m/s  wall speed

# CI mode: only run cases where safe criterion is predicted to pass
# safe = v_wall * dt < r_particle
CI_DT_VALUES    = [5e-5]                       # only the smallest dt
CI_SPEED_VALUES = [0.1, 0.5, 1.0]             # speeds expected to be safe

# Steps per case — enough for the floor to attempt crossing the particle
STEPS_PER_CASE = 300


def run_case(dt: float, v_wall: float) -> dict:
    """Run a single (dt, v_wall) case and return a result dict."""
    config = SimConfig(
        num_particles=1,
        particle_radius=PARTICLE_RADIUS,
        particle_density=2500.0,
        young_modulus=1e7,
        poisson_ratio=0.3,
        restitution=0.3,
        friction_static=0.5,
        friction_dynamic=0.4,
        friction_rolling=0.01,
        dt=dt,
        gravity=(0.0, 0.0, 0.0),   # no gravity — particle is fixed by zero g
        global_damping=0.0,
        device="cuda:0",
    )
    sim = Simulation(config)

    floor_z0 = -0.5   # floor starts below the particle, moves upward
    gap      = PARTICLE_RADIUS * 2  # particle starts at z = PARTICLE_RADIUS*2

    sim.initialize_particles(
        np.array([[0.0, 0.0, floor_z0 + gap + PARTICLE_RADIUS]], dtype=np.float32)
    )
    floor = create_rect_mesh(-0.2, 0.2, -0.2, 0.2, z=floor_z0, device=config.device)
    sim.add_mesh(floor, linear_velocity=(0.0, 0.0, v_wall))

    tunneled       = False
    min_gap_seen   = float("inf")
    n_contacts     = 0
    per_step_disp  = v_wall * dt
    safe_criterion = per_step_disp < PARTICLE_RADIUS

    for _ in range(STEPS_PER_CASE):
        sim.step()

    wp.synchronize()
    pos       = sim.get_positions()[0]
    floor_z   = sim.get_mesh_poses()[0]["pos"][2] + floor_z0
    gap_final = float(pos[2]) - float(floor_z) - PARTICLE_RADIUS

    # Tunneled if particle centre is below (floor surface + particle radius)
    # by more than half a radius
    if float(pos[2]) < float(floor_z) + PARTICLE_RADIUS * 0.5:
        tunneled = True

    return {
        "dt":            dt,
        "v_wall":        v_wall,
        "per_step_disp": per_step_disp,
        "safe_criterion": safe_criterion,
        "tunneled":      tunneled,
        "particle_z":    float(pos[2]),
        "floor_z_final": float(floor_z),
        "gap_final":     gap_final,
    }


def run(ci: bool = False):
    dt_vals    = CI_DT_VALUES    if ci else DT_VALUES
    speed_vals = CI_SPEED_VALUES if ci else SPEED_VALUES

    print("=" * 70)
    print(" Tunneling Study — Phase 1 geometry dynamics")
    print(f" r_particle = {PARTICLE_RADIUS * 1000:.0f} mm  |  "
          f"safe criterion: v_wall * dt < r_particle")
    print("=" * 70)
    print(f"{'dt (s)':>10s}  {'v_wall (m/s)':>13s}  "
          f"{'step_disp (m)':>14s}  {'safe?':>6s}  {'result':>8s}  {'gap (m)':>9s}")
    print("-" * 70)

    results = []
    t0 = time.perf_counter()
    for dt in dt_vals:
        for v in speed_vals:
            r = run_case(dt, v)
            results.append(r)
            flag    = "safe" if r["safe_criterion"] else "unsafe"
            outcome = "TUNNEL" if r["tunneled"] else "OK"
            print(f"  {dt:.1e}     {v:>8.1f}        {r['per_step_disp']:.4f}        "
                  f"{flag:>6s}  {outcome:>8s}  {r['gap_final']:>9.4f}")

    wall = time.perf_counter() - t0
    print(f"\n{len(results)} cases in {wall:.1f}s")

    # CI check: all CI cases must not tunnel
    if ci:
        failures = [r for r in results if r["tunneled"]]
        if failures:
            print(f"\nCI: FAIL — {len(failures)} case(s) tunneled:")
            for r in failures:
                print(f"  dt={r['dt']:.1e}  v_wall={r['v_wall']}  "
                      f"particle_z={r['particle_z']:.4f}  floor_z={r['floor_z_final']:.4f}")
            sys.exit(1)
        print(f"\nCI: PASS — 0 tunneling events in {len(results)} safe cases")
        sys.exit(0)

    # Produce summary table + PNG
    _print_table(results, dt_vals, speed_vals)
    _render_png(results, dt_vals, speed_vals)


def _print_table(results, dt_vals, speed_vals):
    print("\n  Pass/Fail matrix  (OK = no tunneling, TUNNEL = particle passed through)")
    header = "          " + "".join(f"  v={v:.1f}" for v in speed_vals)
    print(header)
    for dt in dt_vals:
        row = f"  dt={dt:.0e} "
        for v in speed_vals:
            r = next(x for x in results if x["dt"] == dt and x["v_wall"] == v)
            cell = " OK   " if not r["tunneled"] else " TUNN "
            row += cell
        print(row)
    print()

    print("  Safe-criterion matrix  (Y = v*dt < r_particle, N = expected tunneling)")
    print(header)
    for dt in dt_vals:
        row = f"  dt={dt:.0e} "
        for v in speed_vals:
            disp = v * dt
            row += "  Y    " if disp < PARTICLE_RADIUS else "  N    "
        print(row)


def _render_png(results, dt_vals, speed_vals):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping PNG")
        return

    nd = len(dt_vals)
    nv = len(speed_vals)
    grid = np.zeros((nd, nv), dtype=int)

    for i, dt in enumerate(dt_vals):
        for j, v in enumerate(speed_vals):
            r = next(x for x in results if x["dt"] == dt and x["v_wall"] == v)
            # 0 = OK, 1 = tunneled, 2 = unsafe-criterion-but-ok (lucky)
            if r["tunneled"]:
                grid[i, j] = 2
            elif not r["safe_criterion"]:
                grid[i, j] = 1
            else:
                grid[i, j] = 0

    cmap = plt.matplotlib.colors.ListedColormap(["#22c55e", "#f59e0b", "#ef4444"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                   extent=[-0.5, nv - 0.5, nd - 0.5, -0.5])

    ax.set_xticks(range(nv))
    ax.set_xticklabels([f"{v:.1f}" for v in speed_vals], fontsize=10)
    ax.set_yticks(range(nd))
    ax.set_yticklabels([f"{dt:.0e}" for dt in dt_vals], fontsize=9)
    ax.set_xlabel("Wall speed  $v_{wall}$ (m/s)", fontsize=12)
    ax.set_ylabel("Timestep  $dt$ (s)", fontsize=12)
    ax.set_title(
        f"Tunneling study  —  $r_{{particle}}$ = {PARTICLE_RADIUS * 1000:.0f} mm\n"
        f"Safe criterion:  $v_{{wall}} \\cdot dt < r_{{particle}}$",
        fontsize=12,
    )

    # Annotate each cell
    for i in range(nd):
        for j in range(nv):
            r = next(x for x in results if x["dt"] == dt_vals[i]
                     and x["v_wall"] == speed_vals[j])
            txt = "OK" if not r["tunneled"] else "TUNNEL"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if r["tunneled"] else "black", fontweight="bold")

    patches = [
        mpatches.Patch(color="#22c55e", label="No tunneling (safe)"),
        mpatches.Patch(color="#f59e0b", label="Criterion not met (but no tunnel)"),
        mpatches.Patch(color="#ef4444", label="Tunneling detected"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0.9)

    # Draw safe-criterion boundary as a dashed line
    # v_wall * dt = r  =>  v_wall = r / dt
    v_crit = [PARTICLE_RADIUS / dt for dt in dt_vals]
    # Plot on right axis (or just annotate with a note)
    ax.text(0.02, 0.03,
            f"Criterion: $v_{{wall}} \\cdot dt < {PARTICLE_RADIUS*1000:.0f}$ mm",
            transform=ax.transAxes, fontsize=9, color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#dbeafe", alpha=0.9))

    fig.tight_layout()
    fig.savefig(str(OUTPUT_PNG), dpi=150)
    plt.close(fig)
    print(f"  wrote {OUTPUT_PNG.stat().st_size / 1024:.0f} KB -> {OUTPUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ci", action="store_true",
                        help="Run only safe cases — CI acceptance test (exits 0/1)")
    args = parser.parse_args()
    run(ci=args.ci)
