"""
Test: JKR Cohesion — Angle of Repose (Cylinder Lift Method)

Built by VeloxSim Tech Pty Ltd and Sam Wong.

Setup (standard granular test):
  - 200 iron-ore particles (d=35 mm, rho=4500 kg/m^3) hex-packed inside
    a cylinder of radius ~100 mm sitting on a flat plate.
  - The cylinder is "lifted" (removed) and particles slump under gravity,
    forming a conical pile whose slope angle is the angle of repose.
  - Two cases compared:
      Case A  cohesion_energy =  0.0 J/m^2  (free-flowing)
      Case B  cohesion_energy = 50.0 J/m^2  (sticky, Bond# ~ 2.1)

  Since meshes are static in VeloxSim-DEM, the lift is simulated by:
    Phase 1 — settle particles inside a cylindrical shell mesh
    Phase 2 — restart with only the floor; particles slump from the
              settled cylinder shape into a natural pile.

PASS/FAIL assertions:
  1. No floor penetration             (z_min > 0 for both)
  2. Both piles form a heap           (angle > 5 deg)
  3. Cohesive angle > non-cohesive
  4. Angle increase >= 5 deg          (clear JKR effect)

Saves final state to repose_data.json for visualisation.
"""

import sys
import math
import json
import numpy as np
import warp as wp

sys.stdout.reconfigure(encoding="utf-8")

from veloxsim_dem import Simulation, SimConfig, create_plane_mesh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS   = 0.0175       # m  (17.5 mm, diameter 35 mm)
DENSITY  = 4500.0       # kg/m^3  (iron ore)
N        = 1500
G        = 9.81
DT       = 2.0e-5       # s

# Cylinder dimensions
CYL_RADIUS = 0.20       # m  (200 mm inner radius)
CYL_HEIGHT = 1.00       # m  (1000 mm tall — enough for all layers)

FLOOR_SIZE = 10.0       # m  (large enough that no particle escapes)

# Settling inside cylinder
PHASE1_STEPS       = 400_000   # 8.0 s — settle inside cylinder
PHASE1_KE_THRESH   = 1.0e-3   # J

# Slumping after cylinder removal
PHASE2_STEPS       = 400_000   # 8.0 s — slump and settle
PHASE2_START_CHECK =  50_000   # 1.0 s before checking KE
PHASE2_KE_THRESH   =  1.0e-3  # J

CASES = [
    {
        "label": "25deg",
        "target_angle": 25,
        "cohesion_energy": 0.0,
        "friction_static": 0.50,
        "friction_dynamic": 0.35,
        "friction_rolling": 0.05,
    },
    {
        "label": "40deg",
        "target_angle": 40,
        "cohesion_energy": 25.0,
        "friction_static": 0.80,
        "friction_dynamic": 0.60,
        "friction_rolling": 0.25,
    },
]

mass     = (4.0 / 3.0) * math.pi * RADIUS**3 * DENSITY
R_eff    = RADIUS / 2.0
weight   = mass * G
for c in CASES:
    c["pulloff"]     = 1.5 * math.pi * c["cohesion_energy"] * R_eff
    c["bond_number"] = c["pulloff"] / weight if weight > 0 else 0.0

# ---------------------------------------------------------------------------
# Cylinder mesh (open-top, open-bottom tube)
# ---------------------------------------------------------------------------
def create_cylinder_mesh(center, radius, height, segments=32, device="cuda:0"):
    """Create a cylindrical tube mesh (open top and bottom).

    Normals point inward so particles inside bounce off the walls.
    """
    cx, cy, cz = center
    vertices = []
    indices = []

    for i in range(segments):
        a1 = 2 * math.pi * i / segments
        a2 = 2 * math.pi * ((i + 1) % segments) / segments

        x1 = cx + radius * math.cos(a1)
        y1 = cy + radius * math.sin(a1)
        x2 = cx + radius * math.cos(a2)
        y2 = cy + radius * math.sin(a2)

        base = len(vertices)
        vertices.append([x1, y1, cz])               # bottom-left
        vertices.append([x2, y2, cz])               # bottom-right
        vertices.append([x2, y2, cz + height])      # top-right
        vertices.append([x1, y1, cz + height])      # top-left

        # Winding: CW from outside = CCW from inside → normals point inward
        indices.extend([base, base + 2, base + 1])
        indices.extend([base, base + 3, base + 2])

    verts = np.array(vertices, dtype=np.float32)
    faces = np.array(indices, dtype=np.int32)

    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )

# ---------------------------------------------------------------------------
# Initial positions: hex-packed layers inside the cylinder
# ---------------------------------------------------------------------------
def pack_cylinder(n, cyl_radius, particle_radius, seed=42):
    """Pack particles in hex layers within a cylinder, bottom-up."""
    d = 2 * particle_radius
    r_inner = cyl_radius - particle_radius * 1.1  # clearance from wall
    positions = []
    z = particle_radius  # first layer sits on the floor
    layer_idx = 0

    while len(positions) < n:
        # Hex grid within a circle of radius r_inner
        layer = []
        spacing = d * 1.02  # small gap to avoid initial overlaps

        # Hex grid offsets
        row = 0
        y_pos = -r_inner
        while y_pos <= r_inner:
            # Offset every other row by half spacing
            x_start = -r_inner + (spacing * 0.5 if row % 2 else 0.0)
            x_pos = x_start
            while x_pos <= r_inner:
                if math.sqrt(x_pos**2 + y_pos**2) + particle_radius <= cyl_radius * 0.95:
                    layer.append([x_pos, y_pos, z])
                x_pos += spacing
            y_pos += spacing * math.sqrt(3) / 2
            row += 1

        # Offset alternate layers for better packing
        if layer_idx % 2 == 1:
            dx = d * 0.5 * 0.3
            dy = d * 0.5 * 0.3
            for p in layer:
                p[0] += dx
                p[1] += dy

        for p in layer:
            if len(positions) < n:
                positions.append(p)

        z += d * math.sqrt(2.0 / 3.0)  # hex close-pack layer spacing
        layer_idx += 1

    return np.array(positions[:n], dtype=np.float32)

# ---------------------------------------------------------------------------
# Angle-of-repose measurement (linear fit to radial profile)
# ---------------------------------------------------------------------------
def measure_repose_angle(pos: np.ndarray) -> float:
    """Measure angle of repose from the settled pile.

    Uses apex height vs 90th-percentile radius to ignore outlier
    particles that have rolled far from the main pile.
    """
    cx = float(np.mean(pos[:, 0]))
    cy = float(np.mean(pos[:, 1]))
    r  = np.sqrt((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2)
    z  = pos[:, 2]

    # Use 90th percentile of radial distance as the pile base
    # (ignores stray particles that rolled far away)
    base_r = float(np.percentile(r, 90)) + RADIUS
    apex_h = float(np.max(z)) - RADIUS

    if base_r < 1e-6 or apex_h <= 0.0:
        return 0.0

    return math.degrees(math.atan2(apex_h, base_r))


def make_config(case_params, global_damping=100.0, device="cuda:0"):
    return SimConfig(
        num_particles=N,
        particle_radius=RADIUS,
        particle_density=DENSITY,
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        restitution=0.15,
        friction_static=case_params.get("friction_static", 0.5),
        friction_dynamic=case_params.get("friction_dynamic", 0.35),
        friction_rolling=case_params.get("friction_rolling", 0.05),
        cohesion_energy=case_params.get("cohesion_energy", 0.0),
        global_damping=global_damping,
        dt=DT,
        gravity=(0.0, 0.0, -G),
        max_contacts_per_particle=32,
        hash_grid_dim=64,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print("=" * 60)
print("VeloxSim-DEM  |  Angle of Repose — Cylinder Lift Method")
print("=" * 60)
print(f"  Particles     : {N}  (d={RADIUS*2*1000:.0f} mm)")
print(f"  Radius        : {RADIUS*1000:.1f} mm,  density={DENSITY:.0f} kg/m^3")
print(f"  Mass/particle : {mass*1000:.1f} g,  weight={weight*1000:.1f} mN")
for c in CASES:
    print(f"  Case '{c['label']}': mu_s={c['friction_static']}, mu_r={c['friction_rolling']}, "
          f"gamma={c['cohesion_energy']} J/m^2, "
          f"pulloff={c['pulloff']*1000:.1f} mN, Bond#={c['bond_number']:.2f}")
print(f"  Cylinder      : R={CYL_RADIUS*1000:.0f} mm  H={CYL_HEIGHT*1000:.0f} mm")
print(f"  DT={DT:.0e} s")
print()

# ---------------------------------------------------------------------------
# Pack particles in the cylinder
# ---------------------------------------------------------------------------
print("Packing particles in cylinder...", end="", flush=True)
INIT_POS = pack_cylinder(N, CYL_RADIUS, RADIUS)
print(f" done.  {len(INIT_POS)} particles, "
      f"z: {INIT_POS[:,2].min()*1000:.1f}-{INIT_POS[:,2].max()*1000:.1f} mm")

# ---------------------------------------------------------------------------
# Phase 1: Settle inside cylinder
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 1: Settling particles inside cylinder")
print("=" * 60)

settle_params = {"cohesion_energy": 0.0, "friction_static": 0.5, "friction_dynamic": 0.35, "friction_rolling": 0.05}
config_settle = make_config(settle_params, global_damping=300.0)  # heavy damping to settle fast
sim1 = Simulation(config_settle)
sim1.initialize_particles(INIT_POS.copy())

# Floor
floor = create_plane_mesh(
    origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
    size=FLOOR_SIZE, device=config_settle.device,
)
sim1.add_mesh(floor)

# Cylinder wall
cyl_mesh = create_cylinder_mesh(
    center=(0.0, 0.0, 0.0), radius=CYL_RADIUS,
    height=CYL_HEIGHT, segments=48, device=config_settle.device,
)
sim1.add_mesh(cyl_mesh)

print("  Compiling Warp kernels ...", end="", flush=True)
sim1.step()
wp.synchronize()
print(" done.")

for step in range(1, PHASE1_STEPS + 1):
    sim1.step()

    if step % 25_000 == 0:
        wp.synchronize()
        ke = sim1.get_kinetic_energy()
        pos = sim1.get_positions()
        print(f"  step {step:>7d}  t={sim1.sim_time:.3f} s  "
              f"KE={ke:.3e} J  z_range={pos[:,2].min()*1000:.1f}-{pos[:,2].max()*1000:.1f} mm")

    if step >= 50_000 and step % 10_000 == 0:
        wp.synchronize()
        ke = sim1.get_kinetic_energy()
        if ke < PHASE1_KE_THRESH:
            print(f"  Settled at step {step} (KE={ke:.3e} J)")
            break

wp.synchronize()
SETTLED_POS = sim1.get_positions().copy()
print(f"  Settled column: z_range={SETTLED_POS[:,2].min()*1000:.1f}-"
      f"{SETTLED_POS[:,2].max()*1000:.1f} mm")
print()

# ---------------------------------------------------------------------------
# Phase 2: Remove cylinder (slump test) for each cohesion case
# ---------------------------------------------------------------------------
results = {}

for case in CASES:
    label = case["label"]
    gamma = case["cohesion_energy"]

    print(f"{'='*60}")
    print(f"PHASE 2: Cylinder removed — {label}  (gamma={gamma} J/m^2)")
    print(f"{'='*60}")

    config = make_config(case, global_damping=100.0)
    sim = Simulation(config)
    sim.initialize_particles(SETTLED_POS.copy())

    # Only the floor — no cylinder!
    floor2 = create_plane_mesh(
        origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        size=FLOOR_SIZE, device=config.device,
    )
    sim.add_mesh(floor2)

    settled_step = None

    for step in range(1, PHASE2_STEPS + 1):
        sim.step()

        if step % 25_000 == 0:
            wp.synchronize()
            ke = sim.get_kinetic_energy()
            pos = sim.get_positions()
            r_max = float(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2).max())
            print(f"  step {step:>7d}  t={sim.sim_time:.3f} s  "
                  f"KE={ke:.3e} J  z_max={pos[:,2].max()*1000:.1f} mm  "
                  f"r_max={r_max*1000:.0f} mm")

        if step >= PHASE2_START_CHECK and step % 10_000 == 0:
            wp.synchronize()
            ke = sim.get_kinetic_energy()
            if ke < PHASE2_KE_THRESH:
                settled_step = step
                print(f"  Settled at step {step}  "
                      f"(t={sim.sim_time:.3f} s  KE={ke:.3e} J)")
                break

    wp.synchronize()
    pos      = sim.get_positions()
    final_ke = sim.get_kinetic_energy()
    angle    = measure_repose_angle(pos)

    if settled_step is None:
        print(f"  Note: KE threshold not reached  (final KE={final_ke:.3e} J)")

    print(f"  z_min={pos[:,2].min()*1000:.3f} mm  "
          f"z_max={pos[:,2].max()*1000:.1f} mm  "
          f"angle={angle:.2f} deg")
    print()

    results[label] = {
        "cohesion_energy": gamma,
        "bond_number":     case["bond_number"],
        "repose_angle":    angle,
        "settled":         settled_step is not None,
        "final_ke":        float(final_ke),
        "z_min":           float(pos[:, 2].min()),
        "positions":       pos.tolist(),
    }

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
with open("repose_data.json", "w") as f:
    json.dump({"radius": RADIUS, "density": DENSITY, "n": N,
               "cases": results}, f, separators=(",", ":"))
print("Saved repose_data.json\n")

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
passes, fails = [], []

def check(name, condition, detail=""):
    (passes if condition else fails).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}]  {name}")
    if detail:
        print(f"          {detail}")

ang_25 = results["25deg"]["repose_angle"]
ang_40 = results["40deg"]["repose_angle"]
diff   = ang_40 - ang_25

print("=" * 60)
print("Assertions")
print("=" * 60)

check("No floor penetration",
      results["25deg"]["z_min"] > 0.0 and results["40deg"]["z_min"] > 0.0,
      f"25deg z_min={results['25deg']['z_min']*1000:.3f} mm  "
      f"40deg z_min={results['40deg']['z_min']*1000:.3f} mm")

check("25 deg case within ±5 deg of target",
      20.0 <= ang_25 <= 30.0,
      f"measured={ang_25:.1f} deg  target=25 deg")

check("40 deg case within ±5 deg of target",
      35.0 <= ang_40 <= 45.0,
      f"measured={ang_40:.1f} deg  target=40 deg")

check("40 deg pile is steeper than 25 deg pile",
      ang_40 > ang_25,
      f"40deg={ang_40:.1f} deg  25deg={ang_25:.1f} deg  diff={diff:+.1f} deg")

check("Angle difference >= 10 deg",
      diff >= 10.0,
      f"delta={diff:.1f} deg  (threshold=10 deg)")

print()
print("=" * 60)
total = len(passes) + len(fails)
if fails:
    print(f"RESULT: FAIL  ({len(passes)}/{total} checks passed)")
    print(f"Failed: {', '.join(fails)}")
    sys.exit(1)
else:
    print(f"RESULT: PASS  ({len(passes)}/{total} checks passed)")
