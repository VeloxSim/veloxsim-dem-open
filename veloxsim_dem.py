"""
VeloxSim-DEM: Discrete Element Method simulation engine built on NVIDIA Warp.

Features:
    - Hertz-Mindlin contact model (normal + tangential with damping)
    - Coulomb sliding friction with static (mu_s) and dynamic (mu_d) coefficients
    - Type C EPSD rolling resistance (elastic-plastic spring-dashpot, history-dependent)
    - JKR-inspired cohesion/adhesion model
    - Particle-particle collision with spatial hash grid broad-phase
    - Particle-mesh collision using BVH mesh queries
    - Moving mesh surfaces (conveyor belts) via surface_velocity parameter
    - Kinematic geometry dynamics Phase 1: meshes translate at a prescribed
      linear_velocity; BVH refitted each step so contacts remain correct
    - Kinematic geometry dynamics Phase 2: meshes rotate at a prescribed
      angular_velocity about an origin; rest-frame vertices are transformed
      by the current quaternion each step; contact history is rotated to
      stay frame-correct; rotational arm (omega x r) included in contact
      velocity so friction acts against the moving surface
    - Velocity Verlet time integration
    - 3D triangular mesh import (OBJ/STL via trimesh)
    - Particle size distribution (PSD): user-specified discrete size classes with
      per-particle radii, masses, and inertias; Hertz-Mindlin uses effective radius
      R* = R_i*R_j/(R_i+R_j) and effective mass m* = m_i*m_j/(m_i+m_j) per contact
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import warp as wp

wp.init()

# ---------------------------------------------------------------------------
# Particle Size Distribution
# ---------------------------------------------------------------------------

class ParticleSizeDistribution:
    """Discrete PSD: a list of ``(radius_m, percentage)`` pairs.

    Parameters
    ----------
    size_classes:
        Sequence of ``(radius, percentage)`` tuples where *radius* is in
        metres and *percentage* is a value in ``[0, 100]``.  Percentages
        must sum to **100** (within 0.001 tolerance).

    Raises
    ------
    ValueError
        If percentages do not sum to 100, any radius is non-positive, or
        any percentage is negative.

    Example
    -------
    >>> psd = ParticleSizeDistribution([(0.01, 30), (0.015, 50), (0.025, 20)])
    >>> psd.max_radius
    0.025
    """

    def __init__(self, size_classes: Sequence[Tuple[float, float]]) -> None:
        size_classes = list(size_classes)
        if not size_classes:
            raise ValueError("size_classes must not be empty")

        radii, pcts = zip(*size_classes)
        radii = [float(r) for r in radii]
        pcts  = [float(p) for p in pcts]

        if any(r <= 0.0 for r in radii):
            raise ValueError("All radii must be positive")
        if any(p < 0.0 for p in pcts):
            raise ValueError("All percentages must be non-negative")

        total = sum(pcts)
        if abs(total - 100.0) > 1e-3:
            raise ValueError(
                f"PSD percentages must sum to 100 (got {total:.6g})"
            )

        self._radii: List[float] = radii
        self._pcts:  List[float] = pcts

    @property
    def size_classes(self) -> List[Tuple[float, float]]:
        """Return the ``(radius, percentage)`` pairs as a list."""
        return list(zip(self._radii, self._pcts))

    @property
    def max_radius(self) -> float:
        """Largest radius in the distribution (metres)."""
        return max(self._radii)

    @property
    def min_radius(self) -> float:
        """Smallest radius in the distribution (metres)."""
        return min(self._radii)

    def assign_radii(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Return a ``float32`` array of length *n* with radii drawn from
        the PSD.

        Particle counts are rounded to the nearest integer; the last size
        class absorbs any rounding remainder so the total is always exactly
        *n*.  The resulting array is shuffled so radii are not spatially
        ordered.

        Parameters
        ----------
        n:   Number of particles.
        rng: Optional NumPy random generator for reproducible results.
        """
        if rng is None:
            rng = np.random.default_rng()

        counts: List[int] = []
        remaining = n
        for pct in self._pcts[:-1]:
            k = int(round(n * pct / 100.0))
            k = max(0, min(k, remaining))
            counts.append(k)
            remaining -= k
        counts.append(remaining)  # last class absorbs rounding remainder

        out = np.empty(n, dtype=np.float32)
        idx = 0
        for r, k in zip(self._radii, counts):
            out[idx : idx + k] = np.float32(r)
            idx += k

        rng.shuffle(out)
        return out


# ---------------------------------------------------------------------------
# Warp struct definitions
# ---------------------------------------------------------------------------

@wp.struct
class MaterialParams:
    young_modulus: wp.float32
    poisson_ratio: wp.float32
    restitution: wp.float32
    friction_static: wp.float32
    friction_rolling: wp.float32
    cohesion_energy: wp.float32  # surface energy gamma (J/m^2)
    particle_radius: wp.float32  # representative radius (used when psd=None)
    particle_density: wp.float32


@wp.struct
class DerivedParams:
    E_eff: wp.float32              # effective Young's modulus E*
    G_eff: wp.float32              # effective shear modulus G*
    beta: wp.float32               # damping coefficient from restitution
    cohesion_energy_gamma: wp.float32  # surface energy γ (J/m²); pull-off computed per contact


# ---------------------------------------------------------------------------
# Warp device-side helper functions
# ---------------------------------------------------------------------------

@wp.func
def compute_relative_velocity(
    vi: wp.vec3, vj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    n: wp.vec3, ri: float, rj: float,
) -> wp.vec3:
    """Relative velocity at contact point including angular contribution."""
    v_rel = (vi - vj) + wp.cross(wi, n * ri) + wp.cross(wj, n * rj)
    return v_rel


@wp.func
def hertz_mindlin_force(
    xi: wp.vec3, xj: wp.vec3,
    vi: wp.vec3, vj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    tangent_disp: wp.vec3,
    ri: float, rj: float,          # per-particle radii
    mi: float, mj: float,          # per-particle masses
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    dt: float,
) -> wp.vec3:
    """
    Compute Hertz-Mindlin contact force between two spheres.
    Returns the force on particle i due to particle j.

    Uses effective radius R* = R_i*R_j / (R_i+R_j) and
    effective mass m* = m_i*m_j / (m_i+m_j) for mixed-size pairs.

    Coulomb friction: static limit mu_s*|F_n| → when exceeded, kinetic force
    is capped at mu_d*|F_n| (mu_d <= mu_s).
    """
    diff = xj - xi
    dist = wp.length(diff)

    if dist < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist  # unit normal from i to j

    # Effective radius and mass for this contact pair
    R_eff = (ri * rj) / (ri + rj)
    m_eff = (mi * mj) / (mi + mj)

    # JKR-inspired cohesion pull-off force: F_po = 1.5 * pi * gamma * R*
    cohesion_pulloff = wp.float32(1.5) * wp.float32(3.14159265) * dp.cohesion_energy_gamma * R_eff

    delta_n = (ri + rj) - dist  # overlap (positive in contact)

    # Cohesion range: allow attractive force slightly beyond contact
    cohesion_range = wp.float32(0.1) * (ri + rj)
    if delta_n < -cohesion_range:
        return wp.vec3(0.0, 0.0, 0.0)

    force = wp.vec3(0.0, 0.0, 0.0)

    # Cohesion/adhesion: attractive force when within range but separating
    if delta_n < wp.float32(0.0):
        cohesion_frac = wp.float32(1.0) + delta_n / cohesion_range
        force = n * cohesion_pulloff * cohesion_frac
        return force

    # --- In contact (delta_n >= 0) ---
    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(R_eff)

    # Normal stiffness and force (Hertz)
    S_n = wp.float32(2.0) * dp.E_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (wp.float32(4.0) / wp.float32(3.0)) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    # Relative velocity at contact (arms: ri on i, rj on j)
    v_rel = compute_relative_velocity(vi, vj, wi, wj, n, ri, rj)
    v_n_mag = wp.dot(v_rel, n)
    v_n = n * v_n_mag
    v_t = v_rel - v_n

    # Normal damping
    damp_n_coeff = -wp.float32(2.0) * wp.sqrt(wp.float32(5.0) / wp.float32(6.0)) * dp.beta * wp.sqrt(S_n * m_eff)
    F_nd = n * (damp_n_coeff * v_n_mag)

    # Cohesion at contact: additive pull-off (attractive, along +n toward j)
    F_cohesion = n * cohesion_pulloff

    # Total normal force (repulsive + damping + cohesion)
    F_normal = n * (-F_n_mag) + F_nd + F_cohesion

    # Tangential stiffness
    S_t = wp.float32(8.0) * dp.G_eff * sqrt_R_eff * sqrt_delta

    # Update tangential displacement incrementally
    new_tangent = tangent_disp + v_t * dt

    # Tangential spring force
    F_t_spring = new_tangent * (-S_t)

    # Tangential damping
    damp_t_coeff = -wp.float32(2.0) * wp.sqrt(wp.float32(5.0) / wp.float32(6.0)) * dp.beta * wp.sqrt(S_t * m_eff)
    F_t_damp = v_t * damp_t_coeff

    F_t_total = F_t_spring + F_t_damp

    # Coulomb friction: static → kinetic when exceeded
    F_t_mag = wp.length(F_t_total)
    F_n_total_mag = F_n_mag - damp_n_coeff * v_n_mag
    static_limit = mu_s * wp.abs(F_n_total_mag)

    if F_t_mag > static_limit and F_t_mag > wp.float32(1.0e-12):
        kinetic_limit = mu_d * wp.abs(F_n_total_mag)
        F_t_total = F_t_total * (kinetic_limit / F_t_mag)

    force = F_normal + F_t_total
    return force


@wp.func
def hertz_mindlin_update_tangent(
    xi: wp.vec3, xj: wp.vec3,
    vi: wp.vec3, vj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    tangent_disp: wp.vec3,
    ri: float, rj: float,
    mi: float, mj: float,
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    dt: float,
) -> wp.vec3:
    """Compute updated tangential displacement for contact history tracking.

    When sliding (F_t > mu_s * F_n), spring is rescaled to kinetic limit
    mu_d * F_n so the stored displacement remains physically consistent.
    """
    diff = xj - xi
    dist = wp.length(diff)
    if dist < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist

    R_eff = (ri * rj) / (ri + rj)
    m_eff = (mi * mj) / (mi + mj)

    delta_n = (ri + rj) - dist
    if delta_n <= wp.float32(0.0):
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(R_eff)

    S_n = wp.float32(2.0) * dp.E_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (wp.float32(4.0) / wp.float32(3.0)) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta
    S_t = wp.float32(8.0) * dp.G_eff * sqrt_R_eff * sqrt_delta

    v_rel = compute_relative_velocity(vi, vj, wi, wj, n, ri, rj)
    v_n_mag = wp.dot(v_rel, n)
    v_t = v_rel - n * v_n_mag

    new_tangent = tangent_disp + v_t * dt

    # Check Coulomb limit on tangential spring
    F_t_spring_mag = wp.length(new_tangent) * S_t
    damp_n_coeff = -wp.float32(2.0) * wp.sqrt(wp.float32(5.0) / wp.float32(6.0)) * dp.beta * wp.sqrt(S_n * m_eff)
    F_n_total_mag = F_n_mag - damp_n_coeff * v_n_mag
    static_limit = mu_s * wp.abs(F_n_total_mag)

    if F_t_spring_mag > static_limit and F_t_spring_mag > wp.float32(1.0e-12):
        kinetic_limit = mu_d * wp.abs(F_n_total_mag)
        new_tangent = new_tangent * (kinetic_limit / (wp.length(new_tangent) * S_t + wp.float32(1.0e-12)))

    return new_tangent


@wp.func
def compute_rolling_torque_pp(
    xi: wp.vec3, xj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    roll_disp: wp.vec3,
    ri: float, rj: float,
    dp: DerivedParams,
    mu_r: float,
) -> wp.vec3:
    """Type C EPSD rolling resistance torque on particle i (particle-particle).

    Full formulation using per-particle ri, rj to compute R* and delta.

    Elastic-plastic spring-dashpot model:
      k_r = 0.25 * S_t * R_eff^2   (rolling stiffness)
      M_r = -k_r * roll_disp        (elastic torque)
      M_r_max = mu_r * R_eff * |F_n|  (plastic cap)
    """
    if mu_r < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    diff = xj - xi
    dist = wp.length(diff)
    if dist < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    R_eff = (ri * rj) / (ri + rj)
    delta_n = (ri + rj) - dist
    if delta_n <= wp.float32(0.0):
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(R_eff)
    S_t = wp.float32(8.0) * dp.G_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (wp.float32(4.0) / wp.float32(3.0)) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    k_r = wp.float32(0.25) * S_t * R_eff * R_eff
    M_r = roll_disp * (-k_r)
    M_r_mag = wp.length(M_r)
    M_r_max = mu_r * R_eff * wp.abs(F_n_mag)

    if M_r_mag > M_r_max and M_r_mag > wp.float32(1.0e-12):
        M_r = M_r * (M_r_max / M_r_mag)

    return M_r


@wp.func
def update_rolling_disp_pp(
    xi: wp.vec3, xj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    roll_disp: wp.vec3,
    ri: float, rj: float,
    dp: DerivedParams,
    mu_r: float,
    dt: float,
) -> wp.vec3:
    """Update rolling displacement for Type C EPSD model (particle-particle).

    Accumulates relative rolling angular velocity (tangential to contact plane)
    and applies plastic cap when elastic torque exceeds mu_r * R_eff * |F_n|.
    """
    if mu_r < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    diff = xj - xi
    dist = wp.length(diff)
    if dist < wp.float32(1.0e-12):
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist
    R_eff = (ri * rj) / (ri + rj)
    delta_n = (ri + rj) - dist
    if delta_n <= wp.float32(0.0):
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(R_eff)
    S_t = wp.float32(8.0) * dp.G_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (wp.float32(4.0) / wp.float32(3.0)) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    # Relative rolling angular velocity: tangential component only (exclude spin)
    omega_rel = wi - wj
    omega_roll = omega_rel - n * wp.dot(omega_rel, n)

    new_roll = roll_disp + omega_roll * dt

    k_r = wp.float32(0.25) * S_t * R_eff * R_eff
    M_r_max = mu_r * R_eff * wp.abs(F_n_mag)
    M_r_spring_mag = wp.length(new_roll) * k_r

    if M_r_spring_mag > M_r_max and M_r_spring_mag > wp.float32(1.0e-12):
        new_roll = new_roll * (M_r_max / M_r_spring_mag)

    return new_roll


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def clear_forces_kernel(
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    forces[tid] = wp.vec3(0.0, 0.0, 0.0)
    torques[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_gravity_kernel(
    forces: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
):
    tid = wp.tid()
    forces[tid] = forces[tid] + gravity * masses[tid]


@wp.kernel
def apply_global_damping_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    damping: float,
    masses: wp.array(dtype=wp.float32),
    inertias: wp.array(dtype=wp.float32),
):
    """Apply global viscous damping (drag) proportional to velocity.

    Models aerodynamic drag / interstitial-fluid resistance.
    F_drag = -damping * mass * velocity
    """
    tid = wp.tid()
    v = velocities[tid]
    w = angular_velocities[tid]
    m = masses[tid]
    I = inertias[tid]
    forces[tid]  = forces[tid]  - v * (damping * m)
    torques[tid] = torques[tid] - w * (damping * I)


@wp.kernel
def compute_particle_forces_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    tangent_disps: wp.array(dtype=wp.vec3, ndim=2),
    roll_disps: wp.array(dtype=wp.vec3, ndim=2),
    contact_ids: wp.array(dtype=wp.int32, ndim=2),
    contact_counts: wp.array(dtype=wp.int32),
    radii: wp.array(dtype=wp.float32),
    masses: wp.array(dtype=wp.float32),
    grid: wp.uint64,
    max_radius: float,             # largest particle radius in simulation
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    mu_r: float,
    dt: float,
    num_particles: int,
    max_contacts: int,
):
    """Compute particle-particle contact forces using hash grid neighbor search.

    Per-particle radii and masses are used to compute per-contact effective
    radius R* = R_i*R_j/(R_i+R_j) and effective mass m* = m_i*m_j/(m_i+m_j).
    The hash grid search radius is ri + max_radius to guarantee that contacts
    between a small and the largest particle are never missed.

    Includes Hertz-Mindlin contact (with static/dynamic Coulomb friction) and
    Type C EPSD rolling resistance.
    """
    i = wp.tid()

    xi = positions[i]
    vi = velocities[i]
    wi = angular_velocities[i]
    ri = radii[i]
    mi = masses[i]

    # Search radius: own radius + largest possible neighbour radius
    query_radius = ri + max_radius

    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_torque = wp.vec3(0.0, 0.0, 0.0)

    new_count = int(0)

    query = wp.hash_grid_query(grid, xi, query_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        j = index
        if j == i:
            continue

        xj = positions[j]
        rj = radii[j]

        diff = xj - xi
        dist = wp.length(diff)

        # Actual contact + cohesion range uses per-particle sum of radii
        sum_r = ri + rj
        cohesion_range = wp.float32(0.1) * sum_r

        if dist > sum_r + cohesion_range:
            continue

        mj = masses[j]
        vj = velocities[j]
        wj = angular_velocities[j]

        # Find existing tangential and rolling displacements for this contact
        tangent = wp.vec3(0.0, 0.0, 0.0)
        roll    = wp.vec3(0.0, 0.0, 0.0)
        old_count = contact_counts[i]
        for k in range(max_contacts):
            if k >= old_count:
                break
            if contact_ids[i, k] == j:
                tangent = tangent_disps[i, k]
                roll    = roll_disps[i, k]
                break

        # Compute contact force (Hertz-Mindlin with per-particle R*, m*)
        f = hertz_mindlin_force(
            xi, xj, vi, vj, wi, wj, tangent,
            ri, rj, mi, mj,
            dp, mu_s, mu_d, dt,
        )
        total_force = total_force + f

        # Tangential torque: r x F_t at contact point (arm = ri * n)
        if dist > wp.float32(1.0e-12):
            n = diff / dist
            f_n_comp = n * wp.dot(f, n)
            f_t_comp = f - f_n_comp
            total_torque = total_torque + wp.cross(n * ri, f_t_comp)

        # Rolling resistance torque (Type C EPSD) using per-particle radii
        M_r = compute_rolling_torque_pp(xi, xj, wi, wj, roll, ri, rj, dp, mu_r)
        total_torque = total_torque + M_r

        # Update contact history
        if new_count < max_contacts:
            new_tangent = hertz_mindlin_update_tangent(
                xi, xj, vi, vj, wi, wj, tangent,
                ri, rj, mi, mj,
                dp, mu_s, mu_d, dt,
            )
            new_roll = update_rolling_disp_pp(xi, xj, wi, wj, roll, ri, rj, dp, mu_r, dt)
            contact_ids[i, new_count] = j
            tangent_disps[i, new_count] = new_tangent
            roll_disps[i, new_count] = new_roll
            new_count = new_count + 1

    # Clear remaining old contact slots
    for k in range(new_count, max_contacts):
        contact_ids[i, k] = -1
        tangent_disps[i, k] = wp.vec3(0.0, 0.0, 0.0)
        roll_disps[i, k]    = wp.vec3(0.0, 0.0, 0.0)

    contact_counts[i] = new_count

    wp.atomic_add(forces, i, total_force)
    wp.atomic_add(torques, i, total_torque)


@wp.kernel
def compute_mesh_forces_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    mesh_tangent_disps: wp.array(dtype=wp.vec3),
    mesh_roll_disps: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    masses: wp.array(dtype=wp.float32),
    mesh_id: wp.uint64,
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    mu_r: float,
    dt: float,
    surface_velocity: wp.vec3,
    linear_velocity: wp.vec3,
    angular_velocity: wp.vec3,
    mesh_origin: wp.vec3,
):
    """Compute particle-mesh contact forces using BVH mesh query.

    Per-particle radius and mass are used.  For particle-wall contacts the
    wall has infinite radius (R_eff = R_particle) and infinite mass
    (m_eff = m_particle).

    Wall velocity at the contact point is:
        v_wall = linear_velocity + cross(angular_velocity, mesh_point - mesh_origin)
                 + surface_velocity
    Friction and rolling resistance act relative to that wall velocity.
    Includes Type C EPSD rolling resistance for particle-wall contacts.
    """
    i = wp.tid()

    xi = positions[i]
    vi = velocities[i]
    wi = angular_velocities[i]
    radius = radii[i]             # per-particle radius
    particle_mass = masses[i]     # per-particle mass

    max_dist = radius * wp.float32(1.5)

    # JKR-inspired cohesion: wall has infinite R → R_eff = radius
    cohesion_pulloff = wp.float32(1.5) * wp.float32(3.14159265) * dp.cohesion_energy_gamma * radius

    query = wp.mesh_query_point_sign_normal(mesh_id, xi, max_dist)

    if query.result:
        mesh_point = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        to_particle = xi - mesh_point
        dist = wp.length(to_particle)

        if dist < wp.float32(1.0e-12):
            return

        n = to_particle / dist  # outward normal (from mesh toward particle)

        delta_n = radius - dist

        if delta_n <= wp.float32(0.0):
            cohesion_range = wp.float32(0.1) * radius
            if delta_n > -cohesion_range and cohesion_pulloff > wp.float32(0.0):
                cohesion_frac = wp.float32(1.0) + delta_n / cohesion_range
                f_cohesion = n * (-cohesion_pulloff * cohesion_frac)
                wp.atomic_add(forces, i, f_cohesion)
            return

        sqrt_delta = wp.sqrt(delta_n)
        sqrt_R = wp.sqrt(radius)  # R_eff for particle-wall = R_particle (wall has infinite radius)

        # Hertz normal stiffness (particle-wall: R_eff = R, m_eff = m_particle)
        S_n = wp.float32(2.0) * dp.E_eff * sqrt_R * sqrt_delta
        F_n_mag = (wp.float32(4.0) / wp.float32(3.0)) * dp.E_eff * sqrt_R * delta_n * sqrt_delta

        # Wall velocity at contact: rigid-body motion + rotational arm + belt slip
        arm = mesh_point - mesh_origin
        v_wall = linear_velocity + wp.cross(angular_velocity, arm) + surface_velocity
        v_rel = vi - v_wall
        v_n_mag = wp.dot(v_rel, n)
        v_n = n * v_n_mag
        v_t = v_rel - v_n

        # Normal damping using full particle mass (wall has infinite mass → m_eff = m_particle)
        damp_n_coeff = -wp.float32(2.0) * wp.sqrt(wp.float32(5.0) / wp.float32(6.0)) * dp.beta * wp.sqrt(S_n * particle_mass)
        F_nd_mag = damp_n_coeff * v_n_mag

        F_cohesion_mag = -cohesion_pulloff

        F_normal = n * (F_n_mag + F_nd_mag - F_cohesion_mag)
        F_n_total_mag = F_n_mag + F_nd_mag

        S_t = wp.float32(8.0) * dp.G_eff * sqrt_R * sqrt_delta
        old_tangent = mesh_tangent_disps[i]
        new_tangent = old_tangent + v_t * dt

        F_t_spring = new_tangent * (-S_t)
        damp_t_coeff = -wp.float32(2.0) * wp.sqrt(wp.float32(5.0) / wp.float32(6.0)) * dp.beta * wp.sqrt(S_t * particle_mass)
        F_t_damp = v_t * damp_t_coeff
        F_t_total = F_t_spring + F_t_damp

        F_t_mag = wp.length(F_t_total)
        static_limit = mu_s * wp.abs(F_n_total_mag)

        if F_t_mag > static_limit and F_t_mag > wp.float32(1.0e-12):
            kinetic_limit = mu_d * wp.abs(F_n_total_mag)
            F_t_total = F_t_total * (kinetic_limit / F_t_mag)
            new_tangent = new_tangent * (kinetic_limit / (wp.length(new_tangent) * S_t + wp.float32(1.0e-12)))

        mesh_tangent_disps[i] = new_tangent

        # Rolling resistance (Type C EPSD) — particle vs. static/moving wall
        omega_roll = wi - n * wp.dot(wi, n)
        old_roll = mesh_roll_disps[i]
        new_roll = old_roll + omega_roll * dt

        k_r = wp.float32(0.25) * S_t * radius * radius
        M_r_limit = mu_r * radius * wp.abs(F_n_total_mag)
        M_r_spring = new_roll * (-k_r)
        M_r_spring_mag = wp.length(M_r_spring)

        if M_r_spring_mag > M_r_limit and M_r_spring_mag > wp.float32(1.0e-12):
            scale = M_r_limit / M_r_spring_mag
            M_r_spring = M_r_spring * scale
            new_roll = new_roll * scale

        mesh_roll_disps[i] = new_roll

        contact_torque = wp.cross(n * (-radius), F_t_total) + M_r_spring
        total_force = F_normal + F_t_total

        wp.atomic_add(forces, i, total_force)
        wp.atomic_add(torques, i, contact_torque)
    else:
        mesh_tangent_disps[i] = wp.vec3(0.0, 0.0, 0.0)
        mesh_roll_disps[i]    = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def integrate_phase1_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertias: wp.array(dtype=wp.float32),
    dt: float,
):
    """Velocity Verlet phase 1: half-step velocity, full-step position."""
    tid = wp.tid()

    v = velocities[tid]
    w = angular_velocities[tid]
    f = forces[tid]
    t = torques[tid]
    inv_m = inv_masses[tid]
    inv_I = inv_inertias[tid]

    v = v + f * (wp.float32(0.5) * dt * inv_m)
    w = w + t * (wp.float32(0.5) * dt * inv_I)

    positions[tid] = positions[tid] + v * dt

    velocities[tid] = v
    angular_velocities[tid] = w


@wp.kernel
def integrate_phase2_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertias: wp.array(dtype=wp.float32),
    dt: float,
):
    """Velocity Verlet phase 2: complete the velocity step with new forces."""
    tid = wp.tid()
    inv_m = inv_masses[tid]
    inv_I = inv_inertias[tid]
    velocities[tid] = velocities[tid] + forces[tid] * (wp.float32(0.5) * dt * inv_m)
    angular_velocities[tid] = angular_velocities[tid] + torques[tid] * (wp.float32(0.5) * dt * inv_I)


@wp.kernel
def translate_mesh_vertices_kernel(
    points: wp.array(dtype=wp.vec3),
    velocity: wp.vec3,
    dt: float,
):
    """Shift every mesh vertex by velocity * dt for kinematic rigid-body translation."""
    i = wp.tid()
    points[i] = points[i] + velocity * dt


@wp.kernel
def transform_mesh_vertices_kernel(
    rest_points: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    quat: wp.quat,
    translation: wp.vec3,
):
    """Apply a full rigid-body transform to mesh vertices from rest-frame positions.

    p_world = rotate(quat, p_rest - origin) + origin + translation
    """
    i = wp.tid()
    p_local = rest_points[i] - origin
    p_rotated = wp.quat_rotate(quat, p_local)
    points[i] = p_rotated + origin + translation


@wp.kernel
def rotate_contact_history_kernel(
    tangent_disps: wp.array(dtype=wp.vec3),
    roll_disps: wp.array(dtype=wp.vec3),
    step_quat: wp.quat,
):
    """Rotate per-particle contact spring displacements by the mesh's step rotation.

    Without this, world-space tangent/roll vectors become incorrect after the
    mesh rotates, causing friction force errors proportional to the rotation
    per step.
    """
    i = wp.tid()
    tangent_disps[i] = wp.quat_rotate(step_quat, tangent_disps[i])
    roll_disps[i]    = wp.quat_rotate(step_quat, roll_disps[i])


@wp.kernel
def compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    inertias: wp.array(dtype=wp.float32),
    energy: wp.array(dtype=float),
):
    """Compute per-particle kinetic energy for diagnostics."""
    tid = wp.tid()
    v = velocities[tid]
    w = angular_velocities[tid]
    m = masses[tid]
    I = inertias[tid]
    ke = wp.float32(0.5) * m * wp.dot(v, v) + wp.float32(0.5) * I * wp.dot(w, w)
    wp.atomic_add(energy, 0, ke)


# ---------------------------------------------------------------------------
# Host-side mesh loading
# ---------------------------------------------------------------------------

def load_mesh(filepath: str, scale: float = 1.0, device: str = "cuda:0") -> wp.Mesh:
    """Load a triangular mesh from OBJ or STL file using trimesh."""
    import trimesh

    mesh = trimesh.load(filepath, force="mesh")
    vertices = np.array(mesh.vertices, dtype=np.float32) * scale
    faces = np.array(mesh.faces, dtype=np.int32).flatten()

    wp_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    return wp_mesh


def create_box_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5),
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a simple box mesh (open-top) for testing."""
    cx, cy, cz = center
    hx, hy, hz = half_extents

    vertices = np.array([
        [cx - hx, cy - hy, cz - hz],  # 0
        [cx + hx, cy - hy, cz - hz],  # 1
        [cx + hx, cy + hy, cz - hz],  # 2
        [cx - hx, cy + hy, cz - hz],  # 3
        [cx - hx, cy - hy, cz + hz],  # 4
        [cx + hx, cy - hy, cz + hz],  # 5
        [cx + hx, cy + hy, cz + hz],  # 6
        [cx - hx, cy + hy, cz + hz],  # 7
    ], dtype=np.float32)

    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32).flatten()

    wp_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    return wp_mesh


def create_plane_mesh(
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    size: float = 10.0,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a large planar mesh (two triangles)."""
    n = np.array(normal, dtype=np.float32)
    n = n / np.linalg.norm(n)

    if abs(n[0]) < 0.9:
        u = np.cross(n, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(n, np.array([0.0, 1.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    o = np.array(origin, dtype=np.float32)
    hs = size * 0.5

    vertices = np.array([
        o - hs * u - hs * v,
        o + hs * u - hs * v,
        o + hs * u + hs * v,
        o - hs * u + hs * v,
    ], dtype=np.float32)

    faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

    wp_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    return wp_mesh


def create_rect_mesh(
    x0: float, x1: float,
    y0: float, y1: float,
    z: float,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a horizontal rectangular mesh (z = const), facing +z."""
    vertices = np.array([
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
    ], dtype=np.float32)
    faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )


def create_cylinder_mesh(
    radius: float = 0.15,
    length: float = 0.30,
    n_theta: int = 32,
    end_caps: bool = True,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a cylindrical drum mesh with inward-facing normals.

    The cylinder axis lies along Y.  Particles placed inside the drum contact
    the inner surface.  End caps (flat circles) are optionally added so the
    drum is fully enclosed.

    Args:
        radius: Cylinder inner radius (m).
        length: Cylinder length along Y (m).
        n_theta: Number of circumferential segments.
        end_caps: If True, flat circular end caps are included.
        device: Warp device string.

    Returns:
        wp.Mesh centred at the world origin with inward-facing normals.
    """
    half_L = length * 0.5
    thetas  = [2.0 * math.pi * k / n_theta for k in range(n_theta)]

    verts: list[list[float]] = []
    # Ring 0: y = -half_L  (indices 0 .. n_theta-1)
    for t in thetas:
        verts.append([radius * math.cos(t), -half_L, radius * math.sin(t)])
    # Ring 1: y = +half_L  (indices n_theta .. 2*n_theta-1)
    for t in thetas:
        verts.append([radius * math.cos(t),  half_L, radius * math.sin(t)])

    faces: list[int] = []
    # Lateral faces — inward normals (CW from outside)
    # Winding: [BL, BR, TR], [BL, TR, TL]  →  normal points inward ✓
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        BL, BR = k,           k1
        TL, TR = k + n_theta, k1 + n_theta
        faces += [BL, BR, TR,  BL, TR, TL]

    if end_caps:
        # Bottom cap at y = -half_L, normal = +Y (toward interior)
        bot_ctr = len(verts)
        verts.append([0.0, -half_L, 0.0])
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            faces += [bot_ctr, k, k1]           # CCW from +Y → normal = +Y

        # Top cap at y = +half_L, normal = -Y (toward interior)
        top_ctr = len(verts)
        verts.append([0.0, half_L, 0.0])
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            TL_i = k  + n_theta
            TR_i = k1 + n_theta
            faces += [top_ctr, TR_i, TL_i]      # reversed → normal = -Y

    verts_np = np.array(verts, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts_np, dtype=wp.vec3, device=device),
        indices=wp.array(faces_np, dtype=wp.int32, device=device),
    )


def create_drum_with_lifters_mesh(
    radius: float = 0.15,
    length: float = 0.30,
    n_theta: int = 48,
    n_lifters: int = 6,
    lifter_height: float = 0.035,
    lifter_half_angle: float = 0.06,
    end_caps: bool = True,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a cylindrical drum mesh with rectangular lifter blades.

    Lifters are radial rectangular fins evenly spaced around the inner
    circumference.  They protrude inward by *lifter_height* and have an
    angular half-width of *lifter_half_angle* radians.  The cylinder axis
    is along Y, centred at the world origin with inward-facing normals.

    Args:
        radius:            Drum inner radius (m).
        length:            Drum length along Y (m).
        n_theta:           Circumferential segments for the cylinder wall.
        n_lifters:         Number of evenly spaced lifter blades.
        lifter_height:     Lifter protrusion from the drum wall (m).
        lifter_half_angle: Angular half-width of each lifter (radians).
        end_caps:          If True, flat end caps are included.
        device:            Warp device string.

    Returns:
        wp.Mesh centred at the world origin with inward-facing normals.
    """
    half_L  = length * 0.5
    thetas  = [2.0 * math.pi * k / n_theta for k in range(n_theta)]

    verts: list[list[float]] = []
    faces: list[int] = []

    # ── Cylinder lateral surface ──────────────────────────────────────────────
    for t in thetas:
        verts.append([radius * math.cos(t), -half_L, radius * math.sin(t)])  # ring 0
    for t in thetas:
        verts.append([radius * math.cos(t),  half_L, radius * math.sin(t)])  # ring 1

    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        BL, BR = k,           k1
        TL, TR = k + n_theta, k1 + n_theta
        faces += [BL, BR, TR,  BL, TR, TL]         # inward normals ✓

    # ── End caps ─────────────────────────────────────────────────────────────
    if end_caps:
        bot_ctr = len(verts)
        verts.append([0.0, -half_L, 0.0])
        for k in range(n_theta):
            faces += [bot_ctr, k, (k + 1) % n_theta]

        top_ctr = len(verts)
        verts.append([0.0, half_L, 0.0])
        for k in range(n_theta):
            TL_i = k  + n_theta
            TR_i = (k + 1) % n_theta + n_theta
            faces += [top_ctr, TR_i, TL_i]

    # ── Lifter blades ─────────────────────────────────────────────────────────
    r_tip = radius - lifter_height
    dth   = lifter_half_angle

    for k in range(n_lifters):
        theta = 2.0 * math.pi * k / n_lifters
        v0 = len(verts)

        # 8 corners: foot (at R) and inner tip (at r_tip), left/right × bot/top
        # Indices relative to v0:
        #   0=flb  1=flt  2=frb  3=frt   (foot: left/right × bot/top)
        #   4=ilb  5=ilt  6=irb  7=irt   (inner: left/right × bot/top)
        for th, y in [(theta - dth, -half_L), (theta - dth, half_L),
                      (theta + dth, -half_L), (theta + dth, half_L)]:
            verts.append([radius * math.cos(th), y, radius * math.sin(th)])
        for th, y in [(theta - dth, -half_L), (theta - dth, half_L),
                      (theta + dth, -half_L), (theta + dth, half_L)]:
            verts.append([r_tip  * math.cos(th), y, r_tip  * math.sin(th)])

        flb, flt, frb, frt = v0,   v0+1, v0+2, v0+3
        ilb, ilt, irb, irt = v0+4, v0+5, v0+6, v0+7

        # Inner (tip) face — the primary particle-contact surface
        faces += [ilb, irb, irt,  ilb, irt, ilt]

        # Left side face
        faces += [flb, ilb, ilt,  flb, ilt, flt]

        # Right side face
        faces += [frb, frt, irt,  frb, irt, irb]

    verts_np = np.array(verts, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts_np, dtype=wp.vec3, device=device),
        indices=wp.array(faces_np, dtype=wp.int32, device=device),
    )


def create_vertical_wall_mesh(
    x0: float, x1: float,
    z0: float, z1: float,
    y: float,
    inward_y: float = 1.0,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a vertical wall at y=const, spanning x0..x1, z0..z1."""
    if inward_y >= 0.0:
        vertices = np.array([
            [x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0],
        ], dtype=np.float32)
    else:
        vertices = np.array([
            [x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1],
        ], dtype=np.float32)
    faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )


def create_endwall_mesh(
    x: float,
    y0: float, y1: float,
    z0: float, z1: float,
    inward_x: float = 1.0,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a vertical end-wall at x=const, spanning y0..y1, z0..z1."""
    if inward_x >= 0.0:
        vertices = np.array([
            [x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1],
        ], dtype=np.float32)
    else:
        vertices = np.array([
            [x, y0, z0], [x, y0, z1], [x, y1, z1], [x, y1, z0],
        ], dtype=np.float32)
    faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )


def create_angled_plate_mesh(
    top_edge: tuple[float, float, float],
    bottom_edge: tuple[float, float, float],
    width: float,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create an angled rectangular plate defined by two edge midpoints."""
    te = np.array(top_edge, dtype=np.float32)
    be = np.array(bottom_edge, dtype=np.float32)
    hw = width * 0.5

    vertices = np.array([
        [te[0], te[1] - hw, te[2]],
        [te[0], te[1] + hw, te[2]],
        [be[0], be[1] + hw, be[2]],
        [be[0], be[1] - hw, be[2]],
    ], dtype=np.float32)

    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[0]
    normal = np.cross(edge1, edge2)

    if normal[0] > 0:
        faces = np.array([0, 2, 1, 0, 3, 2], dtype=np.int32)
    else:
        faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=np.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Quaternion helpers (host-side, used for kinematic mesh integration)
# ---------------------------------------------------------------------------

def _quat_step(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion q by angular velocity omega over timestep dt.

    q  : (x, y, z, w) numpy array — current orientation
    omega : (3,) angular velocity in rad/s (world frame)
    Returns normalised (x, y, z, w) array.
    """
    angle = float(np.linalg.norm(omega)) * dt
    if angle < 1.0e-10:
        return q.copy()
    axis = omega / np.linalg.norm(omega)
    half = angle * 0.5
    s = math.sin(half)
    sq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64)
    # Hamilton product: step_q ⊗ q  (x,y,z,w convention)
    ax, ay, az, aw = sq
    bx, by, bz, bw = q
    result = np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)
    return result / np.linalg.norm(result)


# ---------------------------------------------------------------------------
# MeshBody — kinematic rigid-body state for a collision mesh
# ---------------------------------------------------------------------------

class MeshBody:
    """Holds a wp.Mesh together with its kinematic pose and velocity state.

    Stores rest-frame vertex positions on the GPU so that the full rigid
    transform (translate + rotate) can be applied each timestep without
    accumulating floating-point drift.

    Attributes
    ----------
    mesh             : wp.Mesh — the BVH-backed collision surface
    rest_points_wp   : wp.array(dtype=vec3) — GPU copy of initial vertices
    linear_velocity  : [vx, vy, vz] in m/s — translational rigid-body motion
    angular_velocity : [wx, wy, wz] in rad/s — rotational rigid-body motion
    surface_velocity : [vx, vy, vz] — extra surface slip for friction only
                       (conveyor belt add-on, does not move vertices)
    origin           : (3,) np.float64 — rotation centre in world space
    position         : (3,) np.float64 — accumulated translation from initial pose
    quaternion       : (4,) np.float64 — current orientation (x, y, z, w)
    """

    def __init__(
        self,
        mesh: "wp.Mesh",
        linear_velocity: tuple = (0.0, 0.0, 0.0),
        angular_velocity: tuple = (0.0, 0.0, 0.0),
        surface_velocity: tuple = (0.0, 0.0, 0.0),
        origin: tuple = (0.0, 0.0, 0.0),
        device: str = "cuda:0",
    ):
        self.mesh             = mesh
        self.linear_velocity  = [float(v) for v in linear_velocity]
        self.angular_velocity = [float(v) for v in angular_velocity]
        self.surface_velocity = [float(v) for v in surface_velocity]
        self.origin           = np.array(origin,            dtype=np.float64)
        self.position         = np.zeros(3,                 dtype=np.float64)
        self.quaternion       = np.array([0., 0., 0., 1.],  dtype=np.float64)

        # GPU copy of initial (rest-frame) vertex positions — never mutated
        verts_np = mesh.points.numpy()      # (N, 3) float32
        self.rest_points_wp = wp.array(verts_np, dtype=wp.vec3, device=device)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    num_particles: int = 1000
    particle_radius: float = 0.005       # meters — used when psd is None
    psd: Optional[ParticleSizeDistribution] = None  # particle size distribution
    particle_density: float = 2500.0     # kg/m^3 (glass beads)
    young_modulus: float = 1.0e7         # Pa
    poisson_ratio: float = 0.3
    restitution: float = 0.5
    friction_static: float = 0.5         # mu_s
    friction_dynamic: float = 0.4        # mu_d
    friction_rolling: float = 0.01       # mu_r
    cohesion_energy: float = 0.0         # J/m^2 (0 = no cohesion)
    dt: float = 1.0e-5                   # seconds
    gravity: tuple = (0.0, 0.0, -9.81)
    max_contacts_per_particle: int = 32
    hash_grid_dim: int = 128
    global_damping: float = 0.0          # 1/s viscous drag coefficient
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    """DEM simulation engine managing state, contacts, and time integration.

    Supports both uniform radius (backward-compatible) and particle size
    distributions (PSD).  When ``config.psd`` is set, each particle receives
    an individual radius drawn from the PSD; masses, inertias, and the hash
    grid cell size are all derived from the per-particle radii.

    Example — uniform radius (unchanged API)::

        config = SimConfig(num_particles=1000, particle_radius=0.005)
        sim = Simulation(config)

    Example — PSD::

        psd = ParticleSizeDistribution([(0.01, 30), (0.015, 50), (0.025, 20)])
        config = SimConfig(num_particles=1000, psd=psd)
        sim = Simulation(config)
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.device = config.device
        self.num_particles = config.num_particles
        self.active_count = config.num_particles
        self.sim_time = 0.0
        self.step_count = 0

        # --- Material constants (same for all particles / contacts) ---
        E  = config.young_modulus
        nu = config.poisson_ratio
        e  = config.restitution
        rho = config.particle_density

        self.E_eff = E / (2.0 * (1.0 - nu * nu))
        self.G_eff = E / (2.0 * (2.0 - nu) * (1.0 + nu))

        if e > 0.0:
            ln_e = math.log(e)
            self.beta = -ln_e / math.sqrt(ln_e ** 2 + math.pi ** 2)
        else:
            self.beta = 1.0

        # --- Per-particle radii and derived quantities ---
        N = self.num_particles

        if config.psd is not None:
            # PSD path: assign radii according to the distribution
            radii_np = config.psd.assign_radii(N)
            self.max_radius = config.psd.max_radius
        else:
            # Uniform radius (backward-compatible path)
            radii_np = np.full(N, float(config.particle_radius), dtype=np.float32)
            self.max_radius = float(config.particle_radius)

        masses_np   = (rho * (4.0 / 3.0) * math.pi * radii_np ** 3).astype(np.float32)
        inertias_np = (0.4 * masses_np * radii_np ** 2).astype(np.float32)

        # Scalar representatives kept for backward-compatible diagnostics
        self.particle_mass    = float(masses_np[0])
        self.particle_inertia = float(inertias_np[0])

        # Pre-computed inverse arrays used in Velocity Verlet integration
        inv_masses_np   = (1.0 / masses_np).astype(np.float32)
        inv_inertias_np = (1.0 / inertias_np).astype(np.float32)

        # JKR cohesion: representative pull-off for diagnostics only
        R_eff_rep = self.max_radius / 2.0
        self.cohesion_pulloff = 1.5 * math.pi * config.cohesion_energy * R_eff_rep

        # --- Warp structs ---
        self.mat_params = MaterialParams()
        self.mat_params.young_modulus   = float(E)
        self.mat_params.poisson_ratio   = float(nu)
        self.mat_params.restitution     = float(e)
        self.mat_params.friction_static = float(config.friction_static)
        self.mat_params.friction_rolling = float(config.friction_rolling)
        self.mat_params.cohesion_energy = float(config.cohesion_energy)
        self.mat_params.particle_radius = float(self.max_radius)
        self.mat_params.particle_density = float(rho)

        self.derived_params = DerivedParams()
        self.derived_params.E_eff                = float(self.E_eff)
        self.derived_params.G_eff                = float(self.G_eff)
        self.derived_params.beta                 = float(self.beta)
        self.derived_params.cohesion_energy_gamma = float(config.cohesion_energy)

        self.gravity_vec = wp.vec3(*config.gravity)

        # --- Allocate state arrays ---
        MC = config.max_contacts_per_particle

        self.positions          = wp.zeros(N, dtype=wp.vec3,    device=self.device)
        self.velocities         = wp.zeros(N, dtype=wp.vec3,    device=self.device)
        self.angular_velocities = wp.zeros(N, dtype=wp.vec3,    device=self.device)
        self.forces             = wp.zeros(N, dtype=wp.vec3,    device=self.device)
        self.torques            = wp.zeros(N, dtype=wp.vec3,    device=self.device)

        # Per-particle physical properties
        self.radii        = wp.array(radii_np,       dtype=wp.float32, device=self.device)
        self.particle_masses   = wp.array(masses_np,      dtype=wp.float32, device=self.device)
        self.particle_inertias = wp.array(inertias_np,    dtype=wp.float32, device=self.device)
        self.inv_masses   = wp.array(inv_masses_np,  dtype=wp.float32, device=self.device)
        self.inv_inertias = wp.array(inv_inertias_np, dtype=wp.float32, device=self.device)

        # Particle-particle contact history
        self.tangent_disps  = wp.zeros((N, MC), dtype=wp.vec3,    device=self.device)
        self.roll_disps     = wp.zeros((N, MC), dtype=wp.vec3,    device=self.device)
        self.contact_ids    = wp.full((N, MC),  value=-1, dtype=wp.int32, device=self.device)
        self.contact_counts = wp.zeros(N, dtype=wp.int32, device=self.device)

        # Per-mesh contact history lists
        self._mesh_tangent_disps_list: list[wp.array] = []
        self._mesh_roll_disps_list: list[wp.array]    = []

        # Hash grid — cell size based on max_radius so that contacts between the
        # smallest and largest particles are never missed by the broad-phase.
        dim = config.hash_grid_dim
        self.hash_grid = wp.HashGrid(dim, dim, dim, device=self.device)
        self.cell_size = 2.0 * self.max_radius * 2.1

        # Mesh list — each entry is a MeshBody (see class definition above)
        self.meshes: list[MeshBody] = []

        # Diagnostic energy buffer
        self._energy_buf = wp.zeros(1, dtype=float, device=self.device)

    def initialize_particles(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None,
    ):
        """Set initial particle positions (and optionally velocities).

        Args:
            positions: (N, 3) float32 array of particle centers.
            velocities: (N, 3) float32 array, default zeros.
            angular_velocities: (N, 3) float32 array, default zeros.
        """
        assert positions.shape == (self.num_particles, 3), \
            f"Expected shape ({self.num_particles}, 3), got {positions.shape}"

        self.positions = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=self.device)

        if velocities is not None:
            self.velocities = wp.array(velocities.astype(np.float32), dtype=wp.vec3, device=self.device)

        if angular_velocities is not None:
            self.angular_velocities = wp.array(
                angular_velocities.astype(np.float32), dtype=wp.vec3, device=self.device
            )

    def add_mesh(
        self,
        mesh: wp.Mesh,
        linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Add a collision mesh to the simulation.

        Args:
            mesh: Warp Mesh object.
            linear_velocity: Rigid-body translational velocity (vx, vy, vz) in m/s.
                Vertices are physically moved and the BVH is refitted each step.
            angular_velocity: Rigid-body angular velocity (wx, wy, wz) in rad/s.
                Rotation is applied about *origin* each step.
            surface_velocity: Additional surface slip (vx, vy, vz) in m/s for the
                friction calculation only — geometry does not move.  Use for
                conveyor-belt surfaces that are not themselves translating.
            origin: Rotation centre (x, y, z) in world space.  Ignored when
                angular_velocity is zero.
        """
        body = MeshBody(
            mesh,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            surface_velocity=surface_velocity,
            origin=origin,
            device=self.device,
        )
        self.meshes.append(body)
        N = self.num_particles
        self._mesh_tangent_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        self._mesh_roll_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))

    def add_mesh_from_file(
        self,
        filepath: str,
        scale: float = 1.0,
        linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> wp.Mesh:
        """Load and add a mesh from an OBJ or STL file."""
        mesh = load_mesh(filepath, scale=scale, device=self.device)
        body = MeshBody(
            mesh,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            surface_velocity=surface_velocity,
            origin=origin,
            device=self.device,
        )
        self.meshes.append(body)
        N = self.num_particles
        self._mesh_tangent_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        self._mesh_roll_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        return mesh

    def set_mesh_velocity(
        self,
        mesh_idx: int,
        linear_velocity: tuple[float, float, float],
    ):
        """Change the linear velocity of a mesh during the simulation.

        Args:
            mesh_idx: Index of the mesh (order it was added via add_mesh).
            linear_velocity: New (vx, vy, vz) velocity in m/s.
        """
        self.meshes[mesh_idx].linear_velocity = [float(v) for v in linear_velocity]

    def set_mesh_angular_velocity(
        self,
        mesh_idx: int,
        angular_velocity: tuple[float, float, float],
        origin: tuple[float, float, float] | None = None,
    ):
        """Change the angular velocity of a mesh during the simulation.

        Args:
            mesh_idx: Index of the mesh (order it was added via add_mesh).
            angular_velocity: New (wx, wy, wz) angular velocity in rad/s.
            origin: Rotation centre (x, y, z) in world space.  If None the
                existing origin is kept.
        """
        body = self.meshes[mesh_idx]
        body.angular_velocity = [float(v) for v in angular_velocity]
        if origin is not None:
            body.origin = np.array(origin, dtype=np.float64)

    def get_mesh_poses(self) -> list[dict]:
        """Return the current pose of every mesh as a list of dicts.

        Each dict has:
            pos  -- [x, y, z] accumulated translation from the initial position
            quat -- [qx, qy, qz, qw] current orientation quaternion

        Suitable for embedding in the animation JSON consumed by hopper_viewer.py.
        The viewer applies these as a Three.js position + quaternion on the STL
        mesh which was loaded at its rest-frame vertex positions.
        """
        poses = []
        for body in self.meshes:
            p = body.position
            q = body.quaternion
            poses.append({
                "pos":  [round(float(p[0]), 5),
                         round(float(p[1]), 5),
                         round(float(p[2]), 5)],
                "quat": [round(float(q[0]), 6),
                         round(float(q[1]), 6),
                         round(float(q[2]), 6),
                         round(float(q[3]), 6)],
            })
        return poses

    def step(self):
        """Advance simulation by one timestep using Velocity Verlet integration.

        Only processes the first ``self.active_count`` particles.
        """
        N  = self.active_count
        dt = self.config.dt
        MC = self.config.max_contacts_per_particle

        mu_s = self.config.friction_static
        mu_d = self.config.friction_dynamic
        mu_r = self.config.friction_rolling

        # Phase 1: half-step velocity + full position update
        wp.launch(
            integrate_phase1_kernel,
            dim=N,
            inputs=[
                self.positions, self.velocities, self.angular_velocities,
                self.forces, self.torques,
                self.inv_masses, self.inv_inertias, dt,
            ],
            device=self.device,
        )

        # Kinematic mesh motion — move vertices to their t+dt position and refit
        # the BVH so contacts are resolved against the current geometry.
        for i_body, body in enumerate(self.meshes):
            lv = body.linear_velocity
            av = body.angular_velocity
            has_rot = av[0] != 0.0 or av[1] != 0.0 or av[2] != 0.0
            has_lin = lv[0] != 0.0 or lv[1] != 0.0 or lv[2] != 0.0

            if has_rot:
                omega = np.array(av, dtype=np.float64)
                # Compute step quaternion (exact axis-angle rotation for this dt)
                angle = float(np.linalg.norm(omega)) * dt
                if angle >= 1.0e-10:
                    axis = omega / np.linalg.norm(omega)
                    half = angle * 0.5
                    s = math.sin(half)
                    step_q_arr = np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(half)])
                    step_q_wp = wp.quat(
                        float(step_q_arr[0]), float(step_q_arr[1]),
                        float(step_q_arr[2]), float(step_q_arr[3]),
                    )
                    # Rotate stored contact spring displacements into the new frame
                    tang = self._mesh_tangent_disps_list[i_body]
                    roll = self._mesh_roll_disps_list[i_body]
                    wp.launch(
                        rotate_contact_history_kernel,
                        dim=N,
                        inputs=[tang, roll, step_q_wp],
                        device=self.device,
                    )
                    # Integrate quaternion: new_q = step_q ⊗ old_q
                    body.quaternion = _quat_step(body.quaternion, omega, dt)

                body.position += np.array(lv, dtype=np.float64) * dt

                # Apply full rigid transform from rest-frame vertices
                q = body.quaternion
                quat_wp = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                orig_wp = wp.vec3(float(body.origin[0]), float(body.origin[1]), float(body.origin[2]))
                pos_wp  = wp.vec3(float(body.position[0]), float(body.position[1]), float(body.position[2]))
                wp.launch(
                    transform_mesh_vertices_kernel,
                    dim=body.mesh.points.shape[0],
                    inputs=[body.rest_points_wp, body.mesh.points, orig_wp, quat_wp, pos_wp],
                    device=self.device,
                )
                body.mesh.refit()

            elif has_lin:
                # Pure translation — cheaper path
                lv_wp = wp.vec3(lv[0], lv[1], lv[2])
                wp.launch(
                    translate_mesh_vertices_kernel,
                    dim=body.mesh.points.shape[0],
                    inputs=[body.mesh.points, lv_wp, dt],
                    device=self.device,
                )
                body.mesh.refit()
                body.position[0] += lv[0] * dt
                body.position[1] += lv[1] * dt
                body.position[2] += lv[2] * dt

        # Clear forces
        wp.launch(
            clear_forces_kernel,
            dim=N,
            inputs=[self.forces, self.torques],
            device=self.device,
        )

        # Apply gravity (per-particle mass)
        wp.launch(
            apply_gravity_kernel,
            dim=N,
            inputs=[self.forces, self.particle_masses, self.gravity_vec],
            device=self.device,
        )

        # Apply global viscous damping (if configured)
        if self.config.global_damping > 0.0:
            wp.launch(
                apply_global_damping_kernel,
                dim=N,
                inputs=[
                    self.velocities, self.angular_velocities,
                    self.forces, self.torques,
                    self.config.global_damping,
                    self.particle_masses, self.particle_inertias,
                ],
                device=self.device,
            )

        # Rebuild hash grid — cell size based on max_radius (set in __init__)
        self.hash_grid.build(self.positions, self.cell_size)

        # Particle-particle forces (Hertz-Mindlin + Coulomb + Type C rolling)
        wp.launch(
            compute_particle_forces_kernel,
            dim=N,
            inputs=[
                self.positions, self.velocities, self.angular_velocities,
                self.forces, self.torques,
                self.tangent_disps, self.roll_disps,
                self.contact_ids, self.contact_counts,
                self.radii, self.particle_masses,
                self.hash_grid.id,
                self.max_radius,
                self.derived_params,
                mu_s, mu_d, mu_r, dt, N, MC,
            ],
            device=self.device,
        )

        # Particle-mesh forces
        for body, tang, roll in zip(
            self.meshes, self._mesh_tangent_disps_list, self._mesh_roll_disps_list
        ):
            lv = body.linear_velocity
            av = body.angular_velocity
            sv = body.surface_velocity
            og = body.origin
            wp.launch(
                compute_mesh_forces_kernel,
                dim=N,
                inputs=[
                    self.positions, self.velocities, self.angular_velocities,
                    self.forces, self.torques,
                    tang, roll,
                    self.radii, self.particle_masses,
                    body.mesh.id,
                    self.derived_params,
                    mu_s, mu_d, mu_r, dt,
                    wp.vec3(sv[0], sv[1], sv[2]),
                    wp.vec3(lv[0], lv[1], lv[2]),
                    wp.vec3(av[0], av[1], av[2]),
                    wp.vec3(float(og[0]), float(og[1]), float(og[2])),
                ],
                device=self.device,
            )

        # Phase 2: complete velocity step
        wp.launch(
            integrate_phase2_kernel,
            dim=N,
            inputs=[
                self.velocities, self.angular_velocities,
                self.forces, self.torques,
                self.inv_masses, self.inv_inertias, dt,
            ],
            device=self.device,
        )

        self.sim_time += dt
        self.step_count += 1

    def advance(self, num_steps: int):
        """Run multiple timesteps."""
        for _ in range(num_steps):
            self.step()

    def get_positions(self) -> np.ndarray:
        """Return particle positions as (N, 3) numpy array."""
        return self.positions.numpy()

    def get_velocities(self) -> np.ndarray:
        """Return particle velocities as (N, 3) numpy array."""
        return self.velocities.numpy()

    def get_angular_velocities(self) -> np.ndarray:
        """Return particle angular velocities as (N, 3) numpy array."""
        return self.angular_velocities.numpy()

    def get_radii(self) -> np.ndarray:
        """Return per-particle radii as (N,) float32 numpy array."""
        return self.radii.numpy()

    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy (translational + rotational)."""
        self._energy_buf.zero_()
        wp.launch(
            compute_kinetic_energy_kernel,
            dim=self.num_particles,
            inputs=[
                self.velocities, self.angular_velocities,
                self.particle_masses, self.particle_inertias,
                self._energy_buf,
            ],
            device=self.device,
        )
        return float(self._energy_buf.numpy()[0])


# ---------------------------------------------------------------------------
# Utility: generate particle positions
# ---------------------------------------------------------------------------

def generate_grid_positions(
    num_particles: int,
    radius: float,
    spacing_factor: float = 2.2,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Generate particle positions on a cubic grid.

    Args:
        num_particles: Target number of particles.
        radius: Particle radius (or max_radius for PSD simulations).
        spacing_factor: Multiplier on radius for grid spacing (>2.0 avoids overlap).
        origin: Center of the bottom layer.

    Returns:
        (N, 3) float32 array of positions.
    """
    spacing = radius * spacing_factor
    n_side = int(math.ceil(num_particles ** (1.0 / 3.0)))

    positions = []
    ox, oy, oz = origin
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) >= num_particles:
                    break
                x = ox + (ix - n_side / 2.0 + 0.5) * spacing
                y = oy + (iy - n_side / 2.0 + 0.5) * spacing
                z = oz + (iz + 0.5) * spacing
                positions.append([x, y, z])
            if len(positions) >= num_particles:
                break
        if len(positions) >= num_particles:
            break

    return np.array(positions[:num_particles], dtype=np.float32)


def generate_psd_positions(
    num_particles: int,
    psd: ParticleSizeDistribution,
    spacing_factor: float = 2.2,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Generate particle positions for a PSD simulation.

    Uses ``psd.max_radius`` as the grid spacing reference so that the
    initial packing is overlap-free regardless of the size distribution.

    Args:
        num_particles: Number of particles.
        psd: Particle size distribution (only max_radius is used for spacing).
        spacing_factor: Grid spacing multiplier on max_radius (default 2.2).
        origin: Centre of the bottom layer.

    Returns:
        (N, 3) float32 array of positions.
    """
    return generate_grid_positions(
        num_particles,
        radius=psd.max_radius,
        spacing_factor=spacing_factor,
        origin=origin,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_PARTICLES = 8000
    RADIUS = 0.005  # 5 mm
    NUM_STEPS = 2000
    REPORT_INTERVAL = 200

    print("=" * 60)
    print("VeloxSim-DEM: Discrete Element Method Simulation")
    print(f"  Particles : {NUM_PARTICLES}")
    print(f"  Radius    : {RADIUS * 1000:.1f} mm")
    print(f"  Steps     : {NUM_STEPS}")
    print("=" * 60)

    config = SimConfig(
        num_particles=NUM_PARTICLES,
        particle_radius=RADIUS,
        particle_density=2500.0,
        young_modulus=1.0e7,
        poisson_ratio=0.3,
        restitution=0.5,
        friction_static=0.5,
        friction_dynamic=0.4,
        friction_rolling=0.01,
        cohesion_energy=0.0,
        dt=1.0e-5,
        gravity=(0.0, 0.0, -9.81),
        max_contacts_per_particle=32,
        hash_grid_dim=128,
    )

    sim = Simulation(config)

    positions = generate_grid_positions(
        NUM_PARTICLES, RADIUS, spacing_factor=2.5, origin=(0.0, 0.0, 0.15)
    )
    sim.initialize_particles(positions)

    floor = create_plane_mesh(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        size=2.0,
        device=config.device,
    )
    sim.add_mesh(floor)

    print(f"\nParticle mass   : {sim.particle_mass:.6e} kg")
    print(f"E_eff           : {sim.E_eff:.3e} Pa")
    print(f"G_eff           : {sim.G_eff:.3e} Pa")
    print(f"Damping (beta)  : {sim.beta:.4f}")
    print(f"Timestep        : {config.dt:.1e} s")
    print()

    print("Compiling Warp kernels...")
    t0 = time.perf_counter()
    sim.step()
    wp.synchronize()
    compile_time = time.perf_counter() - t0
    print(f"Kernel compilation: {compile_time:.2f}s\n")

    print(f"{'Step':>8s}  {'Time (ms)':>10s}  {'KE (J)':>12s}  {'Z_min (mm)':>10s}  {'Z_max (mm)':>10s}")
    print("-" * 60)

    t_start = time.perf_counter()
    for step_i in range(1, NUM_STEPS + 1):
        sim.step()

        if step_i % REPORT_INTERVAL == 0:
            wp.synchronize()
            elapsed = (time.perf_counter() - t_start) * 1000.0
            ke = sim.get_kinetic_energy()
            pos = sim.get_positions()
            z_min = float(pos[:, 2].min()) * 1000.0
            z_max = float(pos[:, 2].max()) * 1000.0
            print(f"{step_i:>8d}  {elapsed:>10.1f}  {ke:>12.4e}  {z_min:>10.2f}  {z_max:>10.2f}")
            t_start = time.perf_counter()
