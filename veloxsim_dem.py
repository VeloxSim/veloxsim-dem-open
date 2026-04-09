"""
VeloxSim-DEM: Discrete Element Method simulation engine built on NVIDIA Warp.

Built by VeloxSim Tech Pty Ltd and Sam Wong.

Features:
    - Hertz-Mindlin contact model (normal + tangential with damping)
    - Coulomb sliding friction with static (mu_s) and dynamic (mu_d) coefficients
    - Type C EPSD rolling resistance (elastic-plastic spring-dashpot, history-dependent)
    - JKR-inspired cohesion/adhesion model
    - Particle-particle collision with spatial hash grid broad-phase
    - Particle-mesh collision using BVH mesh queries
    - Moving mesh surfaces (conveyor belts) via surface_velocity parameter
    - Velocity Verlet time integration
    - 3D triangular mesh import (OBJ/STL via trimesh)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import warp as wp

wp.init()

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
    particle_radius: wp.float32
    particle_density: wp.float32


@wp.struct
class DerivedParams:
    E_eff: wp.float32       # effective Young's modulus E*
    G_eff: wp.float32       # effective shear modulus G*
    R_eff: wp.float32       # effective radius (R/2 for equal spheres)
    m_eff: wp.float32       # effective mass (m/2 for equal spheres)
    beta: wp.float32        # damping coefficient from restitution
    cohesion_pulloff: wp.float32  # JKR pull-off force 1.5*pi*gamma*R_eff
    particle_mass: wp.float32
    particle_inertia: wp.float32  # 2/5 * m * R^2


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
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    dt: float,
) -> wp.vec3:
    """
    Compute Hertz-Mindlin contact force between two equal-radius spheres.
    Returns the force on particle i due to particle j.

    Coulomb friction: static limit mu_s*|F_n| → when exceeded, kinetic force
    is capped at mu_d*|F_n| (mu_d <= mu_s).
    """
    diff = xj - xi
    dist = wp.length(diff)

    # Avoid division by zero
    if dist < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist  # normal from i to j
    radius = dp.R_eff * 2.0  # single particle radius = 2 * R_eff
    delta_n = 2.0 * radius - dist  # overlap (positive in contact)

    # Cohesion range: allow attractive force slightly beyond contact
    cohesion_range = 0.1 * radius
    if delta_n < -cohesion_range:
        return wp.vec3(0.0, 0.0, 0.0)

    force = wp.vec3(0.0, 0.0, 0.0)

    # Cohesion/adhesion: attractive force when within range but separating
    if delta_n < 0.0:
        # Linear ramp of cohesion force from full pull-off at contact to zero at range limit
        cohesion_frac = 1.0 + delta_n / cohesion_range  # 1 at delta_n=0, 0 at delta_n=-range
        force = n * dp.cohesion_pulloff * cohesion_frac
        return force

    # --- In contact (delta_n >= 0) ---
    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(dp.R_eff)

    # Normal stiffness and force (Hertz)
    S_n = 2.0 * dp.E_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (4.0 / 3.0) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    # Relative velocity at contact
    v_rel = compute_relative_velocity(vi, vj, wi, wj, n, radius, radius)
    v_n_mag = wp.dot(v_rel, n)
    v_n = n * v_n_mag
    v_t = v_rel - v_n

    # Normal damping
    damp_n_coeff = -2.0 * wp.sqrt(5.0 / 6.0) * dp.beta * wp.sqrt(S_n * dp.m_eff)
    F_nd = n * (damp_n_coeff * v_n_mag)

    # Cohesion: additive pull-off (attractive, along +n direction toward j)
    F_cohesion = n * dp.cohesion_pulloff

    # Total normal force (repulsive + damping + cohesion attraction)
    F_normal = n * (-F_n_mag) + F_nd + F_cohesion

    # Tangential stiffness
    S_t = 8.0 * dp.G_eff * sqrt_R_eff * sqrt_delta

    # Update tangential displacement incrementally
    new_tangent = tangent_disp + v_t * dt

    # Tangential force (spring)
    F_t_spring = new_tangent * (-S_t)

    # Tangential damping
    damp_t_coeff = -2.0 * wp.sqrt(5.0 / 6.0) * dp.beta * wp.sqrt(S_t * dp.m_eff)
    F_t_damp = v_t * damp_t_coeff

    F_t_total = F_t_spring + F_t_damp

    # Coulomb friction limit: static → kinetic when exceeded
    F_t_mag = wp.length(F_t_total)
    F_n_total_mag = F_n_mag - damp_n_coeff * v_n_mag  # elastic + damping normal magnitude
    static_limit = mu_s * wp.abs(F_n_total_mag)

    if F_t_mag > static_limit and F_t_mag > 1.0e-12:
        # Kinetic (sliding) regime: cap at mu_d * |F_n|
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
    if dist < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist
    radius = dp.R_eff * 2.0
    delta_n = 2.0 * radius - dist

    if delta_n <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(dp.R_eff)

    S_n = 2.0 * dp.E_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (4.0 / 3.0) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta
    S_t = 8.0 * dp.G_eff * sqrt_R_eff * sqrt_delta

    v_rel = compute_relative_velocity(vi, vj, wi, wj, n, radius, radius)
    v_n_mag = wp.dot(v_rel, n)
    v_t = v_rel - n * v_n_mag

    new_tangent = tangent_disp + v_t * dt

    # Check Coulomb limit on tangential spring
    F_t_spring_mag = wp.length(new_tangent) * S_t
    damp_n_coeff = -2.0 * wp.sqrt(5.0 / 6.0) * dp.beta * wp.sqrt(S_n * dp.m_eff)
    F_n_total_mag = F_n_mag - damp_n_coeff * v_n_mag
    static_limit = mu_s * wp.abs(F_n_total_mag)

    if F_t_spring_mag > static_limit and F_t_spring_mag > 1.0e-12:
        # Rescale spring to kinetic limit displacement
        kinetic_limit = mu_d * wp.abs(F_n_total_mag)
        new_tangent = new_tangent * (kinetic_limit / (wp.length(new_tangent) * S_t + 1.0e-12))

    return new_tangent


@wp.func
def compute_rolling_torque_pp(
    xi: wp.vec3, xj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    roll_disp: wp.vec3,
    dp: DerivedParams,
    mu_r: float,
) -> wp.vec3:
    """Type C EPSD rolling resistance torque on particle i (particle-particle).

    Elastic-plastic spring-dashpot model:
      k_r = 0.25 * S_t * R_eff^2   (rolling stiffness)
      M_r = -k_r * roll_disp        (elastic torque)
      M_r_max = mu_r * R_eff * |F_n|  (plastic cap)
    """
    if mu_r < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    diff = xj - xi
    dist = wp.length(diff)
    if dist < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    radius = dp.R_eff * 2.0
    delta_n = 2.0 * radius - dist
    if delta_n <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(dp.R_eff)
    S_t = 8.0 * dp.G_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (4.0 / 3.0) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    # Rolling stiffness (Wensrich & Katterfeld factor)
    k_r = 0.25 * S_t * dp.R_eff * dp.R_eff

    # Elastic rolling torque from accumulated spring displacement
    M_r = roll_disp * (-k_r)
    M_r_mag = wp.length(M_r)
    M_r_max = mu_r * dp.R_eff * wp.abs(F_n_mag)

    if M_r_mag > M_r_max and M_r_mag > 1.0e-12:
        M_r = M_r * (M_r_max / M_r_mag)

    return M_r


@wp.func
def update_rolling_disp_pp(
    xi: wp.vec3, xj: wp.vec3,
    wi: wp.vec3, wj: wp.vec3,
    roll_disp: wp.vec3,
    dp: DerivedParams,
    mu_r: float,
    dt: float,
) -> wp.vec3:
    """Update rolling displacement for Type C EPSD model (particle-particle).

    Accumulates relative rolling angular velocity (tangential to contact plane)
    and applies plastic cap when elastic torque exceeds mu_r * R_eff * |F_n|.
    """
    if mu_r < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    diff = xj - xi
    dist = wp.length(diff)
    if dist < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)

    n = diff / dist
    radius = dp.R_eff * 2.0
    delta_n = 2.0 * radius - dist
    if delta_n <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    sqrt_delta = wp.sqrt(delta_n)
    sqrt_R_eff = wp.sqrt(dp.R_eff)
    S_t = 8.0 * dp.G_eff * sqrt_R_eff * sqrt_delta
    F_n_mag = (4.0 / 3.0) * dp.E_eff * sqrt_R_eff * delta_n * sqrt_delta

    # Relative rolling angular velocity: tangential component only (exclude spin)
    omega_rel = wi - wj
    omega_roll = omega_rel - n * wp.dot(omega_rel, n)

    # Increment rolling spring
    new_roll = roll_disp + omega_roll * dt

    # Plastic cap: limit rolling spring displacement
    k_r = 0.25 * S_t * dp.R_eff * dp.R_eff
    M_r_max = mu_r * dp.R_eff * wp.abs(F_n_mag)
    M_r_spring_mag = wp.length(new_roll) * k_r

    if M_r_spring_mag > M_r_max and M_r_spring_mag > 1.0e-12:
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
    gravity: wp.vec3,
    mass: float,
):
    tid = wp.tid()
    forces[tid] = forces[tid] + gravity * mass


@wp.kernel
def apply_global_damping_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    damping: float,
    mass: float,
    inertia: float,
):
    """Apply global viscous damping (drag) proportional to velocity.

    Models aerodynamic drag / interstitial-fluid resistance.
    F_drag = -damping * mass * velocity
    """
    tid = wp.tid()
    v = velocities[tid]
    w = angular_velocities[tid]
    forces[tid]  = forces[tid]  - v * (damping * mass)
    torques[tid] = torques[tid] - w * (damping * inertia)


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
    grid: wp.uint64,
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    mu_r: float,
    dt: float,
    num_particles: int,
    max_contacts: int,
):
    """Compute particle-particle contact forces using hash grid neighbor search.

    Includes Hertz-Mindlin contact (with static/dynamic Coulomb friction) and
    Type C EPSD rolling resistance.
    """
    i = wp.tid()

    xi = positions[i]
    vi = velocities[i]
    wi = angular_velocities[i]
    radius = dp.R_eff * 2.0
    query_radius = 2.0 * radius * 1.05  # slight margin

    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_torque = wp.vec3(0.0, 0.0, 0.0)

    # Track which contacts are still active this step
    new_count = int(0)

    # Query hash grid for neighbors
    query = wp.hash_grid_query(grid, xi, query_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        j = index
        if j == i:
            continue

        xj = positions[j]
        vj = velocities[j]
        wj = angular_velocities[j]

        diff = xj - xi
        dist = wp.length(diff)
        cohesion_range = 0.1 * radius

        if dist > 2.0 * radius + cohesion_range:
            continue

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

        # Compute contact force (Hertz-Mindlin with Coulomb friction)
        f = hertz_mindlin_force(xi, xj, vi, vj, wi, wj, tangent, dp, mu_s, mu_d, dt)
        total_force = total_force + f

        # Tangential torque: r x F_t at contact point
        if dist > 1.0e-12:
            n = diff / dist
            f_n_comp = n * wp.dot(f, n)
            f_t_comp = f - f_n_comp
            total_torque = total_torque + wp.cross(n * radius, f_t_comp)

        # Rolling resistance torque (Type C EPSD)
        M_r = compute_rolling_torque_pp(xi, xj, wi, wj, roll, dp, mu_r)
        total_torque = total_torque + M_r

        # Update contact history and store
        if new_count < max_contacts:
            new_tangent = hertz_mindlin_update_tangent(
                xi, xj, vi, vj, wi, wj, tangent, dp, mu_s, mu_d, dt
            )
            new_roll = update_rolling_disp_pp(xi, xj, wi, wj, roll, dp, mu_r, dt)
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
    mesh_id: wp.uint64,
    dp: DerivedParams,
    mu_s: float,
    mu_d: float,
    mu_r: float,
    cohesion_pulloff: float,
    dt: float,
    surface_velocity: wp.vec3,
):
    """Compute particle-mesh contact forces using BVH mesh query.

    surface_velocity: velocity of the mesh surface (e.g. conveyor belt).
    Friction and rolling resistance act relative to that surface velocity.
    Includes Type C EPSD rolling resistance for particle-wall contacts.
    """
    i = wp.tid()

    xi = positions[i]
    vi = velocities[i]
    wi = angular_velocities[i]
    radius = dp.R_eff * 2.0
    max_dist = radius * 1.5

    # Query closest point on mesh
    query = wp.mesh_query_point_sign_normal(mesh_id, xi, max_dist)

    if query.result:
        mesh_point = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        to_particle = xi - mesh_point
        dist = wp.length(to_particle)

        if dist < 1.0e-12:
            return

        n = to_particle / dist  # outward normal (from mesh toward particle)

        # Use unsigned distance for penetration depth.
        # The sign from mesh_query_point_sign_normal is unreliable for thin
        # single-layer meshes (chute plates, open shells) where the particle
        # can approach from either side.  Using unsigned distance means we
        # always repel particles away from the surface regardless of which
        # side they are on.
        delta_n = radius - dist

        if delta_n <= 0.0:
            # Check cohesion range
            cohesion_range = 0.1 * radius
            if delta_n > -cohesion_range and cohesion_pulloff > 0.0:
                cohesion_frac = 1.0 + delta_n / cohesion_range
                f_cohesion = n * (-cohesion_pulloff * cohesion_frac)
                wp.atomic_add(forces, i, f_cohesion)
            return

        sqrt_delta = wp.sqrt(delta_n)
        sqrt_R = wp.sqrt(radius)  # R_eff for particle-wall = R (wall has infinite radius)

        # Hertz normal stiffness (particle-wall: R_eff = R, m_eff = m)
        S_n = 2.0 * dp.E_eff * sqrt_R * sqrt_delta
        F_n_mag = (4.0 / 3.0) * dp.E_eff * sqrt_R * delta_n * sqrt_delta

        # Relative translational velocity between particle and mesh surface.
        # For mesh (belt/wall) friction we use the translational component only —
        # the angular term is absorbed by the rolling-resistance model (Type C EPSD).
        # This matches common DEM conveyor-belt implementations and lets particles
        # accelerate cleanly to belt_speed via kinetic friction.
        v_rel = vi - surface_velocity
        v_n_mag = wp.dot(v_rel, n)
        v_n = n * v_n_mag
        v_t = v_rel - v_n

        # Normal damping (full particle mass for particle-wall)
        damp_n_coeff = -2.0 * wp.sqrt(5.0 / 6.0) * dp.beta * wp.sqrt(S_n * dp.particle_mass)
        F_nd_mag = damp_n_coeff * v_n_mag

        # Cohesion at contact
        F_cohesion_mag = -cohesion_pulloff

        # Total normal force
        F_normal = n * (F_n_mag + F_nd_mag - F_cohesion_mag)
        F_n_total_mag = F_n_mag + F_nd_mag

        # Tangential spring-dashpot
        S_t = 8.0 * dp.G_eff * sqrt_R * sqrt_delta
        old_tangent = mesh_tangent_disps[i]
        new_tangent = old_tangent + v_t * dt

        F_t_spring = new_tangent * (-S_t)
        damp_t_coeff = -2.0 * wp.sqrt(5.0 / 6.0) * dp.beta * wp.sqrt(S_t * dp.particle_mass)
        F_t_damp = v_t * damp_t_coeff
        F_t_total = F_t_spring + F_t_damp

        # Coulomb limit: static → kinetic when exceeded
        F_t_mag = wp.length(F_t_total)
        static_limit = mu_s * wp.abs(F_n_total_mag)

        if F_t_mag > static_limit and F_t_mag > 1.0e-12:
            kinetic_limit = mu_d * wp.abs(F_n_total_mag)
            F_t_total = F_t_total * (kinetic_limit / F_t_mag)
            new_tangent = new_tangent * (kinetic_limit / (wp.length(new_tangent) * S_t + 1.0e-12))

        mesh_tangent_disps[i] = new_tangent

        # Rolling resistance (Type C EPSD) — particle vs. static/moving wall
        # omega_roll = tangential component of particle angular velocity
        omega_roll = wi - n * wp.dot(wi, n)
        old_roll = mesh_roll_disps[i]
        new_roll = old_roll + omega_roll * dt

        k_r = 0.25 * S_t * radius * radius
        M_r_limit = mu_r * radius * wp.abs(F_n_total_mag)
        M_r_spring = new_roll * (-k_r)
        M_r_spring_mag = wp.length(M_r_spring)

        if M_r_spring_mag > M_r_limit and M_r_spring_mag > 1.0e-12:
            scale = M_r_limit / M_r_spring_mag
            M_r_spring = M_r_spring * scale
            new_roll = new_roll * scale

        mesh_roll_disps[i] = new_roll

        # Combine and apply
        contact_torque = wp.cross(n * (-radius), F_t_total) + M_r_spring
        total_force = F_normal + F_t_total

        wp.atomic_add(forces, i, total_force)
        wp.atomic_add(torques, i, contact_torque)
    else:
        # No contact — reset tangential and rolling displacement
        mesh_tangent_disps[i] = wp.vec3(0.0, 0.0, 0.0)
        mesh_roll_disps[i]    = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def integrate_phase1_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    inv_mass: float,
    inv_inertia: float,
    dt: float,
):
    """Velocity Verlet phase 1: half-step velocity, full-step position."""
    tid = wp.tid()

    v = velocities[tid]
    w = angular_velocities[tid]
    f = forces[tid]
    t = torques[tid]

    # Half-step velocity
    v = v + f * (0.5 * dt * inv_mass)
    w = w + t * (0.5 * dt * inv_inertia)

    # Full-step position
    positions[tid] = positions[tid] + v * dt

    velocities[tid] = v
    angular_velocities[tid] = w


@wp.kernel
def integrate_phase2_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    inv_mass: float,
    inv_inertia: float,
    dt: float,
):
    """Velocity Verlet phase 2: complete the velocity step with new forces."""
    tid = wp.tid()

    velocities[tid] = velocities[tid] + forces[tid] * (0.5 * dt * inv_mass)
    angular_velocities[tid] = angular_velocities[tid] + torques[tid] * (0.5 * dt * inv_inertia)


@wp.kernel
def compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    mass: float,
    inertia: float,
    energy: wp.array(dtype=float),
):
    """Compute per-particle kinetic energy for diagnostics."""
    tid = wp.tid()
    v = velocities[tid]
    w = angular_velocities[tid]
    ke = 0.5 * mass * wp.dot(v, v) + 0.5 * inertia * wp.dot(w, w)
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


def create_plane_mesh(
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    size: float = 10.0,
    device: str = "cuda:0",
) -> wp.Mesh:
    """Create a large planar mesh (two triangles)."""
    n = np.array(normal, dtype=np.float32)
    n = n / np.linalg.norm(n)

    # Find two orthogonal vectors in the plane
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


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    num_particles: int = 1000
    particle_radius: float = 0.005       # meters
    particle_density: float = 2500.0     # kg/m^3 (glass beads)
    young_modulus: float = 1.0e7         # Pa
    poisson_ratio: float = 0.3
    restitution: float = 0.5
    friction_static: float = 0.5         # mu_s — Coulomb static friction coefficient
    friction_dynamic: float = 0.4        # mu_d — Coulomb kinetic friction coefficient
    friction_rolling: float = 0.01       # mu_r — Type C EPSD rolling resistance coefficient
    cohesion_energy: float = 0.0         # J/m^2 (0 = no cohesion)
    dt: float = 1.0e-5                   # seconds
    gravity: tuple = (0.0, 0.0, -9.81)
    max_contacts_per_particle: int = 32
    hash_grid_dim: int = 128
    global_damping: float = 0.0          # 1/s viscous drag coeff (F=-d*m*v)
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    """DEM simulation engine managing state, contacts, and time integration."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.device = config.device
        self.num_particles = config.num_particles
        self.active_count = config.num_particles  # can be reduced for dormant-pool insertion
        self.sim_time = 0.0
        self.step_count = 0

        # --- Derived physical parameters ---
        E = config.young_modulus
        nu = config.poisson_ratio
        R = config.particle_radius
        rho = config.particle_density
        e = config.restitution

        self.E_eff = E / (2.0 * (1.0 - nu * nu))
        self.G_eff = E / (2.0 * (2.0 - nu) * (1.0 + nu))
        self.R_eff = R / 2.0  # equal spheres
        mass = (4.0 / 3.0) * math.pi * R**3 * rho
        self.particle_mass = mass
        self.m_eff = mass / 2.0
        self.particle_inertia = (2.0 / 5.0) * mass * R**2

        if e > 0.0:
            ln_e = math.log(e)
            self.beta = -ln_e / math.sqrt(ln_e**2 + math.pi**2)
        else:
            self.beta = 1.0  # perfectly inelastic

        self.cohesion_pulloff = 1.5 * math.pi * config.cohesion_energy * self.R_eff

        # --- Warp structs ---
        self.mat_params = MaterialParams()
        self.mat_params.young_modulus = float(E)
        self.mat_params.poisson_ratio = float(nu)
        self.mat_params.restitution = float(e)
        self.mat_params.friction_static = float(config.friction_static)
        self.mat_params.friction_rolling = float(config.friction_rolling)
        self.mat_params.cohesion_energy = float(config.cohesion_energy)
        self.mat_params.particle_radius = float(R)
        self.mat_params.particle_density = float(rho)

        self.derived_params = DerivedParams()
        self.derived_params.E_eff = float(self.E_eff)
        self.derived_params.G_eff = float(self.G_eff)
        self.derived_params.R_eff = float(self.R_eff)
        self.derived_params.m_eff = float(self.m_eff)
        self.derived_params.beta = float(self.beta)
        self.derived_params.cohesion_pulloff = float(self.cohesion_pulloff)
        self.derived_params.particle_mass = float(self.particle_mass)
        self.derived_params.particle_inertia = float(self.particle_inertia)

        self.gravity_vec = wp.vec3(*config.gravity)
        self.inv_mass = 1.0 / self.particle_mass
        self.inv_inertia = 1.0 / self.particle_inertia

        # --- Allocate state arrays ---
        N = self.num_particles
        MC = config.max_contacts_per_particle

        self.positions = wp.zeros(N, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(N, dtype=wp.vec3, device=self.device)
        self.angular_velocities = wp.zeros(N, dtype=wp.vec3, device=self.device)
        self.forces = wp.zeros(N, dtype=wp.vec3, device=self.device)
        self.torques = wp.zeros(N, dtype=wp.vec3, device=self.device)

        # Particle-particle contact history (tangential + rolling)
        self.tangent_disps = wp.zeros((N, MC), dtype=wp.vec3, device=self.device)
        self.roll_disps    = wp.zeros((N, MC), dtype=wp.vec3, device=self.device)
        self.contact_ids   = wp.full((N, MC), value=-1, dtype=wp.int32, device=self.device)
        self.contact_counts = wp.zeros(N, dtype=wp.int32, device=self.device)

        # Per-mesh contact history lists (populated in add_mesh / add_mesh_from_file)
        self._mesh_tangent_disps_list: list[wp.array] = []
        self._mesh_roll_disps_list: list[wp.array] = []

        # Hash grid for broad-phase
        dim = config.hash_grid_dim
        self.hash_grid = wp.HashGrid(dim, dim, dim, device=self.device)
        self.cell_size = 2.0 * R * 2.1  # slightly larger than particle diameter

        # Mesh list: each entry is (wp.Mesh, surface_velocity_tuple)
        self.meshes: list[tuple[wp.Mesh, tuple[float, float, float]]] = []

        # Diagnostic energy array
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
        surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Add a collision mesh to the simulation.

        Args:
            mesh: wp.Mesh collision geometry.
            surface_velocity: (vx, vy, vz) surface velocity of the mesh for
                conveyor belt simulation.  Friction and rolling resistance act
                relative to this velocity.  Default (0,0,0) = static mesh.
        """
        self.meshes.append((mesh, surface_velocity))
        N = self.num_particles
        self._mesh_tangent_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        self._mesh_roll_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))

    def add_mesh_from_file(
        self,
        filepath: str,
        scale: float = 1.0,
        surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> wp.Mesh:
        """Load and add a mesh from an OBJ or STL file."""
        mesh = load_mesh(filepath, scale=scale, device=self.device)
        self.meshes.append((mesh, surface_velocity))
        N = self.num_particles
        self._mesh_tangent_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        self._mesh_roll_disps_list.append(wp.zeros(N, dtype=wp.vec3, device=self.device))
        return mesh

    def step(self):
        """Advance simulation by one timestep using Velocity Verlet integration.

        Only processes the first `self.active_count` particles. Particles
        beyond that index are dormant (no forces, no integration, no mesh
        queries). Set `self.active_count` to control how many particles
        participate — useful for dynamic particle insertion.
        """
        N = self.active_count
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
                self.inv_mass, self.inv_inertia, dt,
            ],
            device=self.device,
        )

        # Clear forces
        wp.launch(
            clear_forces_kernel,
            dim=N,
            inputs=[self.forces, self.torques],
            device=self.device,
        )

        # Apply gravity
        wp.launch(
            apply_gravity_kernel,
            dim=N,
            inputs=[self.forces, self.gravity_vec, self.particle_mass],
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
                    self.config.global_damping, self.particle_mass, self.particle_inertia,
                ],
                device=self.device,
            )

        # Rebuild hash grid with current positions
        self.hash_grid.build(self.positions, self.cell_size)

        # Particle-particle forces (Hertz-Mindlin + Coulomb friction + Type C rolling)
        wp.launch(
            compute_particle_forces_kernel,
            dim=N,
            inputs=[
                self.positions, self.velocities, self.angular_velocities,
                self.forces, self.torques,
                self.tangent_disps, self.roll_disps,
                self.contact_ids, self.contact_counts,
                self.hash_grid.id, self.derived_params,
                mu_s, mu_d, mu_r, dt, N, MC,
            ],
            device=self.device,
        )

        # Particle-mesh forces (surface_velocity support for conveyor belts)
        for (mesh, surf_vel), tang, roll in zip(
            self.meshes, self._mesh_tangent_disps_list, self._mesh_roll_disps_list
        ):
            wp.launch(
                compute_mesh_forces_kernel,
                dim=N,
                inputs=[
                    self.positions, self.velocities, self.angular_velocities,
                    self.forces, self.torques,
                    tang, roll,
                    mesh.id,
                    self.derived_params,
                    mu_s, mu_d, mu_r,
                    self.cohesion_pulloff, dt,
                    wp.vec3(surf_vel[0], surf_vel[1], surf_vel[2]),
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
                self.inv_mass, self.inv_inertia, dt,
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

    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy (translational + rotational)."""
        self._energy_buf.zero_()
        wp.launch(
            compute_kinetic_energy_kernel,
            dim=self.num_particles,
            inputs=[
                self.velocities, self.angular_velocities,
                self.particle_mass, self.particle_inertia,
                self._energy_buf,
            ],
            device=self.device,
        )
        return float(self._energy_buf.numpy()[0])
