"""Fisher--Rao reweighting of a weighted particle ensemble.

A mirror (multiplicative) step on the particle weights along the discrete
free-energy gradient, followed, when the maximum entitlement N w_i exceeds a
threshold or the ESS falls below a fraction of N, by a balanced residual
realization that returns to equal weights.
"""

from __future__ import annotations

import numpy as np


def weighted_mean_and_variance(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * np.square(values - mean)))
    return mean, variance


def fisher_rao_weight_step(
    particle_weights, free_energy_gradient, *, step_size: float
):
    """Apply one positivity-preserving Fisher--Rao mirror step.

    ``free_energy_gradient`` is the derivative of one normalized discrete
    free-energy objective with respect to the particle masses.  Centering is
    the simplex Lagrange multiplier; the exponential update is the exact
    positivity-preserving discretization of ``d log(w_i) / dt = -G_i``.
    """

    if step_size <= 0.0:
        raise ValueError("Fisher--Rao step size must be positive")
    weights = np.asarray(particle_weights, dtype=np.float64).copy()
    gradient = np.asarray(free_energy_gradient, dtype=np.float64).copy()
    if weights.ndim != 1 or gradient.shape != weights.shape:
        raise ValueError("weights and free-energy gradient must be 1D and aligned")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Fisher--Rao weights must be positive and finite")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Fisher--Rao free-energy gradient must be finite")
    weights /= np.sum(weights)
    centered = gradient - np.sum(weights * gradient)
    log_weights = np.log(weights) - step_size * centered
    log_weights -= np.max(log_weights)
    # Keep the host-side mirror representation strictly positive even when a
    # rejected singular-energy outlier has an astronomically small mass.
    # JAX runs this experiment in float32 by default.  Keep the host mirror
    # above the float32 normal floor so a subsequent device transfer cannot
    # silently turn a positive quadrature mass into an exact zero.
    log_weights = np.maximum(log_weights, np.log(np.finfo(np.float32).tiny))
    updated = np.exp(log_weights)
    updated /= np.sum(updated)
    return updated, centered


def balanced_residual_resample_population(
    coordinates,
    bandwidth,
    particle_weights,
    lineage,
    *,
    rng,
    roughening_scale: float = 0.0,
):
    """Low-variance equal-weight realization of ``N * particle_weights``.

    Integer offspring entitlements are assigned deterministically with
    ``floor(N w_i)``.  The remaining slots use one systematic draw from the
    fractional residuals.  The returned parent map keeps one copy in its
    original slot whenever its offspring count is nonzero, then replaces only
    zero-offspring victims with the surplus copies.  This minimizes physical
    slot churn for the realized counts while retaining the unbiased residual-
    systematic measure.  Optional roughening replaces exact coordinate
    copies by a zero-barycenter cloud inside the parent's local kernel
    neighborhood.  It is a support-regularization step, not a Metropolis
    transition: the maximum torus displacement is
    ``roughening_scale * bandwidth[parent]`` and therefore vanishes with the
    local bandwidth.
    """

    coordinates = np.asarray(coordinates)
    bandwidth = np.asarray(bandwidth)
    weights = np.asarray(particle_weights, dtype=np.float64).copy()
    lineage = np.asarray(lineage, dtype=np.int64)
    replicas = len(coordinates)
    if bandwidth.shape != (replicas,):
        raise ValueError("balanced realization requires one bandwidth per replica")
    if weights.shape != (replicas,) or lineage.shape != (replicas,):
        raise ValueError("balanced realization arrays must align with coordinates")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("balanced realization weights must be finite and non-negative")
    if roughening_scale < 0.0 or not np.isfinite(roughening_scale):
        raise ValueError("balanced realization roughening scale must be finite and non-negative")
    total_weight = float(np.sum(weights))
    if not np.isclose(total_weight, 1.0):
        raise ValueError("balanced realization weights must be normalized")
    weights /= total_weight

    entitlement = replicas * weights
    offspring_counts = np.floor(entitlement).astype(np.int64)
    residual_slots = int(replicas - np.sum(offspring_counts))
    residual_entitlement = entitlement - offspring_counts
    if residual_slots:
        residual_total = float(np.sum(residual_entitlement))
        if not np.isclose(residual_total, residual_slots):
            raise RuntimeError("fractional offspring entitlement is inconsistent")
        positions = (rng.random() + np.arange(residual_slots)) / residual_slots
        residual_parents = np.searchsorted(
            np.cumsum(residual_entitlement / residual_total),
            positions,
            side="right",
        )
        residual_parents = np.minimum(residual_parents, replicas - 1)
        offspring_counts += np.bincount(
            residual_parents, minlength=replicas
        )

    victims = np.flatnonzero(offspring_counts == 0)
    surplus_parents = np.repeat(
        np.arange(replicas, dtype=np.int64),
        np.maximum(offspring_counts - 1, 0),
    )
    if len(victims) != len(surplus_parents):
        raise RuntimeError("balanced offspring and victim counts disagree")
    parent_indices = np.arange(replicas, dtype=np.int64)
    parent_indices[victims] = surplus_parents
    realized_counts = np.bincount(parent_indices, minlength=replicas)
    if not np.array_equal(realized_counts, offspring_counts):
        raise RuntimeError("balanced parent map does not realize offspring counts")

    realized_coordinates = coordinates[parent_indices].copy()
    roughened_groups = 0
    roughened_particles = 0
    squared_displacement_sum = 0.0
    maximum_displacement = 0.0
    if roughening_scale > 0.0 and len(surplus_parents):
        dimensions = int(np.prod(coordinates.shape[1:]))
        for parent in np.flatnonzero(offspring_counts > 1):
            slots = np.flatnonzero(parent_indices == parent)
            offspring = len(slots)
            pairs = offspring // 2
            offsets = np.zeros((offspring, dimensions), dtype=np.float64)
            if pairs:
                directions = rng.normal(size=(pairs, dimensions))
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                zero_norm = norms[:, 0] == 0.0
                while np.any(zero_norm):
                    directions[zero_norm] = rng.normal(
                        size=(int(np.count_nonzero(zero_norm)), dimensions)
                    )
                    norms = np.linalg.norm(directions, axis=1, keepdims=True)
                    zero_norm = norms[:, 0] == 0.0
                directions /= norms
                radii = (
                    roughening_scale
                    * float(bandwidth[parent])
                    * np.power(rng.random(pairs), 1.0 / dimensions)
                )
                pair_offsets = directions * radii[:, None]
                offsets[: 2 * pairs : 2] = pair_offsets
                offsets[1 : 2 * pairs : 2] = -pair_offsets
                offsets = offsets[rng.permutation(offspring)]
            offsets = offsets.reshape((offspring,) + coordinates.shape[1:])
            realized_coordinates[slots] = (
                realized_coordinates[slots] + offsets
            )
            realized_coordinates[slots] = (
                realized_coordinates[slots] + np.pi
            ) % (2.0 * np.pi) - np.pi
            displacement = np.linalg.norm(
                offsets.reshape(offspring, -1), axis=1
            )
            roughened_groups += 1
            roughened_particles += offspring
            squared_displacement_sum += float(np.sum(np.square(displacement)))
            maximum_displacement = max(
                maximum_displacement, float(np.max(displacement))
            )

    equal_weights = np.full(replicas, 1.0 / replicas, dtype=np.float64)
    statistics = {
        "operations": int(len(victims)),
        "residual_slots": residual_slots,
        "unique_output_parents": int(np.count_nonzero(offspring_counts)),
        "maximum_output_offspring": int(np.max(offspring_counts)),
        "entitlement_l1_error_particles": float(
            np.sum(np.abs(offspring_counts - entitlement))
        ),
        "realized_weight_l1_error": float(
            np.sum(np.abs(offspring_counts / replicas - weights))
        ),
        "roughening_scale": float(roughening_scale),
        "roughened_groups": roughened_groups,
        "roughened_particles": roughened_particles,
        "roughening_rms_displacement_rad": (
            float(np.sqrt(squared_displacement_sum / roughened_particles))
            if roughened_particles
            else 0.0
        ),
        "roughening_max_displacement_rad": maximum_displacement,
    }
    return (
        realized_coordinates,
        bandwidth[parent_indices].copy(),
        equal_weights,
        lineage[parent_indices].copy(),
        parent_indices,
        statistics,
    )


def balanced_realization_trigger(
    particle_weights,
    *,
    ess_fraction: float,
    maximum_entitlement_threshold: float,
):
    """Decide when continuous FR mass warrants integer particle support.

    The local guard is family-free: a particle is under-resolved once its
    equal-weight entitlement ``N*w_i`` reaches the requested threshold.  The
    conventional global ESS guard remains as a fallback.  Between triggers,
    continuous weights accumulate without residual-rounding noise.
    """

    weights = np.asarray(particle_weights, dtype=np.float64).reshape(-1)
    if (
        not len(weights)
        or np.any(weights < 0.0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(
            "balanced trigger weights must be finite and non-negative"
        )
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("balanced trigger weights must have positive mass")
    weights /= total_weight
    replicas = len(weights)
    ess = float(np.reciprocal(np.sum(np.square(weights))))
    maximum_entitlement = float(replicas * np.max(weights))
    ess_triggered = ess <= ess_fraction * replicas
    entitlement_triggered = (
        maximum_entitlement >= maximum_entitlement_threshold
    )
    statistics = {
        "maximum_entitlement": maximum_entitlement,
        "maximum_entitlement_threshold": maximum_entitlement_threshold,
        "entitlement_triggered": bool(entitlement_triggered),
        "ess_triggered": bool(ess_triggered),
    }
    return bool(ess_triggered or entitlement_triggered), statistics
