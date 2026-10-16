"""Exact per-particle row-mass bandwidths and the resulting KDE score.

For every particle i the bandwidth h_i solves

    sum_{j != i} exp[-d_ij^2 / (2 h_i^2)] = K

(the row-mass estimator of the main text).  The root is bracketed in log h and
found by a fixed number of vectorized bisection steps, so the update is
deterministic and jit-compatible.

The torus helpers use the summed chordal metric
d_ij^2 = sum_k 2 [1 - cos(theta_ik - theta_jk)], whose leave-one-out score is

    s_i = h_i^-2 sum_j w_ij sin(theta_j - theta_i),   w_ij = G_ij / sum_j G_ij.

(``Gaussian`` divides the same distance by the dimension; the bandwidths
differ by the factor sqrt(D) and the scores agree.)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..type import JaxArray


def solve_row_mass_log_bandwidth(
    squared_distance: JaxArray,
    row_mass: float,
    *,
    log_lower: float,
    log_upper: float,
    iterations: int = 40,
) -> tuple[JaxArray, JaxArray]:
    """Bisection bracket (lo, hi) of log h_i for the target row mass.

    ``squared_distance`` must already exclude the diagonal (set it to +inf).
    The row mass increases with h, so after ``iterations`` halvings the root
    lies in [lo, hi]; ``exp(lo + hi)`` is the midpoint estimate of h_i^2.
    """
    n = squared_distance.shape[0]

    def mass(log_h):
        return jnp.sum(
            jnp.exp(-squared_distance / (2.0 * jnp.exp(2.0 * log_h)[:, None])), axis=1
        )

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_small = mass(mid) < row_mass
        return jnp.where(too_small, mid, lo), jnp.where(too_small, hi, mid)

    return jax.lax.fori_loop(
        0,
        iterations,
        body,
        (jnp.full(n, log_lower, squared_distance.dtype), jnp.full(n, log_upper, squared_distance.dtype)),
    )


def torus_chordal_squared_distance(theta: JaxArray) -> JaxArray:
    """sum_k 2[1 - cos(theta_ik - theta_jk)] for theta of shape (N, D), diagonal +inf."""
    n, dim = theta.shape
    c, s = jnp.cos(theta), jnp.sin(theta)
    d2 = jnp.maximum(2.0 * dim - 2.0 * (c @ c.T + s @ s.T), 0.0)
    return d2.at[jnp.arange(n), jnp.arange(n)].set(jnp.inf)


def torus_row_mass_score(
    theta: JaxArray,
    row_mass: float,
    *,
    log_lower: float,
    log_upper: float | None = None,
    iterations: int = 40,
) -> tuple[JaxArray, JaxArray, JaxArray]:
    """Leave-one-out score with exact per-particle row-mass bandwidths on the torus.

    Args:
        theta: (N, D) angles.
        row_mass: target off-diagonal row mass K.
        log_lower, log_upper: bisection bracket of log h (summed chordal
            metric); ``log_upper`` defaults to log(4 sqrt(D)).

    Returns:
        (score (N, D), h (N,), achieved row mass (N,)).
    """
    dim = theta.shape[1]
    if log_upper is None:
        log_upper = jnp.log(4.0 * jnp.sqrt(dim))
    c, s = jnp.cos(theta), jnp.sin(theta)
    d2 = torus_chordal_squared_distance(theta)
    lo, hi = solve_row_mass_log_bandwidth(
        d2, row_mass, log_lower=log_lower, log_upper=log_upper, iterations=iterations
    )
    h2 = jnp.exp(lo + hi)
    g = jnp.exp(-d2 / (2.0 * h2[:, None]))
    mass = jnp.sum(g, axis=1)
    w = g / mass[:, None]
    score = (c * (w @ s) - s * (w @ c)) / h2[:, None]
    return score, jnp.sqrt(h2), mass


@partial(jax.jit, static_argnames=("fit_iterations",))
def solve_row_mass_bandwidth_hybrid(
    squared_distance: jax.Array,
    *,
    base_bandwidth: float,
    selector_factor_min: float,
    selector_factor_max: float,
    target_kernel_mass: float,
    fit_iterations: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fit exact row masses with safeguarded Newton steps in log bandwidth.

    ``squared_distance`` is the full pair matrix (the diagonal is excluded
    here).  The bracket is [base_bandwidth * selector_factor_min,
    base_bandwidth * selector_factor_max]; rows whose target lies outside it
    are returned at the bracket edge and flagged as saturated.  Returns
    (bandwidth, fitted row mass, lower_saturated, upper_saturated).

    The row mass is monotone in ``log(h)``.  Each iteration updates the same
    numerical bracket used by the bisection implementation and accepts a
    Newton proposal only when it remains inside that bracket; otherwise it
    falls back to the bracket midpoint.  A fixed iteration count keeps the
    map deterministic and history-free while avoiding dozens of full dense
    kernel evaluations per Euler step.
    """

    dtype = squared_distance.dtype
    particle_count = squared_distance.shape[0]
    lower = jnp.full(
        (particle_count,),
        jnp.log(jnp.asarray(base_bandwidth * selector_factor_min, dtype=dtype)),
    )
    upper = jnp.full(
        (particle_count,),
        jnp.log(jnp.asarray(base_bandwidth * selector_factor_max, dtype=dtype)),
    )
    target = jnp.asarray(target_kernel_mass, dtype=dtype)
    self_mask = jnp.eye(particle_count, dtype=bool)
    lower_bound = lower
    upper_bound = upper

    def mass_and_derivative(log_bandwidth: jax.Array):
        inverse_bandwidth_squared = jnp.exp(-2.0 * log_bandwidth)[:, None]
        scaled_squared = squared_distance * inverse_bandwidth_squared
        weight = jnp.exp(-0.5 * scaled_squared)
        weight = jnp.where(self_mask, 0.0, weight)
        mass = jnp.sum(weight, axis=1)
        derivative = jnp.sum(weight * scaled_squared, axis=1)
        return mass, derivative

    lower_mass, _ = mass_and_derivative(lower)
    upper_mass, _ = mass_and_derivative(upper)
    lower_saturated = target <= lower_mass
    upper_saturated = target >= upper_mass
    # Six safeguarded bisections localize the root inside the protocol's
    # numerical bracket; the remaining iterations use the monotone Newton
    # update.  The production 16-iteration solve therefore retains ten
    # safeguarded Newton corrections after localization.
    initial_bisection_iterations = min(6, fit_iterations)

    def bisect(_, bounds):
        current_lower, current_upper = bounds
        midpoint = 0.5 * (current_lower + current_upper)
        midpoint_mass, _ = mass_and_derivative(midpoint)
        move_lower = midpoint_mass < target
        return (
            jnp.where(move_lower, midpoint, current_lower),
            jnp.where(move_lower, current_upper, midpoint),
        )

    lower, upper = jax.lax.fori_loop(
        0,
        initial_bisection_iterations,
        bisect,
        (lower, upper),
    )
    current = 0.5 * (lower + upper)
    current_mass, _ = mass_and_derivative(current)
    best = current
    best_mass = current_mass
    best_error = jnp.abs(current_mass - target)

    def iterate(_, state):
        (
            current_lower,
            current_upper,
            current_value,
            current_best,
            current_best_mass,
            current_best_error,
        ) = state
        current_mass, derivative = mass_and_derivative(current_value)
        current_error = jnp.abs(current_mass - target)
        improved = current_error < current_best_error
        next_best = jnp.where(improved, current_value, current_best)
        next_best_mass = jnp.where(
            improved,
            current_mass,
            current_best_mass,
        )
        next_best_error = jnp.where(
            improved,
            current_error,
            current_best_error,
        )
        move_lower = current_mass < target
        next_lower = jnp.where(move_lower, current_value, current_lower)
        next_upper = jnp.where(move_lower, current_upper, current_value)
        safe_derivative = jnp.maximum(
            derivative,
            jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
        )
        newton = current_value - (current_mass - target) / safe_derivative
        midpoint = 0.5 * (next_lower + next_upper)
        newton_is_safe = (
            jnp.isfinite(newton)
            & (newton > next_lower)
            & (newton < next_upper)
        )
        next_value = jnp.where(newton_is_safe, newton, midpoint)
        return (
            next_lower,
            next_upper,
            next_value,
            next_best,
            next_best_mass,
            next_best_error,
        )

    lower, upper, current, best, best_mass, best_error = jax.lax.fori_loop(
        initial_bisection_iterations,
        fit_iterations,
        iterate,
        (lower, upper, current, best, best_mass, best_error),
    )
    # The final loop update has not yet been scored.  Reuse the mass evaluation
    # that was previously performed only after selecting ``best`` so the last
    # safeguarded Newton/bisection candidate participates in the comparison at
    # no additional all-pairs pass.
    current_mass, _ = mass_and_derivative(current)
    current_error = jnp.abs(current_mass - target)
    final_improved = current_error < best_error
    interior_log_bandwidth = jnp.where(final_improved, current, best)
    interior_mass = jnp.where(final_improved, current_mass, best_mass)
    interior = jnp.exp(interior_log_bandwidth)
    bandwidth = jnp.where(
        lower_saturated,
        jnp.exp(lower_bound),
        jnp.where(upper_saturated, jnp.exp(upper_bound), interior),
    )
    fitted_mass = jnp.where(
        lower_saturated,
        lower_mass,
        jnp.where(upper_saturated, upper_mass, interior_mass),
    )
    return bandwidth, fitted_mass, lower_saturated, upper_saturated


__all__ = [
    "solve_row_mass_bandwidth_hybrid",
    "solve_row_mass_log_bandwidth",
    "torus_chordal_squared_distance",
    "torus_row_mass_score",
]
