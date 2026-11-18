"""Shared pieces of the three-dimensional eight-well benchmark.

U_3(theta) = exp{3 [3 - sum_j sin^4 theta_j]} - 1 on [-pi, pi)^3.  The eight
minima sit at theta_j = +-pi/2; at beta_0 = (e^3 - 1)^-1 the minimum-to-saddle
barrier is k_B T.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fp_transformer import CoordTopology, EnergySystem, Sampler
from fp_transformer.kernel import Gaussian
from fp_transformer.potential import PeriodicSin4
from fp_transformer.utils import torus_pairwise_distance

BETA_0 = 1.0 / math.expm1(3.0)
NUM_UPDATES = 13500
MAIN_SEEDS = tuple(20260737 + 10 * k for k in range(8))
AUDIT_SEEDS = (20260737, 20260814, 20260815, 20260816)


def wrap_to_pi(values):
    return jnp.arctan2(jnp.sin(values), jnp.cos(values))


def energy_single(x: jax.Array) -> jax.Array:
    return jnp.exp(3.0 * (3.0 - jnp.sum(jnp.sin(x) ** 4))) - 1.0


ENERGY_GRAD = jax.jit(jax.vmap(jax.grad(energy_single)))


def energy_numpy(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, dtype=float)
    return np.exp(3.0 * (3.0 - np.sum(np.sin(values) ** 4, axis=-1))) - 1.0


def well_assignments(samples: np.ndarray) -> np.ndarray:
    """Index (0..7) of the nearest minimum in the toroidal minimum-image metric."""
    values = np.asarray(wrap_to_pi(jnp.asarray(samples)))
    to_negative = np.abs(np.arctan2(np.sin(values + math.pi / 2), np.cos(values + math.pi / 2)))
    to_positive = np.abs(np.arctan2(np.sin(values - math.pi / 2), np.cos(values - math.pi / 2)))
    bits = (to_positive < to_negative).astype(np.int64)
    return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]


def basin_boltzmann_initialization(seed: int, *, replicas: int, beta: float,
                                   resolution: int = 192) -> np.ndarray:
    """Draws from the Boltzmann law restricted to the initial well (all theta_k < 0).

    Cell masses on a regular grid of the negative octant, then a uniform jitter
    inside the chosen cell; the initial well is the one centred at (-pi/2)^3.
    """
    rng = np.random.default_rng(seed)
    spacing = math.pi / resolution
    axis = -math.pi + spacing * (0.5 + np.arange(resolution))
    sin4 = np.sin(axis) ** 4
    energy = np.exp(3.0 * (3.0 - sin4[:, None, None] - sin4[None, :, None]
                           - sin4[None, None, :])) - 1.0
    log_mass = -beta * energy
    mass = np.exp(log_mass - log_mass.max()).ravel()
    cells = rng.choice(mass.size, size=replicas, p=mass / mass.sum())
    index = np.stack(np.unravel_index(cells, (resolution,) * 3), axis=1)
    coordinate = axis[index] + spacing * (rng.random((replicas, 3)) - 0.5)
    return np.asarray(wrap_to_pi(jnp.asarray(coordinate)))


@partial(jax.jit, static_argnames=("steps",))
def langevin_chunk(coord, key, *, beta, step_size, steps):
    """Euler--Maruyama overdamped Langevin dynamics, wrapped to the torus."""

    def step(carry, _):
        current, current_key = carry
        current_key, noise_key = jax.random.split(current_key)
        noise = jax.random.normal(noise_key, current.shape)
        updated = current - step_size * beta * ENERGY_GRAD(current) + jnp.sqrt(2.0 * step_size) * noise
        return (wrap_to_pi(updated), current_key), None

    (coord, key), _ = jax.lax.scan(step, (coord, key), xs=None, length=steps)
    return coord, key


@jax.jit
def _noscore_gradient(coord, beta):
    return beta * ENERGY_GRAD(coord)


def make_noscore_stepper(learning_rate: float, optimizer_name: str = "rmsprop"):
    """The FPT update with the score removed: the same optimizer on beta grad U."""
    optimizer = optax.sgd(learning_rate) if optimizer_name == "sgd" else optax.rmsprop(learning_rate)

    @partial(jax.jit, static_argnames=("steps",))
    def chunk(coord, opt_state, beta, steps):
        def step(carry, _):
            current, state = carry
            update, state = optimizer.update(_noscore_gradient(current, beta), state)
            return (wrap_to_pi(optax.apply_updates(current, update)), state), None

        (coord, opt_state), _ = jax.lax.scan(step, (coord, opt_state), xs=None, length=steps)
        return coord, opt_state

    return optimizer, chunk


def median_bandwidth(coord: np.ndarray) -> float:
    """h^(0) = sqrt(median of pairwise chordal distances)."""
    distance = np.asarray(torus_pairwise_distance(jnp.asarray(coord)[:, None, :]))
    return float(np.sqrt(np.median(distance[np.triu_indices(distance.shape[0], k=1)])))


def build_fpt_sampler(
    coord: np.ndarray,
    *,
    beta: float,
    seed: int,
    learning_rate: float,
    optimizer_name: str = "rmsprop",
    kernel_min: float = 2.0,
    kernel_max: float = 4.0,
    update_pace: int = 1,
) -> Sampler:
    """Shared-bandwidth FPT: leave-one-out pointwise score, global row-mass band."""
    coordinate = jnp.asarray(coord[:, None, :])
    kernel = Gaussian(
        coordinate.shape[0],
        bandwidth=median_bandwidth(coord),
        distance_func=torus_pairwise_distance,
        update_bw_method="rescale",
        kernel_min=float(kernel_min),
        kernel_max=float(kernel_max),
    )
    optimizer = optax.sgd(learning_rate) if optimizer_name == "sgd" else optax.rmsprop(learning_rate)
    sampler = Sampler(
        system=EnergySystem(CoordTopology(coordinate), PeriodicSin4(c=3.0, p=1.0)),
        kernel=kernel,
        optimizer=optimizer,
        temperature=1.0,
        kB=1.0,
        update_pace=int(update_pace),
        seed=seed,
    )
    sampler.beta = float(beta)
    return sampler


def sampler_snapshot(sampler: Sampler) -> np.ndarray:
    return np.asarray(wrap_to_pi(sampler.system.coordinate))[:, 0, :]


# --- Grid reference of the target and equilibrium-audit metrics -------------


def make_reference(beta: float, resolution: int = 128) -> dict[str, np.ndarray]:
    spacing = 2.0 * math.pi / resolution
    axis = -math.pi + spacing * (0.5 + np.arange(resolution))
    sin4 = np.sin(axis) ** 4
    energy = np.exp(3.0 * (3.0 - sin4[:, None, None] - sin4[None, :, None] - sin4[None, None, :])) - 1.0
    log_mass = -beta * energy
    mass = np.exp(log_mass - float(np.max(log_mass)))
    mass /= np.sum(mass)
    flat_energy, flat_mass = energy.ravel(), mass.ravel()
    order = np.argsort(flat_energy, kind="stable")
    return {
        "axis": axis,
        "flat_mass_cdf": np.cumsum(flat_mass),
        "sorted_energy": flat_energy[order],
        "sorted_energy_cdf": np.cumsum(flat_mass[order]),
    }


def sample_reference(rng: np.random.Generator, reference: dict, count: int) -> np.ndarray:
    axis = reference["axis"]
    flat_indices = np.searchsorted(reference["flat_mass_cdf"], rng.random(count), side="right")
    index = np.unravel_index(flat_indices, (axis.size,) * 3)
    return np.column_stack([axis[i] for i in index])


def energy_ks(coordinate: np.ndarray, reference: dict) -> float:
    values = np.sort(energy_numpy(coordinate))
    indices = np.searchsorted(reference["sorted_energy"], values, side="right") - 1
    target_cdf = np.where(indices >= 0, reference["sorted_energy_cdf"][np.maximum(indices, 0)], 0.0)
    count = values.size
    upper = np.arange(1, count + 1, dtype=float) / count
    lower = np.arange(count, dtype=float) / count
    return float(max(np.max(upper - target_cdf), np.max(target_cdf - lower)))


def occupancy_metrics(coordinate: np.ndarray) -> dict:
    count = np.bincount(well_assignments(coordinate), minlength=8)
    probability = count / np.sum(count)
    positive = probability > 0.0
    entropy = -np.sum(probability[positive] * np.log(probability[positive])) / np.log(8)
    return {
        "counts": count.tolist(),
        "covered_modes": int(np.count_nonzero(count)),
        "normalized_entropy": float(entropy),
        "basin_tv": float(0.5 * np.sum(np.abs(probability - 0.125))),
    }


def sample_metrics(coordinate: np.ndarray, reference: dict) -> dict:
    return {**occupancy_metrics(coordinate), "energy_ks": energy_ks(coordinate, reference)}


def reference_mean_bands(*, reference, replicas, seed_count, group_count, rng_seed=20260817) -> dict:
    """Quantiles of four-ensemble means of energy KS and basin TV for iid target samples."""
    rng = np.random.default_rng(rng_seed)
    mean_ks, mean_tv = [], []
    for _ in range(group_count):
        metrics = [sample_metrics(sample_reference(rng, reference, replicas), reference) for _ in range(seed_count)]
        mean_ks.append(float(np.mean([m["energy_ks"] for m in metrics])))
        mean_tv.append(float(np.mean([m["basin_tv"] for m in metrics])))
    return {
        "group_count": group_count,
        "ensembles_per_group": seed_count,
        "energy_ks_mean_95": np.quantile(mean_ks, (0.025, 0.975)).tolist(),
        "energy_ks_mean_99_upper": float(np.quantile(mean_ks, 0.99)),
        "basin_tv_mean_95": np.quantile(mean_tv, (0.025, 0.975)).tolist(),
        "basin_tv_mean_99_upper": float(np.quantile(mean_tv, 0.99)),
    }
