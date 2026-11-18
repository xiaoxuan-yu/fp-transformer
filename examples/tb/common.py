"""Two-dimensional Tiwary--Berne three-well potential and target metrics.

U(x) = 20 - sum_r A_r exp(-2 |x - c_r|^2) + (x_1^6 + x_2^6) / 2
(Tiwary and Berne, J. Chem. Phys. 147, 152701 (2017), Eq. (10)).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

TB_CENTERS = np.asarray(((-0.5, 0.5), (0.8, 1.2), (0.5, -0.3)))
TB_DEPTHS = np.asarray((16.0, 18.0, 16.0))


def tb_energy_single(x: jax.Array) -> jax.Array:
    centers = jnp.asarray(TB_CENTERS)
    depths = jnp.asarray(TB_DEPTHS)
    squared = jnp.sum((x[None, :] - centers) ** 2, axis=1)
    wells = -jnp.sum(depths * jnp.exp(-2.0 * squared))
    return 20.0 + wells + 0.5 * jnp.sum(x**6)


TB_GRAD = jax.jit(jax.vmap(jax.grad(tb_energy_single)))


def tb_energy(samples: np.ndarray) -> np.ndarray:
    squared = np.sum((samples[:, None, :] - TB_CENTERS[None, :, :]) ** 2, axis=-1)
    return 20.0 - np.sum(TB_DEPTHS[None, :] * np.exp(-2.0 * squared), axis=-1) + 0.5 * np.sum(samples**6, axis=-1)


def tb_assignments(samples: np.ndarray) -> np.ndarray:
    """Nearest-center basin index."""
    squared = np.sum((samples[:, None, :] - TB_CENTERS[None, :, :]) ** 2, axis=-1)
    return np.argmin(squared, axis=1)


def tb_exact_grid(beta: float, resolution: int = 320) -> dict[str, np.ndarray]:
    """Boltzmann density on a regular grid covering the support."""
    x = np.linspace(-1.45, 1.55, resolution)
    y = np.linspace(-1.15, 1.85, resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx, yy), axis=-1)
    squared = np.sum((points[..., None, :] - TB_CENTERS) ** 2, axis=-1)
    energy = 20.0 - np.sum(TB_DEPTHS * np.exp(-2.0 * squared), axis=-1) + 0.5 * (xx**6 + yy**6)
    density = np.exp(-beta * (energy - np.min(energy)))
    density /= np.sum(density)
    assignments = np.argmin(squared, axis=-1)
    basin_mass = np.asarray([np.sum(density[assignments == index]) for index in range(3)])
    return {"x": x, "y": y, "energy": energy, "density": density, "basin_mass": basin_mass}


def grid_edges(centers: np.ndarray) -> np.ndarray:
    spacing = np.diff(centers)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("Expected a uniform exact-density grid")
    delta = float(spacing[0])
    return np.concatenate(([centers[0] - 0.5 * delta], 0.5 * (centers[:-1] + centers[1:]), [centers[-1] + 0.5 * delta]))


def smoothed_js_from_histogram(histogram_yx: np.ndarray, exact_density: np.ndarray) -> float:
    estimate = gaussian_filter(histogram_yx.astype(float), sigma=1.2, mode="nearest")
    estimate = np.maximum(estimate, 1e-12)
    estimate /= np.sum(estimate)
    target = np.maximum(exact_density.astype(float), 1e-12)
    target /= np.sum(target)
    midpoint = 0.5 * (estimate + target)
    return float(0.5 * np.sum(estimate * np.log(estimate / midpoint)) + 0.5 * np.sum(target * np.log(target / midpoint)))


def weighted_cdf_reference(exact_energy: np.ndarray, exact_density: np.ndarray, beta: float):
    q = beta * (exact_energy - np.min(exact_energy))
    order = np.argsort(q, axis=None)
    probability = exact_density.ravel()[order].astype(float)
    probability /= np.sum(probability)
    return q.ravel()[order], np.cumsum(probability)


def cdf_at(values_sorted: np.ndarray, cdf: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(values_sorted, query, side="right") - 1
    result = np.zeros_like(query, dtype=float)
    keep = indices >= 0
    result[keep] = cdf[indices[keep]]
    return result


def energy_ks(sample_q: np.ndarray, exact_q: np.ndarray, exact_cdf: np.ndarray) -> float:
    sample_sorted = np.sort(sample_q)
    reference = cdf_at(exact_q, exact_cdf, sample_sorted)
    n = sample_sorted.size
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(n, dtype=float) / n
    return float(max(np.max(np.abs(upper - reference)), np.max(np.abs(lower - reference))))


def metrics(samples, *, beta, energy_min, exact_q, exact_cdf, exact_density, exact_basin_mass, x_edges, y_edges):
    """Density JS, energy KS and basin TV of a sample against the Boltzmann grid."""
    sample_q = beta * (tb_energy(samples) - energy_min)
    histogram, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=(x_edges, y_edges))
    mass = np.bincount(tb_assignments(samples), minlength=3) / samples.shape[0]
    return {
        "density_js": smoothed_js_from_histogram(histogram.T, exact_density),
        "energy_ks": energy_ks(sample_q, exact_q, exact_cdf),
        "basin_tv": float(0.5 * np.sum(np.abs(mass - exact_basin_mass))),
        "basin_mass": mass.tolist(),
        "energy_q99": float(np.quantile(sample_q, 0.99)),
        **{f"tail_{threshold}": float(np.mean(sample_q > threshold)) for threshold in (3, 4, 5, 6)},
    }
