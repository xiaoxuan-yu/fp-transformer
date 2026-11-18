#!/usr/bin/env python3
"""Target-preserving endpoint continuation on the 3D eight-well target (SI).

The endpoints come from fixed-temperature shared-bandwidth FPT advanced by
explicit Euler with the Langevin step (13500 updates from the Boltzmann law
restricted to the initial well).  Four development FPT endpoints select one proposal scale by label-free
importance-weight ESS. Four disjoint confirmatory endpoints are then continued
with a frozen, normalized wrapped-Gaussian mixture and a complete independence
Metropolis--Hastings correction. Basin labels and exact references are used
only after scale selection for evaluation.

Paper protocol:

    python run_endpoint_continuation.py --output-dir DATA/n3_shared_euler_bb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


from examples.periodic3d.common import (
    BETA_0,
    NUM_UPDATES,
    basin_boltzmann_initialization,
    build_fpt_sampler,
    energy_numpy,
    energy_single,
    sampler_snapshot,
    well_assignments,
    wrap_to_pi,
)

DEFAULT_DEVELOPMENT_SEEDS = (20260740, 20260741, 20260742, 20260743)
DEFAULT_CONFIRMATORY_SEEDS = (20260744, 20260745, 20260746, 20260747)
DEFAULT_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_CHECKPOINTS = (0, 1, 2, 5, 10, 25, 50, 100, 200)
TARGET_BETA = BETA_0  # set to beta_scale * BETA_0 in main()
PERIOD = 2.0 * math.pi
PERIODIC_ENERGY = jax.jit(jax.vmap(energy_single))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_DEVELOPMENT_SEEDS,
    )
    parser.add_argument(
        "--confirmatory-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIRMATORY_SEEDS,
    )
    parser.add_argument(
        "--scale-candidates",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
    )
    parser.add_argument(
        "--selected-scale",
        type=float,
        default=None,
        help="Skip development selection and use this pre-frozen scale.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--num-updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--fpt-learning-rate", type=float, default=0.008)
    parser.add_argument("--fpt-optimizer", choices=("rmsprop", "sgd"), default="sgd",
                        help="rmsprop, or sgd = explicit Euler with step --fpt-learning-rate")
    parser.add_argument("--fpt-kernel-min", type=float, default=2.0)
    parser.add_argument("--fpt-kernel-max", type=float, default=4.0)
    parser.add_argument("--fpt-update-pace", type=int, default=1)
    parser.add_argument("--beta-scale", type=float, default=3.0,
                        help="target beta in units of beta_0; barrier = beta_scale k_BT")
    parser.add_argument("--pilot-proposals", type=int, default=4096)
    parser.add_argument("--pilot-seed-offset", type=int, default=51001)
    parser.add_argument("--chain-seed-offset", type=int, default=52001)
    parser.add_argument("--sampler-seed-offset", type=int, default=53001)
    parser.add_argument("--bandwidth-floor", type=float, default=1e-6)
    parser.add_argument("--image-tail-tolerance", type=float, default=1e-12)
    parser.add_argument("--logpdf-batch-size", type=int, default=512)
    parser.add_argument("--reference-resolution", type=int, default=96)
    parser.add_argument("--reference-ensembles", type=int, default=256)
    parser.add_argument("--reference-seed", type=int, default=20260748)
    parser.add_argument(
        "--force-endpoints",
        action="store_true",
        help="Regenerate cached FPT endpoints even if matching records exist.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    checkpoints = list(args.checkpoints)
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("--checkpoints must start at zero")
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("--checkpoints must be sorted and unique")
    if args.replicas <= 0 or args.num_updates <= 0:
        raise ValueError("replicas and number of updates must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if args.checkpoint_interval % args.fpt_update_pace != 0:
        raise ValueError(
            "checkpoint interval must align with the FPT kernel refresh pace"
        )
    if args.num_updates % args.checkpoint_interval != 0:
        raise ValueError("number of updates must be divisible by checkpoint interval")
    if len(set(args.development_seeds) & set(args.confirmatory_seeds)):
        raise ValueError("development and confirmatory seeds must be disjoint")
    if any(scale <= 0.0 for scale in args.scale_candidates):
        raise ValueError("proposal scales must be positive")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def endpoint_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "replicas": int(args.replicas),
        "initial_law": "basin_boltzmann",
        "num_updates": int(args.num_updates),
        "checkpoint_interval": int(args.checkpoint_interval),
        "target_beta": TARGET_BETA,
        "fpt_learning_rate": float(args.fpt_learning_rate),
        "optimizer": args.fpt_optimizer,
        "kernel_refresh_pace": int(args.fpt_update_pace),
        "kernel_band": [float(args.fpt_kernel_min), float(args.fpt_kernel_max)],
        "beta_scale": float(args.beta_scale),
    }


def endpoint_cache_matches(report: dict[str, Any], expected: dict[str, Any]) -> bool:
    return report.get("schema_version") == 1 and report.get("protocol") == expected


def run_fpt_endpoint(
    initial: np.ndarray,
    *,
    transport_seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    sampler = build_fpt_sampler(
        initial,
        learning_rate=args.fpt_learning_rate,
        beta=TARGET_BETA,
        seed=transport_seed + args.sampler_seed_offset,
        optimizer_name=args.fpt_optimizer,
        kernel_min=args.fpt_kernel_min,
        kernel_max=args.fpt_kernel_max,
        update_pace=args.fpt_update_pace,
    )
    total_steps = 0
    start = time.perf_counter()
    while total_steps < args.num_updates:
        sampler.run(
            args.checkpoint_interval,
            verbose=False,
            continue_from_last_state=total_steps > 0,
        )
        total_steps += args.checkpoint_interval
    endpoint = sampler_snapshot(sampler)
    bandwidth = np.asarray(sampler.kernel.bandwidth, dtype=float).reshape(-1)
    elapsed = time.perf_counter() - start
    if total_steps != args.num_updates:
        raise RuntimeError(f"FPT ended at {total_steps} rather than {args.num_updates}")
    if endpoint.shape != (args.replicas, 3):
        raise RuntimeError(f"unexpected endpoint shape {endpoint.shape}")
    if bandwidth.shape != (args.replicas,):
        raise RuntimeError(f"unexpected bandwidth shape {bandwidth.shape}")
    if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(bandwidth)):
        raise RuntimeError("non-finite FPT endpoint state")
    return endpoint, bandwidth, elapsed


def load_or_generate_endpoint(
    phase: str,
    seed: int,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache_dir = args.output_dir / "endpoints" / phase / f"seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = cache_dir / "report.json"
    arrays_path = cache_dir / "endpoints.npz"
    protocol = endpoint_protocol(args)
    expected = {
        **protocol,
        "phase": phase,
        "initial_seed": int(seed),
        "sampler_seed": int(seed + args.sampler_seed_offset),
    }
    if report_path.exists() and arrays_path.exists() and not args.force_endpoints:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if endpoint_cache_matches(report, expected):
            arrays = np.load(arrays_path)
            return {
                "seed": seed,
                "fixed_endpoint": np.asarray(arrays["fixed_endpoint"]),
                "fixed_bandwidth": np.asarray(arrays["fixed_bandwidth"]),
                "report": report,
            }

    initial = basin_boltzmann_initialization(seed, replicas=args.replicas, beta=TARGET_BETA)
    fixed_endpoint, fixed_bandwidth, fixed_wall = run_fpt_endpoint(
        initial,
        transport_seed=seed,
        args=args,
    )
    np.savez_compressed(
        arrays_path,
        initial=initial,
        fixed_endpoint=fixed_endpoint,
        fixed_bandwidth=fixed_bandwidth,
    )
    report = {
        "schema_version": 1,
        "protocol": expected,
        "wall_time_s": {"fixed": fixed_wall},
        "bandwidth_summary": {"fixed": bandwidth_summary(fixed_bandwidth)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "seed": seed,
        "fixed_endpoint": fixed_endpoint,
        "fixed_bandwidth": fixed_bandwidth,
        "report": report,
    }


def bandwidth_summary(bandwidth: np.ndarray) -> dict[str, float]:
    values = np.asarray(bandwidth, dtype=float)
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def choose_image_radius(max_sigma: float, tolerance: float) -> int:
    if max_sigma <= 0.0 or not math.isfinite(max_sigma):
        raise ValueError(f"invalid maximum sigma {max_sigma}")
    radius = 1
    while math.exp(-0.5 * (PERIOD * radius / max_sigma) ** 2) > tolerance:
        radius += 1
    return radius


def image_offsets(radius: int, dtype: jnp.dtype) -> jax.Array:
    return PERIOD * jnp.arange(-radius, radius + 1, dtype=dtype)


@jax.jit
def wrapped_endpoint_logpdf(
    query: jax.Array,
    centers: jax.Array,
    bandwidths: jax.Array,
    scale: jax.Array,
    offsets: jax.Array,
) -> jax.Array:
    """Evaluate a normalized wrapped isotropic variable-bandwidth mixture."""

    sigma = scale * bandwidths
    residual = (
        query[:, None, :, None]
        - centers[None, :, :, None]
        + offsets[None, None, None, :]
    )
    axis_logpdf = (
        -0.5 * jnp.square(residual / sigma[None, :, None, None])
        - jnp.log(sigma[None, :, None, None])
        - 0.5 * math.log(2.0 * math.pi)
    )
    wrapped_axis_logpdf = logsumexp(axis_logpdf, axis=3)
    component_logpdf = jnp.sum(wrapped_axis_logpdf, axis=2)
    return logsumexp(component_logpdf, axis=1) - math.log(centers.shape[0])


def wrapped_logpdf_batched(
    query: np.ndarray,
    centers: np.ndarray,
    bandwidths: np.ndarray,
    scale: float,
    offsets: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, query.shape[0], batch_size):
        stop = min(start + batch_size, query.shape[0])
        outputs.append(
            np.asarray(
                wrapped_endpoint_logpdf(
                    jnp.asarray(query[start:stop]),
                    jnp.asarray(centers),
                    jnp.asarray(bandwidths),
                    jnp.asarray(scale, dtype=jnp.asarray(centers).dtype),
                    jnp.asarray(offsets, dtype=jnp.asarray(centers).dtype),
                )
            )
        )
    return np.concatenate(outputs)


@partial(jax.jit, static_argnames=("proposal_count",))
def draw_wrapped_endpoint_kernel(
    key: jax.Array,
    centers: jax.Array,
    bandwidths: jax.Array,
    scale: jax.Array,
    proposal_count: int,
) -> jax.Array:
    component_key, noise_key = jax.random.split(key)
    components = jax.random.randint(
        component_key,
        shape=(proposal_count,),
        minval=0,
        maxval=centers.shape[0],
    )
    noise = jax.random.normal(
        noise_key,
        shape=(proposal_count, centers.shape[1]),
        dtype=centers.dtype,
    )
    proposal = centers[components] + (
        scale * bandwidths[components]
    )[:, None] * noise
    return wrap_to_pi(proposal)


@partial(jax.jit, static_argnames=("steps",))
def wrapped_endpoint_imh_trace(
    initial: jax.Array,
    key: jax.Array,
    centers: jax.Array,
    bandwidths: jax.Array,
    scale: jax.Array,
    beta: jax.Array,
    offsets: jax.Array,
    steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the frozen wrapped-mixture independence MH transition."""

    initial_energy = PERIODIC_ENERGY(initial)
    initial_log_q = wrapped_endpoint_logpdf(
        initial, centers, bandwidths, scale, offsets
    )

    def step(carry, _):
        current, current_energy, current_log_q, current_key = carry
        current_key, proposal_key, uniform_key = jax.random.split(current_key, 3)
        proposal = draw_wrapped_endpoint_kernel(
            proposal_key,
            centers,
            bandwidths,
            scale,
            current.shape[0],
        )
        proposal_energy = PERIODIC_ENERGY(proposal)
        proposal_log_q = wrapped_endpoint_logpdf(
            proposal, centers, bandwidths, scale, offsets
        )
        log_ratio = (
            -beta * (proposal_energy - current_energy)
            + current_log_q
            - proposal_log_q
        )
        finite = (
            jnp.all(jnp.isfinite(proposal), axis=1)
            & jnp.isfinite(proposal_energy)
            & jnp.isfinite(proposal_log_q)
            & jnp.isfinite(log_ratio)
        )
        bounded_log_ratio = jnp.where(
            finite, jnp.minimum(0.0, log_ratio), -jnp.inf
        )
        log_uniform = -jax.random.exponential(
            uniform_key,
            shape=(current.shape[0],),
            dtype=current.dtype,
        )
        accepted = log_uniform < bounded_log_ratio
        updated = jnp.where(accepted[:, None], proposal, current)
        updated_energy = jnp.where(accepted, proposal_energy, current_energy)
        updated_log_q = jnp.where(accepted, proposal_log_q, current_log_q)
        displacement = jnp.arctan2(
            jnp.sin(updated - current), jnp.cos(updated - current)
        )
        squared_displacement = jnp.sum(jnp.square(displacement), axis=1)
        next_carry = (
            updated,
            updated_energy,
            updated_log_q,
            current_key,
        )
        return next_carry, (updated, accepted, squared_displacement)

    _carry, outputs = jax.lax.scan(
        step,
        (initial, initial_energy, initial_log_q, key),
        xs=None,
        length=steps,
    )
    return outputs


def importance_weight_diagnostics(log_weight: np.ndarray) -> dict[str, float]:
    values = np.asarray(log_weight, dtype=float)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))
    squared_total = float(np.sum(np.square(shifted)))
    ess = total * total / squared_total
    normalized = shifted / total
    entropy = -float(np.sum(normalized * np.log(np.maximum(normalized, 1e-300))))
    return {
        "importance_ess": ess,
        "importance_ess_fraction": ess / values.size,
        "maximum_weight_share": float(np.max(normalized)),
        "weight_perplexity_fraction": math.exp(entropy) / values.size,
    }


def tune_scale(
    records: list[dict[str, Any]],
    *,
    scales: list[float],
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    for record in records:
        centers = np.asarray(record["fixed_endpoint"])
        bandwidths = np.maximum(
            np.asarray(record["fixed_bandwidth"], dtype=float),
            args.bandwidth_floor,
        )
        proposal_rows.append(
            {
                "phase": "development",
                "seed": int(record["seed"]),
                **bandwidth_summary(bandwidths),
            }
        )
        pilot_key = jax.random.PRNGKey(record["seed"] + args.pilot_seed_offset)
        for scale in scales:
            radius = choose_image_radius(
                scale * float(np.max(bandwidths)), args.image_tail_tolerance
            )
            offsets = np.asarray(
                image_offsets(radius, jnp.asarray(centers).dtype)
            )
            candidate = np.asarray(
                draw_wrapped_endpoint_kernel(
                    pilot_key,
                    jnp.asarray(centers),
                    jnp.asarray(bandwidths),
                    jnp.asarray(scale, dtype=jnp.asarray(centers).dtype),
                    args.pilot_proposals,
                )
            )
            log_q = wrapped_logpdf_batched(
                candidate,
                centers,
                bandwidths,
                scale,
                offsets,
                args.logpdf_batch_size,
            )
            energy = np.asarray(PERIODIC_ENERGY(jnp.asarray(candidate)))
            diagnostics = importance_weight_diagnostics(-TARGET_BETA * energy - log_q)
            seed_rows.append(
                {
                    "row_type": "seed",
                    "seed": int(record["seed"]),
                    "scale": float(scale),
                    "pilot_proposals": int(args.pilot_proposals),
                    "image_radius": radius,
                    **diagnostics,
                }
            )
    aggregate_rows = []
    for scale in scales:
        selected_rows = [row for row in seed_rows if row["scale"] == scale]
        fractions = np.asarray(
            [row["importance_ess_fraction"] for row in selected_rows]
        )
        aggregate_rows.append(
            {
                "row_type": "aggregate",
                "scale": float(scale),
                "geometric_mean_ess_fraction": float(
                    np.exp(np.mean(np.log(np.maximum(fractions, 1e-300))))
                ),
                "mean_ess_fraction": float(np.mean(fractions)),
                "minimum_ess_fraction": float(np.min(fractions)),
                "maximum_weight_share": float(
                    max(row["maximum_weight_share"] for row in selected_rows)
                ),
            }
        )
    selected = max(
        aggregate_rows,
        key=lambda row: (
            row["geometric_mean_ess_fraction"],
            row["minimum_ess_fraction"],
            -row["scale"],
        ),
    )
    for row in seed_rows + aggregate_rows:
        row["selected"] = bool(row["scale"] == selected["scale"])
    return float(selected["scale"]), seed_rows + aggregate_rows, proposal_rows


def build_energy_reference(resolution: int) -> dict[str, np.ndarray]:
    spacing = PERIOD / resolution
    values = np.linspace(-math.pi, math.pi, resolution, endpoint=False) + 0.5 * spacing
    mesh = np.meshgrid(values, values, values, indexing="ij")
    coordinates = np.stack(mesh, axis=-1).reshape(-1, 3)
    energy = energy_numpy(coordinates)
    log_weight = -TARGET_BETA * energy
    weight = np.exp(log_weight - np.max(log_weight))
    probability = weight / np.sum(weight)
    order = np.argsort(energy)
    return {
        "energy": energy[order],
        "cdf": np.cumsum(probability[order]),
        "probability": probability,
        "unsorted_energy": energy,
    }


def weighted_energy_ks(
    sample_energy: np.ndarray,
    reference_energy: np.ndarray,
    reference_cdf: np.ndarray,
) -> float:
    sample = np.sort(np.asarray(sample_energy, dtype=float))
    indices = np.searchsorted(reference_energy, sample, side="right") - 1
    target_cdf = np.where(indices >= 0, reference_cdf[np.maximum(indices, 0)], 0.0)
    count = sample.size
    upper = np.arange(1, count + 1, dtype=float) / count
    lower = np.arange(0, count, dtype=float) / count
    return float(max(np.max(upper - target_cdf), np.max(target_cdf - lower)))


def occupancy_metrics(samples: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(well_assignments(samples), minlength=8)
    probability = counts / np.sum(counts)
    positive = probability[probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)) / math.log(8.0))
    return {
        "counts": counts.tolist(),
        "covered_modes": int(np.count_nonzero(counts)),
        "occupancy_entropy": entropy,
        "basin_tv": float(0.5 * np.sum(np.abs(probability - 0.125))),
    }


def evaluate_snapshot(samples: np.ndarray, reference: dict[str, np.ndarray]) -> dict[str, Any]:
    energy = energy_numpy(samples)
    return {
        **occupancy_metrics(samples),
        "energy_ks": weighted_energy_ks(
            energy,
            reference["energy"],
            reference["cdf"],
        ),
        "mean_energy": float(np.mean(energy)),
    }


def finite_iid_reference(
    reference: dict[str, np.ndarray],
    *,
    replicas: int,
    ensemble_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    basin_counts = rng.multinomial(replicas, np.full(8, 0.125), size=ensemble_count)
    basin_probability = basin_counts / replicas
    entropy = -np.sum(
        np.where(
            basin_probability > 0.0,
            basin_probability * np.log(np.maximum(basin_probability, 1e-300)),
            0.0,
        ),
        axis=1,
    ) / math.log(8.0)
    basin_tv = 0.5 * np.sum(np.abs(basin_probability - 0.125), axis=1)
    sampled_indices = rng.choice(
        reference["probability"].size,
        size=(ensemble_count, replicas),
        replace=True,
        p=reference["probability"],
    )
    sampled_energy = reference["unsorted_energy"][sampled_indices]
    energy_ks = np.asarray(
        [
            weighted_energy_ks(row, reference["energy"], reference["cdf"])
            for row in sampled_energy
        ]
    )
    result = {}
    for name, values in (
        ("occupancy_entropy", entropy),
        ("basin_tv", basin_tv),
        ("energy_ks", energy_ks),
    ):
        result[name] = {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return result


def run_confirmatory(
    records: list[dict[str, Any]],
    *,
    selected_scale: float,
    reference: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    saved: dict[str, list[np.ndarray]] = {"fixed": []}
    maximum_step = max(args.checkpoints)
    for record in records:
        for schedule in ("fixed",):
            endpoint = np.asarray(record[f"{schedule}_endpoint"])
            bandwidths = np.maximum(
                np.asarray(record[f"{schedule}_bandwidth"], dtype=float),
                args.bandwidth_floor,
            )
            radius = choose_image_radius(
                selected_scale * float(np.max(bandwidths)),
                args.image_tail_tolerance,
            )
            offsets = image_offsets(radius, jnp.asarray(endpoint).dtype)
            proposal_rows.append(
                {
                    "phase": "confirmatory",
                    "seed": int(record["seed"]),
                    "schedule": schedule,
                    "selected_scale": selected_scale,
                    "image_radius": radius,
                    **bandwidth_summary(bandwidths),
                }
            )
            start = time.perf_counter()
            full_trace, full_accepted, full_displacement = wrapped_endpoint_imh_trace(
                jnp.asarray(endpoint),
                jax.random.PRNGKey(record["seed"] + args.chain_seed_offset),
                jnp.asarray(endpoint),
                jnp.asarray(bandwidths),
                jnp.asarray(selected_scale, dtype=jnp.asarray(endpoint).dtype),
                jnp.asarray(TARGET_BETA, dtype=jnp.asarray(endpoint).dtype),
                offsets,
                maximum_step,
            )
            trace = np.empty((maximum_step + 1, *endpoint.shape), dtype=endpoint.dtype)
            trace[0] = endpoint
            trace[1:] = np.asarray(full_trace)
            accepted = np.asarray(full_accepted)
            displacement = np.asarray(full_displacement)
            runtime = time.perf_counter() - start
            saved[schedule].append(trace[np.asarray(args.checkpoints)])
            for checkpoint in args.checkpoints:
                metric_rows.append(
                    {
                        "seed": int(record["seed"]),
                        "schedule": schedule,
                        "continuation_steps": int(checkpoint),
                        **evaluate_snapshot(trace[checkpoint], reference),
                    }
                )
                if checkpoint > 0:
                    transition_rows.append(
                        {
                            "seed": int(record["seed"]),
                            "schedule": schedule,
                            "continuation_steps": int(checkpoint),
                            "accepted": int(np.sum(accepted[:checkpoint])),
                            "proposals": int(checkpoint * args.replicas),
                            "acceptance_rate": float(np.mean(accepted[:checkpoint])),
                            "mean_squared_torus_displacement": float(
                                np.mean(displacement[:checkpoint])
                            ),
                            "runtime_s_for_full_trace": runtime,
                        }
                    )
    saved_arrays = {
        schedule: np.asarray(schedule_samples)
        for schedule, schedule_samples in saved.items()
    }
    return metric_rows, transition_rows, saved_arrays, proposal_rows


def summarize_metrics(
    rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    checkpoints: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for schedule in ("fixed",):
        summary[schedule] = {}
        for checkpoint in checkpoints:
            selected = [
                row
                for row in rows
                if row["schedule"] == schedule
                and row["continuation_steps"] == checkpoint
            ]
            metrics = {}
            for name in ("occupancy_entropy", "basin_tv", "energy_ks"):
                values = np.asarray([row[name] for row in selected], dtype=float)
                metrics[name] = {
                    "mean": float(np.mean(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            if checkpoint > 0:
                transitions = [
                    row
                    for row in transition_rows
                    if row["schedule"] == schedule
                    and row["continuation_steps"] == checkpoint
                ]
                metrics["acceptance_rate"] = {
                    "mean": float(
                        np.mean([row["acceptance_rate"] for row in transitions])
                    ),
                    "minimum": float(
                        np.min([row["acceptance_rate"] for row in transitions])
                    ),
                    "maximum": float(
                        np.max([row["acceptance_rate"] for row in transitions])
                    ),
                }
            summary[schedule][str(checkpoint)] = metrics
    return summary


def main() -> None:
    global TARGET_BETA
    args = parse_args()
    validate_args(args)
    TARGET_BETA = args.beta_scale * BETA_0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = [
        load_or_generate_endpoint(
            "development",
            seed,
            args=args,
        )
        for seed in args.development_seeds
    ]
    confirmatory = [
        load_or_generate_endpoint(
            "confirmatory",
            seed,
            args=args,
        )
        for seed in args.confirmatory_seeds
    ]

    if args.selected_scale is None:
        selected_scale, scale_rows, development_proposals = tune_scale(
            development,
            scales=list(args.scale_candidates),
            args=args,
        )
    else:
        if args.selected_scale <= 0.0:
            raise ValueError("--selected-scale must be positive")
        selected_scale = float(args.selected_scale)
        scale_rows = []
        development_proposals = []

    reference = build_energy_reference(args.reference_resolution)
    finite_iid = finite_iid_reference(
        reference,
        replicas=args.replicas,
        ensemble_count=args.reference_ensembles,
        seed=args.reference_seed,
    )
    metric_rows, transition_rows, saved_arrays, confirmatory_proposals = (
        run_confirmatory(
            confirmatory,
            selected_scale=selected_scale,
            reference=reference,
            args=args,
        )
    )
    summary = {
        "schema_version": 1,
        "protocol": {
            **endpoint_protocol(args),
            "development_seeds": list(args.development_seeds),
            "confirmatory_seeds": list(args.confirmatory_seeds),
            "scale_candidates": list(args.scale_candidates),
            "selected_scale": selected_scale,
            "scale_selection": (
                "pre_frozen" if args.selected_scale is not None else "development_is_ess"
            ),
            "pilot_proposals": int(args.pilot_proposals),
            "checkpoints": list(args.checkpoints),
            "wrapped_period": PERIOD,
            "image_tail_tolerance": float(args.image_tail_tolerance),
            "reference_resolution": int(args.reference_resolution),
            "reference_ensembles": int(args.reference_ensembles),
            "basin_labels_used_for_scale_selection": False,
            "exact_reference_used_for_scale_selection": False,
            "target_preserving": True,
            "proposal_frozen_before_confirmation": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "finite_iid_reference": finite_iid,
        "metric_summary": summarize_metrics(
            metric_rows, transition_rows, list(args.checkpoints)
        ),
    }
    write_csv(args.output_dir / "scale_selection.csv", scale_rows)
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "transitions.csv", transition_rows)
    write_csv(
        args.output_dir / "proposal_fits.csv",
        development_proposals + confirmatory_proposals,
    )
    np.savez_compressed(
        args.output_dir / "samples.npz",
        checkpoints=np.asarray(args.checkpoints),
        seeds=np.asarray(args.confirmatory_seeds),
        fixed=saved_arrays["fixed"],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "Saved wrapped PeriodicSin4 endpoint-kernel continuation to "
        f"{args.output_dir} (selected scale {selected_scale:g})"
    )


if __name__ == "__main__":
    main()
