"""Archive-based low-score search with Monte Carlo and FPT on torsional Vina scores.

Both methods retain each particle's best conformation at every device step, so
transient low-score visits between host checkpoints are not lost.  First-hit
steps are recorded on the device; first-hit times are interpolated within the
timed 100-step chunks.  ``TOTAL_THRESHOLDS`` and ``BINDING_THRESHOLDS`` are
set per system by ``run_benchmark``.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

KB_KCAL_PER_MOL_K = 0.00198720425864083
REPLICAS = 512
CHECKPOINT_EVERY = 100
TEMPERATURE = 300.0
MC_PROPOSAL_SCALE = 0.5
CLUSTER_RMS_THRESHOLD_RADIANS = 0.25
FIXED_TIME_BUDGETS_SECONDS = (0.05, 0.15, 0.5, 1.0, 2.0)
# (easy, medium, strict); assigned per system by run_benchmark.
TOTAL_THRESHOLDS: tuple[float, ...] = ()
BINDING_THRESHOLDS: tuple[float, ...] = ()


def wrap_angle(theta: jax.Array) -> jax.Array:
    """Map angles to [-pi, pi) without changing their periodic meaning."""

    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def _threshold_label(threshold: float) -> str:
    sign = "m" if threshold < 0 else "p"
    magnitude = f"{abs(threshold):.3f}".replace(".", "p")
    return f"{sign}{magnitude}"


class SearchArchive(NamedTuple):
    best_total: jax.Array
    best_total_theta: jax.Array
    best_binding: jax.Array
    best_binding_theta: jax.Array
    first_total_step: jax.Array
    first_binding_step: jax.Array


def _initial_first_hit(scores: jax.Array, thresholds: jax.Array) -> jax.Array:
    hit = jnp.any(scores[:, None] <= thresholds[None, :], axis=0)
    return jnp.where(hit, 0, -1).astype(jnp.int32)


def init_search_archive(potential, theta: jax.Array) -> tuple[SearchArchive, jax.Array]:
    total, binding = potential.total_and_binding_score(theta)
    total = total[:, 0]
    binding = binding[:, 0]
    archive = SearchArchive(
        best_total=total,
        best_total_theta=theta,
        best_binding=binding,
        best_binding_theta=theta,
        first_total_step=_initial_first_hit(total, jnp.asarray(TOTAL_THRESHOLDS)),
        first_binding_step=_initial_first_hit(
            binding, jnp.asarray(BINDING_THRESHOLDS)
        ),
    )
    return archive, total[:, None]


def update_search_archive(
    archive: SearchArchive,
    theta: jax.Array,
    total: jax.Array,
    binding: jax.Array,
    step: jax.Array,
) -> SearchArchive:
    better_total = total < archive.best_total
    better_binding = binding < archive.best_binding
    total_hit = jnp.any(
        total[:, None] <= jnp.asarray(TOTAL_THRESHOLDS)[None, :], axis=0
    )
    binding_hit = jnp.any(
        binding[:, None] <= jnp.asarray(BINDING_THRESHOLDS)[None, :], axis=0
    )
    first_total = jnp.where(
        (archive.first_total_step < 0) & total_hit,
        step,
        archive.first_total_step,
    )
    first_binding = jnp.where(
        (archive.first_binding_step < 0) & binding_hit,
        step,
        archive.first_binding_step,
    )
    return SearchArchive(
        best_total=jnp.where(better_total, total, archive.best_total),
        best_total_theta=jnp.where(
            better_total[:, None, None], theta, archive.best_total_theta
        ),
        best_binding=jnp.where(better_binding, binding, archive.best_binding),
        best_binding_theta=jnp.where(
            better_binding[:, None, None], theta, archive.best_binding_theta
        ),
        first_total_step=first_total,
        first_binding_step=first_binding,
    )


def make_mc_search_chunk(potential, beta: float):
    def chunk(theta, energy, key, archive, base_step, n_steps):
        def body(carry, offset):
            current, current_energy, rng_key, accepted_count, current_archive = carry
            rng_key, proposal_key, index_key, accept_key = jax.random.split(
                rng_key, 4
            )
            torsion_index = jax.random.randint(
                index_key,
                shape=(current.shape[0],),
                minval=0,
                maxval=current.shape[-1],
            )
            selected = jax.nn.one_hot(
                torsion_index, current.shape[-1], dtype=current.dtype
            )[:, None, :]
            proposal = wrap_angle(
                current
                + MC_PROPOSAL_SCALE
                * jax.random.normal(
                    proposal_key, shape=(current.shape[0], 1, 1)
                )
                * selected
            )
            proposal_energy = potential(proposal)
            log_uniform = jnp.log(
                jax.random.uniform(
                    accept_key,
                    shape=current_energy.shape,
                    minval=jnp.finfo(current.dtype).tiny,
                    maxval=1.0,
                )
            )
            accepted = log_uniform < -beta * (proposal_energy - current_energy)
            next_theta = jnp.where(accepted[:, :, None], proposal, current)
            next_energy = jnp.where(accepted, proposal_energy, current_energy)
            binding = potential.binding_score(next_theta)[:, 0]
            next_archive = update_search_archive(
                current_archive,
                next_theta,
                next_energy[:, 0],
                binding,
                base_step + offset + 1,
            )
            return (
                next_theta,
                next_energy,
                rng_key,
                accepted_count + jnp.sum(accepted),
                next_archive,
            ), None

        initial = (
            theta,
            energy,
            key,
            jnp.asarray(0, dtype=jnp.int32),
            archive,
        )
        final, _ = jax.lax.scan(
            body, initial, jnp.arange(n_steps, dtype=jnp.int32)
        )
        return final

    return jax.jit(chunk, static_argnames=("n_steps",))


def make_fpt_search_chunk(sampler, potential):
    """FPT archive chunk with the Vina value-and-gradient evaluated once per update.

    Requires a fixed bandwidth (no kernel refresh inside the chunk).
    """

    def score_and_free_energy_gradient(state):
        (total, binding), system_grad = potential.value_and_grad(
            state.coordinate, sampler.beta
        )
        kernel_state = state.kernel_state
        kernel_bw = (
            kernel_state.bandwidth
            if kernel_state is not None
            else sampler.kernel.bandwidth
        )
        free_energy_grad, next_kernel_state = (
            sampler.combine_system_kernel_grad_with_state_core_no_noise(
                state.coordinate,
                system_grad,
                sampler.beta,
                kernel_state,
                kernel_bw,
            )
        )
        return (
            total[:, 0],
            binding[:, 0],
            free_energy_grad,
            next_kernel_state,
        )

    def apply_cached_gradient(state, gradient, next_kernel_state):
        next_key, _ = jax.random.split(state.rng_key)
        coordinate, opt_state = sampler.opt_update(
            state.coordinate, gradient, state.opt_state
        )
        return replace(
            state,
            coordinate=coordinate,
            opt_state=opt_state,
            rng_key=next_key,
            kernel_state=next_kernel_state,
        )

    def chunk(state, archive, base_step, n_steps):
        if n_steps < 1:
            raise ValueError("pipelined FPT chunks require n_steps >= 1")

        _, _, gradient, next_kernel_state = score_and_free_energy_gradient(state)

        def body(carry, offset):
            (
                current_state,
                current_archive,
                current_grad,
                current_kernel_state,
            ) = carry
            next_state = apply_cached_gradient(
                current_state, current_grad, current_kernel_state
            )
            total, binding, next_grad, following_kernel_state = (
                score_and_free_energy_gradient(next_state)
            )
            next_archive = update_search_archive(
                current_archive,
                next_state.coordinate,
                total,
                binding,
                base_step + offset + 1,
            )
            return (
                next_state,
                next_archive,
                next_grad,
                following_kernel_state,
            ), None

        carry, _ = jax.lax.scan(
            body,
            (state, archive, gradient, next_kernel_state),
            jnp.arange(n_steps - 1, dtype=jnp.int32),
        )
        state, archive, gradient, next_kernel_state = carry
        state = apply_cached_gradient(state, gradient, next_kernel_state)
        total, binding = potential.total_and_binding_score(
            state.coordinate, sampler.beta
        )
        archive = update_search_archive(
            archive,
            state.coordinate,
            total[:, 0],
            binding[:, 0],
            base_step + n_steps,
        )
        return state, archive

    return jax.jit(chunk, static_argnames=("n_steps",))


def _pair_distance(samples: np.ndarray) -> np.ndarray:
    cosine = np.cos(samples)
    sine = np.sin(samples)
    dot = cosine @ cosine.T + sine @ sine.T
    return np.maximum(2.0 - 2.0 * dot / samples.shape[1], 0.0)


def _unique_cluster_count(samples: np.ndarray, scores: np.ndarray) -> int:
    centers: list[np.ndarray] = []
    for index in np.argsort(scores):
        candidate = samples[index]
        if centers:
            delta = candidate[None, :] - np.asarray(centers)
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            rms = np.sqrt(np.mean(np.square(delta), axis=1))
            if np.any(rms <= CLUSTER_RMS_THRESHOLD_RADIANS):
                continue
        centers.append(candidate)
    return len(centers)


def _conditional_archive_metrics(
    theta: np.ndarray,
    scores: np.ndarray,
    thresholds: tuple[float, ...],
    score_name: str,
) -> dict[str, float]:
    samples = np.asarray(wrap_angle(theta)).reshape(theta.shape[0], -1)
    distance = _pair_distance(samples)
    result: dict[str, float] = {}
    for threshold in thresholds:
        prefix = f"{score_name}_le_{_threshold_label(threshold)}"
        selected = np.flatnonzero(scores <= threshold)
        result[f"{prefix}_archive_count"] = float(selected.size)
        result[f"{prefix}_archive_fraction"] = float(selected.size / len(scores))
        if selected.size:
            result[f"{prefix}_unique_clusters"] = float(
                _unique_cluster_count(samples[selected], scores[selected])
            )
        else:
            result[f"{prefix}_unique_clusters"] = 0.0
        if selected.size >= 2:
            selected_distance = distance[np.ix_(selected, selected)]
            off_diagonal = selected_distance[
                ~np.eye(selected.size, dtype=bool)
            ]
            result[f"{prefix}_mean_pairwise_torus_distance"] = float(
                np.mean(off_diagonal)
            )
            result[f"{prefix}_median_pairwise_torus_distance"] = float(
                np.median(off_diagonal)
            )
        else:
            result[f"{prefix}_mean_pairwise_torus_distance"] = float("nan")
            result[f"{prefix}_median_pairwise_torus_distance"] = float("nan")
    return result


def archive_record(
    method: str,
    seed: int,
    step: int,
    elapsed: float,
    archive: SearchArchive,
    *,
    acceptance_rate: float = float("nan"),
) -> dict[str, float | int | str]:
    host = jax.device_get(archive)
    record: dict[str, float | int | str] = {
        "method": method,
        "seed": seed,
        "step": step,
        "elapsed_seconds": elapsed,
        "acceptance_rate": acceptance_rate,
        "best_total": float(np.min(host.best_total)),
        "best_binding": float(np.min(host.best_binding)),
    }
    record.update(
        _conditional_archive_metrics(
            np.asarray(host.best_total_theta),
            np.asarray(host.best_total),
            TOTAL_THRESHOLDS,
            "sampling_energy",
        )
    )
    record.update(
        _conditional_archive_metrics(
            np.asarray(host.best_binding_theta),
            np.asarray(host.best_binding),
            BINDING_THRESHOLDS,
            "binding_score",
        )
    )
    for threshold, first_step in zip(TOTAL_THRESHOLDS, host.first_total_step):
        prefix = f"sampling_energy_le_{_threshold_label(threshold)}"
        record[f"{prefix}_first_hit_step"] = int(first_step)
    for threshold, first_step in zip(BINDING_THRESHOLDS, host.first_binding_step):
        prefix = f"binding_score_le_{_threshold_label(threshold)}"
        record[f"{prefix}_first_hit_step"] = int(first_step)
    return record


def _update_hit_times(
    hit_times: dict[str, float | None],
    archive: SearchArchive,
    base_step: int,
    elapsed_before: float,
    chunk_seconds: float,
) -> None:
    host = jax.device_get(archive)
    groups = (
        ("sampling_energy", TOTAL_THRESHOLDS, host.first_total_step),
        ("binding_score", BINDING_THRESHOLDS, host.first_binding_step),
    )
    for score_name, thresholds, first_steps in groups:
        for threshold, first_step in zip(thresholds, first_steps):
            key = f"{score_name}_le_{_threshold_label(threshold)}"
            first_step = int(first_step)
            if hit_times[key] is None and first_step >= 0:
                if first_step == 0:
                    hit_times[key] = 0.0
                else:
                    fraction = (first_step - base_step) / CHECKPOINT_EVERY
                    hit_times[key] = elapsed_before + chunk_seconds * fraction


def _empty_hit_times() -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for score_name, thresholds in (
        ("sampling_energy", TOTAL_THRESHOLDS),
        ("binding_score", BINDING_THRESHOLDS),
    ):
        for threshold in thresholds:
            result[f"{score_name}_le_{_threshold_label(threshold)}"] = None
    return result


def run_mc_seed(potential, chunk, theta, seed: int, steps: int):
    archive, energy = init_search_archive(potential, theta)
    key = jax.random.PRNGKey(seed + 1)
    records = [archive_record("mc", seed, 0, 0.0, archive)]
    hit_times = _empty_hit_times()
    _update_hit_times(hit_times, archive, 0, 0.0, 0.0)
    elapsed = 0.0
    accepted = 0
    for base_step in range(0, steps, CHECKPOINT_EVERY):
        started = time.perf_counter()
        theta, energy, key, chunk_accepted, archive = chunk(
            theta,
            energy,
            key,
            archive,
            jnp.asarray(base_step, dtype=jnp.int32),
            n_steps=CHECKPOINT_EVERY,
        )
        jax.block_until_ready(archive.best_total)
        chunk_seconds = time.perf_counter() - started
        _update_hit_times(hit_times, archive, base_step, elapsed, chunk_seconds)
        elapsed += chunk_seconds
        accepted += int(chunk_accepted)
        step = base_step + CHECKPOINT_EVERY
        records.append(
            archive_record(
                "mc",
                seed,
                step,
                elapsed,
                archive,
                acceptance_rate=accepted / (step * REPLICAS),
            )
        )
    return records, archive, hit_times


def run_fpt_seed(
    potential,
    sampler,
    chunk,
    initial_kernel_state,
    theta,
    seed: int,
    steps: int,
):
    sampler._sync_kernel_from_state(initial_kernel_state)
    state = sampler.init_state(theta)
    state = replace(
        state,
        rng_key=jax.random.PRNGKey(seed + 2),
        kernel_state=initial_kernel_state,
    )
    archive, _ = init_search_archive(potential, theta)
    records = [archive_record("fpt", seed, 0, 0.0, archive)]
    hit_times = _empty_hit_times()
    _update_hit_times(hit_times, archive, 0, 0.0, 0.0)
    elapsed = 0.0
    for base_step in range(0, steps, CHECKPOINT_EVERY):
        started = time.perf_counter()
        coordinate, kernel_state, nb_list = sampler._host_refresh_runtime_state(
            state.coordinate
        )
        state = replace(
            state,
            coordinate=coordinate,
            kernel_state=kernel_state,
            nb_list=nb_list,
        )
        state, archive = chunk(
            state,
            archive,
            jnp.asarray(base_step, dtype=jnp.int32),
            n_steps=CHECKPOINT_EVERY,
        )
        jax.block_until_ready(archive.best_total)
        chunk_seconds = time.perf_counter() - started
        _update_hit_times(hit_times, archive, base_step, elapsed, chunk_seconds)
        elapsed += chunk_seconds
        step = base_step + CHECKPOINT_EVERY
        records.append(archive_record("fpt", seed, step, elapsed, archive))
    return records, archive, hit_times


def _seed_summary(records, hit_times):
    final = dict(records[-1])
    for key, value in hit_times.items():
        final[f"{key}_first_hit_seconds_estimate"] = value
    return final


def _aggregate(seed_summaries, records):
    runtime_quality = []
    for method in ("mc", "fpt"):
        selected = [row for row in seed_summaries if row["method"] == method]
        runtimes = np.asarray(
            [row["elapsed_seconds"] for row in selected], dtype=np.float64
        )
        median_runtime = float(np.median(runtimes))
        runtime_quality.append(
            {
                "method": method,
                "median_seconds": median_runtime,
                "outlier_factor": 1.5,
                "outlier_seeds": [
                    int(row["seed"])
                    for row in selected
                    if row["elapsed_seconds"] > 1.5 * median_runtime
                ],
            }
        )
    hit_statistics = []
    for method in ("mc", "fpt"):
        selected = [row for row in seed_summaries if row["method"] == method]
        for score_name, thresholds in (
            ("sampling_energy", TOTAL_THRESHOLDS),
            ("binding_score", BINDING_THRESHOLDS),
        ):
            for threshold in thresholds:
                prefix = f"{score_name}_le_{_threshold_label(threshold)}"
                times = [
                    row[f"{prefix}_first_hit_seconds_estimate"]
                    for row in selected
                    if row[f"{prefix}_first_hit_seconds_estimate"] is not None
                ]
                steps = [
                    row[f"{prefix}_first_hit_step"]
                    for row in selected
                    if row[f"{prefix}_first_hit_step"] >= 0
                ]
                hit_statistics.append(
                    {
                        "method": method,
                        "score": score_name,
                        "threshold": threshold,
                        "successes": len(times),
                        "seeds": len(selected),
                        "success_rate": len(times) / len(selected),
                        "median_first_hit_seconds_estimate_among_successes": (
                            float(np.median(times)) if times else None
                        ),
                        "median_first_hit_step_among_successes": (
                            float(np.median(steps)) if steps else None
                        ),
                        "first_hit_step_iqr_among_successes": (
                            [
                                float(np.quantile(steps, 0.25)),
                                float(np.quantile(steps, 0.75)),
                            ]
                            if steps
                            else None
                        ),
                        "median_final_archive_count": float(
                            np.median(
                                [row[f"{prefix}_archive_count"] for row in selected]
                            )
                        ),
                        "median_final_unique_clusters": float(
                            np.median(
                                [row[f"{prefix}_unique_clusters"] for row in selected]
                            )
                        ),
                    }
                )

    fixed_budget_statistics = []
    for budget in FIXED_TIME_BUDGETS_SECONDS:
        for method in ("mc", "fpt"):
            for threshold in TOTAL_THRESHOLDS:
                prefix = f"sampling_energy_le_{_threshold_label(threshold)}"
                at_budget = []
                for seed in sorted({row["seed"] for row in records}):
                    candidates = [
                        row
                        for row in records
                        if row["method"] == method
                        and row["seed"] == seed
                        and row["elapsed_seconds"] <= budget
                    ]
                    at_budget.append(candidates[-1])
                counts = [row[f"{prefix}_archive_count"] for row in at_budget]
                clusters = [row[f"{prefix}_unique_clusters"] for row in at_budget]
                summaries = [
                    row
                    for row in seed_summaries
                    if row["method"] == method
                ]
                exact_hit_times = [
                    row[f"{prefix}_first_hit_seconds_estimate"]
                    for row in summaries
                ]
                fixed_budget_statistics.append(
                    {
                        "budget_seconds": budget,
                        "method": method,
                        "threshold": threshold,
                        "success_rate_from_exact_hit": float(
                            np.mean(
                                [
                                    value is not None and value <= budget
                                    for value in exact_hit_times
                                ]
                            )
                        ),
                        "median_archive_count": float(np.median(counts)),
                        "median_unique_clusters": float(np.median(clusters)),
                    }
                )
    return {
        "runtime_quality": runtime_quality,
        "first_hit_and_final_archive": hit_statistics,
        "fixed_time_budgets": fixed_budget_statistics,
    }
