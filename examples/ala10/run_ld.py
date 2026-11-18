"""Overdamped Langevin dynamics (LD) for neutral ECEPP/3 Ala10 (main text Fig. 5).

Euler--Maruyama, theta <- wrap(theta - dt beta grad U + sqrt(2 dt) xi), on the
20 periodic internal coordinates with the flat measure on the torus.  After
the steps listed in ``--target-block-move-at`` the same target-temperature
residue-block Metropolis kernel as in run_fpt.py is applied, so that LD
replays the block-move event times of a paired FPT run (run_ld_controls.py).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.ecepp import ECEPP3System, wrap_angles

from examples.ala10.lib.block_moves import make_annealed_block_exploration
from examples.ala10.lib.snapshots import save_metric_snapshot_bundle, write_jsonl_records
from examples.ala10.lib.system import (
    KB_KCAL_PER_MOL_K,
    InternalECEPPPotential,
    alpha_backbone_hbond_counts,
    alpha_backbone_hbond_pairs,
    circular_rms_degrees,
    coil_state,
    helicity,
    residue_block_structure,
)


Array = jax.Array
ForceEvaluator = Callable[[Array], tuple[Array, Array]]


def normalize_target_block_move_schedule(
    event_steps: tuple[int, ...] | list[int],
    *,
    total_steps: int,
    chunk_size: int,
) -> tuple[int, ...]:
    """Validate an externally specified matched-mutation schedule."""

    schedule = tuple(int(step) for step in event_steps)
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError(
            "target block-move steps must be unique and strictly increasing"
        )
    if any(step <= 0 or step > total_steps for step in schedule):
        raise ValueError(
            "target block-move steps must lie in (0, total_steps]"
        )
    if any(step % chunk_size for step in schedule):
        raise ValueError(
            "target block-move steps must be divisible by chunk_size"
        )
    return schedule


class LangevinState(NamedTuple):
    """Cached Markov state.

    ``dimensionless_gradient`` is beta times the physical energy gradient.
    """

    coordinates: Array
    energy_kcal_mol: Array
    dimensionless_gradient: Array
    key: Array
    accepted: Array
    attempted: Array


def overdamped_langevin_proposal(
    coordinates: Array,
    dimensionless_gradient: Array,
    standard_normal_noise: Array,
    time_step: float,
) -> Array:
    """Euler--Maruyama proposal for dtheta=-beta grad(U)dt+sqrt(2)dW."""

    time_step_array = jnp.asarray(time_step, dtype=coordinates.dtype)
    return wrap_angles(
        coordinates
        - time_step_array * dimensionless_gradient
        + jnp.sqrt(2.0 * time_step_array) * standard_normal_noise
    )


def make_force_evaluator(
    system: ECEPP3System, beta: float
) -> tuple[InternalECEPPPotential, ForceEvaluator]:
    """Build a batched physical-energy and beta-gradient evaluator."""

    potential = InternalECEPPPotential(system)

    def evaluate_force(coordinates: Array) -> tuple[Array, Array]:
        def summed_energy(values: Array):
            energies = potential(values[:, None, :])[:, 0]
            return jnp.sum(energies), energies

        (_, energies), physical_gradient = jax.value_and_grad(
            summed_energy, has_aux=True
        )(coordinates)
        dimensionless_gradient = (
            jnp.asarray(beta, dtype=physical_gradient.dtype)
            * physical_gradient
        )
        return energies, dimensionless_gradient

    return potential, jax.jit(evaluate_force)


def make_chunk_runner(
    method: str,
    evaluate_force: ForceEvaluator,
    beta: float,
    time_step: float,
    chunk_size: int,
):
    """Compile a fixed-length chunk with one new force call per Markov step."""

    if method != "ula":
        raise ValueError(f"unknown method: {method}")
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def one_step(state: LangevinState, _):
        next_key, noise_key, accept_key = jax.random.split(state.key, 3)
        noise = jax.random.normal(
            noise_key, state.coordinates.shape, dtype=state.coordinates.dtype
        )
        proposed_coordinates = overdamped_langevin_proposal(
            state.coordinates,
            state.dimensionless_gradient,
            noise,
            time_step,
        )
        proposed_energy, proposed_gradient = evaluate_force(proposed_coordinates)

        # accept_key is unused here; the three-way split keeps the random
        # stream of the stored Langevin runs.
        del accept_key
        next_state = LangevinState(
            coordinates=proposed_coordinates,
            energy_kcal_mol=proposed_energy,
            dimensionless_gradient=proposed_gradient,
            key=next_key,
            accepted=state.accepted,
            attempted=state.attempted,
        )
        return next_state, jnp.asarray(1.0, dtype=state.coordinates.dtype)

    @jax.jit
    def run_chunk(state: LangevinState):
        return jax.lax.scan(one_step, state, xs=None, length=chunk_size)

    return run_chunk


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _diagnostics_from_arrays(
    system: ECEPP3System,
    coordinates: np.ndarray,
    energies: np.ndarray,
    finite_fraction: float,
    accepted: int,
    attempted: int,
    step: int,
    method: str,
    temperature: float,
    time_step: float,
    alpha_reference: np.ndarray,
    coil_reference: np.ndarray,
    backbone_mask: np.ndarray,
    hbond_pairs: np.ndarray,
    dynamics_elapsed: float,
) -> dict:
    coordinates = np.asarray(coordinates)
    energies = np.asarray(energies)
    helicity_values = helicity(coordinates, system.geometry.variable_names)
    committed = np.where(
        helicity_values >= 0.7, 1, np.where(helicity_values <= 0.3, -1, 0)
    )
    alpha_distance = circular_rms_degrees(
        coordinates, alpha_reference, backbone_mask
    )
    coil_distance = circular_rms_degrees(
        coordinates, coil_reference, backbone_mask
    )
    hbond_counts = alpha_backbone_hbond_counts(
        system, coordinates, hbond_pairs
    )
    force_evaluations = coordinates.shape[0] * (step + 1)
    result = {
        "step": step,
        "langevin_time": float(step * time_step),
        "method": method,
        "replicas": int(coordinates.shape[0]),
        "force_evaluations_including_initial": int(force_evaluations),
        "force_evaluations_per_replica_including_initial": int(step + 1),
        "dynamics_wall_seconds": float(dynamics_elapsed),
        "finite_fraction": float(finite_fraction),
        "energy_mean_kcal_mol": float(np.mean(energies)),
        "energy_std_kcal_mol": float(np.std(energies)),
        "energy_min_kcal_mol": float(np.min(energies)),
        "energy_q01_kcal_mol": _percentile(energies, 1.0),
        "energy_q50_kcal_mol": _percentile(energies, 50.0),
        "energy_q99_kcal_mol": _percentile(energies, 99.0),
        "heat_capacity_kcal_mol_kelvin": float(
            np.var(energies)
            / (KB_KCAL_PER_MOL_K * float(temperature) ** 2)
        ),
        "helicity_mean": float(np.mean(helicity_values)),
        "helicity_q90": _percentile(helicity_values, 90.0),
        "helicity_q99": _percentile(helicity_values, 99.0),
        "helicity_max": float(np.max(helicity_values)),
        "particles_with_any_alpha_unit": int(
            np.count_nonzero(helicity_values > 0.0)
        ),
        "coil_count": int(np.count_nonzero(committed == -1)),
        "intermediate_count": int(np.count_nonzero(committed == 0)),
        "helix_count": int(np.count_nonzero(committed == 1)),
        "coil_probability": float(np.mean(committed == -1)),
        "intermediate_probability": float(np.mean(committed == 0)),
        "helix_probability": float(np.mean(committed == 1)),
        "alpha_backbone_rms_degrees_mean": float(np.mean(alpha_distance)),
        "alpha_backbone_rms_degrees_min": float(np.min(alpha_distance)),
        "coil_backbone_rms_degrees_mean": float(np.mean(coil_distance)),
        "coil_backbone_rms_degrees_min": float(np.min(coil_distance)),
        "alpha_hbond_count_mean": float(np.mean(hbond_counts)),
        "alpha_hbond_count_max": int(np.max(hbond_counts)),
        "particles_with_all_six_alpha_hbonds": int(
            np.count_nonzero(hbond_counts == 6)
        ),
    }
    result.update(
        {
            "accepted_proposals": None,
            "attempted_proposals": None,
            "acceptance_rate": None,
        }
    )
    return result


def _diagnostics(
    system: ECEPP3System,
    state: LangevinState,
    step: int,
    method: str,
    temperature: float,
    time_step: float,
    alpha_reference: np.ndarray,
    coil_reference: np.ndarray,
    backbone_mask: np.ndarray,
    hbond_pairs: np.ndarray,
    dynamics_elapsed: float,
) -> dict:
    """Compatibility wrapper for diagnostics on a live state."""

    coordinates = np.asarray(state.coordinates)
    energies = np.asarray(state.energy_kcal_mol)
    finite = np.isfinite(energies) & np.all(
        np.isfinite(np.asarray(state.dimensionless_gradient)), axis=-1
    )
    return _diagnostics_from_arrays(
        system,
        coordinates,
        energies,
        float(np.mean(finite)),
        int(np.asarray(state.accepted)),
        int(np.asarray(state.attempted)),
        step,
        method,
        temperature,
        time_step,
        alpha_reference,
        coil_reference,
        backbone_mask,
        hbond_pairs,
        dynamics_elapsed,
    )


def _save_checkpoint(
    output: Path, state: LangevinState, step: int, method: str
) -> Path:
    path = output / f"checkpoint_{step:07d}.npz"
    np.savez_compressed(
        path,
        method=np.asarray(method),
        step=np.asarray(step, dtype=np.int64),
        coordinates=np.asarray(state.coordinates)[:, None, :],
        energy_kcal_mol=np.asarray(state.energy_kcal_mol),
        dimensionless_gradient=np.asarray(state.dimensionless_gradient),
        key=np.asarray(state.key),
        accepted=np.asarray(state.accepted),
        attempted=np.asarray(state.attempted),
    )
    return path


def _capture_metric_snapshot(
    state: LangevinState,
    step: int,
    context: dict,
) -> dict:
    """Transfer only the arrays needed by post-sampling diagnostics."""

    finite_fraction = jnp.mean(
        (
            jnp.isfinite(state.energy_kcal_mol)
            & jnp.all(
                jnp.isfinite(state.dimensionless_gradient),
                axis=-1,
            )
        ).astype(state.coordinates.dtype)
    )
    (
        coordinates,
        energies,
        accepted,
        attempted,
        finite,
    ) = jax.device_get(
        (
            state.coordinates,
            state.energy_kcal_mol,
            state.accepted,
            state.attempted,
            finite_fraction,
        )
    )
    return {
        "step": np.asarray(step, dtype=np.int64),
        "coordinates": np.asarray(coordinates),
        "energy_kcal_mol": np.asarray(energies),
        "accepted": np.asarray(accepted),
        "attempted": np.asarray(attempted),
        "finite_fraction": np.asarray(finite),
        "context": dict(context),
    }


def _initial_coordinates(
    system: ECEPP3System,
    replicas: int,
    initial_sigma: float,
    seed: int,
) -> Array:
    initial = coil_state(system)
    noise = jax.random.normal(
        jax.random.PRNGKey(seed),
        (replicas, 1, initial.shape[0]),
        dtype=initial.dtype,
    )
    return wrap_angles(initial[None, None, :] + initial_sigma * noise)[:, 0, :]


def run_method(
    system: ECEPP3System,
    initial_coordinates: Array,
    method: str,
    output: Path,
    steps: int,
    temperature: float,
    time_step: float,
    seed: int,
    initial_sigma: float,
    chunk_size: int,
    metrics_every: int,
    checkpoint_every: int,
    target_block_move_at: tuple[int, ...],
    target_block_move_steps: int,
    target_block_move_scales: tuple[float, ...],
    target_block_move_uniform_probability: float,
) -> dict:
    """Run one baseline and write compatible progress/checkpoint artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    config_path = output / "config.json"
    summary_path = output / "summary.json"
    if progress_path.exists() or config_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing run in {output}"
        )

    beta = 1.0 / (KB_KCAL_PER_MOL_K * temperature)
    potential, evaluate_force = make_force_evaluator(system, beta)
    initial_energy, initial_gradient = evaluate_force(initial_coordinates)
    initial_energy.block_until_ready()
    state = LangevinState(
        coordinates=initial_coordinates,
        energy_kcal_mol=initial_energy,
        dimensionless_gradient=initial_gradient,
        key=jax.random.fold_in(jax.random.PRNGKey(seed), 1),
        accepted=jnp.asarray(0, dtype=jnp.int32),
        attempted=jnp.asarray(0, dtype=jnp.int32),
    )
    run_chunk = make_chunk_runner(
        method,
        evaluate_force,
        beta,
        time_step,
        chunk_size,
    )
    target_block_move = None
    target_block_move_at_set = frozenset(target_block_move_at)
    target_block_move_scales_array = jnp.asarray(
        target_block_move_scales,
        dtype=initial_coordinates.dtype,
    )
    if target_block_move_at:
        residue_block_ids, _ = residue_block_structure(
            system.geometry.variable_names
        )
        target_block_move = make_annealed_block_exploration(
            potential,
            beta=beta,
            block_ids=residue_block_ids,
            n_steps=target_block_move_steps,
            uniform_probability=target_block_move_uniform_probability,
            candidate_batching="dense",
        )

    variable_names = np.asarray(system.geometry.variable_names)
    backbone_mask = np.isin(variable_names, ("phi", "psi"))
    alpha_reference = np.asarray(system.geometry.initial_variables)
    coil_reference = np.asarray(coil_state(system))
    hbond_pairs = alpha_backbone_hbond_pairs(system)
    config = {
        "method": method,
        "system": "neutral ECEPP/3 Ala10 internal-coordinate model",
        "target": "exp(-beta U(theta)) dtheta on the flat 20D torus",
        "temperature_kelvin": float(temperature),
        "beta_mol_per_kcal": float(beta),
        "replicas": int(initial_coordinates.shape[0]),
        "steps": int(steps),
        "time_step": float(time_step),
        "initial_state": "coil(phi=-150 deg, psi=150 deg)",
        "initial_sigma_radians": float(initial_sigma),
        "seed": int(seed),
        "dynamics_seed": "jax.random.fold_in(PRNGKey(seed), 1)",
        "chunk_size": int(chunk_size),
        "metrics_every": int(metrics_every),
        "checkpoint_every": int(checkpoint_every),
        "target_block_move_at": list(target_block_move_at),
        "target_block_move_steps": int(target_block_move_steps),
        "target_block_move_scales_rad": list(target_block_move_scales),
        "target_block_move_uniform_probability": float(
            target_block_move_uniform_probability
        ),
        "target_block_move_target_invariant": bool(target_block_move_at),
        "target_block_move_effective_temperature_kelvin": (
            float(temperature) if target_block_move_at else None
        ),
        "metrics_mode": "post-sampling-offline",
        "metric_snapshot_file": "metric_snapshots.npz",
        "wrapped_normal_image_radius": None,
        "force_accounting": (
            "one initial force evaluation plus one force evaluation per "
            "particle per dynamics step. Each matched target block-move "
            "event separately records its energy-only proposals and one "
            "full force refresh per particle."
        ),
        "algorithm": (
            "theta <- wrap(theta - dt beta grad(U) + sqrt(2 dt) Normal(0,I))"
            if method == "ula"
            else (
                "same drifted wrapped-normal proposal followed by the exact "
                "Hastings correction for the truncated wrapped-normal image sum"
            )
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Compile the finite-state sentinel and exercise the small device-to-host
    # snapshot transfer before any benchmark timer starts.
    _capture_metric_snapshot(state, 0, {})
    if target_block_move is not None:
        (
            compiled_coordinates,
            _,
            _,
            _,
            _,
        ) = target_block_move(
            state.coordinates[:, None, :],
            state.key,
            jnp.asarray(1.0, dtype=state.coordinates.dtype),
            jnp.asarray(1.0, dtype=state.coordinates.dtype),
            target_block_move_scales_array,
        )
        compiled_energy, compiled_gradient = evaluate_force(
            compiled_coordinates[:, 0, :]
        )
        jax.block_until_ready((compiled_energy, compiled_gradient))

    dynamics_elapsed = 0.0
    snapshot_capture_elapsed = 0.0
    snapshot_write_elapsed = 0.0
    checkpoint_elapsed = 0.0
    target_block_move_event_count = 0
    target_block_move_attempt_count = 0
    target_block_move_accept_count = 0
    target_block_move_energy_evaluation_count = 0
    target_block_move_force_refresh_count = 0
    last_target_block_move = None
    total_start = time.perf_counter()
    metric_snapshots: list[dict] = []
    snapshot_start = time.perf_counter()
    initial_snapshot = _capture_metric_snapshot(
        state,
        0,
        {
            "sampling_wall_seconds": 0.0,
            "checkpoint_wall_seconds": 0.0,
        },
    )
    snapshot_capture_elapsed += time.perf_counter() - snapshot_start
    initial_snapshot["context"].update(
        {
            "run_wall_seconds": float(time.perf_counter() - total_start),
            "snapshot_wall_seconds": float(snapshot_capture_elapsed),
        }
    )
    metric_snapshots.append(initial_snapshot)

    saved_steps: set[int] = set()
    for step in range(chunk_size, steps + 1, chunk_size):
        chunk_start = time.perf_counter()
        state, _ = run_chunk(state)
        state.coordinates.block_until_ready()
        dynamics_elapsed += time.perf_counter() - chunk_start

        if target_block_move is not None and step in target_block_move_at_set:
            move_start = time.perf_counter()
            (
                moved_coordinates,
                moved_key,
                move_acceptance,
                move_accepted,
                move_attempted,
            ) = target_block_move(
                state.coordinates[:, None, :],
                state.key,
                jnp.asarray(1.0, dtype=state.coordinates.dtype),
                jnp.asarray(1.0, dtype=state.coordinates.dtype),
                target_block_move_scales_array,
            )
            moved_coordinates = moved_coordinates[:, 0, :]
            moved_energy, moved_gradient = evaluate_force(moved_coordinates)
            jax.block_until_ready((moved_energy, moved_gradient))
            state = LangevinState(
                coordinates=moved_coordinates,
                energy_kcal_mol=moved_energy,
                dimensionless_gradient=moved_gradient,
                key=moved_key,
                accepted=state.accepted,
                attempted=state.attempted,
            )
            dynamics_elapsed += time.perf_counter() - move_start
            accepted_value = int(move_accepted)
            attempted_value = int(move_attempted)
            target_block_move_event_count += 1
            target_block_move_accept_count += accepted_value
            target_block_move_attempt_count += attempted_value
            target_block_move_energy_evaluation_count += (
                int(initial_coordinates.shape[0])
                * (target_block_move_steps + 1)
            )
            target_block_move_force_refresh_count += int(
                initial_coordinates.shape[0]
            )
            last_target_block_move = {
                "step": int(step),
                "acceptance": float(move_acceptance),
                "accepted": accepted_value,
                "attempted": attempted_value,
                "inverse_temperature_scale": 1.0,
                "effective_temperature_kelvin": float(temperature),
            }

        if step % metrics_every == 0 or step == steps:
            snapshot_start = time.perf_counter()
            snapshot = _capture_metric_snapshot(
                state,
                step,
                {
                    "sampling_wall_seconds": float(dynamics_elapsed),
                    "checkpoint_wall_seconds": float(checkpoint_elapsed),
                    "target_block_move_events": (
                        target_block_move_event_count
                    ),
                    "target_block_move_attempts": (
                        target_block_move_attempt_count
                    ),
                    "target_block_move_accepts": (
                        target_block_move_accept_count
                    ),
                    "target_block_move_energy_evaluations": (
                        target_block_move_energy_evaluation_count
                    ),
                    "target_block_move_force_refresh_evaluations": (
                        target_block_move_force_refresh_count
                    ),
                    "last_target_block_move": last_target_block_move,
                },
            )
            snapshot_capture_elapsed += time.perf_counter() - snapshot_start
            snapshot["context"].update(
                {
                    "run_wall_seconds": float(
                        time.perf_counter() - total_start
                    ),
                    "snapshot_wall_seconds": float(
                        snapshot_capture_elapsed
                    ),
                }
            )
            metric_snapshots.append(snapshot)

        if step % checkpoint_every == 0 or step == steps:
            checkpoint_start = time.perf_counter()
            _save_checkpoint(output, state, step, method)
            checkpoint_elapsed += time.perf_counter() - checkpoint_start
            saved_steps.add(step)

    if steps not in saved_steps:
        checkpoint_start = time.perf_counter()
        _save_checkpoint(output, state, steps, method)
        checkpoint_elapsed += time.perf_counter() - checkpoint_start

    online_wall_seconds = time.perf_counter() - total_start
    snapshot_write_start = time.perf_counter()
    save_metric_snapshot_bundle(
        output / "metric_snapshots.npz",
        metric_snapshots,
    )
    snapshot_write_elapsed = time.perf_counter() - snapshot_write_start

    offline_analysis_start = time.perf_counter()
    records: list[dict] = []
    first_helix_observed_step = None
    for snapshot in metric_snapshots:
        context = snapshot["context"]
        record = _diagnostics_from_arrays(
            system,
            snapshot["coordinates"],
            snapshot["energy_kcal_mol"],
            float(snapshot["finite_fraction"]),
            int(snapshot["accepted"]),
            int(snapshot["attempted"]),
            int(snapshot["step"]),
            method,
            temperature,
            time_step,
            alpha_reference,
            coil_reference,
            backbone_mask,
            hbond_pairs,
            float(context["sampling_wall_seconds"]),
        )
        record.update(
            {
                "run_wall_seconds": float(context["run_wall_seconds"]),
                "sampling_wall_seconds": float(
                    context["sampling_wall_seconds"]
                ),
                "snapshot_wall_seconds": float(
                    context["snapshot_wall_seconds"]
                ),
                "checkpoint_wall_seconds": float(
                    context["checkpoint_wall_seconds"]
                ),
                "target_block_move_events": int(
                    context.get("target_block_move_events", 0)
                ),
                "target_block_move_attempts": int(
                    context.get("target_block_move_attempts", 0)
                ),
                "target_block_move_accepts": int(
                    context.get("target_block_move_accepts", 0)
                ),
                "target_block_move_energy_evaluations": int(
                    context.get("target_block_move_energy_evaluations", 0)
                ),
                "target_block_move_force_refresh_evaluations": int(
                    context.get(
                        "target_block_move_force_refresh_evaluations", 0
                    )
                ),
                "last_target_block_move": context.get(
                    "last_target_block_move"
                ),
            }
        )
        record["force_evaluations_including_initial"] += record[
            "target_block_move_force_refresh_evaluations"
        ]
        record["force_evaluations_per_replica_including_initial"] += record[
            "target_block_move_events"
        ]
        record.update(
            {
                "interval_accepted_proposals": None,
                "interval_attempted_proposals": None,
                "interval_acceptance_rate": None,
            }
        )
        records.append(record)
        if (
            first_helix_observed_step is None
            and record["helix_count"] > 0
        ):
            first_helix_observed_step = int(record["step"])
        print(json.dumps(record, sort_keys=True), flush=True)
    write_jsonl_records(progress_path, records, sort_keys=True)
    offline_analysis_elapsed = time.perf_counter() - offline_analysis_start
    total_elapsed = time.perf_counter() - total_start
    initial_record = records[0]
    final_record = records[-1]
    summary = {
        **config,
        "total_wall_seconds": float(total_elapsed),
        "dynamics_wall_seconds": float(dynamics_elapsed),
        "sampling_wall_seconds": float(dynamics_elapsed),
        "online_wall_seconds": float(online_wall_seconds),
        "snapshot_capture_wall_seconds": float(
            snapshot_capture_elapsed
        ),
        "snapshot_write_wall_seconds": float(snapshot_write_elapsed),
        "snapshot_wall_seconds": float(
            snapshot_capture_elapsed + snapshot_write_elapsed
        ),
        "checkpoint_wall_seconds": float(checkpoint_elapsed),
        "offline_analysis_wall_seconds": float(
            offline_analysis_elapsed
        ),
        "diagnostic_and_io_wall_seconds": float(
            total_elapsed - dynamics_elapsed
        ),
        "force_evaluations_including_initial": int(
            initial_coordinates.shape[0]
            * (steps + 1 + target_block_move_event_count)
        ),
        "target_block_move_events": target_block_move_event_count,
        "target_block_move_attempts": target_block_move_attempt_count,
        "target_block_move_accepts": target_block_move_accept_count,
        "target_block_move_acceptance": (
            target_block_move_accept_count / target_block_move_attempt_count
            if target_block_move_attempt_count
            else None
        ),
        "target_block_move_energy_evaluations": (
            target_block_move_energy_evaluation_count
        ),
        "target_block_move_force_refresh_evaluations": (
            target_block_move_force_refresh_count
        ),
        "force_evaluations_per_dynamics_second": float(
            initial_coordinates.shape[0] * steps / dynamics_elapsed
        ),
        "first_helix_observed_step_at_metrics_resolution": (
            first_helix_observed_step
        ),
        "peak_observed_helicity": float(
            max(record["helicity_max"] for record in records)
        ),
        "peak_observed_helix_count": int(
            max(record["helix_count"] for record in records)
        ),
        "initial": initial_record,
        "final": final_record,
        "max_resident_set_size_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", type=Path, default=Path(__file__).resolve().parent / "data" / "system.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=4_096)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=462.0)
    parser.add_argument("--initial-sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--time-step", type=float, default=2.5e-5,
                        help="Euler--Maruyama dt in squared-radian Langevin time")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--metrics-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--target-block-move-at", type=int, nargs="*", default=[],
                        help="dynamics steps after which to apply the target-temperature block-move kernel")
    parser.add_argument("--target-block-move-steps", type=int, default=128,
                        help="Metropolis rounds per block-move event")
    parser.add_argument("--target-block-move-scales-rad", type=float, nargs="+", default=[0.35, 0.7, 1.4])
    parser.add_argument("--target-block-move-uniform-probability", type=float, default=0.25)
    args = parser.parse_args()

    for name in ("replicas", "steps", "chunk_size", "metrics_every", "checkpoint_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("steps", "metrics_every", "checkpoint_every"):
        if getattr(args, name) % args.chunk_size:
            parser.error(f"--{name.replace('_', '-')} must be divisible by --chunk-size")
    if args.time_step <= 0:
        parser.error("--time-step must be positive")
    try:
        target_block_move_at = normalize_target_block_move_schedule(
            args.target_block_move_at, total_steps=args.steps, chunk_size=args.chunk_size)
    except ValueError as error:
        parser.error(str(error))
    if any(scale <= 0.0 for scale in args.target_block_move_scales_rad):
        parser.error("target-block-move-scales-rad must all be positive")
    if not 0.0 <= args.target_block_move_uniform_probability <= 1.0:
        parser.error("target-block-move-uniform-probability must lie in [0, 1]")

    system = ECEPP3System.from_npz(args.system)
    initial_coordinates = _initial_coordinates(system, args.replicas, args.initial_sigma, args.seed)
    summary = run_method(
        system=system,
        initial_coordinates=initial_coordinates,
        method="ula",
        output=args.output,
        steps=args.steps,
        temperature=args.temperature,
        time_step=args.time_step,
        seed=args.seed,
        initial_sigma=args.initial_sigma,
        chunk_size=args.chunk_size,
        metrics_every=args.metrics_every,
        checkpoint_every=args.checkpoint_every,
        target_block_move_at=target_block_move_at,
        target_block_move_steps=args.target_block_move_steps if target_block_move_at else 0,
        target_block_move_scales=tuple(args.target_block_move_scales_rad),
        target_block_move_uniform_probability=args.target_block_move_uniform_probability,
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
