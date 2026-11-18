#!/usr/bin/env python3
"""FPT with Fisher--Rao reweighting on neutral Ala10 (main text Fig. 5, SI Ala10).

Every ``--update-pace`` updates the particlewise bandwidths are refreshed so
that each nonself row mass sum_{j!=i} exp(-D_ij / 2 h_ij^2), h_ij^2 =
(h_i^2 + h_j^2)/2, lies in [--row-mass-min, --row-mass-max] (log relaxation
with a pull toward the mean log bandwidth); particles then move by the
replica-global RMS optimizer on beta grad U - s, with s the pointwise score of
the weighted leave-one-out product-von-Mises density.  Every
``--fisher-rao-every`` updates the weights take a mirror step along the
discrete free-energy gradient; when the maximum entitlement N w_i exceeds
``--entitlement-threshold`` (or the ESS falls below half of N) the ensemble is
realized to equal weights by balanced residual resampling and the realized
particles receive ``--block-move-rounds`` rounds of target-temperature
residue-block Metropolis moves.

Paper protocol: seeds 20260812--20260819 with the defaults,

    python -m examples.ala10.run_fpt --seed S --output data/ala10/fpt/band30_50_m128_seedS

Outputs: config.json, progress.jsonl (one record per snapshot),
metric_snapshots.npz (coordinates, weights, bandwidths every
``--metrics-every`` updates), checkpoint_*.npz, summary.json.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer import CoordTopology, EnergySystem, Sampler
from fp_transformer.ecepp import ECEPP3System, wrap_angles
from fp_transformer.kernel import Gaussian, PerReplicaRescaleConfig
from fp_transformer.optim import replica_global_rms
from fp_transformer.utils import torus_pairwise_distance

from examples.ala10.lib.block_moves import make_annealed_block_exploration
from examples.ala10.lib.estimator import make_von_mises_estimator
from examples.ala10.lib.fisher_rao import (
    balanced_realization_trigger,
    balanced_residual_resample_population,
    fisher_rao_weight_step,
    weighted_mean_and_variance,
)
from examples.ala10.lib.snapshots import save_metric_snapshot_bundle, write_jsonl_records
from examples.ala10.lib.system import (
    KB_KCAL_PER_MOL_K,
    InternalECEPPPotential,
    coil_state,
    evaluate,
    helicity,
    macro_state_masses,
    residue_block_structure,
)

DEFAULT_SYSTEM = Path(__file__).resolve().parent / "data" / "system.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", type=Path, default=DEFAULT_SYSTEM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--replicas", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=462.0)
    parser.add_argument("--initial-sigma", type=float, default=0.05,
                        help="std (rad) of the wrapped Gaussian about the coil conformation")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--update-pace", type=int, default=20, help="updates between bandwidth refreshes")
    parser.add_argument("--bandwidth", type=float, default=0.1, help="initial bandwidth (rad)")
    parser.add_argument("--row-mass-min", type=float, default=30.0)
    parser.add_argument("--row-mass-max", type=float, default=50.0)
    parser.add_argument("--bandwidth-min", type=float, default=0.001)
    parser.add_argument("--bandwidth-max", type=float, default=1.0)
    parser.add_argument("--bandwidth-adaptation-rate", type=float, default=0.2)
    parser.add_argument("--bandwidth-global-pull", type=float, default=0.02)
    parser.add_argument("--fisher-rao-every", type=int, default=200)
    parser.add_argument("--fisher-rao-step-size", type=float, default=0.01)
    parser.add_argument("--resample-ess-fraction", type=float, default=0.5)
    parser.add_argument("--entitlement-threshold", type=float, default=1.7)
    parser.add_argument("--block-move-rounds", type=int, default=128)
    parser.add_argument("--block-move-scales-rad", type=float, nargs="+", default=[0.35, 0.7, 1.4])
    parser.add_argument("--block-move-uniform-probability", type=float, default=0.25)
    parser.add_argument("--energy-minibatch", type=int, default=4096)
    parser.add_argument("--metrics-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    for name in ("fisher_rao_every", "metrics_every", "checkpoint_every", "steps"):
        if getattr(args, name) % args.update_pace:
            parser.error(f"--{name.replace('_', '-')} must be a multiple of --update-pace")
    if args.replicas % args.energy_minibatch:
        parser.error("--replicas must be a multiple of --energy-minibatch")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    system = ECEPP3System.from_npz(args.system)
    block_ids, block_labels = residue_block_structure(system.geometry.variable_names)
    beta = 1.0 / (KB_KCAL_PER_MOL_K * args.temperature)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config |= {"residue_block_ids": list(block_ids), "residue_block_labels": list(block_labels),
               "jax_devices": [str(device) for device in jax.devices()]}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    progress_path = args.output / "progress.jsonl"

    # Initial ensemble: wrapped Gaussian about the coil conformation, equal weights.
    initial = coil_state(system)
    coordinates = wrap_angles(initial[None, None, :] + args.initial_sigma
                              * jax.random.normal(jax.random.PRNGKey(args.seed), (args.replicas, 1, len(initial))))
    particle_weights = np.full(args.replicas, 1.0 / args.replicas, dtype=np.float32)
    lineage = np.arange(args.replicas, dtype=np.int64)

    potential = InternalECEPPPotential(system)
    energy_jit = jax.jit(potential)
    energy_system = EnergySystem(CoordTopology(coordinates), potential, minibatch_size=args.energy_minibatch,
                                 assert_divisible_minibatch=True, assume_batch_separable=True)
    coordinate_score, log_density, free_energy_weight_value_and_gradient = make_von_mises_estimator()
    # The bandwidth controller acts on the row sum including the unit self
    # term, so the nonself band [K_min, K_max] is [1 + K_min, 1 + K_max] here;
    # the score is the leave-one-out estimator supplied through score_fn.
    gaussian = Gaussian(
        args.replicas,
        bandwidth=args.bandwidth,
        distance_func=torus_pairwise_distance,
        kernel_min=1.0 + args.row_mass_min,
        kernel_max=1.0 + args.row_mass_max,
        update_bw_method="rescale",
        include_self=True,
        rescale_per_replica_config=PerReplicaRescaleConfig(
            enabled=True,
            fixed_row_sum_min=1.0 + args.row_mass_min,
            fixed_row_sum_max=1.0 + args.row_mass_max,
            bw_min=args.bandwidth_min,
            bw_max=args.bandwidth_max,
            log_step_size=args.bandwidth_adaptation_rate,
            global_pull=args.bandwidth_global_pull,
        ),
        pair_bandwidth="symmetric",
        score_fn=coordinate_score,
    )
    gaussian.particle_weights = jnp.asarray(particle_weights)
    refresh_bandwidth = jax.jit(gaussian.update_with_state)
    optimizer = replica_global_rms(args.learning_rate, b1=args.adam_beta1, b2=args.adam_beta2, eps=args.adam_eps)
    sampler = Sampler(system=energy_system, kernel=gaussian, optimizer=optimizer, temperature=args.temperature,
                      kB=KB_KCAL_PER_MOL_K, update_pace=args.update_pace, seed=args.seed + 2)
    block_moves = make_annealed_block_exploration(
        potential, beta=beta, block_ids=block_ids, n_steps=args.block_move_rounds,
        uniform_probability=args.block_move_uniform_probability, candidate_batching="dense")
    block_scales = jnp.asarray(args.block_move_scales_rad, dtype=jnp.float32)
    population_rng = np.random.default_rng(args.seed + 3)

    def body(_, current_state):
        return sampler.step_core_no_noise_no_metrics(current_state, sampler.beta, sampler.gamma)[0]

    run_chunk = jax.jit(lambda current_state: jax.lax.fori_loop(0, args.update_pace, body, current_state))

    state = sampler.init_state(coordinates)
    counters = {"fisher_rao_events": 0, "balanced_realizations": 0, "balanced_operations": 0,
                "block_move_events": 0, "block_move_attempts": 0.0, "block_move_accepts": 0.0}
    last_fisher_rao = None
    metric_snapshots = []
    start_wall = time.perf_counter()
    for step in range(args.update_pace, args.steps + 1, args.update_pace):
        state = replace(state, kernel_state=refresh_bandwidth(state.coordinate, state.kernel_state))
        gaussian._set_state_fast(state.kernel_state)
        state = run_chunk(state)

        if step % args.fisher_rao_every == 0:
            current_coordinate = np.asarray(state.coordinate)
            current_bandwidth = np.asarray(state.kernel_state.bandwidth).reshape(-1)
            dimensionless_energy = np.asarray(energy_jit(state.coordinate))[:, 0] / (KB_KCAL_PER_MOL_K * args.temperature)
            normalized_weights = particle_weights / np.sum(particle_weights)
            committed = np.where((h := helicity(current_coordinate[:, 0, :], system.geometry.variable_names)) >= 0.7,
                                 1, np.where(h <= 0.3, -1, 0))
            free_energy_before, weight_gradient = free_energy_weight_value_and_gradient(
                state.coordinate, jnp.asarray(current_bandwidth), jnp.asarray(normalized_weights),
                jnp.asarray(dimensionless_energy))
            flowed_weights, centered_gradient = fisher_rao_weight_step(
                normalized_weights, np.asarray(weight_gradient), step_size=args.fisher_rao_step_size)
            realization_weights = np.asarray(flowed_weights, dtype=np.float64).reshape(-1)
            realization_weights = realization_weights / float(np.sum(realization_weights))
            realize, trigger_stats = balanced_realization_trigger(
                realization_weights, ess_fraction=args.resample_ess_fraction,
                maximum_entitlement_threshold=args.entitlement_threshold)
            moved_coordinate, moved_key = state.coordinate, state.rng_key
            move_acceptance = None
            if realize:
                (resampled_coordinate, resampled_bandwidth, particle_weights, lineage, parent_indices,
                 realization_stats) = balanced_residual_resample_population(
                    current_coordinate, current_bandwidth, realization_weights, lineage,
                    rng=population_rng, roughening_scale=0.0)
                counters["balanced_realizations"] += 1
                counters["balanced_operations"] += realization_stats["operations"]
                moved_coordinate, moved_key, acceptance, accepted, attempted = block_moves(
                    jnp.asarray(resampled_coordinate), state.rng_key, 1.0,
                    jnp.asarray(1.0, dtype=jnp.float32), block_scales)
                move_acceptance = float(acceptance)
                counters["block_move_events"] += 1
                counters["block_move_accepts"] += float(accepted)
                counters["block_move_attempts"] += float(attempted)
            else:
                resampled_bandwidth = current_bandwidth
                particle_weights = flowed_weights
            kernel_state = replace(state.kernel_state, bandwidth=jnp.asarray(resampled_bandwidth),
                                   particle_weights=jnp.asarray(particle_weights))
            kernel_state = refresh_bandwidth(moved_coordinate, kernel_state)
            gaussian._set_state_fast(kernel_state)
            state = replace(state, coordinate=moved_coordinate, opt_state=optimizer.init(moved_coordinate),
                            rng_key=moved_key, kernel_state=kernel_state)
            counters["fisher_rao_events"] += 1
            last_fisher_rao = {
                "step": step,
                "free_energy_before": float(free_energy_before),
                "weight_gradient_variance": float(np.sum(normalized_weights * np.square(centered_gradient))),
                "ess_before": float(1.0 / np.sum(np.square(normalized_weights))),
                "ess_after_weight_flow": float(1.0 / np.sum(np.square(flowed_weights))),
                "balanced_trigger": trigger_stats,
                "resampled": bool(realize),
                "block_move_acceptance": move_acceptance,
                "macro_mass_before": macro_state_masses(committed, normalized_weights),
            }

        if step % args.metrics_every == 0 or step == args.steps:
            state = replace(state, coordinate=wrap_angles(state.coordinate[:, 0, :])[:, None, :])
            coordinates_host, bandwidth_host = jax.device_get((state.coordinate, state.kernel_state.bandwidth))
            context = {"step": step, "elapsed_seconds": time.perf_counter() - start_wall,
                       "exploration_events": counters["block_move_events"], **counters}
            if last_fisher_rao is not None:
                context["last_fisher_rao"] = deepcopy(last_fisher_rao)
            metric_snapshots.append({
                "step": np.asarray(step, dtype=np.int64),
                "coordinates": np.asarray(coordinates_host),
                "bandwidth": np.asarray(bandwidth_host),
                "raw_bandwidth": np.asarray(bandwidth_host),
                "particle_weights": np.asarray(particle_weights),
                "lineage": np.asarray(lineage),
                "context": context,
            })
        if step % args.checkpoint_every == 0 or step == args.steps:
            np.savez_compressed(args.output / f"checkpoint_{step:06d}.npz",
                                coordinates=np.asarray(state.coordinate),
                                bandwidth=np.asarray(state.kernel_state.bandwidth),
                                particle_weights=np.asarray(particle_weights),
                                lineage=np.asarray(lineage), step=step)
    sampling_seconds = time.perf_counter() - start_wall
    save_metric_snapshot_bundle(args.output / "metric_snapshots.npz", metric_snapshots)

    records = []
    for snapshot in metric_snapshots:
        energies, helicity_values, committed = evaluate(
            energy_jit, system, jnp.asarray(snapshot["coordinates"]), args.energy_minibatch)
        weights = np.asarray(snapshot["particle_weights"]) / np.sum(snapshot["particle_weights"])
        energy_mean, _ = weighted_mean_and_variance(energies, weights)
        records.append({
            **snapshot["context"],
            "energy_min_kcal_mol": float(np.min(energies)),
            "weighted_energy_mean_kcal_mol": energy_mean,
            "helicity_mean": float(np.mean(helicity_values)),
            "helix_replicas": int(np.count_nonzero(committed == 1)),
            "weighted_helix_probability": float(np.sum(weights[committed == 1])),
            "particle_weight_ess": float(1.0 / np.sum(np.square(weights))),
            "bandwidth": float(np.mean(snapshot["bandwidth"])),
        })
    write_jsonl_records(progress_path, records)
    summary = {"status": "complete", "replicas": args.replicas, "steps": args.steps,
               "sampling_wall_seconds": sampling_seconds, **counters,
               "block_move_acceptance": (counters["block_move_accepts"] / counters["block_move_attempts"]
                                         if counters["block_move_attempts"] else None)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
