#!/usr/bin/env python3
"""MC/FPT low-score search on the cleaned PoseBusters systems (main text Fig. 6, SI).

Every system uses 512 replicas, 1000 updates at T = 300 K, and seeds
6300--6304 shared by both methods.  MC is random-walk Metropolis over ligand
torsions (proposal scale 0.5 rad; ``--mc-temperature`` changes only its
acceptance temperature).  FPT uses the pointwise leave-one-out torus KDE score
with fixed bandwidth h = 0.4 (fused row pass) and Adam with learning rate 0.03.

Paper protocols:

    python -m examples.vina.run_benchmark --num-seeds 5 \
        --output-dir data/vina/full_row_mc300
    python -m examples.vina.run_benchmark --num-seeds 5 --mc-temperature 75 \
        --output-dir data/vina/full_row_mc75
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fp_transformer import CoordTopology, EnergySystem, Sampler
from fp_transformer.kernel import Gaussian
from fp_transformer.utils import torus_pairwise_distance

from examples.vina import search
from examples.vina.dataset import load_posebusters_systems
from examples.vina.pdbqt import build_pdbqt_potential
from examples.vina.search import KB_KCAL_PER_MOL_K

TOTAL_EXCESS = (1.50, 0.75, 0.25)
BINDING_EXCESS = (1.00, 0.50, 0.25)
THRESHOLD_LEVELS = ("easy", "medium", "strict")
OFFICIAL_VINA_COMMIT = "3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645"
BASELINE_VERSION = "adam_single_scale_loo_fixed_bw_pipeline_v3"
BASELINE_REPLICAS = search.REPLICAS
BASELINE_STEPS = 1000
BASELINE_TEMPERATURE = search.TEMPERATURE
BASELINE_MC_PROPOSAL_SCALE = search.MC_PROPOSAL_SCALE
BASELINE_FPT_LEARNING_RATE = 0.03
BASELINE_FPT_BANDWIDTH = 0.4
MAX_STEPS = 1000
ALL_SYSTEMS = load_posebusters_systems()


def make_fpt_sampler(potential, template_theta, *, seed: int) -> Sampler:
    """Fixed-bandwidth FPT with the pointwise leave-one-out score and Adam."""
    kernel = Gaussian(
        template_theta.shape[0],
        bandwidth=BASELINE_FPT_BANDWIDTH,
        distance_func=torus_pairwise_distance,
        update_bw_method="fixed",
        score_backend="fused",
    )
    return Sampler(
        system=EnergySystem(CoordTopology(template_theta), potential),
        kernel=kernel,
        optimizer=optax.adam(learning_rate=BASELINE_FPT_LEARNING_RATE),
        temperature=BASELINE_TEMPERATURE,
        kB=KB_KCAL_PER_MOL_K,
        sample_cv=None,
        update_pace=search.CHECKPOINT_EVERY,
        seed=seed,
    )


def frozen_baseline_metadata():
    """Return the versioned algorithm configuration embedded in every run."""

    return {
        "version": BASELINE_VERSION,
        "degrees_of_freedom": "ligand_torsions_only",
        "replicas": BASELINE_REPLICAS,
        "steps": BASELINE_STEPS,
        "temperature_kelvin": BASELINE_TEMPERATURE,
        "mc": {
            "optimizer": "random_walk_metropolis",
            "proposal_scale_radians": BASELINE_MC_PROPOSAL_SCALE,
        },
        "fpt": {
            "optimizer": "adam",
            "learning_rate": BASELINE_FPT_LEARNING_RATE,
            "kernel": "single_scale_gaussian_loo",
            "bandwidth": BASELINE_FPT_BANDWIDTH,
            "bandwidth_update": "fixed",
            "value_and_grad_pipeline": True,
            "stochastic_noise": False,
            "kernel_weight": 1.0,
            "kernel_activation_step": 0,
        },
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _thresholds(zero_score: float, excess: tuple[float, ...]):
    return tuple(float(zero_score + value) for value in excess)


def _hard_start_theta(potential, shape, seed: int) -> jax.Array:
    """Draw uniform torsions, rejecting easy hits whenever feasible.

    Low-dimensional systems can have no practically reachable region outside
    the union of the easy total and binding thresholds. After the bounded
    rejection budget, retain the shared MC/FPT initial particles and report
    their step-zero hits instead of dropping the system from the benchmark.
    """

    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(
        key, shape=shape, minval=-jnp.pi, maxval=jnp.pi
    )
    for _ in range(32):
        total = potential(theta)[:, 0]
        binding = potential.binding_score(theta)[:, 0]
        invalid = (total <= search.TOTAL_THRESHOLDS[0]) | (
            binding <= search.BINDING_THRESHOLDS[0]
        )
        if not bool(np.any(np.asarray(invalid))):
            return theta
        key, candidate_key = jax.random.split(key)
        candidate = jax.random.uniform(
            candidate_key, shape=shape, minval=-jnp.pi, maxval=jnp.pi
        )
        theta = jnp.where(invalid[:, None, None], candidate, theta)
    total = potential(theta)[:, 0]
    binding = potential.binding_score(theta)[:, 0]
    total_hits = np.asarray(total <= search.TOTAL_THRESHOLDS[0])
    binding_hits = np.asarray(binding <= search.BINDING_THRESHOLDS[0])
    print(
        "hard-start fallback "
        f"seed={seed} total_easy={int(np.sum(total_hits))}/{shape[0]} "
        f"binding_easy={int(np.sum(binding_hits))}/{shape[0]} "
        f"either_easy={int(np.sum(total_hits | binding_hits))}/{shape[0]}",
        flush=True,
    )
    return theta


def _run_system(name, spec, args):
    loaded = build_pdbqt_potential(
        receptor_paths=spec["receptor_paths"],
        ligand_path=spec["ligand_path"],
        interaction_backend=args.interaction_backend,
    )
    potential = loaded.potential
    zero = jnp.zeros((1, 1, potential.num_torsions))
    zero_total = float(potential(zero)[0, 0])
    zero_binding = float(potential.binding_score(zero)[0, 0])
    search.TOTAL_THRESHOLDS = _thresholds(zero_total, TOTAL_EXCESS)
    search.BINDING_THRESHOLDS = _thresholds(zero_binding, BINDING_EXCESS)

    mc_temperature = float(args.mc_temperature)
    beta = 1.0 / (KB_KCAL_PER_MOL_K * mc_temperature)
    mc_chunk = search.make_mc_search_chunk(potential, beta)
    template_theta = jnp.zeros(
        (BASELINE_REPLICAS, 1, potential.num_torsions)
    )
    sampler = make_fpt_sampler(potential, template_theta, seed=args.seed_base + 2)
    initial_kernel_state = sampler._kernel_state_from_kernel()
    fpt_chunk = search.make_fpt_search_chunk(sampler, potential)

    warm_theta = jax.random.uniform(
        jax.random.PRNGKey(args.seed_base),
        shape=template_theta.shape,
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    warm_archive, warm_energy = search.init_search_archive(
        potential, warm_theta
    )
    warm_mc = mc_chunk(
        warm_theta,
        warm_energy,
        jax.random.PRNGKey(args.seed_base + 1),
        warm_archive,
        jnp.asarray(0, dtype=jnp.int32),
        n_steps=search.CHECKPOINT_EVERY,
    )
    jax.block_until_ready(warm_mc[4].best_total)
    sampler._sync_kernel_from_state(initial_kernel_state)
    warm_state = sampler.init_state(warm_theta)
    warm_state = replace(
        warm_state,
        rng_key=jax.random.PRNGKey(args.seed_base + 2),
        kernel_state=initial_kernel_state,
    )
    warm_fpt = fpt_chunk(
        warm_state,
        warm_archive,
        jnp.asarray(0, dtype=jnp.int32),
        n_steps=search.CHECKPOINT_EVERY,
    )
    jax.block_until_ready(warm_fpt[1].best_total)

    records = []
    seed_summaries = []
    system_output = args.output_dir / name
    system_output.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed_base, args.seed_base + args.num_seeds):
        theta = _hard_start_theta(
            potential,
            template_theta.shape,
            seed,
        )
        mc_records, mc_archive, mc_hit_times = search.run_mc_seed(
            potential, mc_chunk, theta, seed, args.steps
        )
        fpt_records, fpt_archive, fpt_hit_times = search.run_fpt_seed(
            potential,
            sampler,
            fpt_chunk,
            initial_kernel_state,
            theta,
            seed,
            args.steps,
        )
        records.extend(mc_records)
        records.extend(fpt_records)
        seed_summaries.append(search._seed_summary(mc_records, mc_hit_times))
        seed_summaries.append(search._seed_summary(fpt_records, fpt_hit_times))
        np.savez_compressed(
            system_output / f"archives_seed{seed}.npz",
            mc_best_total=np.asarray(mc_archive.best_total),
            mc_best_total_theta=np.asarray(mc_archive.best_total_theta),
            mc_best_binding=np.asarray(mc_archive.best_binding),
            mc_best_binding_theta=np.asarray(mc_archive.best_binding_theta),
            fpt_best_total=np.asarray(fpt_archive.best_total),
            fpt_best_total_theta=np.asarray(fpt_archive.best_total_theta),
            fpt_best_binding=np.asarray(fpt_archive.best_binding),
            fpt_best_binding_theta=np.asarray(fpt_archive.best_binding_theta),
        )
        print(
            f"system={name} seed={seed} "
            f"mc={mc_records[-1]['elapsed_seconds']:.3f}s "
            f"fpt={fpt_records[-1]['elapsed_seconds']:.3f}s",
            flush=True,
        )

    metadata = {
        "system": name,
        "description": spec["description"],
        "pose_source": spec["pose_source"],
        "official_vina_source": (
            "https://github.com/ccsb-scripps/AutoDock-Vina"
        ),
        "official_vina_commit": OFFICIAL_VINA_COMMIT,
        "num_torsions": potential.num_torsions,
        "ligand_heavy_atoms": loaded.ligand_heavy_atoms,
        "receptor_heavy_atoms": loaded.receptor_heavy_atoms_total,
        "receptor_pocket_atoms": loaded.receptor_pocket_atoms,
        "zero_total": zero_total,
        "zero_binding": zero_binding,
        "total_thresholds": search.TOTAL_THRESHOLDS,
        "binding_thresholds": search.BINDING_THRESHOLDS,
        "total_threshold_excess": TOTAL_EXCESS,
        "binding_threshold_excess": BINDING_EXCESS,
        "steps": args.steps,
        "replicas": BASELINE_REPLICAS,
        "interaction_backend": args.interaction_backend,
        "fpt_score": "row",
        "mc_temperature_kelvin": mc_temperature,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "initialization": (
            "uniform torsions with bounded rejection of particles meeting "
            "either easy threshold; shared MC/FPT step-zero hits are retained "
            "when rejection is infeasible"
        ),
        "baseline": frozen_baseline_metadata(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    aggregate = {
        "metadata": metadata,
        **search._aggregate(seed_summaries, records),
    }
    _write_csv(system_output / "trajectories.csv", records)
    _write_csv(system_output / "seed_summary.csv", seed_summaries)
    (system_output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    return aggregate


def _suite_rows(aggregates):
    rows = []
    for system, aggregate in aggregates.items():
        metadata = aggregate["metadata"]
        statistics = aggregate["first_hit_and_final_archive"]
        for score_name in ("sampling_energy", "binding_score"):
            score_rows = [row for row in statistics if row["score"] == score_name]
            for level, mc, fpt in zip(
                THRESHOLD_LEVELS,
                [row for row in score_rows if row["method"] == "mc"],
                [row for row in score_rows if row["method"] == "fpt"],
            ):
                mc_step = mc["median_first_hit_step_among_successes"]
                fpt_step = fpt["median_first_hit_step_among_successes"]
                mc_time = mc[
                    "median_first_hit_seconds_estimate_among_successes"
                ]
                fpt_time = fpt[
                    "median_first_hit_seconds_estimate_among_successes"
                ]
                rows.append(
                    {
                        "system": system,
                        "torsions": metadata["num_torsions"],
                        "score": score_name,
                        "threshold_level": level,
                        "threshold": mc["threshold"],
                        "mc_success_rate": mc["success_rate"],
                        "fpt_success_rate": fpt["success_rate"],
                        "mc_median_hit_step": mc_step,
                        "fpt_median_hit_step": fpt_step,
                        "step_speedup_mc_over_fpt": (
                            mc_step / fpt_step
                            if mc_step is not None and fpt_step not in (None, 0)
                            else None
                        ),
                        "mc_median_hit_seconds_estimate": mc_time,
                        "fpt_median_hit_seconds_estimate": fpt_time,
                        "time_speedup_mc_over_fpt": (
                            mc_time / fpt_time
                            if mc_time is not None and fpt_time not in (None, 0)
                            else None
                        ),
                        "mc_median_archive_count": mc[
                            "median_final_archive_count"
                        ],
                        "fpt_median_archive_count": fpt[
                            "median_final_archive_count"
                        ],
                        "mc_median_unique_clusters": mc[
                            "median_final_unique_clusters"
                        ],
                        "fpt_median_unique_clusters": fpt[
                            "median_final_unique_clusters"
                        ],
                    }
                )
    return rows


def _write_markdown_summary(path: Path, rows) -> None:
    lines = [
        "# Real-system short-budget Vina results",
        "",
        "Primary endpoint: first hit of the strict total-score threshold under "
        "the frozen 500-step protocol.",
        "",
        "| System | Torsions | Threshold | MC success | FPT success | "
        "MC median step | FPT median step | Step speedup | "
        "MC/FPT archive count | MC/FPT clusters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = [
        row
        for row in rows
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    for row in selected:
        speedup = row["step_speedup_mc_over_fpt"]
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "NA"
        mc_step = row["mc_median_hit_step"]
        fpt_step = row["fpt_median_hit_step"]
        lines.append(
            f"| {row['system']} | {row['torsions']} | {row['threshold']:.3f} | "
            f"{row['mc_success_rate']:.0%} | {row['fpt_success_rate']:.0%} | "
            f"{mc_step if mc_step is not None else 'NA'} | "
            f"{fpt_step if fpt_step is not None else 'NA'} | "
            f"{speedup_text} | "
            f"{row['mc_median_archive_count']:.1f}/"
            f"{row['fpt_median_archive_count']:.1f} | "
            f"{row['mc_median_unique_clusters']:.1f}/"
            f"{row['fpt_median_unique_clusters']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=6300)
    parser.add_argument(
        "--steps",
        type=int,
        default=BASELINE_STEPS,
        help=f"sampling budget (multiple of 100, maximum {MAX_STEPS})",
    )
    parser.add_argument(
        "--interaction-backend",
        choices=("jax", "pallas"),
        default="pallas",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(ALL_SYSTEMS),
        help="optional explicit subset; default runs every cleaned system",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mc-temperature",
        type=float,
        default=BASELINE_TEMPERATURE,
        help="temperature (K) of the Metropolis acceptance only; FPT and "
        "thresholds are unchanged",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-system outputs after strict config checks",
    )
    args = parser.parse_args()
    if args.num_seeds <= 0:
        parser.error("num-seeds must be positive")
    if not 0 < args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [100, {MAX_STEPS}]")
    if args.steps % search.CHECKPOINT_EVERY:
        parser.error(
            "--steps must be a multiple of "
            f"{search.CHECKPOINT_EVERY}"
        )
    if args.systems is None:
        args.systems = tuple(ALL_SYSTEMS)
    return args


def _load_completed_system(name: str, args) -> dict | None:
    if not args.resume:
        return None
    system_output = args.output_dir / name
    aggregate_path = system_output / "aggregate.json"
    required = [
        aggregate_path,
        system_output / "seed_summary.csv",
        system_output / "trajectories.csv",
    ]
    required.extend(
        system_output / f"archives_seed{seed}.npz"
        for seed in range(args.seed_base, args.seed_base + args.num_seeds)
    )
    if not any(path.exists() for path in required):
        return None
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"incomplete resume output for {name}: missing {missing}"
        )
    aggregate = json.loads(aggregate_path.read_text())
    metadata = aggregate["metadata"]
    expected = {
        "system": name,
        "steps": args.steps,
        "replicas": BASELINE_REPLICAS,
        "interaction_backend": args.interaction_backend,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
    }
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"resume config mismatch for {name}: "
            f"expected {expected}, got {observed}"
        )
    if metadata.get("baseline") != {
        **frozen_baseline_metadata(),
        "steps": args.steps,
    }:
        # Per-system metadata is created by the shared runner, whose frozen
        # baseline records the algorithm's primary 500-step protocol. Budget
        # compatibility is already checked independently above.
        stored_baseline = metadata.get("baseline", {})
        expected_baseline = frozen_baseline_metadata()
        if stored_baseline != expected_baseline:
            raise RuntimeError(
                f"resume baseline mismatch for {name}: "
                f"expected {expected_baseline}, got {stored_baseline}"
            )
    print(f"system={name} resumed", flush=True)
    return aggregate


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = {}
    for name in args.systems:
        aggregate = _load_completed_system(name, args)
        if aggregate is None:
            aggregate = _run_system(name, ALL_SYSTEMS[name], args)
        aggregates[name] = aggregate
    rows = _suite_rows(aggregates)
    _write_csv(args.output_dir / "suite_summary.csv", rows)
    _write_markdown_summary(args.output_dir / "RESULTS.md", rows)
    baseline = frozen_baseline_metadata()
    baseline["steps"] = args.steps
    suite_summary = {
        "protocol": {
            "dataset": "cleaned PoseBusters journal 308-case subset",
            "selection_file": str(
                Path(__file__).with_name("paper_subset_308_ids.txt").name
            ),
            "manifest": Path(__file__).with_name("manifest.json").name,
            "systems": list(args.systems),
            "num_seeds": args.num_seeds,
            "seed_base": args.seed_base,
            "steps": args.steps,
            "replicas": BASELINE_REPLICAS,
            "interaction_backend": args.interaction_backend,
            "fpt_score": "row",
            "mc_temperature_kelvin": args.mc_temperature,
            "baseline": baseline,
        },
        "rows": rows,
    }
    (args.output_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2) + "\n"
    )
    print(json.dumps(suite_summary, indent=2))


if __name__ == "__main__":
    main()
