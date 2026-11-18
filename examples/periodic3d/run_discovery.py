#!/usr/bin/env python3
"""Discovery and redistribution on the 3D eight-well target.

For every initial seed, FPT, Langevin dynamics (LD), and the no-score control
start from the same draw of the Boltzmann law restricted to the initial well
and run for 13500 updates at the fixed target beta.  Every recorded state
stores the well counts and, per well, the number of distinct replicas that
have visited it (``distinct_visitors``), from which split-half identification
is evaluated.  With ``--equilibrium-audit-lrs`` the FPT update is also started
from iid target samples and compared with the 99th percentiles of iid
four-ensemble means (parameter-selection audit).

Paper protocols (SI Table of per-seed times; main-text diagnosis):

    # RMSprop transport protocol, Langevin, no-score, FPT equilibrium audit
    python run_discovery.py --output-dir DATA/periodic_n3_rms_main_bb_visits \
        --save-endpoints --equilibrium-audit-lrs 0.15
    # shared-bandwidth FPT and no-score by explicit Euler at the Langevin step
    python run_discovery.py --output-dir DATA/periodic_n3_shared_euler_bb_visits \
        --fpt-optimizer sgd --fpt-lr 0.008 --methods fpt noscore --save-endpoints
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.periodic3d.common import (
    AUDIT_SEEDS,
    BETA_0,
    MAIN_SEEDS,
    NUM_UPDATES,
    basin_boltzmann_initialization,
    build_fpt_sampler,
    energy_numpy,
    langevin_chunk,
    make_noscore_stepper,
    make_reference,
    reference_mean_bands,
    sample_metrics,
    sample_reference,
    sampler_snapshot,
    well_assignments,
)

# "ula" is the Langevin dynamics (LD) baseline: unadjusted Euler--Maruyama.
METHODS = ("fpt", "ula", "noscore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=MAIN_SEEDS)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--beta-scale", type=float, default=3.0,
                        help="target beta in units of beta_0; the barrier is beta_scale k_BT")
    parser.add_argument("--num-updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--fpt-lr", type=float, default=0.15)
    parser.add_argument("--fpt-optimizer", choices=("rmsprop", "sgd"), default="rmsprop")
    parser.add_argument("--fpt-kernel-min", type=float, default=2.0)
    parser.add_argument("--fpt-kernel-max", type=float, default=4.0)
    parser.add_argument("--fpt-update-pace", type=int, default=1)
    parser.add_argument("--ld-step", type=float, default=0.008)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--save-endpoints", action="store_true")
    parser.add_argument("--equilibrium-audit-lrs", type=float, nargs="*", default=[])
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=AUDIT_SEEDS)
    parser.add_argument("--reference-resolution", type=int, default=128)
    parser.add_argument("--reference-groups", type=int, default=400)
    return parser.parse_args()


def coverage_record(snapshot: np.ndarray) -> tuple[int, float, list[int]]:
    count = np.bincount(well_assignments(snapshot), minlength=8)
    probability = count[count > 0] / count.sum()
    entropy = float(-np.sum(probability * np.log(probability)) / np.log(8))
    return int(np.count_nonzero(count)), entropy, count.tolist()


def sampler_for(args, initial, *, beta, seed, learning_rate=None):
    return build_fpt_sampler(
        initial,
        beta=beta,
        seed=seed,
        learning_rate=args.fpt_lr if learning_rate is None else learning_rate,
        optimizer_name=args.fpt_optimizer,
        kernel_min=args.fpt_kernel_min,
        kernel_max=args.fpt_kernel_max,
        update_pace=args.fpt_update_pace,
    )


def run_localized(args, seed: int, beta_target: float) -> dict:
    """Fixed-beta runs of every method from one restricted-Boltzmann initial ensemble.

    Output keys carry the suffix ``_fixed`` (fixed target temperature).
    """
    initial = basin_boltzmann_initialization(seed, replicas=args.replicas, beta=beta_target)
    state: dict[str, object] = {}
    if "fpt" in args.methods:
        state["fpt_fixed"] = sampler_for(args, initial, beta=beta_target, seed=seed + 1)
    if "ula" in args.methods:
        state["ula_fixed"] = (jnp.asarray(initial), jax.random.PRNGKey(seed + 2))
    if "noscore" in args.methods:
        optimizer, noscore_chunk = make_noscore_stepper(args.fpt_lr, args.fpt_optimizer)
        coord = jnp.asarray(initial)
        state["noscore_fixed"] = (coord, optimizer.init(coord))

    initial_energy = float(beta_target * np.mean(energy_numpy(initial)))
    initial_wells = well_assignments(initial)
    history, visit_mask = {}, {}
    for name in state:
        covered, entropy, counts = coverage_record(initial)
        # Per-replica record of the wells visited at recorded states.
        visit_mask[name] = np.zeros((len(initial), 8), bool)
        visit_mask[name][np.arange(len(initial)), initial_wells] = True
        history[name] = {"checkpoint": [0], "covered": [covered], "entropy": [entropy], "counts": [counts],
                         "distinct_visitors": [visit_mask[name].sum(0).tolist()],
                         "wall_s": [0.0], "beta_target_mean_energy": [initial_energy]}
    elapsed = {name: 0.0 for name in state}
    last_snapshot: dict[str, np.ndarray] = {}
    total = 0
    while total < args.num_updates:
        steps = min(args.checkpoint_interval, args.num_updates - total)
        for name in state:
            method = name.split("_")[0]
            start = time.perf_counter()
            if method == "fpt":
                sampler = state[name]
                sampler.run(steps, verbose=False, continue_from_last_state=total > 0)
                snapshot = sampler_snapshot(sampler)
            elif method == "ula":
                coord, key = state[name]
                coord, key = langevin_chunk(coord, key, beta=beta_target, step_size=args.ld_step,
                                            steps=steps)
                state[name] = (coord, key)
                snapshot = np.asarray(coord)
            else:
                coord, opt_state = state[name]
                coord, opt_state = noscore_chunk(coord, opt_state, beta_target, steps)
                state[name] = (coord, opt_state)
                snapshot = np.asarray(coord)
            elapsed[name] += time.perf_counter() - start
            last_snapshot[name] = snapshot
            covered, entropy, counts = coverage_record(snapshot)
            visit_mask[name][np.arange(len(snapshot)), well_assignments(snapshot)] = True
            record = history[name]
            record["checkpoint"].append(total + steps)
            record["covered"].append(covered)
            record["counts"].append(counts)
            record["distinct_visitors"].append(visit_mask[name].sum(0).tolist())
            record["entropy"].append(entropy)
            record["wall_s"].append(elapsed[name])
            record["beta_target_mean_energy"].append(float(beta_target * np.mean(energy_numpy(snapshot))))
        total += steps
    for record in history.values():
        record["t98"] = next((c for c, h in zip(record["checkpoint"], record["entropy"]) if h >= 0.98), None)
    if args.save_endpoints:
        np.savez_compressed(args.output_dir / f"endpoints_seed{seed}.npz", initial=initial,
                            **{name: np.asarray(x) for name, x in last_snapshot.items()})
    return history


def run_fpt_equilibrium_audit(args, beta_target: float, reference, bands) -> dict:
    total_steps = args.num_updates
    results = {}
    for lr in args.equilibrium_audit_lrs:
        rows = []
        for seed in args.audit_seeds:
            initial = sample_reference(np.random.default_rng(int(seed) + 100_000), reference, args.replicas)
            sampler = sampler_for(args, initial, beta=beta_target, seed=int(seed) + 1, learning_rate=lr)
            sampler.run(total_steps, verbose=False)
            snapshot = sampler_snapshot(sampler)
            metrics = sample_metrics(snapshot, reference)
            rows.append({"seed": int(seed), "stable": bool(np.all(np.isfinite(snapshot))),
                         "energy_ks": metrics["energy_ks"], "basin_tv": metrics["basin_tv"],
                         "initial": sample_metrics(initial, reference)})
        ks = float(np.mean([r["energy_ks"] for r in rows]))
        tv = float(np.mean([r["basin_tv"] for r in rows]))
        admissible = (all(r["stable"] for r in rows) and ks <= bands["energy_ks_mean_99_upper"]
                      and tv <= bands["basin_tv_mean_99_upper"])
        results[f"lr={lr:g}"] = {"learning_rate": lr, "runs": rows, "mean_energy_ks": ks,
                                 "mean_basin_tv": tv, "admissible": admissible}
        print(f"[audit] lr={lr:g} KS={ks:.4f} TV={tv:.4f} admissible={admissible}", flush=True)
    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beta_target = args.beta_scale * BETA_0
    output = {"protocol": {
        "replicas": args.replicas, "seeds": list(args.seeds), "initial_law": "basin_boltzmann",
        "beta_target": beta_target, "beta_scale": args.beta_scale,
        "num_updates": args.num_updates, "checkpoint_interval": args.checkpoint_interval,
        "fpt_lr": args.fpt_lr, "fpt_optimizer": args.fpt_optimizer,
        "fpt_kernel_band": [args.fpt_kernel_min, args.fpt_kernel_max], "fpt_update_pace": args.fpt_update_pace,
        "ula_step": args.ld_step, "methods": args.methods, "seed_offsets": {"fpt": 1, "ula_noise": 2},
    }, "runs": {}}
    out_path = args.output_dir / "summary.json"
    for seed in args.seeds:
        start = time.perf_counter()
        history = run_localized(args, int(seed), beta_target)
        output["runs"][str(seed)] = history
        brief = {k: v["t98"] for k, v in history.items()}
        print(f"[seed {seed}] t98 {brief}  {time.perf_counter() - start:.0f}s", flush=True)
        out_path.write_text(json.dumps(output) + "\n")
    if args.equilibrium_audit_lrs:
        reference = make_reference(beta_target, args.reference_resolution)
        bands = reference_mean_bands(reference=reference, replicas=args.replicas,
                                     seed_count=len(args.audit_seeds), group_count=args.reference_groups)
        output["iid_reference"] = bands
        output["fpt_equilibrium_audit"] = run_fpt_equilibrium_audit(args, beta_target, reference, bands)
        out_path.write_text(json.dumps(output) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
