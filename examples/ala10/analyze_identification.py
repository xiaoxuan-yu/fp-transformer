#!/usr/bin/env python3
"""Prior-free H0--H3 identification for alanine decapeptide (main text Fig. 5).

No helix filter is applied to the method output. Each half of an audit is
clustered blindly with HDBSCAN in the cosine--sine backbone-angle feature
space, with the minimum cluster size set to a fixed fraction of the half, so a
family must carry that share of the output mass to form a cluster. The
canonical atlas is used only afterwards, to name a cluster H_k when most of its
members carry atlas label k. A family is identified when both halves contain a
cluster named after it.

Units follow the 3D/100D protocol: FPT halves are disjoint sets of particles of
the current ensemble, resampled by weight; LD halves are disjoint sets of
trajectories, sampled uniformly from all their recorded frames up to the
horizon (the time-averaged trajectory history).

    python -m examples.ala10.analyze_identification --fpt-root data/ala10/fpt \
        --ld-root data/ala10/ld_controls --output-dir data/ala10/identification
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from examples.ala10.lib.families import (
    FAMILY_COUNT,
    classify_strict,
    fit_canonical_model,
    torus_embedding,
)

DATA = Path(__file__).resolve().parent / "data"
SYSTEM = DATA / "system.npz"
CANONICAL = DATA / "canonical_mc_samples.npz"
SEEDS = tuple(range(20260812, 20260820))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("reference", "fpt", "ld"),
                        default=["reference", "fpt", "ld"])
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(0, 20_001, 1_000)))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--draws", type=int, default=4000, help="configurations per half")
    parser.add_argument("--cluster-fraction", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--selection", choices=("leaf", "eom"), default="leaf")
    parser.add_argument("--fpt-root", type=Path, required=True,
                        help="directory holding band30_50_m128_seed<seed>/metric_snapshots.npz")
    parser.add_argument("--ld-root", type=Path, required=True,
                        help="directory holding seed<seed>/ld_m128/metric_snapshots.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class Identifier:
    def __init__(self, args: argparse.Namespace):
        self.names = np.asarray(np.load(SYSTEM)["variable_names"]).astype(str)
        self.canonical_angles = np.asarray(np.load(CANONICAL)["angles"], np.float32)
        self.canonical = fit_canonical_model(self.canonical_angles, self.names)
        self.fraction = args.cluster_fraction
        self.min_samples = args.min_samples
        self.selection = args.selection

    def __call__(self, angles: np.ndarray) -> tuple[np.ndarray, int]:
        features = torus_embedding(angles[:, self.canonical.backbone]).astype(np.float64)
        strict, family = classify_strict(angles, self.names, self.canonical)
        atlas = np.full(len(angles), -1)
        atlas[strict] = family
        labels = HDBSCAN(
            min_cluster_size=max(2, round(self.fraction * len(angles))),
            min_samples=self.min_samples,
            cluster_selection_method=self.selection,
        ).fit(features).labels_
        found = np.zeros(FAMILY_COUNT, bool)
        clusters = 0
        for cluster in set(labels.tolist()) - {-1}:
            clusters += 1
            members = atlas[labels == cluster]
            share = np.bincount(members[members >= 0], minlength=FAMILY_COUNT) / len(members)
            if share.max() > 0.5:
                found[int(share.argmax())] = True
        return found, clusters


def rows_for(source, seed, step, repeat, left, right, clusters):
    return [{"source": source, "seed": seed, "step": step, "repeat": repeat, "family": k,
             "left": int(left[k]), "right": int(right[k]), "identified": int(left[k] and right[k]),
             "clusters_left": clusters[0], "clusters_right": clusters[1]}
            for k in range(FAMILY_COUNT)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identify = Identifier(args)
    rows: list[dict] = []

    if "reference" in args.methods:
        rng = np.random.default_rng(2026110301)
        reference = identify.canonical_angles
        half = len(reference) // 2
        for repeat in range(args.repeats):
            order = rng.permutation(len(reference))
            halves = [reference[order[i * half:(i + 1) * half]] for i in range(2)]
            draws = [h[rng.integers(0, len(h), args.draws)] for h in halves]
            (left, cl), (right, cr) = identify(draws[0]), identify(draws[1])
            rows += rows_for("reference", "", 20_000, repeat, left, right, (cl, cr))
        print("reference", np.mean([r["identified"] for r in rows]) * FAMILY_COUNT, flush=True)

    for seed in args.seeds:
        for method in [m for m in ("fpt", "ld") if m in args.methods]:
            if method == "ld":
                with np.load(args.ld_root / f"seed{seed}" / "ld_m128"
                             / "metric_snapshots.npz") as data:
                    frames = np.asarray(data["coordinates"], np.float32)
                    steps = np.asarray(data["step"], np.int64)
            else:
                with np.load(args.fpt_root / f"band30_50_m128_seed{seed}" / "metric_snapshots.npz") as data:
                    frames = np.asarray(data["coordinates"], np.float32)[:, :, 0, :]
                    weights = np.asarray(data["particle_weights"], np.float64)
                    steps = np.asarray(data["step"], np.int64)
                with np.load(args.ld_root / f"seed{seed}" / "ld_m128"
                             / "metric_snapshots.npz") as data:
                    start = np.asarray(data["coordinates"][0], np.float32)
                frames = np.concatenate((start[None], frames))
                weights = np.concatenate((np.full((1, frames.shape[1]), 1.0), weights))
                steps = np.concatenate(([0], steps))
            lookup = {int(s): i for i, s in enumerate(steps)}
            units = frames.shape[1]
            for step in args.steps:
                index = lookup[step]
                rng = np.random.default_rng((seed, step, 0 if method == "fpt" else 1))
                hits = []
                for repeat in range(args.repeats):
                    order = rng.permutation(units)
                    samples = []
                    for members in (order[:units // 2], order[units // 2:]):
                        if method == "ld":
                            times = rng.integers(0, index + 1, args.draws)
                            chosen = members[rng.integers(0, len(members), args.draws)]
                            samples.append(frames[times, chosen])
                        else:
                            local = weights[index, members]
                            chosen = members[rng.choice(len(members), args.draws, replace=True,
                                                        p=local / local.sum())]
                            samples.append(frames[index, chosen])
                    (left, cl), (right, cr) = identify(samples[0]), identify(samples[1])
                    rows += rows_for(method, seed, step, repeat, left, right, (cl, cr))
                    hits.append(left & right)
                print(method, seed, step, np.round(np.mean(hits, 0), 2), flush=True)

    path = args.output_dir / f"audits_{'_'.join(args.methods)}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"protocol_{'_'.join(args.methods)}.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   indent=2) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
