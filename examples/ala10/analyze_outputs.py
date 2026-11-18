#!/usr/bin/env python3
"""Strict-helix mass and canonical-PCA projections of the Ala10 outputs (main text Fig. 5).

The canonical Monte Carlo sample defines the strict-helical manifold
(helicity >= 0.70) and the canonical PCA of the backbone cosine--sine
features.  For every seed and recorded horizon this script records the
strict-helix mass of the FPT ensemble (by particle weight) and of the LD
ensemble, and projects the strict-helical FPT endpoint and LD trajectory
frames onto the canonical PCA.  Family identification is computed by
``analyze_identification.py``.

    python -m examples.ala10.analyze_outputs --fpt-root data/ala10/fpt \
        --ld-root data/ala10/ld_controls --output-dir data/ala10/recovery
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from examples.ala10.lib.families import (
    PRIMARY_HELIX_THRESHOLD,
    classify_strict,
    fit_canonical_model,
    helicity,
    normalize_weights,
    torus_embedding,
)

DATA = Path(__file__).resolve().parent / "data"
SYSTEM = DATA / "system.npz"
CANONICAL = DATA / "canonical_mc_samples.npz"
SEEDS = tuple(range(20260812, 20260820))
DEFAULT_STEPS = tuple(range(0, 20_001, 1_000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("m128", "m256"), default="m128")
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--ld-root",
        type=Path,
        required=True,
        help="directory holding seed<seed>/ld_<variant>/metric_snapshots.npz",
    )
    parser.add_argument(
        "--fpt-root",
        type=Path,
        required=True,
        help="directory holding band30_50_<variant>_seed<seed>/metric_snapshots.npz",
    )
    return parser.parse_args()


def flatten_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 3 and coordinates.shape[1] == 1:
        coordinates = coordinates[:, 0, :]
    return coordinates


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fpt_series(
    variant: str,
    seed: int,
    initial_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(fpt_path(variant, seed), allow_pickle=False) as data:
        coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        weights = np.asarray(data["particle_weights"], dtype=np.float64)
        steps = np.asarray(data["step"], dtype=np.int64)
    coordinates = coordinates[:, :, 0, :]
    particle_count = coordinates.shape[1]
    coordinates = np.concatenate(
        (initial_coordinates[None, :, :], coordinates), axis=0
    )
    weights = np.concatenate(
        (
            np.full((1, particle_count), 1.0 / particle_count),
            weights,
        ),
        axis=0,
    )
    steps = np.concatenate((np.asarray([0], dtype=np.int64), steps))
    return coordinates, weights, steps


def load_ld_series(variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ld_path(variant, seed), allow_pickle=False) as data:
        return (
            np.asarray(data["coordinates"], dtype=np.float32),
            np.asarray(data["step"], dtype=np.int64),
        )


def selected_indices(all_steps: np.ndarray, requested_steps: tuple[int, ...]) -> list[int]:
    lookup = {int(step): index for index, step in enumerate(all_steps)}
    missing = [step for step in requested_steps if step not in lookup]
    if missing:
        raise ValueError(f"requested steps absent from snapshots: {missing}")
    return [lookup[step] for step in requested_steps]


def strict_mass(
    coordinates: np.ndarray,
    weights: np.ndarray,
    names: np.ndarray,
) -> float:
    mask = helicity(flatten_coordinates(coordinates), names) >= PRIMARY_HELIX_THRESHOLD
    return float(np.sum(normalize_weights(weights)[mask]))


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    global fpt_path, ld_path
    args = parse_args()
    ld_root, fpt_root = args.ld_root, args.fpt_root

    def ld_path(variant: str, seed: int) -> Path:
        return ld_root / f"seed{seed}" / f"ld_{variant}" / "metric_snapshots.npz"

    def fpt_path(variant: str, seed: int) -> Path:
        return fpt_root / f"band30_50_{variant}_seed{seed}" / "metric_snapshots.npz"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_steps = tuple(sorted(set(args.steps)))
    if requested_steps[0] != 0 or requested_steps[-1] != 20_000:
        raise ValueError("the figure requires steps 0 and 20000")

    with np.load(SYSTEM, allow_pickle=False) as data:
        names = np.asarray(data["variable_names"]).astype(str)
    with np.load(CANONICAL, allow_pickle=False) as data:
        canonical_angles = np.asarray(data["angles"], dtype=np.float32)
    canonical = fit_canonical_model(canonical_angles, names)

    mass_rows: list[dict] = []
    projection_lists: dict[str, list[np.ndarray]] = {
        "fpt_features": [],
        "fpt_labels": [],
        "fpt_weights": [],
        "fpt_seed": [],
        "ld_features": [],
        "ld_labels": [],
        "ld_weights": [],
        "ld_seed": [],
    }

    for seed in args.seeds:
        ld_coordinates, ld_steps = load_ld_series(args.variant, seed)
        fpt_coordinates, fpt_weights, fpt_steps = load_fpt_series(
            args.variant, seed, ld_coordinates[0]
        )
        fpt_indices = selected_indices(fpt_steps, requested_steps)
        ld_indices = selected_indices(ld_steps, requested_steps)
        replicas = ld_coordinates.shape[1]

        for step, fpt_index, ld_index in zip(
            requested_steps, fpt_indices, ld_indices, strict=True
        ):
            fpt_frame = fpt_coordinates[fpt_index]
            fpt_weight = normalize_weights(fpt_weights[fpt_index])
            ld_frame = ld_coordinates[ld_index]
            mass_rows.extend(
                (
                    {
                        "source": "fpt",
                        "seed": seed,
                        "step": step,
                        "strict_helix_mass": strict_mass(
                            fpt_frame, fpt_weight, names
                        ),
                    },
                    {
                        "source": "ld",
                        "seed": seed,
                        "step": step,
                        "strict_helix_mass": strict_mass(
                            ld_frame,
                            np.full(replicas, 1.0 / replicas),
                            names,
                        ),
                    },
                )
            )

        endpoint = fpt_coordinates[-1]
        endpoint_weight = normalize_weights(fpt_weights[-1])
        endpoint_strict, endpoint_labels = classify_strict(
            endpoint, names, canonical
        )
        projection_lists["fpt_features"].append(
            torus_embedding(endpoint[endpoint_strict][:, canonical.backbone])
        )
        projection_lists["fpt_labels"].append(endpoint_labels)
        projection_lists["fpt_weights"].append(
            endpoint_weight[endpoint_strict] / len(args.seeds)
        )
        projection_lists["fpt_seed"].append(
            np.full(np.count_nonzero(endpoint_strict), seed, dtype=np.int64)
        )

        ld_flat = ld_coordinates.reshape(-1, ld_coordinates.shape[-1])
        ld_strict, ld_labels = classify_strict(ld_flat, names, canonical)
        projection_lists["ld_features"].append(
            torus_embedding(ld_flat[ld_strict][:, canonical.backbone])
        )
        projection_lists["ld_labels"].append(ld_labels)
        projection_lists["ld_weights"].append(
            np.full(
                np.count_nonzero(ld_strict),
                1.0 / (len(args.seeds) * len(ld_flat)),
            )
        )
        projection_lists["ld_seed"].append(
            np.full(np.count_nonzero(ld_strict), seed, dtype=np.int64)
        )

    write_csv(mass_rows, args.output_dir / "strict_helix_mass.csv")

    projection = {
        "canonical_xy": canonical.pca.transform(
            canonical.strict_features
        ).astype(np.float32),
        "canonical_label": canonical.strict_labels.astype(np.int8),
        "canonical_medoid_xy": canonical.pca.transform(
            canonical.medoids
        ).astype(np.float32),
        "canonical_population": canonical.populations,
        "explained_variance_ratio": canonical.pca.explained_variance_ratio_,
    }
    for method in ("fpt", "ld"):
        features = np.concatenate(projection_lists[f"{method}_features"])
        projection[f"{method}_xy"] = canonical.pca.transform(features).astype(
            np.float32
        )
        projection[f"{method}_label"] = np.concatenate(
            projection_lists[f"{method}_labels"]
        ).astype(np.int8)
        projection[f"{method}_weight"] = np.concatenate(
            projection_lists[f"{method}_weights"]
        )
        projection[f"{method}_seed"] = np.concatenate(
            projection_lists[f"{method}_seed"]
        )
    np.savez_compressed(args.output_dir / "projection.npz", **projection)

    summary = finite_json(
        {
            "analysis": "strict-helix mass and canonical-PCA projections of the outputs",
            "variant": args.variant,
            "fpt_root": str(args.fpt_root),
            "seeds": list(args.seeds),
            "ld_root": str(args.ld_root),
            "steps": list(requested_steps),
            "strict_helix_threshold": PRIMARY_HELIX_THRESHOLD,
            "canonical": {
                "total_frames": len(canonical_angles),
                "strict_mass": len(canonical.strict_features)
                / len(canonical_angles),
                "conditional_family_populations": canonical.populations.tolist(),
                "pca_explained_variance_ratio": canonical.pca.explained_variance_ratio_.tolist(),
            },
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "mass_rows": len(mass_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
