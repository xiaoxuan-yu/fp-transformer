"""Paired system-level statistics for the PoseBusters MC/FPT benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


SCORES = ("sampling_energy", "binding_score")
LEVELS = ("easy", "medium", "strict")
TORSION_BINS = ("1-2", "3-5", "6-7", "8-12", "13+")
BOOTSTRAP_SEED = 6300
BOOTSTRAP_SAMPLES = 20_000
ENERGY_TIE_TOLERANCE = 1e-3


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _percentile_ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def _exact_two_sided_sign_p(positive: int, negative: int) -> float:
    """Two-sided exact binomial/sign p-value after discarding ties."""
    trials = positive + negative
    if trials == 0:
        return 1.0
    smaller = min(positive, negative)
    tail = sum(math.comb(trials, value) for value in range(smaller + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def _paired_sign_counts(values: np.ndarray, tolerance: float) -> dict[str, int]:
    return {
        "fpt_better": int(np.sum(values > tolerance)),
        "mc_better": int(np.sum(values < -tolerance)),
        "tie": int(np.sum(np.abs(values) <= tolerance)),
    }


def _torsion_bin(torsions: int) -> str:
    if torsions <= 2:
        return "1-2"
    if torsions <= 5:
        return "3-5"
    if torsions <= 7:
        return "6-7"
    if torsions <= 12:
        return "8-12"
    return "13+"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class SystemResult:
    name: str
    torsions: int
    mc_elapsed: float
    fpt_elapsed: float
    mc_best_total: float
    fpt_best_total: float
    mc_best_binding: float
    fpt_best_binding: float
    strict_total_threshold: float
    strict_binding_threshold: float
    mc_seed_rows: tuple[dict[str, str], ...]
    fpt_seed_rows: tuple[dict[str, str], ...]


def _endpoint_prefix(seed_row: dict[str, str], score: str, level: str) -> str:
    keys = [
        key
        for key in seed_row
        if key.startswith(f"{score}_le_") and key.endswith("_first_hit_step")
    ]
    if len(keys) != len(LEVELS):
        raise RuntimeError(
            f"expected {len(LEVELS)} {score} first-hit columns, got {keys}"
        )
    return keys[LEVELS.index(level)].removesuffix("_first_hit_step")


def _system_endpoint_arrays(
    system: SystemResult, score: str, level: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paired_rows = zip(system.mc_seed_rows, system.fpt_seed_rows)
    mc_success, fpt_success = [], []
    mc_steps, fpt_steps = [], []
    mc_capped, fpt_capped = [], []
    for mc_row, fpt_row in paired_rows:
        if mc_row["seed"] != fpt_row["seed"]:
            raise RuntimeError(f"{system.name}: unpaired seed rows")
        mc_prefix = _endpoint_prefix(mc_row, score, level)
        fpt_prefix = _endpoint_prefix(fpt_row, score, level)
        if mc_prefix != fpt_prefix:
            raise RuntimeError(f"{system.name}: mismatched threshold columns")
        prefix = mc_prefix
        values = {}
        for method, row in (("mc", mc_row), ("fpt", fpt_row)):
            step = _optional_float(row[f"{prefix}_first_hit_step"])
            if step is not None and step < 0:
                step = None
            hit_seconds = _optional_float(
                row[f"{prefix}_first_hit_seconds_estimate"]
            )
            success = step is not None and hit_seconds is not None
            if (step is None) != (hit_seconds is None):
                raise RuntimeError(
                    f"{system.name} seed {row['seed']}: inconsistent hit data"
                )
            values[method] = (
                success,
                np.nan if step is None else step,
                hit_seconds if success else float(row["elapsed_seconds"]),
            )
        mc_success.append(values["mc"][0])
        fpt_success.append(values["fpt"][0])
        mc_steps.append(values["mc"][1])
        fpt_steps.append(values["fpt"][1])
        mc_capped.append(values["mc"][2])
        fpt_capped.append(values["fpt"][2])
    return tuple(
        np.asarray(values, dtype=dtype)
        for values, dtype in (
            (mc_success, bool),
            (fpt_success, bool),
            (mc_steps, float),
            (fpt_steps, float),
            (mc_capped, float),
            (fpt_capped, float),
        )
    )


def _paired_seed_improvement(system: SystemResult, metric: str) -> float:
    mc = np.asarray([float(row[metric]) for row in system.mc_seed_rows])
    fpt = np.asarray([float(row[metric]) for row in system.fpt_seed_rows])
    return float(np.median(mc - fpt))


def _load(output_dir: Path) -> tuple[dict, dict[str, SystemResult]]:
    suite = json.loads((output_dir / "suite_summary.json").read_text())
    protocol = suite["protocol"]
    systems = {}
    strict_rows = {
        row["system"]: row
        for row in suite["rows"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    }
    strict_binding_rows = {
        row["system"]: row
        for row in suite["rows"]
        if row["score"] == "binding_score"
        and row["threshold_level"] == "strict"
    }
    for name in protocol["systems"]:
        seed_rows = _read_csv(output_dir / name / "seed_summary.csv")
        methods = {
            method: tuple(
                sorted(
                    (row for row in seed_rows if row["method"] == method),
                    key=lambda row: int(row["seed"]),
                )
            )
            for method in ("mc", "fpt")
        }
        if any(len(rows) != protocol["num_seeds"] for rows in methods.values()):
            raise RuntimeError(f"{name}: incomplete paired seed summaries")
        if [row["seed"] for row in methods["mc"]] != [
            row["seed"] for row in methods["fpt"]
        ]:
            raise RuntimeError(f"{name}: MC/FPT seeds do not match")
        systems[name] = SystemResult(
            name=name,
            torsions=int(strict_rows[name]["torsions"]),
            mc_elapsed=float(median(float(row["elapsed_seconds"]) for row in methods["mc"])),
            fpt_elapsed=float(median(float(row["elapsed_seconds"]) for row in methods["fpt"])),
            mc_best_total=float(median(float(row["best_total"]) for row in methods["mc"])),
            fpt_best_total=float(median(float(row["best_total"]) for row in methods["fpt"])),
            mc_best_binding=float(median(float(row["best_binding"]) for row in methods["mc"])),
            fpt_best_binding=float(median(float(row["best_binding"]) for row in methods["fpt"])),
            strict_total_threshold=float(strict_rows[name]["threshold"]),
            strict_binding_threshold=float(strict_binding_rows[name]["threshold"]),
            mc_seed_rows=methods["mc"],
            fpt_seed_rows=methods["fpt"],
        )
    return suite, systems


def _endpoint_statistics(
    suite: dict,
    systems: dict[str, SystemResult],
    bootstrap_indices: np.ndarray,
) -> list[dict]:
    results = []
    for score in SCORES:
        for level in LEVELS:
            rows = [
                row
                for row in suite["rows"]
                if row["score"] == score and row["threshold_level"] == level
            ]
            system_arrays = [
                _system_endpoint_arrays(systems[row["system"]], score, level)
                for row in rows
            ]
            mc_trial_success = np.stack([values[0] for values in system_arrays])
            fpt_trial_success = np.stack([values[1] for values in system_arrays])
            mc_rates = np.mean(mc_trial_success, axis=1)
            fpt_rates = np.mean(fpt_trial_success, axis=1)
            paired_delta = fpt_rates - mc_rates
            rate_counts = _paired_sign_counts(paired_delta, tolerance=1e-12)
            mc_capped_trials = np.stack([values[4] for values in system_arrays])
            fpt_capped_trials = np.stack([values[5] for values in system_arrays])
            mc_capped = np.mean(mc_capped_trials, axis=1)
            fpt_capped = np.mean(fpt_capped_trials, axis=1)
            bootstrap_coverage = 100 * np.mean(paired_delta[bootstrap_indices], axis=1)
            bootstrap_speedup = np.sum(mc_capped[bootstrap_indices], axis=1) / np.sum(
                fpt_capped[bootstrap_indices], axis=1
            )
            time_difference = fpt_capped - mc_capped
            time_counts = _paired_sign_counts(-time_difference, tolerance=1e-12)
            results.append(
                {
                    "score": score,
                    "threshold_level": level,
                    "systems": len(rows),
                    "trials": int(mc_trial_success.size),
                    "mc_successes": int(np.sum(mc_trial_success)),
                    "fpt_successes": int(np.sum(fpt_trial_success)),
                    "coverage_difference_percentage_points": float(
                        100 * np.mean(paired_delta)
                    ),
                    "coverage_difference_bootstrap_ci_low": _percentile_ci(
                        bootstrap_coverage
                    )[0],
                    "coverage_difference_bootstrap_ci_high": _percentile_ci(
                        bootstrap_coverage
                    )[1],
                    "mc_better_success_rate_systems": rate_counts["mc_better"],
                    "fpt_better_success_rate_systems": rate_counts["fpt_better"],
                    "equal_success_rate_systems": rate_counts["tie"],
                    "paired_system_success_rate_sign_exact_p": _exact_two_sided_sign_p(
                        rate_counts["fpt_better"], rate_counts["mc_better"]
                    ),
                    # Backward-compatible aliases used by the existing tables.
                    "mc_only": rate_counts["mc_better"],
                    "fpt_only": rate_counts["fpt_better"],
                    "mcnemar_exact_p": _exact_two_sided_sign_p(
                        rate_counts["fpt_better"], rate_counts["mc_better"]
                    ),
                    "aggregate_capped_time_speedup_mc_over_fpt": float(
                        np.sum(mc_capped) / np.sum(fpt_capped)
                    ),
                    "aggregate_capped_time_speedup_bootstrap_ci_low": _percentile_ci(
                        bootstrap_speedup
                    )[0],
                    "aggregate_capped_time_speedup_bootstrap_ci_high": _percentile_ci(
                        bootstrap_speedup
                    )[1],
                    "per_system_time_fpt_better": time_counts["fpt_better"],
                    "per_system_time_mc_better": time_counts["mc_better"],
                    "per_system_time_tie": time_counts["tie"],
                    "per_system_time_sign_exact_p": _exact_two_sided_sign_p(
                        time_counts["fpt_better"], time_counts["mc_better"]
                    ),
                }
            )
    return results


def _torsion_statistics(
    suite: dict,
    systems: dict[str, SystemResult],
    rng: np.random.Generator,
) -> list[dict]:
    strict = [
        row
        for row in suite["rows"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    results = []
    for label in TORSION_BINS:
        rows = [row for row in strict if _torsion_bin(int(row["torsions"])) == label]
        energy_improvement = np.array(
            [
                _paired_seed_improvement(systems[row["system"]], "best_total")
                for row in rows
            ]
        )
        sign_counts = _paired_sign_counts(
            energy_improvement, tolerance=ENERGY_TIE_TOLERANCE
        )
        indices = rng.integers(0, len(rows), size=(BOOTSTRAP_SAMPLES, len(rows)))
        energy_bootstrap = np.median(energy_improvement[indices], axis=1)

        step_ratios = []
        mc_capped = []
        fpt_capped = []
        mc_successes = 0
        fpt_successes = 0
        mc_only = 0
        fpt_only = 0
        for row in rows:
            system = systems[row["system"]]
            (
                mc_ok,
                fpt_ok,
                mc_steps,
                fpt_steps,
                mc_trial_capped,
                fpt_trial_capped,
            ) = _system_endpoint_arrays(system, "sampling_energy", "strict")
            mc_successes += int(np.sum(mc_ok))
            fpt_successes += int(np.sum(fpt_ok))
            mc_only += int(np.sum(mc_ok & ~fpt_ok))
            fpt_only += int(np.sum(fpt_ok & ~mc_ok))
            comparable = mc_ok & fpt_ok & (mc_steps > 0) & (fpt_steps > 0)
            if np.any(comparable):
                step_ratios.append(
                    float(np.median(mc_steps[comparable] / fpt_steps[comparable]))
                )
            mc_capped.append(float(np.mean(mc_trial_capped)))
            fpt_capped.append(float(np.mean(fpt_trial_capped)))
        step_ratios = np.array(step_ratios)
        mc_capped = np.array(mc_capped)
        fpt_capped = np.array(fpt_capped)
        time_bootstrap = np.sum(mc_capped[indices], axis=1) / np.sum(
            fpt_capped[indices], axis=1
        )
        step_indices = rng.integers(
            0,
            len(step_ratios),
            size=(BOOTSTRAP_SAMPLES, len(step_ratios)),
        )
        step_bootstrap = np.median(step_ratios[step_indices], axis=1)
        results.append(
            {
                "torsion_bin": label,
                "systems": len(rows),
                "trials": len(rows) * int(suite["protocol"]["num_seeds"]),
                "mc_successes": mc_successes,
                "fpt_successes": fpt_successes,
                "fpt_only": fpt_only,
                "mc_only": mc_only,
                "median_final_energy_improvement": float(median(energy_improvement)),
                "median_final_energy_improvement_bootstrap_ci_low": _percentile_ci(
                    energy_bootstrap
                )[0],
                "median_final_energy_improvement_bootstrap_ci_high": _percentile_ci(
                    energy_bootstrap
                )[1],
                "energy_fpt_better": sign_counts["fpt_better"],
                "energy_mc_better": sign_counts["mc_better"],
                "energy_tie_within_0p001": sign_counts["tie"],
                "energy_sign_exact_p": _exact_two_sided_sign_p(
                    sign_counts["fpt_better"], sign_counts["mc_better"]
                ),
                "median_step_speedup_mc_over_fpt": float(median(step_ratios)),
                "median_step_speedup_bootstrap_ci_low": _percentile_ci(step_bootstrap)[0],
                "median_step_speedup_bootstrap_ci_high": _percentile_ci(step_bootstrap)[1],
                "aggregate_capped_time_speedup_mc_over_fpt": float(
                    np.sum(mc_capped) / np.sum(fpt_capped)
                ),
                "aggregate_capped_time_speedup_bootstrap_ci_low": _percentile_ci(
                    time_bootstrap
                )[0],
                "aggregate_capped_time_speedup_bootstrap_ci_high": _percentile_ci(
                    time_bootstrap
                )[1],
            }
        )
    return results


def _quality_statistics(
    suite: dict,
    systems: dict[str, SystemResult],
    bootstrap_indices: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    ordered = [systems[name] for name in suite["protocol"]["systems"]]
    total_improvement = np.array(
        [_paired_seed_improvement(system, "best_total") for system in ordered]
    )
    binding_improvement = np.array(
        [_paired_seed_improvement(system, "best_binding") for system in ordered]
    )
    total_counts = _paired_sign_counts(total_improvement, ENERGY_TIE_TOLERANCE)
    binding_counts = _paired_sign_counts(binding_improvement, ENERGY_TIE_TOLERANCE)
    strict_rows = [
        row
        for row in suite["rows"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    joint = [
        row
        for row in strict_rows
        if float(row["mc_success_rate"]) >= 0.5
        and float(row["fpt_success_rate"]) >= 0.5
    ]
    mc_clusters = np.array([row["mc_median_unique_clusters"] for row in joint])
    fpt_clusters = np.array([row["fpt_median_unique_clusters"] for row in joint])
    cluster_difference = fpt_clusters - mc_clusters
    cluster_counts = _paired_sign_counts(cluster_difference, tolerance=0)
    cluster_indices = rng.integers(
        0, len(joint), size=(BOOTSTRAP_SAMPLES, len(joint))
    )
    return {
        "energy_tie_tolerance_kcal_per_mol": ENERGY_TIE_TOLERANCE,
        "total_energy": {
            "median_improvement_mc_minus_fpt": float(median(total_improvement)),
            "median_improvement_bootstrap_ci": _percentile_ci(
                np.median(total_improvement[bootstrap_indices], axis=1)
            ),
            **total_counts,
            "sign_exact_p": _exact_two_sided_sign_p(
                total_counts["fpt_better"], total_counts["mc_better"]
            ),
        },
        "binding_score": {
            "median_improvement_mc_minus_fpt": float(median(binding_improvement)),
            "median_improvement_bootstrap_ci": _percentile_ci(
                np.median(binding_improvement[bootstrap_indices], axis=1)
            ),
            **binding_counts,
            "sign_exact_p": _exact_two_sided_sign_p(
                binding_counts["fpt_better"], binding_counts["mc_better"]
            ),
        },
        "strict_joint_hit_cluster_diversity": {
            "systems": len(joint),
            "mc_median_unique_clusters": float(median(mc_clusters)),
            "fpt_median_unique_clusters": float(median(fpt_clusters)),
            "median_paired_difference_fpt_minus_mc": float(median(cluster_difference)),
            "median_paired_difference_bootstrap_ci": _percentile_ci(
                np.median(cluster_difference[cluster_indices], axis=1)
            ),
            **cluster_counts,
            "sign_exact_p": _exact_two_sided_sign_p(
                cluster_counts["fpt_better"], cluster_counts["mc_better"]
            ),
        },
    }


def _neither_hit_statistics(
    suite: dict, systems: dict[str, SystemResult]
) -> list[dict]:
    strict_rows = [
        row
        for row in suite["rows"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
        and float(row["mc_success_rate"]) < 0.5
        and float(row["fpt_success_rate"]) < 0.5
    ]
    results = []
    for row in strict_rows:
        system = systems[row["system"]]
        mc_total_gap = system.mc_best_total - system.strict_total_threshold
        fpt_total_gap = system.fpt_best_total - system.strict_total_threshold
        mc_binding_gap = system.mc_best_binding - system.strict_binding_threshold
        fpt_binding_gap = system.fpt_best_binding - system.strict_binding_threshold
        results.append(
            {
                "system": system.name,
                "torsions": system.torsions,
                "mc_total_gap": mc_total_gap,
                "fpt_total_gap": fpt_total_gap,
                "total_improvement_mc_minus_fpt": mc_total_gap - fpt_total_gap,
                "mc_binding_gap": mc_binding_gap,
                "fpt_binding_gap": fpt_binding_gap,
                "binding_improvement_mc_minus_fpt": mc_binding_gap
                - fpt_binding_gap,
            }
        )
    return results


def _format_ci(low: float, high: float, digits: int = 2) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _write_markdown(summary: dict, output_path: Path) -> None:
    strict = next(
        row
        for row in summary["endpoints"]
        if row["score"] == "sampling_energy" and row["threshold_level"] == "strict"
    )
    quality = summary["quality"]
    systems_count = len(summary["protocol"]["systems"])
    trials_count = systems_count * int(summary["protocol"]["num_seeds"])
    lines = [
        "# PoseBusters paired statistical analysis",
        "",
        "All confidence intervals are 95% paired percentile bootstrap intervals "
        f"over the {systems_count} systems (20,000 resamples). Each system-level "
        f"effect summarizes {summary['protocol']['num_seeds']} paired random seeds; "
        "exact sign tests also treat systems as paired observations.",
        "",
        "## Primary strict sampling-energy endpoint",
        "",
        f"- Coverage: MC {strict['mc_successes']}/{trials_count}; FPT "
        f"{strict['fpt_successes']}/{trials_count}; paired system-mean difference "
        f"{strict['coverage_difference_percentage_points']:.2f} percentage points, "
        f"95% CI {_format_ci(strict['coverage_difference_bootstrap_ci_low'], strict['coverage_difference_bootstrap_ci_high'])}.",
        f"- Systems with higher success rate: FPT {strict['fpt_only']}, MC "
        f"{strict['mc_only']}; exact paired sign p={strict['mcnemar_exact_p']:.4g}.",
        f"- Aggregate timeout-capped speedup MC/FPT: "
        f"{strict['aggregate_capped_time_speedup_mc_over_fpt']:.2f}x, 95% CI "
        f"{_format_ci(strict['aggregate_capped_time_speedup_bootstrap_ci_low'], strict['aggregate_capped_time_speedup_bootstrap_ci_high'])}.",
        f"- Per-system capped-time wins: FPT {strict['per_system_time_fpt_better']}, "
        f"MC {strict['per_system_time_mc_better']}, ties "
        f"{strict['per_system_time_tie']}; exact sign p="
        f"{strict['per_system_time_sign_exact_p']:.3g}, not significant. "
        "The larger aggregate speedup reflects effect magnitude: MC-favoring systems are often cheap, "
        "whereas FPT reduces hard-tail failures.",
        "",
        "## Robustness across all endpoints",
        "",
        "| Endpoint | Coverage gain [95% CI], pp | Paired sign p | Capped-time speedup [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["endpoints"]:
        score_label = "Total" if row["score"] == "sampling_energy" else "Binding"
        lines.append(
            f"| {score_label} {row['threshold_level']} | "
            f"{row['coverage_difference_percentage_points']:.2f} "
            f"{_format_ci(row['coverage_difference_bootstrap_ci_low'], row['coverage_difference_bootstrap_ci_high'])} | "
            f"{row['mcnemar_exact_p']:.4g} | "
            f"{row['aggregate_capped_time_speedup_mc_over_fpt']:.2f} "
            f"{_format_ci(row['aggregate_capped_time_speedup_bootstrap_ci_low'], row['aggregate_capped_time_speedup_bootstrap_ci_high'])} |"
        )
    lines.extend(
        [
        "",
        "## Fixed-budget final quality",
        "",
        f"- Total energy: median paired improvement MC-FPT "
        f"{quality['total_energy']['median_improvement_mc_minus_fpt']:.3f} kcal/mol, "
        f"95% CI {_format_ci(*quality['total_energy']['median_improvement_bootstrap_ci'], digits=3)}; "
        f"FPT better {quality['total_energy']['fpt_better']}, MC better "
        f"{quality['total_energy']['mc_better']}, ties within 0.001 kcal/mol "
        f"{quality['total_energy']['tie']}; exact sign p="
        f"{quality['total_energy']['sign_exact_p']:.3g}.",
        f"- Binding score: median paired improvement MC-FPT "
        f"{quality['binding_score']['median_improvement_mc_minus_fpt']:.3f} kcal/mol, "
        f"95% CI {_format_ci(*quality['binding_score']['median_improvement_bootstrap_ci'], digits=3)}; "
        f"FPT better {quality['binding_score']['fpt_better']}, MC better "
        f"{quality['binding_score']['mc_better']}, ties "
        f"{quality['binding_score']['tie']}; exact sign p="
        f"{quality['binding_score']['sign_exact_p']:.3g}.",
        "",
        "## Torsion-stratified statistics",
        "",
        "| Torsions | Median energy gain [95% CI] | Step speedup [95% CI] | Capped-time speedup [95% CI] |",
        "|---:|---:|---:|---:|",
        ]
    )
    for row in summary["torsion_bins"]:
        lines.append(
            f"| {row['torsion_bin']} | {row['median_final_energy_improvement']:.3f} "
            f"{_format_ci(row['median_final_energy_improvement_bootstrap_ci_low'], row['median_final_energy_improvement_bootstrap_ci_high'], 3)} | "
            f"{row['median_step_speedup_mc_over_fpt']:.2f} "
            f"{_format_ci(row['median_step_speedup_bootstrap_ci_low'], row['median_step_speedup_bootstrap_ci_high'])} | "
            f"{row['aggregate_capped_time_speedup_mc_over_fpt']:.2f} "
            f"{_format_ci(row['aggregate_capped_time_speedup_bootstrap_ci_low'], row['aggregate_capped_time_speedup_bootstrap_ci_high'])} |"
        )
    diversity = quality["strict_joint_hit_cluster_diversity"]
    lines.extend(
        [
            "",
            "## Strict-threshold cluster diversity",
            "",
            f"Among {diversity['systems']} joint hits, median unique-cluster counts "
            f"are MC {diversity['mc_median_unique_clusters']:.1f} and FPT "
            f"{diversity['fpt_median_unique_clusters']:.1f}. Paired median FPT-MC "
            f"difference is {diversity['median_paired_difference_fpt_minus_mc']:.1f}, "
            f"95% CI {_format_ci(*diversity['median_paired_difference_bootstrap_ci'])}; "
            f"exact sign p={diversity['sign_exact_p']:.3g}.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def _latex_ci(low: float, high: float, digits: int = 2) -> str:
    return f"{low:.{digits}f}, {high:.{digits}f}"


def _latex_p(value: float) -> str:
    if value == 0:
        return "$<10^{-300}$"
    if value < 1e-3:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / 10**exponent
        return f"${mantissa:.2f}\\times10^{{{exponent}}}$"
    return f"{value:.4f}"


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _write_latex_tables(summary: dict, output_path: Path) -> None:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[scaled=0.95]{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{document}",
        r"\small",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Target-reaching performance across score definitions and thresholds.}",
        r"\label{tab:endpoint_statistics}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Score & Target & MC hits & FPT hits & $\Delta$ coverage [95\% CI] & Paired sign $p$ & Time speedup [95\% CI] \\",
        r"\midrule",
    ]
    for row in summary["endpoints"]:
        score = "Total energy" if row["score"] == "sampling_energy" else "Binding score"
        coverage = (
            f"{row['coverage_difference_percentage_points']:.2f} "
            f"[{_latex_ci(row['coverage_difference_bootstrap_ci_low'], row['coverage_difference_bootstrap_ci_high'])}]"
        )
        speedup = (
            f"{row['aggregate_capped_time_speedup_mc_over_fpt']:.2f} "
            f"[{_latex_ci(row['aggregate_capped_time_speedup_bootstrap_ci_low'], row['aggregate_capped_time_speedup_bootstrap_ci_high'])}]"
        )
        lines.append(
            f"{score} & {row['threshold_level'].capitalize()} & "
            f"{row['mc_successes']}/{row['trials']} & "
            f"{row['fpt_successes']}/{row['trials']} & {coverage} & "
            f"{_latex_p(row['mcnemar_exact_p'])} & {speedup} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Coverage differences are FPT minus MC, in percentage points. Time speedup is the aggregate timeout-capped MC/FPT ratio; values above one favor FPT. Failures receive the full 1000-step sampling time. Confidence intervals are paired percentile bootstrap intervals over systems (20,000 resamples).\end{minipage}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Strict sampling-energy performance stratified by ligand torsional complexity.}",
            r"\label{tab:torsion_statistics}",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Torsions & $N$ & MC/FPT hits & Energy gain & Step speedup & Time speedup & Sign $p$ \\",
            r"\midrule",
        ]
    )
    for row in summary["torsion_bins"]:
        energy = (
            f"{row['median_final_energy_improvement']:.3f} "
            f"[{_latex_ci(row['median_final_energy_improvement_bootstrap_ci_low'], row['median_final_energy_improvement_bootstrap_ci_high'], 3)}]"
        )
        step = (
            f"{row['median_step_speedup_mc_over_fpt']:.2f} "
            f"[{_latex_ci(row['median_step_speedup_bootstrap_ci_low'], row['median_step_speedup_bootstrap_ci_high'])}]"
        )
        time = (
            f"{row['aggregate_capped_time_speedup_mc_over_fpt']:.2f} "
            f"[{_latex_ci(row['aggregate_capped_time_speedup_bootstrap_ci_low'], row['aggregate_capped_time_speedup_bootstrap_ci_high'])}]"
        )
        lines.append(
            f"{row['torsion_bin']} & {row['systems']} & "
            f"{row['mc_successes']}/{row['fpt_successes']} of {row['trials']} & {energy} & {step} & "
            f"{time} & {_latex_p(row['energy_sign_exact_p'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Brackets report 95\% paired percentile bootstrap confidence intervals. Energy gain is median $E_{\mathrm{MC}}-E_{\mathrm{FPT}}$ in kcal/mol; positive values favor FPT. Step speedup is the median MC/FPT first-hit step ratio among joint positive-step hits. Time speedup is the aggregate timeout-capped MC/FPT ratio. The exact sign test uses a 0.001 kcal/mol energy-tie tolerance.\end{minipage}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Fixed-budget quality and conformational-diversity comparisons.}",
            r"\label{tab:quality_statistics}",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Metric & Systems & Paired effect [95\% CI] & FPT better & MC better & Ties / exact $p$ \\",
            r"\midrule",
        ]
    )
    total = summary["quality"]["total_energy"]
    binding = summary["quality"]["binding_score"]
    diversity = summary["quality"]["strict_joint_hit_cluster_diversity"]
    lines.extend(
        [
            f"Best total energy & {len(summary['protocol']['systems'])} & {total['median_improvement_mc_minus_fpt']:.3f} "
            f"[{_latex_ci(*total['median_improvement_bootstrap_ci'], 3)}] kcal/mol & "
            f"{total['fpt_better']} & {total['mc_better']} & {total['tie']} / {_latex_p(total['sign_exact_p'])} \\\\ ",
            f"Best binding score & {len(summary['protocol']['systems'])} & {binding['median_improvement_mc_minus_fpt']:.3f} "
            f"[{_latex_ci(*binding['median_improvement_bootstrap_ci'], 3)}] kcal/mol & "
            f"{binding['fpt_better']} & {binding['mc_better']} & {binding['tie']} / {_latex_p(binding['sign_exact_p'])} \\\\ ",
            f"Unique clusters & {diversity['systems']} & "
            f"{diversity['median_paired_difference_fpt_minus_mc']:.1f} "
            f"[{_latex_ci(*diversity['median_paired_difference_bootstrap_ci'], 1)}] clusters & "
            f"{diversity['fpt_better']} & {diversity['mc_better']} & "
            f"{diversity['tie']} / {_latex_p(diversity['sign_exact_p'])} \\\\ ",
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Energy effects are median $E_{\mathrm{MC}}-E_{\mathrm{FPT}}$; cluster effects are median FPT minus MC. Energy ties use a 0.001 kcal/mol tolerance. Cluster diversity is evaluated only among systems reaching the strict total-energy target with both methods.\end{minipage}",
            r"\end{table}",
            "",
            r"\begin{table}[!ht]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Final quality of systems missing the strict total-energy target with both methods.}",
            r"\label{tab:neither_hit_statistics}",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"System & Torsions & MC total gap & FPT total gap & Total gain & MC binding gap & FPT binding gap & Binding gain \\",
            r"\midrule",
        ]
    )
    for row in summary["neither_hit_systems"]:
        lines.append(
            f"\\texttt{{{_latex_escape(row['system'])}}} & {row['torsions']} & "
            f"{row['mc_total_gap']:.3f} & {row['fpt_total_gap']:.3f} & "
            f"{row['total_improvement_mc_minus_fpt']:.3f} & "
            f"{row['mc_binding_gap']:.3f} & {row['fpt_binding_gap']:.3f} & "
            f"{row['binding_improvement_mc_minus_fpt']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Gaps are best sampled score minus the system-specific strict threshold in kcal/mol; positive gaps indicate failure. Gains are MC gap minus FPT gap, so positive values favor FPT. Best-total and best-binding values may arise from different conformations.\end{minipage}",
            r"\end{table}",
            "",
            r"\end{document}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def analyze(output_dir: Path, result_dir: Path) -> dict:
    suite, systems = _load(output_dir)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(systems)
    bootstrap_indices = rng.integers(
        0, count, size=(BOOTSTRAP_SAMPLES, count)
    )
    endpoint_rows = _endpoint_statistics(suite, systems, bootstrap_indices)
    torsion_rows = _torsion_statistics(suite, systems, rng)
    quality = _quality_statistics(suite, systems, bootstrap_indices, rng)
    neither_hit_rows = _neither_hit_statistics(suite, systems)
    summary = {
        "protocol": {
            **suite["protocol"],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_unit": "paired systems with within-system paired-seed summaries",
            "random_seed_variability_estimated": suite["protocol"]["num_seeds"] > 1,
        },
        "endpoints": endpoint_rows,
        "torsion_bins": torsion_rows,
        "quality": quality,
        "neither_hit_systems": neither_hit_rows,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "endpoint_statistics.csv", endpoint_rows)
    _write_csv(result_dir / "torsion_statistics.csv", torsion_rows)
    _write_csv(result_dir / "neither_hit_statistics.csv", neither_hit_rows)
    (result_dir / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _write_markdown(summary, result_dir / "STATISTICAL_ANALYSIS.md")
    _write_latex_tables(summary, result_dir / "STATISTICAL_TABLES.tex")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="default: <output-dir>/paper_figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir or args.output_dir / "paper_figures"
    summary = analyze(args.output_dir, result_dir)
    strict = next(
        row
        for row in summary["endpoints"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    )
    print(
        f"strict coverage delta={strict['coverage_difference_percentage_points']:.2f} pp; "
        f"paired sign p={strict['mcnemar_exact_p']:.4g}; "
        f"capped speedup={strict['aggregate_capped_time_speedup_mc_over_fpt']:.3f}x"
    )


if __name__ == "__main__":
    main()
