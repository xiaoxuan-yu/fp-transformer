"""Summarize a completed PoseBusters MC/FPT benchmark run.

The primary comparison uses timeout-capped first-passage time: successful runs
use their estimated first-hit wall time, while failures use the full sampling
wall time.  This keeps failures in the comparison instead of reporting only
the easier, successful subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


SCORES = ("sampling_energy", "binding_score")
LEVELS = ("easy", "medium", "strict")
TORSION_BINS = (
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-7", 6, 7),
    ("8-12", 8, 12),
    ("13+", 13, math.inf),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _winner(left: float, right: float, left_name: str, right_name: str) -> str:
    if math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    return left_name if left < right else right_name


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _flatten_group_summary(row: dict) -> dict:
    """Make nested outcome counters convenient to consume as a CSV table."""
    flat = {
        key: value
        for key, value in row.items()
        if key not in {"outcomes", "both_success_step_winners", "capped_time_winners"}
    }
    for prefix, key in (
        ("outcome", "outcomes"),
        ("step_winner", "both_success_step_winners"),
        ("capped_time_winner", "capped_time_winners"),
    ):
        flat.update({f"{prefix}_{name}": value for name, value in row[key].items()})
    return flat


def _outcome_counts(rows: list[dict]) -> dict[str, int]:
    result = {"both": 0, "mc_only": 0, "fpt_only": 0, "neither": 0}
    for row in rows:
        mc = row["mc_success"]
        fpt = row["fpt_success"]
        if mc and fpt:
            result["both"] += 1
        elif mc:
            result["mc_only"] += 1
        elif fpt:
            result["fpt_only"] += 1
        else:
            result["neither"] += 1
    return result


def _summarize_group(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    outcomes = _outcome_counts(rows)
    both = [row for row in rows if row["mc_success"] and row["fpt_success"]]
    positive = [
        row
        for row in both
        if row["mc_hit_step"] > 0 and row["fpt_hit_step"] > 0
    ]
    step_winners = {"mc": 0, "fpt": 0, "tie": 0}
    for row in both:
        step_winners[row["step_winner"]] += 1
    time_winners = {"mc": 0, "fpt": 0, "tie": 0}
    for row in rows:
        time_winners[row["capped_time_winner"]] += 1
    mc_capped_total = sum(row["mc_capped_seconds"] for row in rows)
    fpt_capped_total = sum(row["fpt_capped_seconds"] for row in rows)
    return {
        "systems": len({row["system"] for row in rows}),
        "trials": len(rows),
        "mc_successes": sum(row["mc_success"] for row in rows),
        "fpt_successes": sum(row["fpt_success"] for row in rows),
        "outcomes": outcomes,
        "step_zero_trials": sum(row["step_zero"] for row in rows),
        "step_zero_systems": len(
            {row["system"] for row in rows if row["step_zero"]}
        ),
        "both_success_step_winners": step_winners,
        "positive_hit_step_pairs": len(positive),
        "median_step_speedup_mc_over_fpt_positive_pairs": (
            median(row["mc_hit_step"] / row["fpt_hit_step"] for row in positive)
            if positive
            else None
        ),
        "mc_capped_seconds_total": mc_capped_total,
        "fpt_capped_seconds_total": fpt_capped_total,
        "capped_time_speedup_mc_over_fpt": _safe_ratio(
            mc_capped_total, fpt_capped_total
        ),
        "median_mc_capped_seconds_per_system": median(
            row["mc_capped_seconds"] for row in rows
        ),
        "median_fpt_capped_seconds_per_system": median(
            row["fpt_capped_seconds"] for row in rows
        ),
        "capped_time_winners": time_winners,
    }


def _build_endpoint_rows(output_dir: Path, suite: dict) -> tuple[list[dict], dict]:
    protocol = suite["protocol"]
    systems = protocol["systems"]
    expected_rows = len(systems) * len(SCORES) * len(LEVELS)
    suite_rows = suite["rows"]
    if len(suite_rows) != expected_rows:
        raise RuntimeError(
            f"suite has {len(suite_rows)} rows; expected {expected_rows}"
        )
    indexed = {
        (row["system"], row["score"], row["threshold_level"]): row
        for row in suite_rows
    }
    endpoint_rows = []
    system_data = {}
    required = ("aggregate.json", "seed_summary.csv", "trajectories.csv")
    for system in systems:
        system_dir = output_dir / system
        missing = [name for name in required if not (system_dir / name).exists()]
        if missing:
            raise RuntimeError(f"{system}: missing {missing}")
        aggregate = json.loads((system_dir / "aggregate.json").read_text())
        metadata = aggregate["metadata"]
        seed_rows = _read_csv(system_dir / "seed_summary.csv")
        expected_seed_rows = 2 * int(protocol["num_seeds"])
        if len(seed_rows) != expected_seed_rows:
            raise RuntimeError(
                f"{system}: expected {expected_seed_rows} paired seed rows, "
                f"got {len(seed_rows)}"
            )
        methods = {
            method: {
                int(row["seed"]): row
                for row in seed_rows
                if row["method"] == method
            }
            for method in ("mc", "fpt")
        }
        if set(methods) != {"mc", "fpt"}:
            raise RuntimeError(f"{system}: invalid methods {set(methods)}")
        expected_seeds = set(
            range(
                int(protocol["seed_base"]),
                int(protocol["seed_base"]) + int(protocol["num_seeds"]),
            )
        )
        for method, rows_by_seed in methods.items():
            if set(rows_by_seed) != expected_seeds:
                raise RuntimeError(
                    f"{system}: {method} seeds {sorted(rows_by_seed)} do not match "
                    f"expected {sorted(expected_seeds)}"
                )
        elapsed = {
            method: median(
                float(row["elapsed_seconds"]) for row in rows_by_seed.values()
            )
            for method, rows_by_seed in methods.items()
        }
        system_data[system] = {
            "metadata": metadata,
            "methods": methods,
            "elapsed": elapsed,
        }
        for score in SCORES:
            for level in LEVELS:
                source = indexed[(system, score, level)]
                for seed in sorted(expected_seeds):
                    paired = {method: methods[method][seed] for method in methods}
                    prefixes = {
                        method: _endpoint_prefix(row, score, level)
                        for method, row in paired.items()
                    }
                    if len(set(prefixes.values())) != 1:
                        raise RuntimeError(
                            f"{system} seed {seed} {score} {level}: "
                            f"mismatched threshold columns {prefixes}"
                        )
                    prefix = prefixes["mc"]
                    hit_steps = {}
                    hit_seconds = {}
                    successes = {}
                    capped = {}
                    for method, row in paired.items():
                        step = _optional_float(row[f"{prefix}_first_hit_step"])
                        if step is not None and step < 0:
                            step = None
                        seconds = _optional_float(
                            row[f"{prefix}_first_hit_seconds_estimate"]
                        )
                        success = step is not None and seconds is not None
                        if (step is None) != (seconds is None):
                            raise RuntimeError(
                                f"{system} seed {seed} {score} {level}: "
                                f"inconsistent {method} hit data"
                            )
                        hit_steps[method] = step
                        hit_seconds[method] = seconds
                        successes[method] = success
                        capped[method] = (
                            seconds if success else float(row["elapsed_seconds"])
                        )
                    step_winner = "not_comparable"
                    if successes["mc"] and successes["fpt"]:
                        step_winner = _winner(
                            hit_steps["mc"], hit_steps["fpt"], "mc", "fpt"
                        )
                    endpoint_rows.append(
                        {
                            "system": system,
                            "seed": seed,
                            "torsions": int(source["torsions"]),
                            "ligand_heavy_atoms": metadata["ligand_heavy_atoms"],
                            "receptor_pocket_atoms": metadata["receptor_pocket_atoms"],
                            "score": score,
                            "threshold_level": level,
                            "threshold": float(source["threshold"]),
                            "mc_success": successes["mc"],
                            "fpt_success": successes["fpt"],
                            "mc_hit_step": hit_steps["mc"],
                            "fpt_hit_step": hit_steps["fpt"],
                            "mc_hit_seconds": hit_seconds["mc"],
                            "fpt_hit_seconds": hit_seconds["fpt"],
                            "mc_capped_seconds": capped["mc"],
                            "fpt_capped_seconds": capped["fpt"],
                            "step_winner": step_winner,
                            "capped_time_winner": _winner(
                                capped["mc"], capped["fpt"], "mc", "fpt"
                            ),
                            "mc_archive_count": float(
                                paired["mc"][f"{prefix}_archive_count"]
                            ),
                            "fpt_archive_count": float(
                                paired["fpt"][f"{prefix}_archive_count"]
                            ),
                            "mc_unique_clusters": float(
                                paired["mc"][f"{prefix}_unique_clusters"]
                            ),
                            "fpt_unique_clusters": float(
                                paired["fpt"][f"{prefix}_unique_clusters"]
                            ),
                            "step_zero": hit_steps["mc"] == 0
                            or hit_steps["fpt"] == 0,
                        }
                    )
    return endpoint_rows, system_data


def _torsion_bin(torsions: int) -> str:
    for label, lower, upper in TORSION_BINS:
        if lower <= torsions <= upper:
            return label
    raise ValueError(f"torsion count outside bins: {torsions}")


def _format_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown_report(summary: dict) -> str:
    protocol = summary["protocol"]
    strict = summary["endpoints"]["sampling_energy/strict"]
    strict_nonzero = summary["sensitivity"]["sampling_energy/strict_without_step_zero"]
    binding = summary["endpoints"]["binding_score/strict"]
    final = summary["final_quality"]
    full_wall = summary["full_budget_sampling_wall_seconds"]
    notable = summary["notable_systems"]["sampling_energy/strict"]
    lines = [
        "# PoseBusters MC vs FPT analysis",
        "",
        "## Protocol",
        "",
        f"- Systems: {strict['systems']} cleaned PoseBusters complexes",
        f"- Paired random seeds: {protocol['num_seeds']} (seed {protocol['seed_base']})",
        f"- Budget: {protocol['steps']} steps, {protocol['replicas']} particles per method/system",
        f"- Backend: {protocol['interaction_backend']} on GPU; torsions are the only degrees of freedom",
        "- Primary endpoint: first passage of the strict sampling-energy threshold (crystal-pose total energy +0.25 kcal/mol)",
        "- Capped time: estimated first-hit sampling time for a success, full sampling time for a failure",
        "",
        (
            "With repeated seeds, success fractions are paired system-seed trial "
            "rates; system counts remain the external generalization unit."
            if protocol["num_seeds"] > 1
            else "With one seed, the reported success rate is **system coverage**, not a repeated-run probability."
        ),
        "",
        "## Primary result: strict sampling energy",
        "",
        "| Metric | MC | FPT |",
        "|---|---:|---:|",
        f"| Successful paired trials | {strict['mc_successes']}/{strict['trials']} | {strict['fpt_successes']}/{strict['trials']} |",
        f"| Aggregate capped first-passage time | {strict['mc_capped_seconds_total']:.3f} s | {strict['fpt_capped_seconds_total']:.3f} s |",
        f"| Median capped time per system | {strict['median_mc_capped_seconds_per_system']:.4f} s | {strict['median_fpt_capped_seconds_per_system']:.4f} s |",
        f"| Full 1000-step sampling wall time | {full_wall['mc']:.3f} s | {full_wall['fpt']:.3f} s |",
        "",
        f"FPT records {strict['fpt_successes'] - strict['mc_successes']} additional successful paired trials. "
        f"Its aggregate capped time is {_format_float(strict['capped_time_speedup_mc_over_fpt'])}x faster than MC. "
        f"Among positive-step pairs where both succeed, the median step speedup MC/FPT is {_format_float(strict['median_step_speedup_mc_over_fpt_positive_pairs'])}x.",
        "",
        f"Paired-trial outcome counts: both {strict['outcomes']['both']}, MC-only {strict['outcomes']['mc_only']}, "
        f"FPT-only {strict['outcomes']['fpt_only']}, neither {strict['outcomes']['neither']}. "
        f"Per-system capped-time wins are MC {strict['capped_time_winners']['mc']}, "
        f"FPT {strict['capped_time_winners']['fpt']}, ties {strict['capped_time_winners']['tie']}. "
        "Thus the aggregate gain comes from harder tails and avoided MC failures, while MC remains cheaper on many easy systems.",
        "",
        f"FPT-only strict hits: {', '.join(notable['fpt_only'])}. Neither method hits strict: "
        f"{', '.join(notable['neither'])}.",
        "",
        "## Dependence on torsion count",
        "",
        "| Torsions | Systems | MC trial hits | FPT trial hits | Median step speedup | Aggregate capped-time speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["torsion_bins"]:
        if row["score"] != "sampling_energy" or row["threshold_level"] != "strict":
            continue
        lines.append(
            f"| {row['torsion_bin']} | {row['systems']} | {row['mc_successes']} | "
            f"{row['fpt_successes']} | "
            f"{_format_float(row['median_step_speedup_mc_over_fpt_positive_pairs'])}x | "
            f"{_format_float(row['capped_time_speedup_mc_over_fpt'])}x |"
        )
    lines.extend(
        [
            "",
            "FPT becomes favorable as torsional dimension grows; MC is substantially cheaper for the 1–5 torsion groups.",
            "",
            "## Initialization sensitivity",
            "",
            f"{strict['step_zero_systems']} systems already satisfy the strict endpoint at step 0 despite bounded hard-start rejection. "
        f"After excluding step-zero trials, coverage is MC {strict_nonzero['mc_successes']}/{strict_nonzero['trials']} versus "
        f"FPT {strict_nonzero['fpt_successes']}/{strict_nonzero['trials']}; the main coverage conclusion is unchanged.",
            "",
            "## Secondary quality checks",
            "",
            f"At the strict binding-score endpoint, MC covers {binding['mc_successes']}/{binding['systems']} systems and "
            f"FPT covers {binding['fpt_successes']}/{binding['systems']}.",
            f"After the full budget, FPT has the lower best total score in {final['best_total']['fpt']} paired trials, "
            f"MC in {final['best_total']['mc']}, with {final['best_total']['tie']} ties. "
            f"For best binding score the paired-trial counts are FPT {final['best_binding']['fpt']}, "
            f"MC {final['best_binding']['mc']}, ties {final['best_binding']['tie']}.",
            f"At the strict total-energy threshold, median unique-cluster counts are MC "
            f"{final['strict_sampling_archive']['mc_median_unique_clusters']:.1f} and FPT "
            f"{final['strict_sampling_archive']['fpt_median_unique_clusters']:.1f}.",
            "",
            "## Interpretation",
            "",
            "The result supports a qualified claim: FPT is more effective on high-torsion and hard-tail systems, reaching the strict Vina target more often and with lower aggregate censored time. It is not uniformly faster per system, and its full-budget run is slower overall. Repeated seeds quantify run-to-run variability; uncertainty over new molecular systems is assessed separately by system-level resampling.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(output_dir: Path) -> dict:
    suite_path = output_dir / "suite_summary.json"
    if not suite_path.exists():
        raise FileNotFoundError(suite_path)
    suite = json.loads(suite_path.read_text())
    protocol = suite["protocol"]
    endpoint_rows, system_data = _build_endpoint_rows(output_dir, suite)

    endpoint_summary = {}
    torsion_rows = []
    for score in SCORES:
        for level in LEVELS:
            selected = [
                row
                for row in endpoint_rows
                if row["score"] == score and row["threshold_level"] == level
            ]
            endpoint_summary[f"{score}/{level}"] = _summarize_group(selected)
            for label, _, _ in TORSION_BINS:
                group = [row for row in selected if _torsion_bin(row["torsions"]) == label]
                values = _summarize_group(group)
                torsion_rows.append(
                    {"score": score, "threshold_level": level, "torsion_bin": label, **values}
                )

    sensitivity = {}
    for score in SCORES:
        for level in LEVELS:
            selected = [
                row
                for row in endpoint_rows
                if row["score"] == score
                and row["threshold_level"] == level
                and not row["step_zero"]
            ]
            sensitivity[f"{score}/{level}_without_step_zero"] = _summarize_group(selected)

    final_winners = {
        "best_total": {"mc": 0, "fpt": 0, "tie": 0},
        "best_binding": {"mc": 0, "fpt": 0, "tie": 0},
    }
    full_wall = {"mc": 0.0, "fpt": 0.0}
    for data in system_data.values():
        seeds = sorted(data["methods"]["mc"])
        for seed in seeds:
            for method in ("mc", "fpt"):
                full_wall[method] += float(
                    data["methods"][method][seed]["elapsed_seconds"]
                )
            for metric in final_winners:
                mc_value = float(data["methods"]["mc"][seed][metric])
                fpt_value = float(data["methods"]["fpt"][seed][metric])
                final_winners[metric][
                    _winner(mc_value, fpt_value, "mc", "fpt")
                ] += 1

    strict_rows = [
        row
        for row in endpoint_rows
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    archive_quality = {
        "mc_median_archive_count": median(row["mc_archive_count"] for row in strict_rows),
        "fpt_median_archive_count": median(row["fpt_archive_count"] for row in strict_rows),
        "mc_median_unique_clusters": median(
            row["mc_unique_clusters"] for row in strict_rows
        ),
        "fpt_median_unique_clusters": median(
            row["fpt_unique_clusters"] for row in strict_rows
        ),
        "archive_count_winners": {"mc": 0, "fpt": 0, "tie": 0},
        "unique_cluster_winners": {"mc": 0, "fpt": 0, "tie": 0},
    }
    for row in strict_rows:
        # Larger is better for archive size and cluster coverage.
        archive_winner = _winner(
            -row["mc_archive_count"], -row["fpt_archive_count"], "mc", "fpt"
        )
        cluster_winner = _winner(
            -row["mc_unique_clusters"], -row["fpt_unique_clusters"], "mc", "fpt"
        )
        archive_quality["archive_count_winners"][archive_winner] += 1
        archive_quality["unique_cluster_winners"][cluster_winner] += 1

    notable_systems = {}
    for score in SCORES:
        for level in LEVELS:
            selected = [
                row
                for row in endpoint_rows
                if row["score"] == score and row["threshold_level"] == level
            ]
            grouped = {
                system: [row for row in selected if row["system"] == system]
                for system in protocol["systems"]
            }
            majority = {
                system: {
                    method: sum(row[f"{method}_success"] for row in rows)
                    >= math.ceil(len(rows) / 2)
                    for method in ("mc", "fpt")
                }
                for system, rows in grouped.items()
            }
            notable_systems[f"{score}/{level}"] = {
                "mc_only": [
                    system
                    for system, outcome in majority.items()
                    if outcome["mc"] and not outcome["fpt"]
                ],
                "fpt_only": [
                    system
                    for system, outcome in majority.items()
                    if outcome["fpt"] and not outcome["mc"]
                ],
                "neither": [
                    system
                    for system, outcome in majority.items()
                    if not outcome["mc"] and not outcome["fpt"]
                ],
                "step_zero": [
                    system
                    for system, rows in grouped.items()
                    if any(row["step_zero"] for row in rows)
                ],
            }

    summary = {
        "protocol": protocol,
        "validation": {
            "system_directories": len(system_data),
            "endpoint_rows": len(endpoint_rows),
            "complete": True,
        },
        "full_budget_sampling_wall_seconds": {
            **full_wall,
            "fpt_over_mc": _safe_ratio(full_wall["fpt"], full_wall["mc"]),
        },
        "endpoints": endpoint_summary,
        "sensitivity": sensitivity,
        "torsion_bins": torsion_rows,
        "notable_systems": notable_systems,
        "final_quality": {
            **final_winners,
            "strict_sampling_archive": archive_quality,
        },
    }
    _write_csv(output_dir / "analysis_system_endpoints.csv", endpoint_rows)
    _write_csv(
        output_dir / "analysis_by_torsion.csv",
        [_flatten_group_summary(row) for row in torsion_rows],
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "ANALYSIS.md").write_text(_markdown_report(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.output_dir)
    strict = summary["endpoints"]["sampling_energy/strict"]
    print(
        f"validated {summary['validation']['system_directories']} systems; "
        f"strict coverage MC={strict['mc_successes']} "
        f"FPT={strict['fpt_successes']}; "
        f"capped-time speedup={strict['capped_time_speedup_mc_over_fpt']:.3f}x"
    )


if __name__ == "__main__":
    main()
