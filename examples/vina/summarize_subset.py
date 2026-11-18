#!/usr/bin/env python3
"""Strict total-energy summary for Vina runs on arbitrary system subsets.

``analyze_results.py`` requires every torsion stratum to be populated; this
reads the per-system ``aggregate.json`` and ``seed_summary.csv`` directly and
reports, per run directory and method, strict-target hits, median first-hit
step among hits, and the mean final gap E_best - E_strict.  Several run
directories that share an FPT configuration can be compared on their MC rows.

MC temperature scan (SI): run the 24 systems in mc_temperature_scan_systems.txt
at each MC temperature, e.g.

    python -m examples.vina.run_benchmark --num-seeds 5 --mc-temperature 75 \
        --systems $(cat examples/vina/mc_temperature_scan_systems.txt) \
        --output-dir data/vina/scan_row_mc75
    python -m examples.vina.summarize_subset data/vina/scan_row_mc{300,150,75,30}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(run: Path):
    rows = []
    for system in sorted(p for p in run.iterdir() if (p / "aggregate.json").exists()):
        meta = json.loads((system / "aggregate.json").read_text())["metadata"]
        strict = min(meta["total_thresholds"])
        tag = f"sampling_energy_le_{strict:.3f}".replace("-", "m").replace(".", "p")
        for row in csv.DictReader((system / "seed_summary.csv").open()):
            hit = row.get(f"{tag}_first_hit_step", "")
            rows.append({
                "system": system.name,
                "torsions": meta["num_torsions"],
                "method": row["method"],
                "seed": int(row["seed"]),
                # misses are stored as -1
                "hit_step": float(hit) if hit not in ("", "None", "nan") and float(hit) >= 0 else None,
                "gap": float(row["best_total"]) - strict,
                "mc_temperature": meta.get("mc_temperature_kelvin", 300.0),
                "fpt_score": meta.get("fpt_score", "library"),
            })
    return rows


def summarize(rows, method, label):
    sel = [r for r in rows if r["method"] == method]
    hits = [r["hit_step"] for r in sel if r["hit_step"] is not None]
    systems = sorted({r["system"] for r in sel})
    per_system_gap = [np.mean([r["gap"] for r in sel if r["system"] == s]) for s in systems]
    print(f"{label:28s} hits {len(hits):3d}/{len(sel)}  median hit step {np.median(hits) if hits else float('nan'):6.0f}"
          f"  mean final gap {np.mean(per_system_gap):+.3f} kcal/mol  systems all-hit {sum(all(r['hit_step'] is not None for r in sel if r['system']==s) for s in systems)}/{len(systems)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for i, run in enumerate(args.runs):
        rows = load(run)
        if not rows:
            print(f"{run}: no completed systems")
            continue
        temperature = rows[0]["mc_temperature"]
        score = rows[0]["fpt_score"]
        summarize(rows, "mc", f"MC {temperature:g} K")
        if i == 0:
            summarize(rows, "fpt", f"FPT ({score} score)")


if __name__ == "__main__":
    main()
