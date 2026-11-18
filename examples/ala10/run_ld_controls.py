#!/usr/bin/env python3
"""Matched Langevin controls for Ala10 FPT runs (main text Fig. 5).

For each FPT run directory (named ..._seed<seed>) the block-move event steps
are read from its progress.jsonl (steps at which ``exploration_events``
increases) and run_ld.py is started from the same seed with that event
schedule.  All other settings are the run_ld.py defaults.

    python -m examples.ala10.run_ld_controls --fpt-runs data/ala10/fpt/band30_50_m128_seed* \
        --output-root data/ala10/ld_controls
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def event_steps(progress: Path) -> list[int]:
    steps, previous = [], 0
    for line in progress.read_text().splitlines():
        record = json.loads(line)
        events = record.get("exploration_events", 0)
        if events > previous:
            steps.append(int(record["step"]))
            previous = events
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fpt-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", default="m128", help="output subdirectory ld_<variant>")
    args = parser.parse_args()
    for run in args.fpt_runs:
        seed = int(re.search(r"seed(\d+)", run.name).group(1))
        schedule = event_steps(run / "progress.jsonl")
        output = args.output_root / f"seed{seed}" / f"ld_{args.variant}"
        print(f"[seed {seed}] {len(schedule)} events: {schedule}", flush=True)
        subprocess.run([sys.executable, "-m", "examples.ala10.run_ld", "--seed", str(seed),
                        "--target-block-move-at", *map(str, schedule), "--output", str(output)], check=True)
        stored = json.loads((output / "config.json").read_text())
        assert stored["target_block_move_at"] == schedule, "schedule not applied"


if __name__ == "__main__":
    main()
