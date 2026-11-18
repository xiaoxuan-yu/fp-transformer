#!/usr/bin/env python3
"""Run the complete Tiwary--Berne transport sequence (main text Fig. 2, SI table).

Every particle number N starts from the same Gaussian law and runs to the same
horizon with the row mass K_N = round(0.16 N^0.6) (16, 24, 36 for N = 2048,
4096, 8192) and four seeds.  Each run is a separate ``run_transport`` process;
completed runs are skipped unless --force is given.

    python -m examples.tb.run_batch --output-dir data/tb_transport
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tb_transport"))
    parser.add_argument("--sizes", type=int, nargs="+", default=(2048, 4096, 8192))
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260820, 20260821, 20260822, 20260823))
    parser.add_argument("--mass-prefactor", type=float, default=0.16)
    parser.add_argument("--mass-exponent", type=float, default=0.6)
    parser.add_argument("--time-step", type=float, default=2.0e-4)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    started = time.perf_counter()
    for n_particles in args.sizes:
        row_mass = int(round(args.mass_prefactor * n_particles**args.mass_exponent))
        row_mass = min(max(row_mass, 2), n_particles - 2)
        for seed in args.seeds:
            run_dir = args.output_dir / f"n{n_particles}" / f"seed{seed}"
            command = [sys.executable, "-m", "examples.tb.run_transport", "--output-dir", str(run_dir),
                       "--replicas", str(n_particles), "--seed", str(seed), "--row-mass", str(row_mass),
                       "--time-step", f"{args.time_step:.17g}", "--final-time", f"{args.final_time:.17g}"]
            run_started = time.perf_counter()
            if (run_dir / "summary.json").exists() and not args.force:
                status = "existing"
            else:
                subprocess.run(command + (["--force"] if args.force else []), check=True)
                status = "completed"
            records.append({"n_particles": n_particles, "target_kernel_mass": row_mass, "seed": seed,
                            "status": status, "wall_time_s": time.perf_counter() - run_started,
                            "directory": str(run_dir), "command": command})
            print(f"[{status}] N={n_particles} K={row_mass} seed={seed}", flush=True)
    manifest = {
        "protocol": {
            "system": "two-dimensional Tiwary--Berne potential", "beta": 1.25,
            "initial_law": "isotropic Gaussian centered at (-0.5, 0.5) with standard deviation 0.3",
            "time_step": args.time_step, "final_time": args.final_time, "sizes": list(args.sizes),
            "seeds": list(args.seeds), "row_mass_schedule": "K_N = round(0.16 N^0.6)",
        },
        "total_wall_time_s": time.perf_counter() - started,
        "runs": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
