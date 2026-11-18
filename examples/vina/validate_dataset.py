"""Validate prepared PoseBusters PDBQT files with differentiable Vina."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np



from examples.vina.pdbqt import build_pdbqt_potential
from examples.vina.manifest_io import (
    write_manifest_csv,
)


HERE = Path(__file__).resolve().parent


def validate_record(record: dict, prepared_dir: Path) -> None:
    if record["status"] != "prepared":
        record["validation_status"] = "not_applicable"
        return
    system_dir = prepared_dir / record["system_id"]
    try:
        loaded = build_pdbqt_potential(
            receptor_paths=system_dir / "receptor.pdbqt",
            ligand_path=system_dir / "ligand.pdbqt",
        )
        zero = jnp.zeros((1, 1, loaded.potential.num_torsions))
        total = loaded.potential(zero)
        binding = loaded.potential.binding_score(zero)
        gradient = jax.grad(
            lambda theta: jnp.sum(loaded.potential(theta))
        )(zero)
        jax.block_until_ready(gradient)
        if not (
            np.all(np.isfinite(np.asarray(total)))
            and np.all(np.isfinite(np.asarray(binding)))
            and np.all(np.isfinite(np.asarray(gradient)))
        ):
            raise ValueError("non-finite score or gradient")
        if loaded.potential.num_torsions != record["pdbqt_torsions"]:
            raise ValueError(
                "torsion mismatch: "
                f"manifest={record['pdbqt_torsions']} "
                f"potential={loaded.potential.num_torsions}"
            )
        record.update(
            validation_status="passed",
            validation_error=None,
            zero_total_score=float(total[0, 0]),
            zero_binding_score=float(binding[0, 0]),
            zero_gradient_l2=float(jnp.linalg.norm(gradient)),
            receptor_heavy_atoms=loaded.receptor_heavy_atoms_total,
            receptor_pocket_atoms=loaded.receptor_pocket_atoms,
        )
    except Exception as error:  # retain heterogeneous parser/geometry failures
        record.update(
            validation_status="failed",
            validation_error=f"{type(error).__name__}: {error}",
            primary_benchmark=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument(
        "--prepared-dir", type=Path, default=HERE / "data/prepared"
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="revalidate only records whose previous validation failed",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    prepared = [
        record
        for record in manifest["systems"]
        if record["status"] == "prepared"
        and (
            not args.failed_only
            or record.get("validation_status") == "failed"
        )
    ]
    for index, record in enumerate(prepared, start=1):
        validate_record(record, args.prepared_dir)
        print(
            f"[{index:03d}/{len(prepared):03d}] "
            f"{record['system_id']} {record['validation_status']}",
            flush=True,
        )
    systems = manifest["systems"]
    for record in systems:
        record["primary_benchmark"] = (
            record["status"] == "prepared"
            and record.get("validation_status") == "passed"
        )
    manifest["primary_system_ids"] = [
        record["system_id"] for record in systems if record["primary_benchmark"]
    ]
    manifest["counts"].update(
        validated=sum(
            record.get("validation_status") == "passed" for record in systems
        ),
        validation_failed=sum(
            record.get("validation_status") == "failed" for record in systems
        ),
        primary_benchmark=sum(record["primary_benchmark"] for record in systems),
    )
    manifest["validation"] = {
        "jax_backend": jax.default_backend(),
        "checks": [
            "PDBQT parsing",
            "torsion-count agreement",
            "finite zero-torsion total score",
            "finite zero-torsion binding score",
            "finite torsion gradient",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    write_manifest_csv(args.manifest.with_suffix(".csv"), systems)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
