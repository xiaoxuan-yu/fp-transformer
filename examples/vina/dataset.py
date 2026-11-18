"""Manifest-backed access to prepared PoseBusters sampling systems."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_PREPARED_DIR = HERE / "data/prepared"


def load_posebusters_systems(
    manifest_path: Path = DEFAULT_MANIFEST,
    prepared_dir: Path = DEFAULT_PREPARED_DIR,
    *,
    validated_only: bool = True,
) -> dict[str, dict]:
    """Return all technically compatible journal-subset systems."""

    manifest = json.loads(manifest_path.read_text())
    systems = {}
    for record in manifest["systems"]:
        if record["status"] != "prepared" or not record["primary_benchmark"]:
            continue
        if validated_only and record.get("validation_status") != "passed":
            continue
        system_id = record["system_id"]
        system_dir = prepared_dir / system_id
        receptor = system_dir / "receptor.pdbqt"
        ligand = system_dir / "ligand.pdbqt"
        if not receptor.exists() or not ligand.exists():
            raise FileNotFoundError(
                f"prepared PDBQT files missing for {system_id}; see README.md"
            )
        systems[system_id.lower()] = {
            "receptor_paths": (receptor,),
            "ligand_path": ligand,
            "pose_source": "PoseBusters crystallographic pose",
            "description": f"PoseBusters {system_id}",
            "posebusters_record": record,
        }
    return systems


__all__ = ["load_posebusters_systems"]
