"""Dependency-free manifest serialization helpers for PoseBusters."""

from __future__ import annotations

import csv
from pathlib import Path


LIST_FIELDS = (
    "pocket_hetero_residues",
    "removed_pocket_hetero_residues",
    "retained_pocket_ions",
    "blocking_pocket_hetero_residues",
)


def write_manifest_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("cannot write an empty manifest CSV")
    fieldnames = list(records[0])
    for record in records[1:]:
        fieldnames.extend(key for key in record if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in LIST_FIELDS:
                row[key] = ";".join(record.get(key, []))
            writer.writerow(row)


__all__ = ["write_manifest_csv"]
