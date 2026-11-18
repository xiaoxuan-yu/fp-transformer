"""Lightweight metric snapshots for post-sampling Ala10 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTEXT_KEY = "context"
CONTEXT_ARRAY_KEY = "context_json"


def _snapshot_array_keys(
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if not snapshots:
        raise ValueError("at least one metric snapshot is required")
    keys = tuple(
        sorted(key for key in snapshots[0] if key != CONTEXT_KEY)
    )
    if not keys:
        raise ValueError("metric snapshots must contain numeric arrays")
    expected = set(keys) | {CONTEXT_KEY}
    for snapshot in snapshots:
        if set(snapshot) != expected:
            raise ValueError(
                "all metric snapshots must have the same array/context keys"
            )
    return keys


def save_metric_snapshot_bundle(
    path: str | Path,
    snapshots: Sequence[Mapping[str, Any]],
) -> Path:
    """Write one uncompressed, pickle-free bundle after sampling completes."""

    path = Path(path)
    array_keys = _snapshot_array_keys(snapshots)
    payload = {
        key: np.stack(
            [np.asarray(snapshot[key]) for snapshot in snapshots],
            axis=0,
        )
        for key in array_keys
    }
    payload[CONTEXT_ARRAY_KEY] = np.asarray(
        [
            json.dumps(
                snapshot[CONTEXT_KEY],
                sort_keys=True,
                separators=(",", ":"),
            )
            for snapshot in snapshots
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def load_metric_snapshot_bundle(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load a bundle produced by :func:`save_metric_snapshot_bundle`."""

    with np.load(Path(path), allow_pickle=False) as data:
        if CONTEXT_ARRAY_KEY not in data:
            raise ValueError("metric snapshot bundle is missing context_json")
        contexts = [
            json.loads(str(value)) for value in data[CONTEXT_ARRAY_KEY]
        ]
        array_keys = tuple(
            key for key in data.files if key != CONTEXT_ARRAY_KEY
        )
        snapshots = []
        for index, context in enumerate(contexts):
            snapshot = {
                key: np.asarray(data[key][index]) for key in array_keys
            }
            snapshot[CONTEXT_KEY] = context
            snapshots.append(snapshot)
    return snapshots


def write_jsonl_records(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
) -> Path:
    """Write completed offline records in one filesystem operation."""

    path = Path(path)
    text = "".join(
        json.dumps(record, sort_keys=sort_keys) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")
    return path
