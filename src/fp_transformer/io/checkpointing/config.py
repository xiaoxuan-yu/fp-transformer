"""Checkpoint runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointConfig:
    """Convenience config object for sampler checkpoint integration."""

    every: int | None = None
    interval_s: float | None = None
    rank: int = 0
    world_size: int = 1
    global_shape: tuple[int, ...] | None = None
    commit: bool = True
    commit_rank: int = 0
    wait_timeout_s: float = 30.0
    sharding_spec: str | None = None
    mesh_axes: tuple[str, ...] | None = None
    meta: dict | None = None
    at_end: bool = True
