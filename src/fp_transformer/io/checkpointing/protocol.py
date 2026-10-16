"""Checkpoint store protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .state import Manifest, RunState


class CheckpointStore(ABC):
    """Abstract interface for multi-rank checkpoint stores."""

    @abstractmethod
    def stage_rank_state(self, state: RunState, *, rank: int) -> Path:
        """Stage one rank shard for `state.step`."""

    @abstractmethod
    def commit_step(
        self,
        *,
        step: int,
        world_size: int,
        global_shape: tuple[int, ...],
        dtype: str,
        sharding_spec: str | None = None,
        mesh_axes: tuple[str, ...] | None = None,
        extra: dict | None = None,
    ) -> Manifest:
        """Commit one step after all required rank shards are staged."""

    @abstractmethod
    def load_rank_state(self, *, step: int, rank: int) -> RunState:
        """Load one rank shard from a committed step."""

    @abstractmethod
    def latest_step(self, *, only_committed: bool = True) -> int | None:
        """Return latest step index, or None when empty."""

