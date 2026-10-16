"""State models for checkpointing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RunState:
    """Per-rank runtime state payload.

    `coords_shard` is expected to be the rank-local shard (or full coords in single-rank).
    """

    step: int
    coords_shard: Any
    sampler_state: Any = None
    kernel_state: Any = None
    its_state: Any = None
    rng_state: Any = None
    metrics: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class ShardMeta:
    """File metadata for one rank shard."""

    rank: int
    path: str
    num_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Manifest:
    """Global manifest for one checkpoint step."""

    version: int
    step: int
    status: str
    world_size: int
    global_shape: list[int]
    dtype: str
    sharding_spec: str | None = None
    mesh_axes: list[str] | None = None
    rank_files: list[ShardMeta] | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["rank_files"] = (
            [item.to_dict() for item in self.rank_files]
            if self.rank_files is not None
            else None
        )
        return out

