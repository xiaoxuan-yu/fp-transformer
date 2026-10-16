"""Local filesystem-backed checkpoint store with staging + commit semantics."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import CheckpointStore
from .state import Manifest, RunState, ShardMeta


class LocalNpzCheckpointStore(CheckpointStore):
    """Checkpoint store using rank-local `.npz` shards plus a global manifest."""

    MANIFEST_VERSION = 1

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _step_dir(self, step: int) -> Path:
        return self.root_dir / f"step_{step:08d}"

    @staticmethod
    def _rank_dir(step_dir: Path, rank: int) -> Path:
        return step_dir / f"rank_{rank:04d}"

    @staticmethod
    def _rank_state_path(step_dir: Path, rank: int) -> Path:
        return LocalNpzCheckpointStore._rank_dir(step_dir, rank) / "state.npz"

    @staticmethod
    def _manifest_path(step_dir: Path) -> Path:
        return step_dir / "manifest.json"

    @staticmethod
    def _commit_marker_path(step_dir: Path) -> Path:
        return step_dir / "COMMIT"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _pack_obj(obj: Any) -> np.ndarray:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return np.frombuffer(payload, dtype=np.uint8)

    @staticmethod
    def _unpack_obj(arr: np.ndarray) -> Any:
        return pickle.loads(arr.tobytes())

    def stage_rank_state(self, state: RunState, *, rank: int) -> Path:
        step_dir = self._step_dir(int(state.step))
        rank_dir = self._rank_dir(step_dir, int(rank))
        rank_dir.mkdir(parents=True, exist_ok=True)

        target = self._rank_state_path(step_dir, int(rank))
        tmp = rank_dir / "state.tmp.npz"

        np.savez_compressed(
            tmp,
            run_state=self._pack_obj(state),
        )
        tmp.replace(target)
        return target

    def _collect_rank_files(self, *, step: int, world_size: int) -> list[ShardMeta]:
        step_dir = self._step_dir(step)
        rank_files: list[ShardMeta] = []
        for rank in range(int(world_size)):
            path = self._rank_state_path(step_dir, rank)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing rank shard before commit: step={step} rank={rank} path={path}"
                )
            rank_files.append(
                ShardMeta(
                    rank=rank,
                    path=str(path.relative_to(step_dir)),
                    num_bytes=int(path.stat().st_size),
                    sha256=self._sha256_file(path),
                )
            )
        return rank_files

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
        step = int(step)
        world_size = int(world_size)
        step_dir = self._step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)

        rank_files = self._collect_rank_files(step=step, world_size=world_size)
        manifest = Manifest(
            version=self.MANIFEST_VERSION,
            step=step,
            status="committed",
            world_size=world_size,
            global_shape=[int(x) for x in global_shape],
            dtype=str(dtype),
            sharding_spec=sharding_spec,
            mesh_axes=list(mesh_axes) if mesh_axes is not None else None,
            rank_files=rank_files,
            extra=extra,
        )

        manifest_path = self._manifest_path(step_dir)
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        with manifest_tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        manifest_tmp.replace(manifest_path)

        commit_path = self._commit_marker_path(step_dir)
        commit_tmp = commit_path.with_suffix(".tmp")
        commit_tmp.write_text("ok\n", encoding="utf-8")
        commit_tmp.replace(commit_path)
        return manifest

    def _assert_committed(self, step_dir: Path) -> None:
        if not self._commit_marker_path(step_dir).exists():
            raise FileNotFoundError(f"Step is not committed: {step_dir}")

    def load_rank_state(self, *, step: int, rank: int) -> RunState:
        step_dir = self._step_dir(int(step))
        self._assert_committed(step_dir)
        shard_path = self._rank_state_path(step_dir, int(rank))
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        with np.load(shard_path, allow_pickle=False) as data:
            return self._unpack_obj(data["run_state"])

    def load_manifest(self, *, step: int) -> Manifest:
        step_dir = self._step_dir(int(step))
        self._assert_committed(step_dir)
        path = self._manifest_path(step_dir)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        rank_items = obj.get("rank_files") or []
        rank_files = [ShardMeta(**item) for item in rank_items]
        return Manifest(
            version=int(obj["version"]),
            step=int(obj["step"]),
            status=str(obj["status"]),
            world_size=int(obj["world_size"]),
            global_shape=[int(x) for x in obj["global_shape"]],
            dtype=str(obj["dtype"]),
            sharding_spec=obj.get("sharding_spec"),
            mesh_axes=obj.get("mesh_axes"),
            rank_files=rank_files,
            extra=obj.get("extra"),
        )

    def list_steps(self) -> list[int]:
        out: list[int] = []
        for path in self.root_dir.glob("step_*"):
            name = path.name
            if not name.startswith("step_"):
                continue
            try:
                out.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
        return sorted(out)

    def latest_step(self, *, only_committed: bool = True) -> int | None:
        steps = self.list_steps()
        if not steps:
            return None
        if not only_committed:
            return steps[-1]
        for step in reversed(steps):
            step_dir = self._step_dir(step)
            if self._commit_marker_path(step_dir).exists():
                return step
        return None
