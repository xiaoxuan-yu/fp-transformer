from __future__ import annotations

import numpy as np
import pytest

from src.fp_transformer.io.checkpointing import LocalNpzCheckpointStore, RunState


def _mk_state(step: int, rank: int) -> RunState:
    return RunState(
        step=step,
        coords_shard=np.full((2, 1, 3), float(rank), dtype=np.float32),
        sampler_state={"rank": rank},
        metrics={"loss": float(rank)},
        meta={"source": "unit-test"},
    )


def test_stage_commit_and_load_rank_states(tmp_path):
    store = LocalNpzCheckpointStore(tmp_path)
    step = 12

    store.stage_rank_state(_mk_state(step, 0), rank=0)
    store.stage_rank_state(_mk_state(step, 1), rank=1)
    manifest = store.commit_step(
        step=step,
        world_size=2,
        global_shape=(4, 1, 3),
        dtype="float32",
        sharding_spec="P('data', None, None)",
        mesh_axes=("node", "device"),
        extra={"exp": "ckpt-smoke"},
    )

    assert manifest.step == step
    assert manifest.world_size == 2
    assert len(manifest.rank_files or []) == 2
    assert store.latest_step() == step

    s0 = store.load_rank_state(step=step, rank=0)
    s1 = store.load_rank_state(step=step, rank=1)
    assert s0.step == step and s1.step == step
    assert np.allclose(s0.coords_shard, 0.0)
    assert np.allclose(s1.coords_shard, 1.0)


def test_commit_requires_all_rank_shards(tmp_path):
    store = LocalNpzCheckpointStore(tmp_path)
    step = 3

    store.stage_rank_state(_mk_state(step, 0), rank=0)
    with pytest.raises(FileNotFoundError):
        store.commit_step(
            step=step,
            world_size=2,
            global_shape=(4, 1, 3),
            dtype="float32",
        )


def test_latest_step_only_committed(tmp_path):
    store = LocalNpzCheckpointStore(tmp_path)

    # staged-only step
    store.stage_rank_state(_mk_state(1, 0), rank=0)
    assert store.latest_step(only_committed=True) is None
    assert store.latest_step(only_committed=False) == 1

    # committed step
    store.stage_rank_state(_mk_state(2, 0), rank=0)
    store.commit_step(
        step=2,
        world_size=1,
        global_shape=(2, 1, 3),
        dtype="float32",
    )
    assert store.latest_step(only_committed=True) == 2
    assert store.latest_step(only_committed=False) == 2

