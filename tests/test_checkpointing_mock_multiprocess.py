import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_local_mock_multiprocess_checkpoint_commit_and_load():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    script = r"""
import json
import os
import time
from pathlib import Path

import numpy as np

from fp_transformer.io.checkpointing import LocalNpzCheckpointStore, RunState

rank = int(os.environ["MOCK_RANK"])
world = int(os.environ["MOCK_WORLD"])
step = int(os.environ["MOCK_STEP"])
root = Path(os.environ["MOCK_CKPT_ROOT"])

store = LocalNpzCheckpointStore(root)
state = RunState(
    step=step,
    coords_shard=np.full((2, 1, 3), float(rank), dtype=np.float32),
    sampler_state={"rank": rank},
    metrics={"loss": float(rank)},
    meta={"mock": True},
)
store.stage_rank_state(state, rank=rank)

step_dir = root / f"step_{step:08d}"
commit_path = step_dir / "COMMIT"

if rank == 0:
    deadline = time.time() + 20.0
    while time.time() < deadline:
        all_ready = True
        for r in range(world):
            p = step_dir / f"rank_{r:04d}" / "state.npz"
            if not p.exists():
                all_ready = False
                break
        if all_ready:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("timed out waiting for rank shards")

    manifest = store.commit_step(
        step=step,
        world_size=world,
        global_shape=(2 * world, 1, 3),
        dtype="float32",
        sharding_spec="P('data', None, None)",
        mesh_axes=("node", "device"),
        extra={"mock_test": True},
    )
    if manifest.world_size != world:
        raise AssertionError("manifest world_size mismatch")
else:
    deadline = time.time() + 20.0
    while time.time() < deadline and not commit_path.exists():
        time.sleep(0.05)
    if not commit_path.exists():
        raise RuntimeError("timed out waiting for COMMIT")

loaded = store.load_rank_state(step=step, rank=rank)
if loaded.step != step:
    raise AssertionError("loaded step mismatch")
if not np.allclose(loaded.coords_shard, float(rank)):
    raise AssertionError("loaded shard mismatch")

print(json.dumps({"ok": True, "rank": rank}))
"""

    with tempfile.TemporaryDirectory(prefix="fpt_ckpt_mock_mp_") as td:
        ckpt_root = Path(td) / "ckpt"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = (
            f"{src_dir}:{base_env.get('PYTHONPATH', '')}"
            if base_env.get("PYTHONPATH")
            else str(src_dir)
        )
        base_env["MOCK_WORLD"] = "2"
        base_env["MOCK_STEP"] = "7"
        base_env["MOCK_CKPT_ROOT"] = str(ckpt_root)

        procs: list[subprocess.Popen] = []
        try:
            for rank in (0, 1):
                env = base_env.copy()
                env["MOCK_RANK"] = str(rank)
                procs.append(
                    subprocess.Popen(
                        [sys.executable, "-c", script],
                        cwd=str(repo_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            outs = []
            for idx, p in enumerate(procs):
                out, err = p.communicate(timeout=60)
                outs.append((idx, out, err))
                if p.returncode != 0:
                    raise AssertionError(
                        f"rank={idx} failed with code {p.returncode}\nstdout:\n{out}\nstderr:\n{err}"
                    )

            for idx, out, _ in outs:
                lines = [ln for ln in out.strip().splitlines() if ln.strip()]
                assert lines, f"rank={idx} produced no output"
                payload = json.loads(lines[-1])
                assert payload["ok"] is True
                assert int(payload["rank"]) == idx
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
