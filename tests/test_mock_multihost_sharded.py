import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_local_mock_multihost_sharded_minibatch():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    port = _pick_free_port()

    script = r"""
import json
import jax
import jax.numpy as jnp
from jax import random
from jax import distributed as jax_dist
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fp_transformer.core.system.system import EnergySystem
from fp_transformer.core.topology import CoordTopology
from fp_transformer.distributed import init_jax_distributed, safe_shutdown


class QuadraticPotential:
    def __call__(self, coord, beta=1.0):
        del beta
        return jnp.sum(coord * coord, axis=(-1, -2))


ctx = init_jax_distributed(jax_dist, jax, cluster_detection_method=None)
rank = int(ctx.global_rank)
world = int(ctx.world_size)
if world != 2:
    raise RuntimeError(f"Expected mocked world_size=2, got {world}")

if ctx.total_devices % ctx.world_size != 0:
    raise RuntimeError(
        f"total_devices ({ctx.total_devices}) must be divisible by world_size ({ctx.world_size})"
    )
mesh = Mesh(
    mesh_utils.create_device_mesh((ctx.world_size, ctx.total_devices // ctx.world_size)),
    axis_names=("node", "device"),
)
sharding = NamedSharding(mesh, P(("node", "device"), None, None))

B, N, D = 16, 5, 3
if rank == 0:
    coord_host = random.normal(random.PRNGKey(0), (B, N, D), dtype=jnp.float32)
else:
    coord_host = jnp.zeros((B, N, D), dtype=jnp.float32)
coord_host = multihost_utils.broadcast_one_to_all(coord_host, is_source=(rank == 0))
coord = jax.device_put(coord_host, sharding)

sys = EnergySystem(
    CoordTopology(coord),
    QuadraticPotential(),
    minibatch_size=4,
    assert_divisible_minibatch=True,
)
sys.minibatch_size = None
e_full = sys.potential(beta=1.0, coord=coord)
sys.minibatch_size = 4
e_chunk = sys.potential(beta=1.0, coord=coord)
if not bool(jnp.allclose(e_full, e_chunk, rtol=1e-6, atol=1e-6)):
    raise AssertionError("Local full/chunk potential mismatch.")

sys.minibatch_size = None
g_full = sys.grad_potential(coord, beta=1.0)
sys.minibatch_size = 4
g_chunk = sys.grad_potential(coord, beta=1.0)

if not bool(jnp.allclose(g_full, g_chunk, rtol=1e-6, atol=1e-6)):
    raise AssertionError("Local full/chunk gradient mismatch.")

coord_spec = str(getattr(getattr(coord, "sharding", None), "spec", None))
g_chunk_spec = str(getattr(getattr(g_chunk, "sharding", None), "spec", None))
if g_chunk_spec != coord_spec:
    raise AssertionError(f"Chunk output spec mismatch: {g_chunk_spec} != {coord_spec}")

all_g = multihost_utils.process_allgather(g_chunk, tiled=True)
if tuple(all_g.shape) != (B, N, D):
    raise AssertionError(f"Unexpected gathered shape: {all_g.shape}")

if rank == 0:
    print(
        json.dumps(
            {
                "ok": True,
                "coord_spec": coord_spec,
                "g_spec": g_chunk_spec,
                "e_shape": tuple(e_chunk.shape),
            }
        )
    )

safe_shutdown()
"""

    base_env = os.environ.copy()
    base_env["JAX_PLATFORM_NAME"] = "cpu"
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["CUDA_VISIBLE_DEVICES"] = ""
    base_env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(port)
    base_env["SLURM_NTASKS"] = "2"
    base_env["SLURM_NTASKS_PER_NODE"] = "2"
    base_env["PYTHONPATH"] = (
        f"{src_dir}:{base_env.get('PYTHONPATH', '')}"
        if base_env.get("PYTHONPATH")
        else str(src_dir)
    )

    procs: list[subprocess.Popen] = []
    try:
        for rank in (0, 1):
            env = base_env.copy()
            env["SLURM_PROCID"] = str(rank)
            env["SLURM_LOCALID"] = str(rank)
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

        outs: list[tuple[int, str, str]] = []
        for idx, p in enumerate(procs):
            out, err = p.communicate(timeout=120)
            outs.append((idx, out, err))
            if p.returncode != 0:
                raise AssertionError(
                    f"rank={idx} failed with code {p.returncode}\nstdout:\n{out}\nstderr:\n{err}"
                )

        rank0_out = outs[0][1].strip().splitlines()
        assert rank0_out, "rank0 produced no output"
        payload = json.loads(rank0_out[-1])
        assert payload["ok"] is True
        assert payload["coord_spec"] == payload["g_spec"]
        assert tuple(payload["e_shape"]) == (16,)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
