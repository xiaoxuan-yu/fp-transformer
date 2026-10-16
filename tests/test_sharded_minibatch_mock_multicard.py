import json
import os
import subprocess
import sys
from pathlib import Path


def test_mock_multicard_sharded_minibatch_scan():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    script = r"""
import json
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fp_transformer.core.system.system import EnergySystem
from fp_transformer.core.topology import CoordTopology


class QuadraticPotential:
    def __call__(self, coord, beta=1.0):
        del beta
        return jnp.sum(coord * coord, axis=(-1, -2))


num_devices = int(jax.device_count())
if num_devices != 4:
    raise RuntimeError(f"Expected 4 mocked CPU devices, got {num_devices}")

mesh = Mesh(mesh_utils.create_device_mesh((1, 4)), axis_names=("node", "device"))
sharding = NamedSharding(mesh, P(("node", "device"), None, None))

B, N, D = 32, 5, 3
coord = random.normal(random.PRNGKey(0), (B, N, D), dtype=jnp.float32)
coord = jax.device_put(coord, sharding)
coord_spec = str(getattr(getattr(coord, "sharding", None), "spec", None))
if coord_spec != str(sharding.spec):
    raise AssertionError(f"Input coord is not sharded as expected: {coord_spec}")

top = CoordTopology(coord)
sys = EnergySystem(
    top,
    QuadraticPotential(),
    minibatch_size=8,
    assert_divisible_minibatch=True,
)

sys.minibatch_size = None
e_full = sys.potential(beta=1.0, coord=coord)
sys.minibatch_size = 8
e_chunk = sys.potential(beta=1.0, coord=coord)
e_ok = bool(jnp.allclose(e_full, e_chunk, rtol=1e-6, atol=1e-6))
if not e_ok:
    raise AssertionError("Sharded minibatch potential does not match full-batch potential.")

sys.minibatch_size = None
g_full = sys.grad_potential(coord, beta=1.0)
sys.minibatch_size = 8
g_chunk = sys.grad_potential(coord, beta=1.0)

ok = bool(jnp.allclose(g_full, g_chunk, rtol=1e-6, atol=1e-6))
if not ok:
    raise AssertionError("Sharded minibatch grad does not match full-batch grad.")

g_full_spec = str(getattr(getattr(g_full, "sharding", None), "spec", None))
g_chunk_spec = str(getattr(getattr(g_chunk, "sharding", None), "spec", None))
if g_full_spec != coord_spec:
    raise AssertionError(f"Full-batch output spec mismatch: {g_full_spec} != {coord_spec}")
if g_chunk_spec != coord_spec:
    raise AssertionError(f"Minibatch output spec mismatch: {g_chunk_spec} != {coord_spec}")

print(
    json.dumps(
        {
            "device_count": num_devices,
            "coord_spec": coord_spec,
            "g_full_spec": g_full_spec,
            "g_chunk_spec": g_chunk_spec,
            "e_ok": e_ok,
            "ok": ok,
        }
    )
)
"""

    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = (
        f"{src_dir}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(src_dir)
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise AssertionError(
            f"Subprocess failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["e_ok"] is True
    assert int(payload["device_count"]) == 4
    assert "node" in payload["coord_spec"] and "device" in payload["coord_spec"]
    assert payload["g_full_spec"] == payload["coord_spec"]
    assert payload["g_chunk_spec"] == payload["coord_spec"]
