from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import optax
from jax import numpy as jnp
from jax import random

sys.path.append(".")

from src.fp_transformer import (
    CoordTopology,
    DynamicITSSampler,
    EnergySystem,
    ITSSampler,
    Sampler,
    kernel,
    potential,
)
from src.fp_transformer.io.checkpointing import CheckpointConfig, LocalNpzCheckpointStore
from src.fp_transformer.utils import torus_pairwise_distance


def _build_sampler(*, rff: bool = False, seed: int = 42) -> Sampler:
    num_replicas = 12
    key = random.PRNGKey(0)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1
    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k_kwargs = dict(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
        rff=rff,
    )
    if rff:
        k_kwargs["rff_dim"] = 64
        k_kwargs["rff_seed"] = 11
    k = kernel.Gaussian(**k_kwargs)
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    return Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=seed)


def _build_dynamic_sampler() -> DynamicITSSampler:
    num_replicas = 12
    key = random.PRNGKey(123)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1
    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k = kernel.Gaussian(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    return DynamicITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        beta=2.0,
        max_num_temperatures=6,
        max_steps=20,
        grad_norm_threshold_mixing=1e-3,
        grad_norm_threshold_annealing=1e-3,
    )


def _build_its_sampler() -> ITSSampler:
    num_replicas = 12
    key = random.PRNGKey(321)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1
    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k = kernel.Gaussian(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    beta_array = jnp.array([0.8, 1.0, 1.2], dtype=jnp.float32)
    return ITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )


def test_sampler_state_dict_roundtrip_with_rff_kernel_state():
    s1 = _build_sampler(rff=True, seed=7)
    st0 = s1.init_state(s1.system.coordinate)
    noise_scale = jnp.asarray(0.0, dtype=st0.coordinate.dtype)
    st1, _ = s1.step_core(st0, s1.beta, noise_scale, s1.gamma)

    s1.system.coordinate = st1.coordinate
    s1.key = st1.rng_key
    s1._sync_kernel_from_state(st1.kernel_state)
    s1._last_sampler_state = st1
    s1.set_beta(2.5)
    s1.gamma = 0.75

    payload = s1.state_dict()

    s2 = _build_sampler(rff=True, seed=999)
    s2.from_state_dict(payload)

    assert jnp.allclose(s2.system.coordinate, s1.system.coordinate)
    assert jnp.array_equal(s2.key, s1.key)
    assert float(s2.beta) == float(s1.beta)
    assert float(s2.gamma) == float(s1.gamma)

    ks1 = s1._kernel_state_from_kernel()
    ks2 = s2._kernel_state_from_kernel()
    assert jnp.allclose(ks2.bandwidth, ks1.bandwidth)
    assert jnp.array_equal(ks2.rff_key, ks1.rff_key)

    st2 = s2.state_dict()["sampler_state"]
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: jnp.allclose(a, b), st2.opt_state, st1.opt_state
        )
    )


def test_dynamic_its_sampler_state_dict_roundtrip():
    s1 = _build_dynamic_sampler()
    its1 = s1.get_its_state()
    its1 = replace(
        its1,
        beta_array=its1.beta_array.at[0].set(1.5).at[1].set(2.0),
        f_array=its1.f_array.at[0].set(0.2).at[1].set(0.4),
        mask=its1.mask.at[0].set(1).at[1].set(1),
        anneal_factor=jnp.asarray(0.61),
        total_steps=jnp.asarray(123),
    )
    s1.set_its_state(its1)

    payload = s1.state_dict()
    s2 = _build_dynamic_sampler()
    s2.from_state_dict(payload)

    its2 = s2.get_its_state()
    assert jnp.allclose(its2.beta_array, its1.beta_array)
    assert jnp.allclose(its2.f_array, its1.f_array)
    assert jnp.array_equal(its2.mask, its1.mask)
    assert float(its2.anneal_factor) == float(its1.anneal_factor)
    assert int(its2.total_steps) == int(its1.total_steps)


def test_mock_multiprocess_sampler_state_checkpoint_roundtrip():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    script = r'''
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random

from fp_transformer import CoordTopology, EnergySystem, Sampler, kernel, potential
from fp_transformer.io.checkpointing import LocalNpzCheckpointStore, RunState
from fp_transformer.utils import torus_pairwise_distance

rank = int(os.environ["MOCK_RANK"])
world = int(os.environ["MOCK_WORLD"])
step = int(os.environ["MOCK_STEP"])
root = Path(os.environ["MOCK_CKPT_ROOT"])

num_replicas = 8
key = random.PRNGKey(rank)
coord = random.normal(key, (num_replicas, 1, 2)) * 0.1
init_dist2 = torus_pairwise_distance(coord)
bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][(num_replicas - 1) * (num_replicas - 2) // 2 // 2]
k = kernel.Gaussian(
    num_replicas=num_replicas,
    bandwidth=bw_init,
    update_bw_method="rescale",
    distance_func=torus_pairwise_distance,
)
top = CoordTopology(coord)
energy_ = potential.PeriodicMultidim(
    centers=jnp.array([[0.0, 0.0]]),
    depths=jnp.array([5.0]),
    concentrations=jnp.array([[3.0, 3.0]]),
)
sys_ = EnergySystem(topology=top, potential_=energy_)
opt = optax.adagrad(0.05, 0.1)
sampler = Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=rank + 10)

st0 = sampler.init_state(sampler.system.coordinate)
sampler.system.coordinate = st0.coordinate
sampler.key = st0.rng_key
sampler._sync_kernel_from_state(st0.kernel_state)
sampler._last_sampler_state = st0

store = LocalNpzCheckpointStore(root)
payload = sampler.state_dict()
state = RunState(
    step=step,
    coords_shard=np.asarray(sampler.system.coordinate),
    sampler_state=payload,
    metrics={"rank": rank},
)
store.stage_rank_state(state, rank=rank)

step_dir = root / f"step_{step:08d}"
commit_path = step_dir / "COMMIT"

if rank == 0:
    deadline = time.time() + 90.0
    while time.time() < deadline:
        if all((step_dir / f"rank_{r:04d}" / "state.npz").exists() for r in range(world)):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("timed out waiting for rank shards")
    store.commit_step(
        step=step,
        world_size=world,
        global_shape=(world * num_replicas, 1, 2),
        dtype="float32",
    )
else:
    deadline = time.time() + 90.0
    while time.time() < deadline and not commit_path.exists():
        time.sleep(0.05)
    if not commit_path.exists():
        raise RuntimeError("timed out waiting for COMMIT")

loaded = store.load_rank_state(step=step, rank=rank)
restored = Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=999)
restored.from_state_dict(loaded.sampler_state)

if not jnp.allclose(restored.system.coordinate, sampler.system.coordinate):
    raise AssertionError("coordinate mismatch after from_state_dict")
if not jnp.array_equal(restored.key, sampler.key):
    raise AssertionError("rng key mismatch after from_state_dict")

print(json.dumps({"ok": True, "rank": rank}))
'''

    with tempfile.TemporaryDirectory(prefix="fpt_ckpt_sampler_mp_") as td:
        ckpt_root = Path(td) / "ckpt"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = (
            f"{src_dir}:{base_env.get('PYTHONPATH', '')}"
            if base_env.get("PYTHONPATH")
            else str(src_dir)
        )
        base_env["MOCK_WORLD"] = "2"
        base_env["MOCK_STEP"] = "9"
        base_env["MOCK_CKPT_ROOT"] = str(ckpt_root)
        base_env["JAX_PLATFORMS"] = "cpu"

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
                out, err = p.communicate(timeout=120)
                outs.append((idx, out, err))
                if p.returncode != 0:
                    raise AssertionError(
                        f"rank={idx} failed with code {p.returncode}\\nstdout:\\n{out}\\nstderr:\\n{err}"
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


def test_sampler_run_with_periodic_checkpoint_and_restore(tmp_path):
    store = LocalNpzCheckpointStore(tmp_path / "ckpt")
    cfg = CheckpointConfig(every=2, rank=0, world_size=1, at_end=True, meta={"phase": "B"})
    s1 = _build_sampler(rff=False, seed=1234)
    s1.run(
        5,
        verbose=False,
        checkpoint_store=store,
        checkpoint_config=cfg,
    )
    assert store.latest_step(only_committed=True) == 5
    for step in (2, 4, 5):
        assert (tmp_path / "ckpt" / f"step_{step:08d}" / "COMMIT").exists()

    s2 = _build_sampler(rff=False, seed=0)
    loaded = s2.load_checkpoint(store=store, step=5, rank=0)
    assert loaded.step == 5
    assert jnp.allclose(s2.system.coordinate, s1.system.coordinate)
    assert jnp.array_equal(s2.key, s1.key)


def test_sampler_run_with_time_checkpoint_trigger(tmp_path):
    store = LocalNpzCheckpointStore(tmp_path / "ckpt_time")
    cfg = CheckpointConfig(
        every=None,
        interval_s=0.0,
        rank=0,
        world_size=1,
        at_end=False,
    )
    s1 = _build_sampler(rff=False, seed=2026)
    s1.run(
        3,
        verbose=False,
        checkpoint_store=store,
        checkpoint_config=cfg,
    )
    assert store.latest_step(only_committed=True) == 3


def test_its_sampler_run_passes_checkpoint_options(monkeypatch):
    sampler = _build_its_sampler()
    calls = []

    def fake_run_fix(self, num_steps, **kwargs):
        calls.append((int(num_steps), kwargs))
        return self.get_its_state()

    monkeypatch.setattr(ITSSampler, "run_fix_ensemble", fake_run_fix)
    cfg = CheckpointConfig(
        every=10,
        rank=1,
        world_size=2,
        commit_rank=0,
        meta={"tag": "its"},
        at_end=False,
    )
    sampler.run(3, verbose=False, checkpoint_store=object(), checkpoint_config=cfg)

    assert calls
    _, kwargs = calls[0]
    assert kwargs["checkpoint_config"] == cfg


def test_dynamic_its_run_passes_checkpoint_options(monkeypatch):
    sampler = _build_dynamic_sampler()
    calls = []

    def fake_mixing(self, **kwargs):
        calls.append(("mix", kwargs))
        return self.system.coordinate

    def fake_annealing(self, **kwargs):
        calls.append(("anneal", kwargs))
        return self.system.coordinate

    monkeypatch.setattr(DynamicITSSampler, "sample_ensemble_mixing", fake_mixing)
    monkeypatch.setattr(DynamicITSSampler, "sample_ensemble_annealing", fake_annealing)

    cfg = CheckpointConfig(
        every=8, rank=0, world_size=3, commit_rank=0, meta={"tag": "dyn"}, at_end=True
    )
    sampler.run(
        num_its_iterations=2,
        verbose=False,
        checkpoint_store=object(),
        checkpoint_config=cfg,
    )

    mix_calls = [c for c in calls if c[0] == "mix"]
    anneal_calls = [c for c in calls if c[0] == "anneal"]
    assert len(mix_calls) == 2
    assert len(anneal_calls) == 1
    for _, kwargs in calls:
        assert kwargs["checkpoint_config"] == cfg
