import time
import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Callable

import jax
import optax
from jax import jit, vmap
from jax import numpy as jnp
from tqdm import tqdm, trange

from ..kernel import Kernel
from ..type import FloatLike, Optimizer
from ..utils import (
    compute_next_anneal_factor,
    # its_grad_coeff,
    its_grad_coeff_naive,
    its_grad_coeff_per_sample,
    logsumexp,
)
from ..io.checkpointing import CheckpointConfig, CheckpointStore, RunState
from .system import BaseSystem
from .state import ITSState, SamplerState
from .runtime import (
    kernel_state_from_kernel as _kernel_state_from_kernel_runtime,
    run_chunked_host_loop as _run_chunked_host_loop_runtime,
    run_postgrad_opt_loop as _run_postgrad_opt_loop_runtime,
    run_pregrad_opt_loop as _run_pregrad_opt_loop_runtime,
    run_step_state_loop as _run_step_state_loop_runtime,
    sync_kernel_from_state as _sync_kernel_from_state_runtime,
    to_host_control_scalar as _to_host_control_scalar_runtime,
    to_host_scalar as _to_host_scalar_runtime,
    validate_initial_coord_for_multiprocess as _validate_initial_coord_for_multiprocess_runtime,
)

kB_default = 1.38064852e-23


def _is_periodic_neighborlist_system(system: BaseSystem) -> bool:
    return bool(getattr(system, "periodic", False)) and all(
        hasattr(system, attr) for attr in ("update_neighbor_list", "nbList")
    )


@jit
def _build_system_kwargs(beta=None, nbList=None):
    """根据 System 类型自动构建 kwargs。

    对于 EnergySystem/ForceSystem 使用 beta 一个 kwarg，
    而对于 AmberSystem 额外传入 system.nbList。

    Args:
        system: 系统对象
        beta: 逆温度参数

    Returns:
        dict: 包含适当参数的 kwargs 字典
    """
    kwargs = {
        "beta": beta,
    }
    if nbList is not None:
        kwargs["nbList"] = nbList

    return kwargs


@jit
def _jit_mean_std(x):
    m = jnp.mean(x)
    s = jnp.std(x)
    return m, s


@jit
def _jit_norm(x):
    return jnp.linalg.norm(x)


def _to_host_scalar(x):
    """Convert scalar-like JAX values for logging only."""
    return _to_host_scalar_runtime(x)


def _to_host_control_scalar(x):
    """Convert scalar-like JAX values for host-side control flow decisions.

    Caller must ensure `x` is globally consistent across ranks (e.g. reduced
    scalar or replicated runtime scalar).
    """
    return _to_host_control_scalar_runtime(x)


def _validate_initial_coord_for_multiprocess(coord):
    """One-time startup validation for multi-process coordinate layout."""
    _validate_initial_coord_for_multiprocess_runtime(coord)


def _safe_global_mean_std(x):
    """使用 jit 的全局归约让 XLA 处理跨设备通信，并返回主机 float。"""
    m, s = _jit_mean_std(x)
    return _to_host_scalar(m), _to_host_scalar(s)


def _kernel_grad_with_bw(kernel: Kernel, coord, bw):
    """Compute kernel gradient with explicit bandwidth to avoid stale captured state."""
    # Non-RFF Gaussian path: use the pre-jitted grad core directly.
    if hasattr(kernel, "grad_jit") and not bool(getattr(kernel, "rff", False)):
        return kernel.grad_jit(coord, bw)  # type: ignore[attr-defined]
    # Fallback for kernels without grad_jit or stateful RFF kernels.
    return kernel.kernel_grad(coord, bw=bw)


def _kernel_grad_with_state(kernel: Kernel, coord, kernel_state, bw):
    """Compute kernel gradient and thread kernel runtime state explicitly."""
    if kernel_state is not None and hasattr(kernel, "kernel_grad_with_state"):
        return kernel.kernel_grad_with_state(coord, kernel_state, bw=bw)
    return _kernel_grad_with_bw(kernel, coord, bw), kernel_state


class BaselineLangevinSampler:  # A naive Langvin sampler with batch for baseline
    def __init__(
        self,
        system: BaseSystem,
        optimizer: Optimizer,
        temperature: FloatLike,
        friction_factor: FloatLike,
        tau: FloatLike,
        kB: FloatLike = kB_default,
        update_pace: int = 20,
    ):
        self.system = system
        if temperature <= 0:
            raise ValueError(
                f"Temperature in Kelvin should be positive but got {temperature}"
            )
        self.beta = jnp.reciprocal(kB * temperature)

        self.update_pace = update_pace
        self.update_nbList_flag = _is_periodic_neighborlist_system(self.system)

        def system_grad_func(coord, **kwargs):
            return self.system.grad_potential(coord, **kwargs)

        self.system_grad = jit(system_grad_func)
        if friction_factor < 0:
            raise ValueError(
                f"Langevin mobility should be non-negative but got {friction_factor}"
            )
        if tau <= 0:
            raise ValueError(f"Langevin time step should be positive but got {tau}")
        self.friction_factor = friction_factor
        self.tau = tau
        self.step_core = jit(self._step_core_)
        self.chunk_core = jit(self._chunk_core_, static_argnames=("n_steps",))
        self.run_step = jit(self.run_step_)

    def run_step_(self, carry):
        coord, key = carry
        # 使用自动构建的 kwargs
        kwargs = _build_system_kwargs(beta=self.beta, nbList=self.system.nbList)
        grad = self.system_grad(coord, **kwargs) * self.beta
        # generate drifting term
        key, subkey = jax.random.split(key)
        out_sharding = getattr(coord, "sharding", None)
        if out_sharding is None:
            drift = jax.random.normal(subkey, shape=coord.shape)
        else:
            drift = jax.random.normal(
                subkey, shape=coord.shape, out_sharding=out_sharding
            )
        # add drift term to coord
        effective_step = self.friction_factor * self.tau
        coord = coord - effective_step * grad + jnp.sqrt(2 * effective_step) * drift
        coord = self.system.project_coordinate(coord)
        return coord, key

    def init_state(self, coord0, key, nb_list=None):
        return SamplerState(
            coordinate=coord0,
            opt_state=None,
            rng_key=key,
            kernel_state=None,
            nb_list=nb_list,
        )

    def _step_core_(self, state, beta, friction_factor, tau):
        kwargs = _build_system_kwargs(beta=beta, nbList=state.nb_list)
        grad = self.system_grad(state.coordinate, **kwargs) * beta
        next_key, subkey = jax.random.split(state.rng_key)
        out_sharding = getattr(state.coordinate, "sharding", None)
        if out_sharding is None:
            drift = jax.random.normal(subkey, shape=state.coordinate.shape)
        else:
            drift = jax.random.normal(
                subkey, shape=state.coordinate.shape, out_sharding=out_sharding
            )
        effective_step = friction_factor * tau
        coord = (
            state.coordinate
            - effective_step * grad
            + jnp.sqrt(2 * effective_step) * drift
        )
        coord = self.system.project_coordinate(coord)
        new_state = SamplerState(
            coordinate=coord,
            opt_state=None,
            rng_key=next_key,
            kernel_state=None,
            nb_list=state.nb_list,
        )
        return new_state, {"grad_norm": _jit_norm(grad)}

    def _chunk_core_(self, state, n_steps, beta, friction_factor, tau):
        def body(st, _):
            st, metrics = self._step_core_(st, beta, friction_factor, tau)
            return st, metrics["grad_norm"]

        new_state, grad_norm_hist = jax.lax.scan(body, state, xs=None, length=n_steps)
        return new_state, {"grad_norm": grad_norm_hist[-1]}

    def run(self, steps, key):
        coord = self.system.coordinate
        if coord is None:
            raise ValueError("System coordinate is None")
        state = self.init_state(
            coord, key, nb_list=getattr(self.system, "nbList", None)
        )

        def _run_chunk(st, n):
            st, _ = self.chunk_core(
                st, int(n), self.beta, self.friction_factor, self.tau
            )
            return st

        def _refresh(st):
            self.system.coordinate = st.coordinate
            self.system.update_neighbor_list(st.coordinate, nbLists=self.system.nbList)
            return replace(
                st, coordinate=self.system.coordinate, nb_list=self.system.nbList
            )

        state = _run_chunked_host_loop_runtime(
            total_steps=int(steps),
            update_pace=self.update_pace if self.update_nbList_flag else 0,
            state=state,
            run_chunk=_run_chunk,
            refresh_runtime=_refresh if self.update_nbList_flag else None,
        )
        self.system.coordinate = state.coordinate


class Sampler:
    def _warn_deprecated_mutation_api(
        self, api_name: str, replacement: str, note: str | None = None
    ) -> None:
        warned = getattr(self, "_deprecated_api_warned", None)
        if warned is None:
            warned = set()
            self._deprecated_api_warned = warned
        if api_name in warned:
            return
        msg = (
            f"`{api_name}` is a legacy mutation-heavy API and is deprecated. "
            f"Prefer `{replacement}` for jit-safe explicit state transitions."
        )
        if note:
            msg = f"{msg} {note}"
        warnings.warn(msg, FutureWarning, stacklevel=2)
        warned.add(api_name)

    def __init__(
        self,
        system: BaseSystem,
        kernel: Kernel,
        optimizer: Optimizer,
        temperature: FloatLike,
        sample_cv: callable = None,
        kB: FloatLike = kB_default,
        update_pace: int = 20,
        gamma: FloatLike = 1.0,
        seed: int = 2026,
        nmax_atoms: int | None = None,
        nmax_bonds: int | None = None,
        nmax_angles: int | None = None,
        nmax_dihedrals: int | None = None,
        nmax_nb14s: int | None = None,
    ):
        self.system = system
        if temperature <= 0:
            raise ValueError(
                f"Temperature in Kelvin should be positive but got {temperature}"
            )
        self.beta = jnp.reciprocal(kB * temperature)  # SI unit
        self.kernel = kernel
        self.optimizer = optimizer
        self.num_replicas = self.kernel.num_replicas

        def system_grad_func(coord, **kwargs):
            return self.system.grad_potential(coord, **kwargs)

        self.system_grad = jit(system_grad_func)
        self.kernel_grad = self.kernel.kernel_grad
        self.opt_update = jit(self.opt_update_)

        # set sample_cv (identity if not provided)
        def identity(x):
            return x

        self.sample_cv = jit(identity) if sample_cv is None else jit(sample_cv)
        self.grad_freeE_core = jit(self._grad_freeE_core_)
        self.grad_freeE = self._grad_freeE_jit
        self.update_pace = update_pace
        self.pullback_cv = jit(self._pullback_cv_func)
        self.grad_freeE_with_state_core = jit(self._grad_freeE_with_state_core_)
        self.grad_freeE_with_state_core_no_noise = jit(
            self._grad_freeE_with_state_core_no_noise_
        )
        self.combine_system_kernel_grad_with_state_core_no_noise = jit(
            self._combine_system_kernel_grad_with_state_core_no_noise_
        )
        self.step_core = jit(self._step_core_)
        self.step_core_no_noise = jit(self._step_core_no_noise_)
        self.step_core_no_metrics = jit(self._step_core_no_metrics_)
        self.step_core_no_noise_no_metrics = jit(self._step_core_no_noise_no_metrics_)
        self.chunk_core = jit(self._chunk_core_, static_argnames=("n_steps",))

        self.update_nbList_flag = _is_periodic_neighborlist_system(self.system)
        self.key = jax.random.PRNGKey(seed)
        self.gamma = gamma
        if self.system.coordinate is not None:
            _validate_initial_coord_for_multiprocess(self.system.coordinate)

        if any(
            v is not None
            for v in (nmax_atoms, nmax_bonds, nmax_angles, nmax_dihedrals, nmax_nb14s)
        ):
            from ..potential.gaff_utils import set_gaff_max_values

            set_gaff_max_values(
                nmax_atoms=nmax_atoms,
                nmax_bonds=nmax_bonds,
                nmax_angles=nmax_angles,
                nmax_dihedrals=nmax_dihedrals,
                nmax_nb14s=nmax_nb14s,
            )

    def _kernel_state_from_kernel(self):
        return _kernel_state_from_kernel_runtime(self.kernel)

    def _sync_kernel_from_state(self, kernel_state):
        _sync_kernel_from_state_runtime(self.kernel, kernel_state)

    def init_state(self, coord0):
        _validate_initial_coord_for_multiprocess(coord0)
        state = SamplerState(
            coordinate=coord0,
            opt_state=self.optimizer.init(coord0),
            rng_key=self.key,
            kernel_state=self._kernel_state_from_kernel(),
            nb_list=getattr(self.system, "nbList", None),
        )
        self._last_sampler_state = state
        return state

    def _snapshot_sampler_state(self) -> SamplerState:
        if self.system.coordinate is None:
            raise ValueError("System coordinate is None; cannot serialize sampler state.")
        last_state = getattr(self, "_last_sampler_state", None)
        rng_key = (
            self.key
            if hasattr(self, "key")
            else (
                last_state.rng_key
                if isinstance(last_state, SamplerState)
                else jax.random.PRNGKey(2026)
            )
        )
        opt_state = (
            last_state.opt_state
            if isinstance(last_state, SamplerState) and last_state.opt_state is not None
            else self.optimizer.init(self.system.coordinate)
        )
        return SamplerState(
            coordinate=self.system.coordinate,
            opt_state=opt_state,
            rng_key=rng_key,
            kernel_state=self._kernel_state_from_kernel(),
            nb_list=getattr(self.system, "nbList", None),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "sampler_state": self._snapshot_sampler_state(),
            "beta": self.beta,
            "gamma": getattr(self, "gamma", 1.0),
            "update_pace": int(self.update_pace),
        }

    def from_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be a mapping, got {type(state)!r}")
        if "sampler_state" not in state:
            raise KeyError("state_dict missing required key: 'sampler_state'")
        sampler_state = state["sampler_state"]
        if not isinstance(sampler_state, SamplerState):
            raise TypeError(
                "state_dict['sampler_state'] must be SamplerState, "
                f"got {type(sampler_state)!r}"
            )
        self.system.coordinate = sampler_state.coordinate
        self.key = sampler_state.rng_key
        if "beta" in state:
            self.set_beta(state["beta"])
        if "gamma" in state:
            self.gamma = state["gamma"]
        if "update_pace" in state:
            self.update_pace = int(state["update_pace"])
        if hasattr(self.system, "nbList"):
            self.system.nbList = sampler_state.nb_list
        self._sync_kernel_from_state(sampler_state.kernel_state)
        self._last_sampler_state = sampler_state

    def _build_run_state(
        self,
        *,
        step: int,
        metrics: dict | None = None,
        meta: dict | None = None,
    ) -> RunState:
        if self.system.coordinate is None:
            raise ValueError("System coordinate is None; cannot build run state.")
        return RunState(
            step=int(step),
            coords_shard=self.system.coordinate,
            sampler_state=self.state_dict(),
            kernel_state=self._kernel_state_from_kernel(),
            metrics=metrics,
            meta=meta,
        )

    def save_checkpoint(
        self,
        *,
        store: CheckpointStore,
        step: int,
        rank: int = 0,
        world_size: int = 1,
        global_shape: tuple[int, ...] | None = None,
        commit: bool = True,
        commit_rank: int = 0,
        wait_timeout_s: float = 30.0,
        wait_interval_s: float = 0.05,
        sharding_spec: str | None = None,
        mesh_axes: tuple[str, ...] | None = None,
        metrics: dict | None = None,
        meta: dict | None = None,
    ):
        run_state = self._build_run_state(step=step, metrics=metrics, meta=meta)
        store.stage_rank_state(run_state, rank=int(rank))
        if not commit or int(rank) != int(commit_rank):
            return None

        if global_shape is None:
            global_shape = tuple(int(x) for x in self.system.coordinate.shape)

        deadline = time.time() + float(wait_timeout_s)
        while True:
            try:
                return store.commit_step(
                    step=int(step),
                    world_size=int(world_size),
                    global_shape=tuple(global_shape),
                    dtype=str(self.system.coordinate.dtype),
                    sharding_spec=sharding_spec,
                    mesh_axes=mesh_axes,
                    extra=meta,
                )
            except FileNotFoundError:
                if time.time() >= deadline:
                    raise
                time.sleep(float(wait_interval_s))

    def load_checkpoint(self, *, store: CheckpointStore, step: int, rank: int = 0):
        run_state = store.load_rank_state(step=int(step), rank=int(rank))
        payload = run_state.sampler_state
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Checkpoint payload `sampler_state` must be a mapping, "
                f"got {type(payload)!r}"
            )
        self.from_state_dict(payload)
        return run_state

    @staticmethod
    def _apply_checkpoint_config(
        checkpoint_config: CheckpointConfig | None,
        *,
        checkpoint_every,
        checkpoint_interval_s,
        checkpoint_rank,
        checkpoint_world_size,
        checkpoint_global_shape,
        checkpoint_commit,
        checkpoint_commit_rank,
        checkpoint_wait_timeout_s,
        checkpoint_sharding_spec,
        checkpoint_mesh_axes,
        checkpoint_meta,
        checkpoint_at_end,
    ):
        if checkpoint_config is None:
            return (
                checkpoint_every,
                checkpoint_interval_s,
                checkpoint_rank,
                checkpoint_world_size,
                checkpoint_global_shape,
                checkpoint_commit,
                checkpoint_commit_rank,
                checkpoint_wait_timeout_s,
                checkpoint_sharding_spec,
                checkpoint_mesh_axes,
                checkpoint_meta,
                checkpoint_at_end,
            )
        return (
            checkpoint_config.every,
            checkpoint_config.interval_s,
            checkpoint_config.rank,
            checkpoint_config.world_size,
            checkpoint_config.global_shape,
            checkpoint_config.commit,
            checkpoint_config.commit_rank,
            checkpoint_config.wait_timeout_s,
            checkpoint_config.sharding_spec,
            checkpoint_config.mesh_axes,
            checkpoint_config.meta,
            checkpoint_config.at_end,
        )

    def _host_refresh_runtime_state(self, coordinate):
        """Host-side runtime refresh at chunk/update boundaries.

        Returns:
            tuple: (updated_coordinate, kernel_state, nb_list)
        """
        kernel_coordinate = self.sample_cv(coordinate)
        if all(
            hasattr(self.kernel, attr)
            for attr in ("get_state", "set_state", "update_with_state")
        ):
            k_state = self.kernel.get_state()
            k_state = self.kernel.update_with_state(kernel_coordinate, k_state)
            if hasattr(self.kernel, "_set_state_fast"):
                self.kernel._set_state_fast(k_state)
            else:
                self.kernel.set_state(k_state)
        else:
            self.kernel.update(kernel_coordinate)
        if self.update_nbList_flag:
            self.system.update_neighbor_list(coordinate, nbLists=self.system.nbList)
            coordinate = self.system.coordinate
        return coordinate, self._kernel_state_from_kernel(), self.system.nbList

    def _run_pregrad_opt_loop(
        self,
        coordinate,
        opt_state,
        kernel_state,
        *,
        max_steps: int,
        grad_norm_threshold: float,
        grad_fn: Callable,
    ):
        return _run_pregrad_opt_loop_runtime(
            coordinate=coordinate,
            opt_state=opt_state,
            kernel_state=kernel_state,
            max_steps=max_steps,
            grad_norm_threshold=grad_norm_threshold,
            grad_fn=grad_fn,
            opt_update=self.opt_update,
            update_pace=self.update_pace,
            refresh_runtime=self._host_refresh_runtime_state,
            norm_fn=_jit_norm,
            control_scalar_fn=_to_host_control_scalar,
        )

    def _run_postgrad_opt_loop(
        self,
        coordinate,
        opt_state,
        kernel_state,
        *,
        num_steps: int,
        grad_fn: Callable,
        grad_norm_threshold: float | None = None,
        on_step: Callable | None = None,
    ):
        return _run_postgrad_opt_loop_runtime(
            coordinate=coordinate,
            opt_state=opt_state,
            kernel_state=kernel_state,
            num_steps=num_steps,
            grad_fn=grad_fn,
            opt_update=self.opt_update,
            update_pace=self.update_pace,
            refresh_runtime=self._host_refresh_runtime_state,
            norm_fn=_jit_norm,
            grad_norm_threshold=grad_norm_threshold,
            on_step=on_step,
            control_scalar_fn=_to_host_control_scalar,
        )

    def _pullback_cv_func(self, coordinate, grad_cv):
        # pullback = jax.linear_transpose(self.sample_cv, coordinate)
        cv, pullback = jax.vjp(self.sample_cv, coordinate)
        return pullback(grad_cv)[0]

    def opt_update_(self, coord, grad_value, opt_state):
        """
        Update the system's coordinates using the optimizer.
        """
        update, new_opt_state = self.optimizer.update(grad_value, opt_state)
        new_coord = optax.apply_updates(coord, update)
        new_coord = self.system.project_coordinate(new_coord)
        return new_coord, new_opt_state

    def _grad_freeE_core_(
        self,
        coordinate,
        beta,
        nb_list,
        kernel_bw,
        noise_key,
        noise_scale,
        gamma,
    ):
        # system potential gradient in coordinate space
        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        grad_val = self.system_grad(coordinate, **kwargs) * beta
        # map coordinate -> collective variable (CV)
        kernel_coordinate = self.sample_cv(coordinate)
        # kernel gradient w.r.t. CV space
        grad_cv = _kernel_grad_with_bw(self.kernel, kernel_coordinate, kernel_bw)
        grad_cv_noise = jax.random.normal(noise_key, shape=grad_cv.shape) * gamma
        grad_cv = grad_cv + noise_scale * grad_cv_noise
        # pull back to coordinate space via VJP of sample_cv
        # _, vjp_fn = jax.vjp(self.sample_cv, coordinate)
        # grad_coord_from_cv = vjp_fn(grad_cv)[0]
        grad_val += self.pullback_cv(coordinate, grad_cv)
        return grad_val

    def _grad_freeE_jit(self, coordinate, noise=False):
        self.key, noise_key = jax.random.split(self.key)
        noise_scale = jnp.asarray(1.0 if noise else 0.0, dtype=coordinate.dtype)
        kernel_state = self._kernel_state_from_kernel()
        grad_val, next_kernel_state = self.grad_freeE_with_state_core(
            coordinate,
            self.beta,
            self.system.nbList,
            kernel_state,
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth,
            noise_key,
            noise_scale,
            self.gamma,
        )
        self._sync_kernel_from_state(next_kernel_state)
        return grad_val

    def grad_freeE_(self, coordinate, noise=False):
        self._warn_deprecated_mutation_api(
            "Sampler.grad_freeE_",
            "Sampler._grad_freeE_jit / Sampler.grad_freeE_with_state_core",
        )
        self.key, noise_key = jax.random.split(self.key)
        noise_scale = jnp.asarray(1.0 if noise else 0.0, dtype=coordinate.dtype)
        return self._grad_freeE_core_(
            coordinate,
            self.beta,
            self.system.nbList,
            self.kernel.bandwidth,
            noise_key,
            noise_scale,
            self.gamma,
        )

    def _step_core_(self, state, beta, noise_scale, gamma):
        next_key, noise_key = jax.random.split(state.rng_key)
        kernel_state = state.kernel_state
        kernel_bw = (
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth
        )
        grad_value, next_kernel_state = self.grad_freeE_with_state_core(
            state.coordinate,
            beta,
            state.nb_list,
            kernel_state,
            kernel_bw,
            noise_key,
            noise_scale,
            gamma,
        )
        new_coord, new_opt_state = self.opt_update(
            state.coordinate, grad_value, state.opt_state
        )
        new_state = SamplerState(
            coordinate=new_coord,
            opt_state=new_opt_state,
            rng_key=next_key,
            kernel_state=next_kernel_state,
            nb_list=state.nb_list,
        )
        metrics = {"grad_norm": _jit_norm(grad_value)}
        return new_state, metrics

    def _step_core_no_noise_(self, state, beta, gamma):
        next_key, noise_key = jax.random.split(state.rng_key)
        kernel_state = state.kernel_state
        kernel_bw = (
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth
        )
        grad_value, next_kernel_state = self.grad_freeE_with_state_core_no_noise(
            state.coordinate,
            beta,
            state.nb_list,
            kernel_state,
            kernel_bw,
            noise_key,
            gamma,
        )
        new_coord, new_opt_state = self.opt_update(
            state.coordinate, grad_value, state.opt_state
        )
        new_state = SamplerState(
            coordinate=new_coord,
            opt_state=new_opt_state,
            rng_key=next_key,
            kernel_state=next_kernel_state,
            nb_list=state.nb_list,
        )
        metrics = {"grad_norm": _jit_norm(grad_value)}
        return new_state, metrics

    def _step_core_no_metrics_(self, state, beta, noise_scale, gamma):
        next_key, noise_key = jax.random.split(state.rng_key)
        kernel_state = state.kernel_state
        kernel_bw = (
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth
        )
        grad_value, next_kernel_state = self.grad_freeE_with_state_core(
            state.coordinate,
            beta,
            state.nb_list,
            kernel_state,
            kernel_bw,
            noise_key,
            noise_scale,
            gamma,
        )
        new_coord, new_opt_state = self.opt_update(
            state.coordinate, grad_value, state.opt_state
        )
        new_state = SamplerState(
            coordinate=new_coord,
            opt_state=new_opt_state,
            rng_key=next_key,
            kernel_state=next_kernel_state,
            nb_list=state.nb_list,
        )
        return new_state, None

    def _step_core_no_noise_no_metrics_(self, state, beta, gamma):
        next_key, noise_key = jax.random.split(state.rng_key)
        kernel_state = state.kernel_state
        kernel_bw = (
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth
        )
        grad_value, next_kernel_state = self.grad_freeE_with_state_core_no_noise(
            state.coordinate,
            beta,
            state.nb_list,
            kernel_state,
            kernel_bw,
            noise_key,
            gamma,
        )
        new_coord, new_opt_state = self.opt_update(
            state.coordinate, grad_value, state.opt_state
        )
        new_state = SamplerState(
            coordinate=new_coord,
            opt_state=new_opt_state,
            rng_key=next_key,
            kernel_state=next_kernel_state,
            nb_list=state.nb_list,
        )
        return new_state, None

    def _grad_freeE_with_state_core_(
        self,
        coordinate,
        beta,
        nb_list,
        kernel_state,
        kernel_bw,
        noise_key,
        noise_scale,
        gamma,
    ):
        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        grad_val = self.system_grad(coordinate, **kwargs) * beta
        kernel_coordinate = self.sample_cv(coordinate)
        grad_cv, next_kernel_state = _kernel_grad_with_state(
            self.kernel, kernel_coordinate, kernel_state, kernel_bw
        )
        grad_cv_noise = jax.random.normal(noise_key, shape=grad_cv.shape) * gamma
        grad_cv = grad_cv + noise_scale * grad_cv_noise
        grad_val += self.pullback_cv(coordinate, grad_cv)
        return grad_val, next_kernel_state

    def _grad_freeE_with_state_core_no_noise_(
        self,
        coordinate,
        beta,
        nb_list,
        kernel_state,
        kernel_bw,
        noise_key,
        gamma,
    ):
        del noise_key, gamma
        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        system_grad = self.system_grad(coordinate, **kwargs)
        return self._combine_system_kernel_grad_with_state_core_no_noise_(
            coordinate,
            system_grad,
            beta,
            kernel_state,
            kernel_bw,
        )

    def _combine_system_kernel_grad_with_state_core_no_noise_(
        self,
        coordinate,
        system_grad,
        beta,
        kernel_state,
        kernel_bw,
    ):
        """Combine a precomputed system gradient with the kernel force."""

        grad_val = system_grad * beta
        kernel_coordinate = self.sample_cv(coordinate)
        grad_cv, next_kernel_state = _kernel_grad_with_state(
            self.kernel, kernel_coordinate, kernel_state, kernel_bw
        )
        grad_val += self.pullback_cv(coordinate, grad_cv)
        return grad_val, next_kernel_state

    def _chunk_core_(self, state, n_steps, beta, noise_scale, gamma):
        def body(st, _):
            st, metrics = self._step_core_(st, beta, noise_scale, gamma)
            return st, metrics["grad_norm"]

        new_state, grad_norm_hist = jax.lax.scan(body, state, xs=None, length=n_steps)
        return new_state, {"grad_norm": grad_norm_hist[-1]}

    # estimate the partition function
    def KDE_density(self, x):
        @jit
        def gaussian(x, mu, sigma):
            return (
                1
                / (2 * jnp.pi * sigma)
                * jnp.exp(-jnp.linalg.norm(x - mu) / (2 * sigma**2))
            )

        @jit
        def gaussian_integral(sigma):
            return 2 * jnp.pi * sigma**2

        Rs = self.system.coordinate
        sigmas = self.kernel.bandwidth
        if sigmas.shape == (1,):
            sigmas = jnp.repeat(sigmas, Rs.shape[0])
        density = vmap(gaussian, in_axes=(None, 0, 0))(x, Rs, sigmas)
        Z = jnp.sum(vmap(gaussian_integral)(sigmas))
        return jnp.sum(density, axis=0) / (Z * Rs.shape[0])

    def effective_potential(self, x):  # NOTE: It is wrong!!
        return -1 / self.beta * jnp.log(self.KDE_density(x))

    def set_beta(self, beta: FloatLike):
        self.beta = beta

    def run(
        self,
        num_steps,
        verbose=False,
        verbose_pace=500,
        noise=False,
        *,
        verbose_metrics=None,
        verbose_print=None,
        skip_first_host_refresh=False,
        continue_from_last_state: bool = False,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        observe_fn=None,
    ):
        """Run ``num_steps`` sampler updates.

        ``observe_fn(step, state)``, if given, is called on the host after every
        update with the 1-based update count and the current ``SamplerState``.
        """
        (
            checkpoint_every,
            checkpoint_interval_s,
            checkpoint_rank,
            checkpoint_world_size,
            checkpoint_global_shape,
            checkpoint_commit,
            checkpoint_commit_rank,
            checkpoint_wait_timeout_s,
            checkpoint_sharding_spec,
            checkpoint_mesh_axes,
            checkpoint_meta,
            checkpoint_at_end,
        ) = self._apply_checkpoint_config(
            checkpoint_config,
            checkpoint_every=None,
            checkpoint_interval_s=None,
            checkpoint_rank=0,
            checkpoint_world_size=1,
            checkpoint_global_shape=None,
            checkpoint_commit=True,
            checkpoint_commit_rank=0,
            checkpoint_wait_timeout_s=30.0,
            checkpoint_sharding_spec=None,
            checkpoint_mesh_axes=None,
            checkpoint_meta=None,
            checkpoint_at_end=True,
        )
        if continue_from_last_state:
            last_state = getattr(self, "_last_sampler_state", None)
            if not isinstance(last_state, SamplerState):
                raise ValueError(
                    "continue_from_last_state=True requires an existing SamplerState. "
                    "Run a stage first or load a checkpoint/state_dict before continuing."
                )
            state = last_state
            self.system.coordinate = state.coordinate
            self.key = state.rng_key
            if hasattr(self.system, "nbList"):
                self.system.nbList = state.nb_list
            self._sync_kernel_from_state(state.kernel_state)
        else:
            state = self.init_state(self.system.coordinate)
        if verbose_metrics is None:
            verbose_metrics = bool(verbose)
        if verbose_print is None:
            verbose_print = bool(verbose)
        checkpoint_enabled = checkpoint_store is not None and (
            checkpoint_every is not None
            or checkpoint_interval_s is not None
            or checkpoint_at_end
        )
        last_checkpoint_time = time.perf_counter()

        indices = trange(num_steps) if verbose_print else range(num_steps)

        def _pre_step(i, st):
            if skip_first_host_refresh and i == 0:
                return st
            if i % self.update_pace != 0:
                return st
            updated_coord, kernel_state, nb_list = self._host_refresh_runtime_state(
                st.coordinate
            )
            return replace(
                st, coordinate=updated_coord, kernel_state=kernel_state, nb_list=nb_list
            )

        if noise:
            noise_scale = jnp.asarray(1.0, dtype=self.system.coordinate.dtype)
            if verbose_metrics:

                def _step_fn(st):
                    return self.step_core(st, self.beta, noise_scale, self.gamma)

            else:

                def _step_fn(st):
                    return self.step_core_no_metrics(
                        st, self.beta, noise_scale, self.gamma
                    )

        else:
            if verbose_metrics:

                def _step_fn(st):
                    return self.step_core_no_noise(st, self.beta, self.gamma)

            else:

                def _step_fn(st):
                    return self.step_core_no_noise_no_metrics(st, self.beta, self.gamma)

        def _on_step(i, _st, metrics):
            nonlocal last_checkpoint_time
            if observe_fn is not None:
                observe_fn(int(i) + 1, _st)
            if verbose_metrics and i % verbose_pace == 0 and verbose_print:
                gn = _to_host_scalar(metrics["grad_norm"])
                tqdm.write(f"Step {i}, Grad Norm: {gn}")

            if not checkpoint_enabled:
                return

            step_idx = int(i) + 1
            is_interval = checkpoint_every is not None and checkpoint_every > 0 and (
                step_idx % int(checkpoint_every) == 0
            )
            is_interval_time = False
            if checkpoint_interval_s is not None:
                now = time.perf_counter()
                is_interval_time = (now - last_checkpoint_time) >= float(
                    checkpoint_interval_s
                )
            is_end = bool(checkpoint_at_end) and step_idx == int(num_steps)
            if not (is_interval or is_interval_time or is_end):
                return

            self.system.coordinate = _st.coordinate
            self.key = _st.rng_key
            self._sync_kernel_from_state(_st.kernel_state)
            self._last_sampler_state = _st

            ckpt_metrics = None
            if isinstance(metrics, Mapping) and "grad_norm" in metrics:
                ckpt_metrics = {"grad_norm": _to_host_scalar(metrics["grad_norm"])}
            self.save_checkpoint(
                store=checkpoint_store,
                step=step_idx,
                rank=checkpoint_rank,
                world_size=checkpoint_world_size,
                global_shape=checkpoint_global_shape,
                commit=checkpoint_commit,
                commit_rank=checkpoint_commit_rank,
                wait_timeout_s=checkpoint_wait_timeout_s,
                sharding_spec=checkpoint_sharding_spec,
                mesh_axes=checkpoint_mesh_axes,
                metrics=ckpt_metrics,
                meta=checkpoint_meta,
            )
            last_checkpoint_time = time.perf_counter()

        state = _run_step_state_loop_runtime(
            indices=indices,
            state=state,
            step_fn=_step_fn,
            pre_step_fn=_pre_step,
            on_step_fn=(
                _on_step
                if (verbose_metrics or checkpoint_enabled or observe_fn is not None)
                else None
            ),
        )
        self.system.coordinate = state.coordinate
        self.key = state.rng_key
        self._sync_kernel_from_state(state.kernel_state)
        self._last_sampler_state = state


class ITSSampler(Sampler):
    def __init__(
        self,
        system: BaseSystem,
        kernel: Kernel,
        optimizer: Optimizer,
        temperature: FloatLike,
        gamma: FloatLike | None = None,
        sample_cv: callable = None,
        kB: FloatLike = kB_default,
        update_pace: int = 20,
        its_iterations: int = 1,
        grad_norm_threshold: float = 1e-3,
        f_array_threshold: float = 1e-1,
        ema_alpha: float = 0.5,
        beta_array: jnp.ndarray | None = None,
        nmax_atoms: int | None = None,
        nmax_bonds: int | None = None,
        nmax_angles: int | None = None,
        nmax_dihedrals: int | None = None,
        nmax_nb14s: int | None = None,
    ):
        if beta_array is None:
            raise ValueError("beta_array must be provided for ITSSampler")
        super().__init__(
            system=system,
            kernel=kernel,
            optimizer=optimizer,
            temperature=temperature,
            sample_cv=sample_cv,
            kB=kB,
            update_pace=update_pace,
            gamma=1.0 if gamma is None else gamma,
            nmax_atoms=nmax_atoms,
            nmax_bonds=nmax_bonds,
            nmax_angles=nmax_angles,
            nmax_dihedrals=nmax_dihedrals,
            nmax_nb14s=nmax_nb14s,
        )

        self.beta_array = jnp.asarray(beta_array)
        self.f_array = jnp.zeros_like(self.beta_array)
        self.mask = jnp.ones_like(self.beta_array, dtype=jnp.int32)

        self.ema_alpha = ema_alpha
        self.grad_norm_threshold = grad_norm_threshold
        self.f_array_threshold = f_array_threshold
        self.its_iterations = its_iterations

        self.total_steps = 0
        self.grad_freeE_core = jit(self._grad_freeE_core_)
        self.grad_freeE = self._grad_freeE_jit
        self.grad_freeE_from_state_core = jit(self._grad_freeE_from_state_core_)
        self.grad_freeE_from_state = self._grad_freeE_from_state_jit
        self.grad_freeE_from_state_with_kernel_state_core = jit(
            self._grad_freeE_from_state_with_kernel_state_core_
        )
        self.grad_freeE_from_state_with_kernel_state = (
            self._grad_freeE_from_state_with_kernel_state_jit
        )
        self.its_update = jit(self.its_update_)
        self.compute_potential_core = jit(self._compute_potential_core_)
        self.compute_potential = self._compute_potential
        self.is_connected = jit(self.is_connected_jax_from_dist_)
        self.its_state = self.get_its_state()

    def init_its_state(self):
        return ITSState(
            beta_array=self.beta_array,
            f_array=self.f_array,
            mask=self.mask,
            total_steps=jnp.asarray(self.total_steps),
        )

    def get_its_state(self):
        return ITSState(
            beta_array=self.beta_array,
            f_array=self.f_array,
            mask=self.mask,
            total_steps=jnp.asarray(self.total_steps),
        )

    def set_its_state(self, state: ITSState):
        self.beta_array = state.beta_array
        self.f_array = state.f_array
        self.mask = state.mask
        if state.total_steps is not None:
            self.total_steps = int(_to_host_control_scalar(state.total_steps))
        self.its_state = state

    def state_dict(self) -> dict[str, object]:
        out = super().state_dict()
        out["its_state"] = self.get_its_state()
        if hasattr(self, "min_energy"):
            out["min_energy"] = self.min_energy
        return out

    def _build_run_state(
        self,
        *,
        step: int,
        metrics: dict | None = None,
        meta: dict | None = None,
    ) -> RunState:
        out = super()._build_run_state(step=step, metrics=metrics, meta=meta)
        return replace(out, its_state=self.get_its_state())

    def from_state_dict(self, state: Mapping[str, object]) -> None:
        super().from_state_dict(state)
        its_state = state.get("its_state")
        if its_state is not None:
            if not isinstance(its_state, ITSState):
                raise TypeError(
                    "state_dict['its_state'] must be ITSState, "
                    f"got {type(its_state)!r}"
                )
            self.set_its_state(its_state)
        if "min_energy" in state:
            self.min_energy = state["min_energy"]

    def _compute_potential_core_(self, coord, nb_list):
        kwargs = _build_system_kwargs(beta=1.0, nbList=nb_list)
        return self.system.potential(coord=coord, **kwargs)

    def _compute_potential(self, coord):
        return self.compute_potential_core(coord, self.system.nbList)

    def its_update_(self, energy, beta_array, f_array, ema_alpha):
        # energy: (n_replicas, 1) or (n_replicas,)
        if hasattr(energy, "ndim") and energy.ndim == 2 and energy.shape[-1] == 1:
            energy = energy[:, 0]
        betak_U = jnp.expand_dims(beta_array, 1) * energy[None, :]
        sum_k_t = -logsumexp(-betak_U, axis=1)
        f_k_t = sum_k_t - sum_k_t[0]
        return ema_alpha * f_array + (1.0 - ema_alpha) * f_k_t

    def _grad_freeE_core_(
        self,
        coordinate,
        beta_array,
        f_array,
        min_energy,
        beta,
        nb_list,
        kernel_bw,
    ):
        beta_array = jnp.where(beta_array < 1e-8, 1e-8, beta_array)
        potential_array = self.compute_potential_core(coordinate, nb_list)
        potential_array = potential_array - min_energy + 10.0
        its_coeff = its_grad_coeff_naive(beta_array, potential_array, f_array)  # (N,)

        coordinate_shape = coordinate.shape
        its_coeff = its_coeff.reshape(-1, *([1] * (len(coordinate_shape) - 1)))

        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        grad_val = self.system_grad(coordinate, **kwargs) * its_coeff

        kernel_coordinate = self.sample_cv(coordinate)
        grad_cv = _kernel_grad_with_bw(self.kernel, kernel_coordinate, kernel_bw)
        grad_val += self.pullback_cv(coordinate, grad_cv)
        return grad_val

    def _grad_freeE_from_state_core_(
        self, coordinate, its_state, min_energy, beta, nb_list, kernel_bw
    ):
        return self._grad_freeE_core_(
            coordinate,
            its_state.beta_array,
            its_state.f_array,
            min_energy,
            beta,
            nb_list,
            kernel_bw,
        )

    def _grad_freeE_from_state_jit(self, coordinate, its_state, min_energy):
        kernel_state = self._kernel_state_from_kernel()
        grad_val, next_kernel_state = self.grad_freeE_from_state_with_kernel_state(
            coordinate,
            its_state,
            min_energy,
            kernel_state,
        )
        self._sync_kernel_from_state(next_kernel_state)
        return grad_val

    def _grad_freeE_from_state_(self, coordinate, its_state, min_energy):
        self._warn_deprecated_mutation_api(
            "ITSSampler._grad_freeE_from_state_",
            "ITSSampler._grad_freeE_from_state_jit",
        )
        return self._grad_freeE_from_state_core_(
            coordinate,
            its_state,
            min_energy,
            self.beta,
            self.system.nbList,
            self.kernel.bandwidth,
        )

    def _grad_freeE_from_state_with_kernel_state_core_(
        self, coordinate, its_state, min_energy, beta, nb_list, kernel_state, kernel_bw
    ):
        beta_array = jnp.where(its_state.beta_array < 1e-8, 1e-8, its_state.beta_array)
        potential_array = self.compute_potential_core(coordinate, nb_list)
        potential_array = potential_array - min_energy + 10.0
        its_coeff = its_grad_coeff_naive(beta_array, potential_array, its_state.f_array)
        coordinate_shape = coordinate.shape
        its_coeff = its_coeff.reshape(-1, *([1] * (len(coordinate_shape) - 1)))

        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        grad_val = self.system_grad(coordinate, **kwargs) * its_coeff

        kernel_coordinate = self.sample_cv(coordinate)
        grad_cv, next_kernel_state = _kernel_grad_with_state(
            self.kernel, kernel_coordinate, kernel_state, kernel_bw
        )
        grad_val += self.pullback_cv(coordinate, grad_cv)
        return grad_val, next_kernel_state

    def _grad_freeE_from_state_with_kernel_state_jit(
        self, coordinate, its_state, min_energy, kernel_state
    ):
        return self.grad_freeE_from_state_with_kernel_state_core(
            coordinate,
            its_state,
            min_energy,
            self.beta,
            self.system.nbList,
            kernel_state,
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth,
        )

    def _grad_freeE_from_state_with_kernel_state_(
        self, coordinate, its_state, min_energy, kernel_state
    ):
        return self._grad_freeE_from_state_with_kernel_state_core_(
            coordinate,
            its_state,
            min_energy,
            self.beta,
            self.system.nbList,
            kernel_state,
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth,
        )

    def _grad_freeE_jit(self, coordinate, beta_array, f_array, min_energy):
        its_state = ITSState(
            beta_array=beta_array,
            f_array=f_array,
            mask=jnp.ones_like(beta_array, dtype=jnp.int32),
            total_steps=None,
        )
        kernel_state = self._kernel_state_from_kernel()
        grad_val, next_kernel_state = self.grad_freeE_from_state_with_kernel_state(
            coordinate,
            its_state,
            min_energy,
            kernel_state,
        )
        self._sync_kernel_from_state(next_kernel_state)
        return grad_val

    def grad_freeE_(self, coordinate, beta_array, f_array, min_energy):
        self._warn_deprecated_mutation_api(
            "ITSSampler.grad_freeE_",
            "ITSSampler._grad_freeE_jit",
        )
        return self._grad_freeE_core_(
            coordinate,
            beta_array,
            f_array,
            min_energy,
            self.beta,
            self.system.nbList,
            self.kernel.bandwidth,
        )

    def is_connected_jax_from_dist_(self, cv_coordinate, radius_factor: float):
        D = self.kernel.distance_func(cv_coordinate)
        r_cut = radius_factor * self.kernel.bandwidth
        N = D.shape[0]
        A = D <= r_cut
        A = A & (~jnp.eye(N, dtype=bool))
        A = A | A.T

        visited = jnp.zeros((N,), dtype=bool).at[0].set(True)
        frontier = visited

        def cond(state):
            visited_, frontier_ = state
            return jnp.any(frontier_) & (~jnp.all(visited_))

        def body(state):
            visited_, frontier_ = state
            nbrs = jnp.any(A & frontier_[:, None], axis=0)
            new = nbrs & (~visited_)
            return visited_ | new, new

        visited, _ = jax.lax.while_loop(cond, body, (visited, frontier))
        return jnp.all(visited)

    def run_fix_ensemble(
        self,
        num_steps: int,
        *,
        verbose: bool,
        verbose_pace: int,
        its_state: ITSState | None = None,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        (
            checkpoint_every,
            checkpoint_interval_s,
            checkpoint_rank,
            checkpoint_world_size,
            checkpoint_global_shape,
            checkpoint_commit,
            checkpoint_commit_rank,
            checkpoint_wait_timeout_s,
            checkpoint_sharding_spec,
            checkpoint_mesh_axes,
            checkpoint_meta,
            checkpoint_at_end,
        ) = self._apply_checkpoint_config(
            checkpoint_config,
            checkpoint_every=None,
            checkpoint_interval_s=None,
            checkpoint_rank=0,
            checkpoint_world_size=1,
            checkpoint_global_shape=None,
            checkpoint_commit=True,
            checkpoint_commit_rank=0,
            checkpoint_wait_timeout_s=30.0,
            checkpoint_sharding_spec=None,
            checkpoint_mesh_axes=None,
            checkpoint_meta=None,
            checkpoint_at_end=True,
        )
        its_state = self.get_its_state() if its_state is None else its_state
        kernel_state = self._kernel_state_from_kernel()
        last_checkpoint_time = time.perf_counter()
        for it in range(self.its_iterations):
            opt_state = self.optimizer.init(self.system.coordinate)
            pbar = trange(num_steps) if verbose else None

            def on_step(step, grad_norm, _kernel_state):
                if pbar is not None:
                    pbar.update(1)
                if step % verbose_pace == 0:
                    kernel_bw = _to_host_scalar(jnp.mean(self.kernel.bandwidth))
                    if verbose:
                        tqdm.write(
                            f"iter {it}, step {step}, grad_norm {grad_norm:.4e}, bw {kernel_bw:.4f}"
                        )

            (
                self.system.coordinate,
                opt_state,
                kernel_state,
                step_count,
                _,
                _,
            ) = self._run_postgrad_opt_loop(
                self.system.coordinate,
                opt_state,
                kernel_state,
                num_steps=num_steps,
                grad_fn=lambda coord, ks: self.grad_freeE_from_state_with_kernel_state(
                    coord, its_state, self.min_energy, ks
                ),
                grad_norm_threshold=self.grad_norm_threshold,
                on_step=on_step,
            )
            if pbar is not None:
                pbar.close()
            its_state = replace(
                its_state,
                total_steps=(its_state.total_steps + step_count)
                if its_state.total_steps is not None
                else jnp.asarray(step_count),
            )

            energy = self.compute_potential(coord=self.system.coordinate)
            new_f_array = self.its_update(
                energy, its_state.beta_array, its_state.f_array, self.ema_alpha
            )
            if (
                jnp.max(jnp.abs(new_f_array - its_state.f_array))
                < self.f_array_threshold
            ):
                its_state = replace(its_state, f_array=new_f_array)
                break
            its_state = replace(its_state, f_array=new_f_array)

            if checkpoint_store is not None:
                total = int(_to_host_control_scalar(its_state.total_steps))
                is_interval_step = (
                    checkpoint_every is not None
                    and checkpoint_every > 0
                    and total % int(checkpoint_every) == 0
                )
                is_interval_time = (
                    checkpoint_interval_s is not None
                    and (time.perf_counter() - last_checkpoint_time)
                    >= float(checkpoint_interval_s)
                )
                if is_interval_step or is_interval_time:
                    self.set_its_state(its_state)
                    self._sync_kernel_from_state(kernel_state)
                    self.save_checkpoint(
                        store=checkpoint_store,
                        step=total,
                        rank=checkpoint_rank,
                        world_size=checkpoint_world_size,
                        global_shape=checkpoint_global_shape,
                        commit=checkpoint_commit,
                        commit_rank=checkpoint_commit_rank,
                        wait_timeout_s=checkpoint_wait_timeout_s,
                        sharding_spec=checkpoint_sharding_spec,
                        mesh_axes=checkpoint_mesh_axes,
                        meta={
                            "phase": "its_fix_ensemble",
                            "iteration": int(it),
                            **(checkpoint_meta or {}),
                        },
                    )
                    last_checkpoint_time = time.perf_counter()
        self._sync_kernel_from_state(kernel_state)
        self.set_its_state(its_state)
        if checkpoint_store is not None and checkpoint_at_end:
            final_step = int(_to_host_control_scalar(its_state.total_steps))
            self.save_checkpoint(
                store=checkpoint_store,
                step=final_step,
                rank=checkpoint_rank,
                world_size=checkpoint_world_size,
                global_shape=checkpoint_global_shape,
                commit=checkpoint_commit,
                commit_rank=checkpoint_commit_rank,
                wait_timeout_s=checkpoint_wait_timeout_s,
                sharding_spec=checkpoint_sharding_spec,
                mesh_axes=checkpoint_mesh_axes,
                meta={"phase": "its_fix_ensemble_end", **(checkpoint_meta or {})},
            )
        return its_state

    def run(
        self,
        num_steps: int,
        *,
        verbose: bool = False,
        verbose_pace: int = 5000,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        self.min_energy = jnp.min(self.compute_potential(coord=self.system.coordinate))
        its_state = self.get_its_state()
        its_state = self.run_fix_ensemble(
            num_steps,
            verbose=verbose,
            verbose_pace=verbose_pace,
            its_state=its_state,
            checkpoint_store=checkpoint_store,
            checkpoint_config=checkpoint_config,
        )

        # Sort by beta and keep f aligned with beta
        order = jnp.argsort(its_state.beta_array)
        its_state = replace(
            its_state,
            beta_array=its_state.beta_array[order],
            f_array=its_state.f_array[order],
        )
        self.set_its_state(its_state)

        output = {
            "coords": [self.system.coordinate],
            "bw": [self.kernel.bandwidth],
            "f": [its_state.f_array],
            "beta": [its_state.beta_array],
        }

        while its_state.beta_array.shape[0] > 1:
            if not bool(self.is_connected(self.sample_cv(self.system.coordinate), 1.0)):
                break
            next_beta = its_state.beta_array[1:]
            next_f = its_state.f_array[1:]
            its_state = replace(
                its_state, beta_array=next_beta, f_array=next_f - next_f[0]
            )
            self.set_its_state(its_state)
            its_state = self.run_fix_ensemble(
                num_steps,
                verbose=verbose,
                verbose_pace=verbose_pace,
                its_state=its_state,
                checkpoint_store=checkpoint_store,
                checkpoint_config=checkpoint_config,
            )
            output["coords"].append(self.system.coordinate)
            output["bw"].append(self.kernel.bandwidth)
            output["f"].append(its_state.f_array)
            output["beta"].append(its_state.beta_array)

        return output


# dynamicITS Sampler
class DynamicITSSampler(Sampler):
    def __init__(
        self,
        system: BaseSystem,
        kernel: Kernel,
        optimizer: Optimizer,
        temperature: FloatLike = None,
        beta: FloatLike = None,
        max_num_temperatures: int = 10,
        anneal_factor: FloatLike = 0.1,
        kl_div_threshold: FloatLike = 0.01,
        ema_alpha: FloatLike = 0.8,
        max_steps: int = 1000,  # 新增最大步数
        grad_norm_threshold_mixing: float = 1e-3,  # 新增梯度范数阈值
        grad_norm_threshold_annealing: float = 1e-3,
        kB: FloatLike = kB_default,
        sample_cv: callable = None,
        update_pace: int = 20,
    ):
        """
        初始化动态ITS采样器

        参数:
            system: 系统对象
            kernel: 核函数对象
            optimizer: 优化器对象
            temperature: 温度 (K)，与beta二选一
            beta: 温度的倒数 (1/kB·T)，与temperature二选一
            max_num_temperatures: 最大温度离散化数量
            num_steps: 每次优化迭代的步数
            max_steps: 每次优化的最大步数
            grad_norm_threshold: 梯度范数收敛阈值
        """
        # 检查初始化参数
        if temperature is None and beta is None:
            raise ValueError("必须提供temperature或beta其中之一")
        if temperature is not None and beta is not None:
            raise ValueError("temperature和beta不能同时提供")

        self.beta_array = jnp.zeros((max_num_temperatures,))
        self.f_array = jnp.zeros_like(self.beta_array)
        self.mask = jnp.zeros_like(self.beta_array, dtype=jnp.int32)
        self.ema_alpha = ema_alpha  # ITS参数的指数滑动平均memory time
        self.kl_div_threshold = kl_div_threshold  # 退火KL散度阈值
        self.anneal_factor = anneal_factor  # 初始退火因子
        self.total_steps = 0  # 新增：记录总步数

        # 保存参数
        self.max_num_temperatures = max_num_temperatures
        self.max_steps = max_steps  # 保存最大步数
        self.grad_norm_threshold_mixing = grad_norm_threshold_mixing
        self.grad_norm_threshold_annealing = grad_norm_threshold_annealing

        # 调用父类初始化
        if beta is not None:
            # 如果提供了beta，则直接设置self.beta并计算对应的temperature
            self.beta = beta
            temperature = 1.0 / (kB * beta)
            super(Sampler, self).__init__()  # 只初始化基类的基类
            # 手动设置父类Sampler的成员变量
            self.system = system
            self.kernel = kernel
            self.optimizer = optimizer
            self.num_replicas = self.kernel.num_replicas

            def system_grad_func(coord, **kwargs):
                return self.system.grad_potential(coord, **kwargs)

            self.system_grad = jit(system_grad_func)
            # self.kernel_grad = jit(self.kernel.kernel_grad)
            self.kernel_grad = self.kernel.kernel_grad
            self.update_pace = update_pace
            self.update_nbList_flag = _is_periodic_neighborlist_system(self.system)
        else:
            # 如果提供了temperature，则使用原始的父类初始化方法
            super().__init__(
                system, kernel, optimizer, temperature, sample_cv, kB, update_pace
            )

        # 确保在未通过父类构造时也有 sample_cv（默认为恒等映射）
        if not hasattr(self, "sample_cv"):
            if sample_cv is None:

                def identity(x):
                    return x

                self.sample_cv = jit(identity)
            else:
                self.sample_cv = jit(sample_cv)

        # vmapped potential: map over beta (axis=0) while keeping coord as non-mapped
        # wrap keyword-only coord into a positional arg for vmap to specify in_axes
        def potential_vmap_func(b, c, **kwargs):
            return self.system.potential(b, coord=c, **kwargs)

        self.potential_vmap = vmap(
            jit(potential_vmap_func),
            in_axes=(0, None),
        )
        self.grad_freeE_core = jit(self._grad_freeE_core_)
        self.grad_freeE = self._grad_freeE_jit
        self.grad_freeE_from_state_core = jit(self._grad_freeE_from_state_core_)
        self.grad_freeE_from_state = self._grad_freeE_from_state_jit
        self.grad_freeE_from_state_with_kernel_state_core = jit(
            self._grad_freeE_from_state_with_kernel_state_core_
        )
        self.grad_freeE_from_state_with_kernel_state = (
            self._grad_freeE_from_state_with_kernel_state_jit
        )
        self.opt_update = jit(self.opt_update_)
        self.ref_energy_core = jit(self._ref_energy_core_)
        self.ref_energy = self._ref_energy
        self.ref_energy_from_state_core = jit(self._ref_energy_from_state_core_)
        self.ref_energy_from_state = self._ref_energy_from_state
        self.dynamic_its_update = jit(self.dynamic_its_update_)
        self.dynamic_anneal = jit(self.dynamic_anneal_)
        self.pullback_cv = jit(self._pullback_cv_func)

        self.kB = kB  # 保存kB常数
        if self.system.coordinate is not None:
            _validate_initial_coord_for_multiprocess(self.system.coordinate)
        self.its_state = self.get_its_state()

    def init_its_state(self):
        return ITSState(
            beta_array=self.beta_array,
            f_array=self.f_array,
            mask=self.mask,
            anneal_factor=jnp.asarray(self.anneal_factor),
            total_steps=jnp.asarray(self.total_steps),
        )

    def get_its_state(self):
        return ITSState(
            beta_array=self.beta_array,
            f_array=self.f_array,
            mask=self.mask,
            anneal_factor=jnp.asarray(self.anneal_factor),
            total_steps=jnp.asarray(self.total_steps),
        )

    def set_its_state(self, state: ITSState):
        self.beta_array = state.beta_array
        self.f_array = state.f_array
        self.mask = state.mask
        if state.anneal_factor is not None:
            self.anneal_factor = float(_to_host_control_scalar(state.anneal_factor))
        if state.total_steps is not None:
            self.total_steps = int(_to_host_control_scalar(state.total_steps))
        self.its_state = state

    def state_dict(self) -> dict[str, object]:
        out = super().state_dict()
        out["its_state"] = self.get_its_state()
        return out

    def _build_run_state(
        self,
        *,
        step: int,
        metrics: dict | None = None,
        meta: dict | None = None,
    ) -> RunState:
        out = super()._build_run_state(step=step, metrics=metrics, meta=meta)
        return replace(out, its_state=self.get_its_state())

    def from_state_dict(self, state: Mapping[str, object]) -> None:
        super().from_state_dict(state)
        its_state = state.get("its_state")
        if its_state is not None:
            if not isinstance(its_state, ITSState):
                raise TypeError(
                    "state_dict['its_state'] must be ITSState, "
                    f"got {type(its_state)!r}"
                )
            self.set_its_state(its_state)

    def _pullback_cv_func(self, coordinate, grad_cv):
        return super()._pullback_cv_func(coordinate, grad_cv)

    def opt_update_(self, coord, grad_value, opt_state):
        """
        Update the system's coordinates using the optimizer.
        """
        update, new_opt_state = self.optimizer.update(grad_value, opt_state)
        new_coord = optax.apply_updates(coord, update)
        new_coord = self.system.project_coordinate(new_coord)
        return new_coord, new_opt_state

    def _grad_freeE_core_(
        self, coord, beta_array, beta, f_array, mask, nb_list, kernel_bw
    ):
        """计算自由能梯度"""
        # beta array 0 to one, avoiding inf
        beta_array = jnp.where(beta_array < 1e-8, 1e-8, beta_array)
        # 准备势能数组（沿 beta 维度映射，coord 保持常量）
        potential_array = self.potential_vmap(beta_array, coord)

        # 计算逐样本 ITS 系数 (N,)
        coeff = its_grad_coeff_per_sample(
            beta, beta_array, potential_array, f_array, mask
        )
        coord_shape = coord.shape
        coeff = coeff.reshape(-1, *([1] * (len(coord_shape) - 1)))

        # 计算原系综下的系统梯度
        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)

        grad_val = self.system_grad(coord, **kwargs) * beta
        # print("grad_val.addressable_shards:", grad_val.addressable_shards)
        # print("coeff.addressable_shards:", coeff.addressable_shards)
        # 应用混合系数
        grad_val = coeff * grad_val

        # 通过 sample_cv 将核函数梯度从 CV 空间拉回到坐标空间
        kernel_coordinate = self.sample_cv(coord)
        grad_cv = _kernel_grad_with_bw(self.kernel, kernel_coordinate, kernel_bw)
        # _, vjp_fn = jax.vjp(self.sample_cv, coord)
        # grad_coord_from_cv = vjp_fn(grad_cv)[0]
        grad_coord_from_cv = self.pullback_cv(coord, grad_cv)
        # print("grad_coord_from_cv.addressable_shards:", grad_coord_from_cv.addressable_shards)
        grad_val += grad_coord_from_cv

        return grad_val

    def _grad_freeE_jit(self, coord, beta_array, beta, f_array, mask):
        its_state = ITSState(
            beta_array=beta_array,
            f_array=f_array,
            mask=mask,
            anneal_factor=None,
            total_steps=None,
        )
        kernel_state = self._kernel_state_from_kernel()
        grad_val, next_kernel_state = self.grad_freeE_from_state_with_kernel_state(
            coord,
            its_state,
            beta,
            kernel_state,
        )
        self._sync_kernel_from_state(next_kernel_state)
        return grad_val

    def grad_freeE_(
        self, coord, beta_array, beta, f_array, mask
    ):  # TODO: check and fix
        self._warn_deprecated_mutation_api(
            "DynamicITSSampler.grad_freeE_",
            "DynamicITSSampler._grad_freeE_jit",
        )
        return self._grad_freeE_core_(
            coord,
            beta_array,
            beta,
            f_array,
            mask,
            self.system.nbList,
            self.kernel.bandwidth,
        )

    def _grad_freeE_from_state_core_(self, coord, its_state, beta, nb_list, kernel_bw):
        return self._grad_freeE_core_(
            coord,
            its_state.beta_array,
            beta,
            its_state.f_array,
            its_state.mask,
            nb_list,
            kernel_bw,
        )

    def _grad_freeE_from_state_jit(self, coord, its_state, beta):
        kernel_state = self._kernel_state_from_kernel()
        grad_val, next_kernel_state = self.grad_freeE_from_state_with_kernel_state(
            coord,
            its_state,
            beta,
            kernel_state,
        )
        self._sync_kernel_from_state(next_kernel_state)
        return grad_val

    def _grad_freeE_from_state_(self, coord, its_state, beta):
        self._warn_deprecated_mutation_api(
            "DynamicITSSampler._grad_freeE_from_state_",
            "DynamicITSSampler._grad_freeE_from_state_jit",
        )
        return self._grad_freeE_from_state_core_(
            coord, its_state, beta, self.system.nbList, self.kernel.bandwidth
        )

    def _grad_freeE_from_state_with_kernel_state_core_(
        self, coord, its_state, beta, nb_list, kernel_state, kernel_bw
    ):
        beta_array = jnp.where(its_state.beta_array < 1e-8, 1e-8, its_state.beta_array)
        potential_array = self.potential_vmap(beta_array, coord)
        coeff = its_grad_coeff_per_sample(
            beta, beta_array, potential_array, its_state.f_array, its_state.mask
        )
        coord_shape = coord.shape
        coeff = coeff.reshape(-1, *([1] * (len(coord_shape) - 1)))
        kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        grad_val = self.system_grad(coord, **kwargs) * beta
        grad_val = coeff * grad_val
        kernel_coordinate = self.sample_cv(coord)
        grad_cv, next_kernel_state = _kernel_grad_with_state(
            self.kernel, kernel_coordinate, kernel_state, kernel_bw
        )
        grad_coord_from_cv = self.pullback_cv(coord, grad_cv)
        grad_val += grad_coord_from_cv
        return grad_val, next_kernel_state

    def _grad_freeE_from_state_with_kernel_state_jit(
        self, coord, its_state, beta, kernel_state
    ):
        return self.grad_freeE_from_state_with_kernel_state_core(
            coord,
            its_state,
            beta,
            self.system.nbList,
            kernel_state,
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth,
        )

    def _grad_freeE_from_state_with_kernel_state_(
        self, coord, its_state, beta, kernel_state
    ):
        return self._grad_freeE_from_state_with_kernel_state_core_(
            coord,
            its_state,
            beta,
            self.system.nbList,
            kernel_state,
            kernel_state.bandwidth
            if kernel_state is not None
            else self.kernel.bandwidth,
        )

    def _ref_energy_core_(self, coord, beta, nb_list):
        # Avoid closing over sharded/non-addressable arrays by passing coord explicitly
        system_kwargs = _build_system_kwargs(beta=beta, nbList=nb_list)
        if "beta" in system_kwargs:
            del system_kwargs["beta"]
        return self.system.potential(beta, coord=coord, **system_kwargs)

    def _ref_energy(self, coord, beta, **kwargs):
        # kwargs kept for compatibility with existing call sites.
        return self.ref_energy_core(coord, beta, self.system.nbList)

    def ref_energy_(self, coord, beta, **kwargs):
        return self._ref_energy_core_(coord, beta, self.system.nbList)

    def _ref_energy_from_state_core_(self, coord, beta, its_state, nb_list):
        del its_state
        return self._ref_energy_core_(coord, beta, nb_list)

    def _ref_energy_from_state(self, coord, beta, its_state, **kwargs):
        del kwargs
        return self.ref_energy_from_state_core(
            coord, beta, its_state, self.system.nbList
        )

    def dynamic_its_update_(
        self, energy, current_beta, beta_array, f_array, ema_alpha, mask
    ):
        """更新ITS参数"""
        # 重塑能量为(1, n_replicas)
        betat_U_its = current_beta * energy
        betat_U_its = jnp.transpose(betat_U_its)

        # 计算所有温度下的能量 (num_temperatures, n_replicas)
        betak_U = jnp.expand_dims(beta_array, 1) * jnp.transpose(energy)

        # 计算logsumexp(beta_t * U - beta_k * U)
        sum_k_t = logsumexp(betat_U_its - betak_U, axis=1)

        # 计算f_k_t，相对于第一个温度
        f_k_t = sum_k_t - sum_k_t[0]

        # 初始化f参数
        init_f_mask = (mask == 1).astype(jnp.int32)
        current_f = init_f_mask * f_k_t + (1 - init_f_mask) * f_array

        # 使用指数滑动平均更新f参数
        fk = ema_alpha * current_f + (1 - ema_alpha) * f_k_t

        # 对未使用的温度进行padding
        pad_f_mask = (mask < 1).astype(jnp.int32)
        fk_final = pad_f_mask * 1e4 + (1 - pad_f_mask) * fk
        beta_final = (1 - pad_f_mask) * beta_array

        return fk_final, beta_final

    def dynamic_anneal_(self, energy, anneal_factor_in, kl_div_threshold):
        """
        动态退火控制 (使用 lax.while 重写，保持原始签名)
        """
        af_now = anneal_factor_in

        # 1. 计算循环的初始状态 (init_val)
        initial_af_increment = jnp.minimum(0.1, (1.0 - af_now))
        initial_af_next, initial_kld = compute_next_anneal_factor(
            af_now, initial_af_increment, energy
        )
        init_state = (initial_af_increment, initial_af_next, initial_kld)

        # 2. 定义条件函数 (cond_fun)，定义在方法内部以捕获外部变量
        def cond_fun(state):
            _, _, kld = state
            return kld > kl_div_threshold

        # 3. 定义循环体函数 (body_fun)，同样定义在方法内部
        def body_fun(state):
            af_increment, _, _ = state

            new_af_increment = af_increment / 2.0
            new_af_next, new_kld = compute_next_anneal_factor(
                af_now, new_af_increment, energy
            )

            return (new_af_increment, new_af_next, new_kld)

        # 4. 执行 lax.while 循环
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # 5. 从最终状态中提取并返回结果
        _, final_af_next, final_kld = final_state
        return final_af_next, final_kld

    def sample_ensemble_mixing(
        self,
        verbose=False,
        *,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        (
            checkpoint_every,
            checkpoint_interval_s,
            checkpoint_rank,
            checkpoint_world_size,
            checkpoint_global_shape,
            checkpoint_commit,
            checkpoint_commit_rank,
            checkpoint_wait_timeout_s,
            checkpoint_sharding_spec,
            checkpoint_mesh_axes,
            checkpoint_meta,
            checkpoint_at_end,
        ) = self._apply_checkpoint_config(
            checkpoint_config,
            checkpoint_every=None,
            checkpoint_interval_s=None,
            checkpoint_rank=0,
            checkpoint_world_size=1,
            checkpoint_global_shape=None,
            checkpoint_commit=True,
            checkpoint_commit_rank=0,
            checkpoint_wait_timeout_s=30.0,
            checkpoint_sharding_spec=None,
            checkpoint_mesh_axes=None,
            checkpoint_meta=None,
            checkpoint_at_end=True,
        )
        """实现系综混合采样"""
        anneal_max = 0.999
        its_state = self.get_its_state()
        kernel_state = self._kernel_state_from_kernel()
        anneal_factor = (
            its_state.anneal_factor
            if its_state.anneal_factor is not None
            else jnp.asarray(self.anneal_factor)
        )
        last_checkpoint_time = time.perf_counter()
        warned_max_temps = False

        # 保存当前坐标的能量（显式传入坐标，避免闭包捕获分片数组）
        energy = self.ref_energy_from_state(
            self.system.coordinate, self.beta, its_state
        )
        if _to_host_control_scalar(jnp.all(jnp.isfinite(energy))) <= 0.5:
            raise FloatingPointError(
                "sample_ensemble_mixing: initial energy contains non-finite values."
            )

        # 为当前混合过程创建日志记录列表
        iteration = 0

        if verbose:
            print("\n===== 开始系综混合过程 =====")
            print(f"初始退火因子: {anneal_factor:.4f}, 目标: {anneal_max:.4f}")

        # calculation must be done on every process.
        _mean_e, _std_e = _safe_global_mean_std(energy)

        if verbose:
            print(f"初始能量均值: {_mean_e:.6f}, 标准差: {_std_e:.6f}")

        start_time = time.time()

        while anneal_factor < anneal_max:
            iteration += 1
            iter_start_time = time.time()

            # 设置当前beta
            beta_set = anneal_factor * self.beta
            beta_set_h = float(_to_host_scalar(beta_set))
            temp_set_h = (
                1.0 / (self.kB * beta_set_h) if beta_set_h > 0 else float("inf")
            )

            # 更新温度数组与mask
            current_beta_idx = jnp.sum(its_state.mask > 0)
            current_beta_idx_h = int(_to_host_control_scalar(current_beta_idx))
            at_capacity = current_beta_idx_h >= int(self.max_num_temperatures)

            if (
                current_beta_idx_h > 0
                and jnp.max(its_state.beta_array) > 0.99 * self.beta
            ):
                # 所有温度都已遍历过，增加计数
                next_mask = its_state.mask.at[:current_beta_idx].set(
                    its_state.mask[:current_beta_idx] + 1
                )
                its_state = replace(its_state, mask=next_mask)
                temp_status = f"更新已有温度 ({current_beta_idx_h} 个)"
            elif at_capacity:
                # 达到温度离散数量上限：不再追加，改为“挤压/更新”最高 beta，
                # 保证仍能推进到 target beta（避免数组越界与永远达不到低温端）
                if not warned_max_temps:
                    warnings.warn(
                        "DynamicITSSampler reached max_num_temperatures="
                        f"{int(self.max_num_temperatures)} before approaching target beta. "
                        "Consider increasing max_num_temperatures to improve temperature resolution.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_max_temps = True
                last = int(self.max_num_temperatures) - 1
                if last >= 0:
                    next_beta_array = its_state.beta_array.at[last].set(beta_set)
                    next_mask = its_state.mask.at[: int(self.max_num_temperatures)].set(
                        its_state.mask[: int(self.max_num_temperatures)] + 1
                    )
                    its_state = replace(
                        its_state, beta_array=next_beta_array, mask=next_mask
                    )
                temp_status = (
                    f"已达上限 {int(self.max_num_temperatures)} 个，"
                    f"更新最高β={beta_set_h:.4e} (T={temp_set_h:.2f})"
                )
            else:
                # 追加新温度
                next_beta_array = its_state.beta_array.at[current_beta_idx].set(
                    beta_set
                )
                next_mask = its_state.mask.at[: current_beta_idx + 1].set(
                    its_state.mask[: current_beta_idx + 1] + 1
                )
                its_state = replace(
                    its_state, beta_array=next_beta_array, mask=next_mask
                )
                temp_status = (
                    f"添加新温度 β={beta_set_h:.4e} (T={temp_set_h:.2f}) "
                    f"(共 {current_beta_idx_h + 1} 个)"
                )

            # 更新f_array和beta_array
            next_f_array, next_beta_array = self.dynamic_its_update(
                energy,
                beta_set,
                its_state.beta_array,
                its_state.f_array,
                self.ema_alpha,
                its_state.mask,
            )
            its_state = replace(
                its_state, f_array=next_f_array, beta_array=next_beta_array
            )

            # 运行优化器
            opt_state = self.optimizer.init(self.system.coordinate)
            self.kernel.bandwidth = 0.1
            (
                self.system.coordinate,
                opt_state,
                kernel_state,
                grad_norms,
                step_count,
                _,
            ) = self._run_pregrad_opt_loop(
                self.system.coordinate,
                opt_state,
                kernel_state,
                max_steps=self.max_steps,
                grad_norm_threshold=self.grad_norm_threshold_mixing,
                grad_fn=lambda coord, ks: self.grad_freeE_from_state_with_kernel_state(
                    coord, its_state, self.beta, ks
                ),
            )
            its_state = replace(
                its_state,
                total_steps=(its_state.total_steps + step_count)
                if its_state.total_steps is not None
                else jnp.asarray(step_count),
            )

            # 更新能量
            energy = self.ref_energy_from_state(
                self.system.coordinate, self.beta, its_state
            )
            if _to_host_control_scalar(jnp.all(jnp.isfinite(energy))) <= 0.5:
                warnings.warn(
                    "sample_ensemble_mixing detected non-finite energy after optimizer "
                    "step; stopping mixing early and keeping last finite state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            # 动态更新退火因子
            old_anneal_factor = anneal_factor
            _af_host = float(_to_host_control_scalar(anneal_factor))
            kl_div_threshold = float(self.kl_div_threshold) * (
                0.1 if _af_host < 0.5 else 1.0
            )
            anneal_factor, kld = self.dynamic_anneal(
                energy, anneal_factor, kl_div_threshold
            )
            its_state = replace(its_state, anneal_factor=anneal_factor)

            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            iter_time = time.time() - iter_start_time

            # 打印进度信息
            _mean_e, _std_e = _safe_global_mean_std(energy)
            if verbose:
                _af0 = float(_to_host_scalar(old_anneal_factor))
                _af1 = float(_to_host_scalar(anneal_factor))
                _kld = float(_to_host_scalar(kld))
                _avg_gn = float(avg_grad_norm)
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                    f"混合迭代 {iteration} - AF: {_af0:.4f} → {_af1:.4f}, "
                    f"KL: {_kld:.4e}, 温度: {temp_status}, "
                    f"能量: {_mean_e:.6f}±{_std_e:.6f}, "
                    f"梯度: {_avg_gn:.4e}, 总步数: {int(_to_host_control_scalar(its_state.total_steps)) if its_state.total_steps is not None else 0}, 耗时: {iter_time:.2f}秒"
                )

            if checkpoint_store is not None:
                total = int(_to_host_control_scalar(its_state.total_steps))
                is_interval_step = (
                    checkpoint_every is not None
                    and checkpoint_every > 0
                    and total % int(checkpoint_every) == 0
                )
                is_interval_time = (
                    checkpoint_interval_s is not None
                    and (time.perf_counter() - last_checkpoint_time)
                    >= float(checkpoint_interval_s)
                )
                if is_interval_step or is_interval_time:
                    self.set_its_state(its_state)
                    self._sync_kernel_from_state(kernel_state)
                    self.save_checkpoint(
                        store=checkpoint_store,
                        step=total,
                        rank=checkpoint_rank,
                        world_size=checkpoint_world_size,
                        global_shape=checkpoint_global_shape,
                        commit=checkpoint_commit,
                        commit_rank=checkpoint_commit_rank,
                        wait_timeout_s=checkpoint_wait_timeout_s,
                        sharding_spec=checkpoint_sharding_spec,
                        mesh_axes=checkpoint_mesh_axes,
                        meta={
                            "phase": "dynamic_mixing",
                            "iteration": int(iteration),
                            **(checkpoint_meta or {}),
                        },
                    )
                    last_checkpoint_time = time.perf_counter()

        # 记录总时间
        total_time = time.time() - start_time
        if verbose:
            print(f"完成系综混合，总共 {iteration} 次迭代，总耗时 {total_time:.2f}秒")

        self.set_its_state(its_state)
        self._sync_kernel_from_state(kernel_state)
        if checkpoint_store is not None and checkpoint_at_end:
            final_step = int(_to_host_control_scalar(its_state.total_steps))
            self.save_checkpoint(
                store=checkpoint_store,
                step=final_step,
                rank=checkpoint_rank,
                world_size=checkpoint_world_size,
                global_shape=checkpoint_global_shape,
                commit=checkpoint_commit,
                commit_rank=checkpoint_commit_rank,
                wait_timeout_s=checkpoint_wait_timeout_s,
                sharding_spec=checkpoint_sharding_spec,
                mesh_axes=checkpoint_mesh_axes,
                meta={"phase": "dynamic_mixing_end", **(checkpoint_meta or {})},
            )
        return self.system.coordinate

    def sample_ensemble_annealing(
        self,
        verbose=False,
        *,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        (
            checkpoint_every,
            checkpoint_interval_s,
            checkpoint_rank,
            checkpoint_world_size,
            checkpoint_global_shape,
            checkpoint_commit,
            checkpoint_commit_rank,
            checkpoint_wait_timeout_s,
            checkpoint_sharding_spec,
            checkpoint_mesh_axes,
            checkpoint_meta,
            checkpoint_at_end,
        ) = self._apply_checkpoint_config(
            checkpoint_config,
            checkpoint_every=None,
            checkpoint_interval_s=None,
            checkpoint_rank=0,
            checkpoint_world_size=1,
            checkpoint_global_shape=None,
            checkpoint_commit=True,
            checkpoint_commit_rank=0,
            checkpoint_wait_timeout_s=30.0,
            checkpoint_sharding_spec=None,
            checkpoint_mesh_axes=None,
            checkpoint_meta=None,
            checkpoint_at_end=True,
        )
        """实现系综退火采样"""
        its_state = self.get_its_state()
        kernel_state = self._kernel_state_from_kernel()
        last_checkpoint_time = time.perf_counter()
        # 获取当前能量（显式传入坐标）
        energy = self.ref_energy_from_state(
            self.system.coordinate, self.beta, its_state
        )
        if _to_host_control_scalar(jnp.all(jnp.isfinite(energy))) <= 0.5:
            raise FloatingPointError(
                "sample_ensemble_annealing: initial energy contains non-finite values."
            )

        # 获取当前使用的温度数量
        num_ensemble = jnp.sum(its_state.mask > 0)
        num_ensemble_h = int(_to_host_control_scalar(num_ensemble))

        _mean_e, _std_e = _safe_global_mean_std(energy)
        if verbose:
            print("\n===== 开始系综退火过程 =====")
            print(f"当前活跃温度数量: {num_ensemble_h}")
            print(f"初始能量均值: {_mean_e:.6f}, 标准差: {_std_e:.6f}")

        start_time = time.time()

        for i in range(num_ensemble_h - 1):
            temp_start_time = time.time()

            # 记录当前温度的信息
            beta_current_h = float(_to_host_scalar(its_state.beta_array[i]))
            beta_next_h = float(_to_host_scalar(its_state.beta_array[i + 1]))
            temp_current = (
                1.0 / (self.kB * beta_current_h) if beta_current_h > 0 else float("inf")
            )
            temp_next = (
                1.0 / (self.kB * beta_next_h) if beta_next_h > 0 else float("inf")
            )

            # 重置当前温度的mask
            next_mask = its_state.mask.at[i].set(0)
            next_beta = its_state.beta_array.at[i].set(0)

            # 增加剩余温度的mask计数
            next_mask = next_mask.at[i + 1 : num_ensemble].set(
                next_mask[i + 1 : num_ensemble] + 1
            )
            its_state = replace(its_state, mask=next_mask, beta_array=next_beta)

            # 更新f_array和beta_array
            next_f_array, next_beta_array = self.dynamic_its_update(
                energy,
                self.beta,
                its_state.beta_array,
                its_state.f_array,
                self.ema_alpha,
                its_state.mask,
            )
            its_state = replace(
                its_state, f_array=next_f_array, beta_array=next_beta_array
            )

            # 运行优化器
            opt_state = self.optimizer.init(self.system.coordinate)
            self.kernel.bandwidth = 0.1
            (
                self.system.coordinate,
                opt_state,
                kernel_state,
                grad_norms,
                step_count,
                _,
            ) = self._run_pregrad_opt_loop(
                self.system.coordinate,
                opt_state,
                kernel_state,
                max_steps=self.max_steps,
                grad_norm_threshold=self.grad_norm_threshold_annealing,
                grad_fn=lambda coord, ks: self.grad_freeE_from_state_with_kernel_state(
                    coord, its_state, self.beta, ks
                ),
            )
            its_state = replace(
                its_state,
                total_steps=(its_state.total_steps + step_count)
                if its_state.total_steps is not None
                else jnp.asarray(step_count),
            )

            # 更新能量
            energy = self.ref_energy_from_state(
                self.system.coordinate, self.beta, its_state
            )
            if _to_host_control_scalar(jnp.all(jnp.isfinite(energy))) <= 0.5:
                warnings.warn(
                    "sample_ensemble_annealing detected non-finite energy after optimizer "
                    "step; stopping annealing early and keeping last finite state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            temp_time = time.time() - temp_start_time

            # 打印进度信息
            _mean_e, _std_e = _safe_global_mean_std(energy)
            if verbose:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                    f"退火温度 {i + 1}/{num_ensemble_h - 1} - "
                    f"Beta: {beta_current_h:.4e} → {beta_next_h:.4e}, "
                    f"温度: {temp_current:.2f} → {temp_next:.2f}, "
                    f"能量: {_mean_e:.6f}±{_std_e:.6f}, "
                    f"梯度: {avg_grad_norm:.4e}, 总步数: {int(_to_host_control_scalar(its_state.total_steps)) if its_state.total_steps is not None else 0}, 耗时: {temp_time:.2f}秒"
                )

            if checkpoint_store is not None:
                total = int(_to_host_control_scalar(its_state.total_steps))
                is_interval_step = (
                    checkpoint_every is not None
                    and checkpoint_every > 0
                    and total % int(checkpoint_every) == 0
                )
                is_interval_time = (
                    checkpoint_interval_s is not None
                    and (time.perf_counter() - last_checkpoint_time)
                    >= float(checkpoint_interval_s)
                )
                if is_interval_step or is_interval_time:
                    self.set_its_state(its_state)
                    self._sync_kernel_from_state(kernel_state)
                    self.save_checkpoint(
                        store=checkpoint_store,
                        step=total,
                        rank=checkpoint_rank,
                        world_size=checkpoint_world_size,
                        global_shape=checkpoint_global_shape,
                        commit=checkpoint_commit,
                        commit_rank=checkpoint_commit_rank,
                        wait_timeout_s=checkpoint_wait_timeout_s,
                        sharding_spec=checkpoint_sharding_spec,
                        mesh_axes=checkpoint_mesh_axes,
                        meta={
                            "phase": "dynamic_annealing",
                            "iteration": int(i),
                            **(checkpoint_meta or {}),
                        },
                    )
                    last_checkpoint_time = time.perf_counter()

        # 记录总时间
        total_time = time.time() - start_time
        if verbose:
            print(
                f"完成系综退火，总共处理 {num_ensemble - 1} 个温度，总耗时 {total_time:.2f}秒"
            )

        self.set_its_state(its_state)
        self._sync_kernel_from_state(kernel_state)
        if checkpoint_store is not None and checkpoint_at_end:
            final_step = int(_to_host_control_scalar(its_state.total_steps))
            self.save_checkpoint(
                store=checkpoint_store,
                step=final_step,
                rank=checkpoint_rank,
                world_size=checkpoint_world_size,
                global_shape=checkpoint_global_shape,
                commit=checkpoint_commit,
                commit_rank=checkpoint_commit_rank,
                wait_timeout_s=checkpoint_wait_timeout_s,
                sharding_spec=checkpoint_sharding_spec,
                mesh_axes=checkpoint_mesh_axes,
                meta={"phase": "dynamic_annealing_end", **(checkpoint_meta or {})},
            )
        return self.system.coordinate

    def run(
        self,
        num_its_iterations=1,
        verbose=False,
        verbose_pace=500,
        *,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        """运行完整的DynamicITS采样过程"""
        start_time = time.time()

        if verbose:
            print("\n===== 开始DynamicITS采样 =====")
            print(
                f"最小温度: {1.0 / (self.kB * self.beta):.2f}K, 最大温度: {1.0 / (self.kB * self.beta_array[0]):.2f}K"
            )
            print(f"ITS迭代次数: {num_its_iterations}")

        # 运行系综混合过程
        for i in range(num_its_iterations):
            if verbose:
                print(f"\n===== 系综混合轮次 {i + 1}/{num_its_iterations} =====")
            self.sample_ensemble_mixing(
                verbose=verbose,
                checkpoint_store=checkpoint_store,
                checkpoint_config=checkpoint_config,
            )

        # 运行系综退火过程
        final_coordinates = self.sample_ensemble_annealing(
            verbose=verbose,
            checkpoint_store=checkpoint_store,
            checkpoint_config=checkpoint_config,
        )

        total_time = time.time() - start_time
        if verbose:
            print("\n===== DynamicITS采样完成 =====")
            print(f"总步数: {self.total_steps}")
            print(f"总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")
            _final_beta = float(_to_host_scalar(self.beta))
            print(f"最终Beta: {_final_beta}")

        return final_coordinates
