from __future__ import annotations

from typing import Callable

import jax
import numpy as np
from jax import numpy as jnp

from .state import KernelState


def to_host_scalar(x):
    """Convert scalar-like JAX values for logging only."""
    try:
        shards = getattr(x, "addressable_shards", None)
        if shards is not None and len(shards) > 0:
            return float(np.asarray(jax.device_get(shards[0].data)))
    except Exception:
        pass
    return float(np.asarray(jax.device_get(x)))


def to_host_control_scalar(x):
    """Convert scalar-like JAX values for host-side control-flow decisions."""
    x_arr = jnp.asarray(x)
    if x_arr.ndim != 0:
        raise ValueError(
            "Control scalar must be 0-D. Reduce to a scalar before control-flow use."
        )
    try:
        host = np.asarray(jax.device_get(x_arr))
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch control scalar to host. Ensure the value is a globally "
            "consistent reduced scalar before using it in control flow."
        ) from exc
    if host.shape != ():
        raise ValueError(
            "Control scalar conversion expects a scalar result. "
            "Apply an explicit reduction first."
        )
    return float(host)


def validate_initial_coord_for_multiprocess(coord):
    """One-time startup validation for multi-process coordinate layout."""
    if jax.process_count() <= 1:
        return
    coord_arr = jnp.asarray(coord)
    sharding = getattr(coord_arr, "sharding", None)
    if sharding is None:
        raise ValueError(
            "In multi-process mode, initial coord must be a global jax.Array with sharding."
        )
    shards = getattr(coord_arr, "addressable_shards", None)
    if shards is None or len(shards) == 0:
        raise ValueError(
            "In multi-process mode, initial coord has no addressable shards on this process."
        )


def kernel_state_from_kernel(kernel):
    if hasattr(kernel, "get_state"):
        return kernel.get_state()
    bandwidth = getattr(kernel, "bandwidth", None)
    if bandwidth is None:
        raise ValueError("Kernel must provide `bandwidth` or `get_state()`")
    return KernelState(
        bandwidth=bandwidth,
        rff_key=getattr(kernel, "rff_key", None),
        aux=getattr(kernel, "gaussian_exp", None),
    )


def sync_kernel_from_state(kernel, kernel_state):
    if hasattr(kernel, "_set_state_fast"):
        kernel._set_state_fast(kernel_state)
        return
    if hasattr(kernel, "set_state"):
        kernel.set_state(kernel_state)
        return
    if isinstance(kernel_state, KernelState):
        kernel.bandwidth = kernel_state.bandwidth
        if hasattr(kernel, "rff_key") and kernel_state.rff_key is not None:
            kernel.rff_key = kernel_state.rff_key
        if hasattr(kernel, "gaussian_exp") and kernel_state.aux is not None:
            kernel.gaussian_exp = kernel_state.aux


def run_chunked_host_loop(
    *,
    total_steps: int,
    update_pace: int,
    state,
    run_chunk: Callable,
    refresh_runtime: Callable | None = None,
):
    """Shared host loop for chunked execution with optional host refresh."""
    if update_pace <= 0 or refresh_runtime is None:
        return run_chunk(state, int(total_steps))

    remaining = int(total_steps)
    cur_state = state
    while remaining > 0:
        n = min(int(update_pace), remaining)
        cur_state = run_chunk(cur_state, int(n))
        cur_state = refresh_runtime(cur_state)
        remaining -= n
    return cur_state


def run_pregrad_opt_loop(
    *,
    coordinate,
    opt_state,
    kernel_state,
    max_steps: int,
    grad_norm_threshold: float,
    grad_fn: Callable,
    opt_update: Callable,
    update_pace: int,
    refresh_runtime: Callable,
    norm_fn: Callable,
    control_scalar_fn: Callable = to_host_control_scalar,
):
    """Shared host-side optimization loop (pre-grad style)."""

    def _all_finite_tree(x) -> bool:
        leaves = jax.tree_util.tree_leaves(x)
        for leaf in leaves:
            if leaf is None:
                continue
            try:
                arr = jnp.asarray(leaf)
            except Exception:
                continue
            # bool/int leaves are always finite for our control purposes.
            if arr.dtype.kind in ("b", "i", "u"):
                continue
            if control_scalar_fn(jnp.all(jnp.isfinite(arr))) <= 0.5:
                return False
        return True

    step = 0
    grad_norms = []
    last_finite_coordinate = coordinate
    last_finite_opt_state = opt_state
    last_finite_kernel_state = kernel_state

    grad_value, kernel_state = grad_fn(coordinate, kernel_state)
    grad_norm = control_scalar_fn(norm_fn(grad_value))
    if not (_all_finite_tree(grad_value) and np.isfinite(grad_norm)):
        return (
            last_finite_coordinate,
            last_finite_opt_state,
            last_finite_kernel_state,
            grad_norms,
            step,
            grad_norm,
        )

    while step < max_steps and np.isfinite(grad_norm) and grad_norm > grad_norm_threshold:
        if step % update_pace == 0:
            coordinate_new, kernel_state_new, _ = refresh_runtime(coordinate)
            if not (_all_finite_tree(coordinate_new) and _all_finite_tree(kernel_state_new)):
                break
            coordinate = coordinate_new
            kernel_state = kernel_state_new
            last_finite_coordinate = coordinate
            last_finite_kernel_state = kernel_state

        grad_value, kernel_state = grad_fn(coordinate, kernel_state)
        grad_norm = control_scalar_fn(norm_fn(grad_value))
        if not (
            _all_finite_tree(grad_value)
            and _all_finite_tree(kernel_state)
            and np.isfinite(grad_norm)
        ):
            break

        grad_norms.append(grad_norm)
        coordinate_new, opt_state_new = opt_update(coordinate, grad_value, opt_state)
        if not (_all_finite_tree(coordinate_new) and _all_finite_tree(opt_state_new)):
            break
        coordinate = coordinate_new
        opt_state = opt_state_new
        last_finite_coordinate = coordinate
        last_finite_opt_state = opt_state
        last_finite_kernel_state = kernel_state
        step += 1

    if not (
        _all_finite_tree(coordinate)
        and _all_finite_tree(opt_state)
        and _all_finite_tree(kernel_state)
    ):
        coordinate = last_finite_coordinate
        opt_state = last_finite_opt_state
        kernel_state = last_finite_kernel_state

    return coordinate, opt_state, kernel_state, grad_norms, step, grad_norm


def run_postgrad_opt_loop(
    *,
    coordinate,
    opt_state,
    kernel_state,
    num_steps: int,
    grad_fn: Callable,
    opt_update: Callable,
    update_pace: int,
    refresh_runtime: Callable,
    norm_fn: Callable,
    grad_norm_threshold: float | None = None,
    on_step: Callable | None = None,
    control_scalar_fn: Callable = to_host_control_scalar,
):
    """Shared host-side optimization loop (post-grad style)."""
    step_count = 0
    last_grad_norm = 0.0
    stop_code = 0
    for step in range(int(num_steps)):
        if step % update_pace == 0:
            coordinate, kernel_state, _ = refresh_runtime(coordinate)
        grad_value, kernel_state = grad_fn(coordinate, kernel_state)
        coordinate, opt_state = opt_update(coordinate, grad_value, opt_state)
        step_count += 1

        has_nan = control_scalar_fn(jnp.any(jnp.isnan(grad_value))) > 0.5
        grad_norm = control_scalar_fn(norm_fn(grad_value))
        last_grad_norm = grad_norm

        if on_step is not None:
            on_step(step, grad_norm, kernel_state)

        if has_nan:
            stop_code = 1
            break
        if grad_norm_threshold is not None and grad_norm < grad_norm_threshold:
            stop_code = 2
            break
    return coordinate, opt_state, kernel_state, step_count, stop_code, last_grad_norm


def run_step_state_loop(
    *,
    indices,
    state,
    step_fn: Callable,
    pre_step_fn: Callable | None = None,
    on_step_fn: Callable | None = None,
):
    """Shared host-side step loop for stateful samplers."""
    cur_state = state
    for i in indices:
        if pre_step_fn is not None:
            cur_state = pre_step_fn(i, cur_state)
        cur_state, metrics = step_fn(cur_state)
        if on_step_fn is not None:
            on_step_fn(i, cur_state, metrics)
    return cur_state
