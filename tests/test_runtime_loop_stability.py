import numpy as np
import jax.numpy as jnp

from src.fp_transformer.core.runtime import run_pregrad_opt_loop


def _control_scalar(x):
    return float(np.asarray(x))


def _norm_fn(x):
    return jnp.linalg.norm(x)


def test_pregrad_loop_breaks_on_nonfinite_grad_and_keeps_last_finite_coordinate():
    coord0 = jnp.ones((4, 1, 2), dtype=jnp.float32)
    opt_state0 = {"m": jnp.zeros_like(coord0)}
    kernel_state0 = jnp.asarray(0, dtype=jnp.int32)
    call_count = {"n": 0}

    def grad_fn(coord, kernel_state):
        call_count["n"] += 1
        if call_count["n"] == 1:
            grad = jnp.ones_like(coord)
        else:
            grad = jnp.ones_like(coord) * jnp.nan
        return grad, kernel_state

    def opt_update(coord, grad, opt_state):
        del opt_state
        return coord - 0.1 * grad, {"m": grad}

    def refresh_runtime(coord):
        return coord, kernel_state0, None

    coord, opt_state, kernel_state, grad_norms, step_count, _ = run_pregrad_opt_loop(
        coordinate=coord0,
        opt_state=opt_state0,
        kernel_state=kernel_state0,
        max_steps=10,
        grad_norm_threshold=0.0,
        grad_fn=grad_fn,
        opt_update=opt_update,
        update_pace=1,
        refresh_runtime=refresh_runtime,
        norm_fn=_norm_fn,
        control_scalar_fn=_control_scalar,
    )

    assert step_count == 0
    assert len(grad_norms) == 0
    assert jnp.all(jnp.isfinite(coord))
    assert jnp.allclose(coord, coord0)
    assert jnp.all(jnp.isfinite(opt_state["m"]))
    assert jnp.all(jnp.isfinite(kernel_state))


def test_pregrad_loop_breaks_on_nonfinite_update_and_rolls_back_coordinate():
    coord0 = jnp.ones((4, 1, 2), dtype=jnp.float32)
    opt_state0 = {"m": jnp.zeros_like(coord0)}
    kernel_state0 = jnp.asarray(0, dtype=jnp.int32)

    def grad_fn(coord, kernel_state):
        return jnp.ones_like(coord), kernel_state

    def opt_update(coord, grad, opt_state):
        del coord, grad, opt_state
        bad = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return bad, {"m": bad}

    def refresh_runtime(coord):
        return coord, kernel_state0, None

    coord, _, _, grad_norms, step_count, _ = run_pregrad_opt_loop(
        coordinate=coord0,
        opt_state=opt_state0,
        kernel_state=kernel_state0,
        max_steps=10,
        grad_norm_threshold=0.0,
        grad_fn=grad_fn,
        opt_update=opt_update,
        update_pace=1,
        refresh_runtime=refresh_runtime,
        norm_fn=_norm_fn,
        control_scalar_fn=_control_scalar,
    )

    # We should abort before committing any non-finite update.
    assert step_count == 0
    assert len(grad_norms) == 1
    assert jnp.all(jnp.isfinite(coord))
    assert jnp.allclose(coord, coord0)
