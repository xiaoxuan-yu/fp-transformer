from __future__ import annotations

from typing import Any, Callable


def minimize_with_energy_fn(
    coordinates,
    energy_fn: Callable[[Any], Any],
    *,
    num_steps: int,
    learning_rate: float,
    beta: float | None = None,
    is_main_rank: bool = True,
):
    import jax
    import jax.numpy as jnp
    import optax

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(coordinates)

    def log(msg: str) -> None:
        if is_main_rank:
            print(msg)

    init_energy = energy_fn(coordinates)
    init_energy_val = jnp.mean(init_energy)
    if not jnp.isfinite(init_energy_val):
        log("[warn] initial energy is non-finite, skip minimization")
        return coordinates
    if beta is not None:
        log(
            f"[min] init E={float(init_energy_val):.6f}, beta*E={float(beta * init_energy_val):.6f}"
        )

    @jax.jit
    def min_step(coord, state):
        e, g = jax.value_and_grad(lambda c: jnp.mean(energy_fn(c)))(coord)
        updates, new_state = optimizer.update(g, state)
        new_coord = optax.apply_updates(coord, updates)
        return new_coord, new_state, e, g

    cur = coordinates
    cur_state = opt_state
    last_finite = coordinates
    for i in range(num_steps):
        cur, cur_state, energy, grad = min_step(cur, cur_state)
        if not jnp.all(jnp.isfinite(cur)):
            log(f"[warn] minimization produced non-finite coords at step {i}, rollback")
            return last_finite
        if not jnp.all(jnp.isfinite(grad)):
            log(f"[warn] minimization produced non-finite grad at step {i}, rollback")
            return last_finite
        if not jnp.isfinite(energy):
            log(f"[warn] minimization produced non-finite energy at step {i}, rollback")
            return last_finite
        last_finite = cur
        if i % 100 == 0:
            if beta is None:
                log(f"[min] step={i}/{num_steps}, energy={float(energy):.6f}")
            else:
                log(
                    f"[min] step={i}/{num_steps}, E={float(energy):.6f}, beta*E={float(beta * energy):.6f}"
                )
    return cur
