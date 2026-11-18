"""Target-temperature residue-block Metropolis moves on the dihedral torus.

Each round proposes, for every replica, a new value of one residue block: a
wrapped Gaussian displacement at one of the local widths or, with the uniform
probability, an independent uniform replacement; proposals are accepted by
the Metropolis rule at the target temperature, so the move leaves the target
invariant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.ecepp import wrap_angles


def annealed_exploration_candidate_count(
    activity: float,
    replica_count: int,
    *,
    batching: str = "dense",
) -> int:
    """Return the static proposal batch for the selected execution mode.

    ``dense`` preserves the original independent activity clocks.  The
    experimental ``power-of-two`` mode bounds JAX compilation shapes while
    keeping proposal-energy work below twice the expected number of active
    replicas; conditional thinning recovers the exact per-replica marginal
    attempt probability.
    """

    probability = float(activity)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("exploration activity must lie in [0, 1]")
    if replica_count < 1:
        raise ValueError("exploration replica count must be positive")
    if batching not in ("dense", "power-of-two"):
        raise ValueError(
            "exploration candidate batching must be dense or power-of-two"
        )
    if probability == 0.0:
        return 0
    if batching == "dense":
        return replica_count
    required = min(
        replica_count,
        max(1, int(np.ceil(probability * replica_count))),
    )
    return min(replica_count, 1 << (required - 1).bit_length())


def make_annealed_block_exploration(
    potential,
    *,
    beta: float,
    block_ids,
    n_steps: int,
    uniform_probability: float,
    candidate_batching: str = "dense",
):
    """Build state-agnostic hot joint-block exploration.

    Every active proposal selects one residue block uniformly and changes all
    of its free dihedrals together.  A fixed mixture combines symmetric
    wrapped-Gaussian random walks with symmetric uniform replacement on the
    selected block.  Metropolis acceptance uses an annealed inverse
    temperature; while hot this is deliberately not target invariant.
    """

    if n_steps < 1:
        raise ValueError("annealed block exploration steps must be positive")
    if not 0.0 <= uniform_probability <= 1.0:
        raise ValueError("uniform exploration probability must be in [0, 1]")
    if candidate_batching not in ("dense", "power-of-two"):
        raise ValueError(
            "exploration candidate batching must be dense or power-of-two"
        )
    block_ids = jnp.asarray(block_ids, dtype=jnp.int32)
    n_blocks = int(np.max(np.asarray(block_ids))) + 1

    def explore_bucket(
        coordinates,
        key,
        activity,
        inverse_temperature_scale,
        scales,
        *,
        candidate_count,
    ):
        current_energy = potential(coordinates)[:, 0]
        accepted = jnp.zeros((), dtype=jnp.int32)
        attempted = jnp.zeros((), dtype=jnp.int32)

        def body(_, carry):
            (
                coordinate,
                energy,
                rng_key,
                accepted_count,
                attempted_count,
            ) = carry
            (
                rng_key,
                activity_key,
                block_key,
                scale_key,
                mode_key,
                noise_key,
                uniform_key,
                accept_key,
            ) = jax.random.split(rng_key, 8)
            replica_count = coordinate.shape[0]
            if candidate_count == replica_count:
                selected_replicas = jnp.arange(replica_count)
                clock_key = activity_key
            else:
                selection_key, clock_key = jax.random.split(activity_key)
                offset = jax.random.randint(
                    selection_key, (), 0, replica_count
                )
                selected_replicas = (
                    offset + jnp.arange(candidate_count)
                ) % replica_count
            selected_coordinates = coordinate[selected_replicas]
            selected_energy = energy[selected_replicas]
            thinning_probability = jnp.minimum(
                jnp.asarray(1.0, dtype=coordinate.dtype),
                activity * replica_count / candidate_count,
            )
            active = (
                jax.random.uniform(clock_key, (candidate_count,))
                < thinning_probability
            )
            selected_block = jax.random.randint(
                block_key, (candidate_count,), 0, n_blocks
            )
            selected_scale = jax.random.randint(
                scale_key, (candidate_count,), 0, len(scales)
            )
            use_uniform = (
                jax.random.uniform(mode_key, (candidate_count,))
                < uniform_probability
            )
            current_angles = selected_coordinates[:, 0, :]
            gaussian_angles = wrap_angles(
                current_angles
                + jax.random.normal(
                    noise_key,
                    current_angles.shape,
                    dtype=current_angles.dtype,
                )
                * scales[selected_scale, None]
            )
            uniform_angles = jax.random.uniform(
                uniform_key,
                current_angles.shape,
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=current_angles.dtype,
            )
            proposed_angles = jnp.where(
                use_uniform[:, None], uniform_angles, gaussian_angles
            )
            mask = block_ids[None, :] == selected_block[:, None]
            proposal = jnp.where(mask, proposed_angles, current_angles)[:, None, :]
            proposal_energy = potential(proposal)[:, 0]
            log_acceptance = (
                -beta
                * inverse_temperature_scale
                * (proposal_energy - selected_energy)
            )
            metropolis_accept = jnp.log(
                jax.random.uniform(accept_key, (candidate_count,))
            ) < jnp.minimum(0.0, log_acceptance)
            accept = active & metropolis_accept
            updated_selected = jnp.where(
                accept[:, None, None], proposal, selected_coordinates
            )
            coordinate = coordinate.at[selected_replicas].set(updated_selected)
            updated_energy = jnp.where(
                accept, proposal_energy, selected_energy
            )
            energy = energy.at[selected_replicas].set(updated_energy)
            return (
                coordinate,
                energy,
                rng_key,
                accepted_count + jnp.count_nonzero(accept),
                attempted_count + jnp.count_nonzero(active),
            )

        coordinate, _, key, accepted, attempted = jax.lax.fori_loop(
            0,
            n_steps,
            body,
            (coordinates, current_energy, key, accepted, attempted),
        )
        acceptance = jnp.where(
            attempted > 0,
            accepted / attempted,
            jnp.asarray(0.0, dtype=coordinates.dtype),
        )
        return coordinate, key, acceptance, accepted, attempted

    compiled_explore_bucket = jax.jit(
        explore_bucket,
        static_argnames=("candidate_count",),
    )

    def explore(
        coordinates,
        key,
        activity,
        inverse_temperature_scale,
        scales,
    ):
        activity_value = float(np.asarray(jax.device_get(activity)))
        candidate_count = annealed_exploration_candidate_count(
            activity_value,
            int(coordinates.shape[0]),
            batching=candidate_batching,
        )
        if candidate_count == 0:
            zero_count = jnp.zeros((), dtype=jnp.int32)
            zero_acceptance = jnp.zeros((), dtype=coordinates.dtype)
            return (
                coordinates,
                key,
                zero_acceptance,
                zero_count,
                zero_count,
            )
        return compiled_explore_bucket(
            coordinates,
            key,
            jnp.asarray(activity_value, dtype=coordinates.dtype),
            inverse_temperature_scale,
            scales,
            candidate_count=candidate_count,
        )

    return explore
