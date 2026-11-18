#!/usr/bin/env python3
"""Canonical Monte Carlo reference for neutral Ala10 (SI Ala10).

Sequential-scan Metropolis: each sweep proposes an independent replacement
from [-pi, pi) for every free dihedral in turn.  Of 100000 sweeps from the coil
conformation the first 20000 are discarded and every tenth subsequent sweep is
retained (8000 canonical frames).  The initial energy is checked against the
SMMP reference stored with the system.

    python -m examples.ala10.run_mc_reference --output examples/ala10/data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.ecepp import ECEPP3System, wrap_angles

KB_KCAL_PER_MOL_K = 0.00198720425864083


def energy_function(system: ECEPP3System):
    return jax.jit(lambda angles: system.energy_variables(angles)[0])


def coil_state(system: ECEPP3System):
    values = np.asarray(system.geometry.initial_variables).copy()
    for index, name in enumerate(system.geometry.variable_names):
        if name == "phi":
            values[index] = np.deg2rad(-150.0)
        elif name == "psi":
            values[index] = np.deg2rad(150.0)
    return jnp.asarray(values)


def _proposal(angles, index, key):
    # A full [-pi, pi] proposal matches the ECEPP MUCA literature. It is also
    # equivalent to SMMP's axvr=2*pi wrapped twist.
    value = jax.random.uniform(key, (), minval=-jnp.pi, maxval=jnp.pi)
    return wrap_angles(angles.at[index].set(value))


def canonical_mc(energy_fn, initial, key, sweeps: int, beta: float):
    n_variables = initial.shape[0]

    def sweep(carry, _):
        angles, energy, key, accepted = carry
        key, sweep_key = jax.random.split(key)
        keys = jax.random.split(sweep_key, 2 * n_variables)

        def update(state, item):
            angles, energy, accepted = state
            index, proposal_key, accept_key = item
            proposal = _proposal(angles, index, proposal_key)
            proposal_energy = energy_fn(proposal)
            log_ratio = -beta * (proposal_energy - energy)
            accept = jnp.log(jax.random.uniform(accept_key, ())) < log_ratio
            return (
                jnp.where(accept, proposal, angles),
                jnp.where(accept, proposal_energy, energy),
                accepted + accept,
            ), None

        (angles, energy, accepted), _ = jax.lax.scan(
            update,
            (angles, energy, accepted),
            (jnp.arange(n_variables), keys[:n_variables], keys[n_variables:]),
        )
        return (angles, energy, key, accepted), (angles, energy)

    initial_energy = energy_fn(initial)
    final, trajectory = jax.lax.scan(
        sweep,
        (initial, initial_energy, key, jnp.asarray(0)),
        None,
        length=sweeps,
    )
    return trajectory, final[3] / (sweeps * n_variables)


def helicity(angles, variable_names):
    names = np.asarray(variable_names)
    phi = np.rad2deg(np.asarray(angles)[:, names == "phi"])
    psi = np.rad2deg(np.asarray(angles)[:, names == "psi"])
    alpha = (
        (phi >= -100.0)
        & (phi <= -30.0)
        & (psi >= -80.0)
        & (psi <= -5.0)
    )
    return alpha.mean(axis=1)


def transition_metrics(helicity_values, sweep_offset: int = 0):
    state = np.where(
        helicity_values >= 0.7, 1, np.where(helicity_values <= 0.3, -1, 0)
    )
    committed = state[state != 0]
    transitions = (
        int(np.count_nonzero(committed[1:] != committed[:-1]))
        if len(committed) > 1
        else 0
    )
    helix = np.flatnonzero(state == 1)
    return {
        "helicity_mean": float(np.mean(helicity_values)),
        "helicity_min": float(np.min(helicity_values)),
        "helicity_max": float(np.max(helicity_values)),
        "helix_frame_fraction": float(np.mean(state == 1)),
        "coil_frame_fraction": float(np.mean(state == -1)),
        "helix_coil_transitions": transitions,
        "first_helix_sweep": (
            int(sweep_offset + helix[0] + 1) if len(helix) else None
        ),
    }


def energy_round_trip_metrics(energies, low=5.0, high=40.0):
    state = np.where(energies <= low, -1, np.where(energies >= high, 1, 0))
    committed = state[state != 0]
    crossings = (
        int(np.count_nonzero(committed[1:] != committed[:-1]))
        if len(committed) > 1
        else 0
    )
    return {
        "energy_low_threshold_kcal_mol": low,
        "energy_high_threshold_kcal_mol": high,
        "energy_low_high_crossings": crossings,
        "energy_round_trips": crossings // 2,
    }


def save_run(
    output: Path,
    method: str,
    angles,
    energies,
    variable_names,
    burn_in: int,
    thin: int,
    elapsed: float,
    extras: dict,
):
    output.mkdir(parents=True, exist_ok=True)
    angles = np.asarray(angles)
    energies = np.asarray(energies)
    helicity_values = helicity(angles, variable_names)
    selection = slice(burn_in, None, thin)
    np.savez_compressed(
        output / f"{method}_trajectory.npz",
        sweeps=np.arange(1, len(energies) + 1)[::thin],
        angles=angles[::thin],
        energies=energies[::thin],
        helicity=helicity_values[::thin],
    )
    np.savez_compressed(
        output / f"{method}_samples.npz",
        sweeps=np.arange(1, len(energies) + 1)[selection],
        angles=angles[selection],
        energies=energies[selection],
        helicity=helicity_values[selection],
    )
    # Report equilibrium observables on the complete post-burn-in production
    # segment.  Thinning is only a storage choice and must not change the
    # estimator or accidentally leave burn-in in structural statistics.
    production = slice(burn_in, None)
    production_energy = energies[production]
    production_helicity = helicity_values[production]
    summary = {
        "method": method,
        "sweep_definition": "sequential scan; one independent Metropolis test per free dihedral",
        "free_dihedrals": int(angles.shape[1]),
        "production_sweeps": int(len(energies)),
        "energy_evaluations": int(len(energies) * angles.shape[1] + 1),
        "burn_in_sweeps": burn_in,
        "thin_sweeps": thin,
        "elapsed_seconds": elapsed,
        "energy_mean_kcal_mol": float(np.mean(production_energy)),
        "energy_std_kcal_mol": float(np.std(production_energy)),
        "energy_min_kcal_mol": float(np.min(production_energy)),
        "energy_max_kcal_mol": float(np.max(production_energy)),
        **transition_metrics(production_helicity, sweep_offset=burn_in),
        **energy_round_trip_metrics(production_energy),
        **extras,
    }
    (output / f"{method}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def validate_gold(system: ECEPP3System):
    actual = {
        name: float(value[0])
        for name, value in system.energy_dict_variables(
            system.geometry.initial_variables
        ).items()
    }
    errors = {
        name: actual[name] - gold for name, gold in system.gold_components.items()
    }
    if max(abs(error) for error in errors.values()) > 2.0e-4:
        raise RuntimeError(f"initial SMMP gold alignment failed: {errors}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", type=Path, default=Path(__file__).resolve().parent / "data" / "system.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=100_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=462.0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not 0 <= args.burn_in < args.sweeps:
        parser.error("require 0 <= burn-in < sweeps")

    system = ECEPP3System.from_npz(args.system)
    gold_errors = validate_gold(system)
    energy_fn = energy_function(system)
    initial = coil_state(system)
    energy_fn(initial).block_until_ready()
    start = time.perf_counter()
    (angles, energies), acceptance = canonical_mc(
        energy_fn, initial, jax.random.fold_in(jax.random.PRNGKey(args.seed), 0), args.sweeps,
        1.0 / (KB_KCAL_PER_MOL_K * args.temperature))
    energies.block_until_ready()
    save_run(
        args.output, "canonical_mc", angles, energies, system.geometry.variable_names,
        args.burn_in, args.thin, time.perf_counter() - start,
        {
            "initial_state": "coil(phi=-150 deg, psi=150 deg)",
            "proposal": "replace selected dihedral with Uniform[-pi, pi]",
            "temperature_kelvin": args.temperature,
            "smmp_initial_gold_max_abs_error_kcal_mol": max(abs(value) for value in gold_errors.values()),
            "acceptance_rate": float(acceptance),
        },
    )


if __name__ == "__main__":
    main()
