"""ECEPP/3 neutral Ala10 on its 20 free dihedrals: potential wrapper and structural metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.ecepp import ECEPP3System, wrap_angles
from fp_transformer.potential.base import Potential

KB_KCAL_PER_MOL_K = 0.00198720425864083


class InternalECEPPPotential(Potential):
    """Expose ECEPP internal variables as an EnergySystem potential."""

    def __init__(self, system: ECEPP3System):
        super().__init__()
        self.system = system
        self.geometry = system.geometry
        self.pair_potential = system.potential
        self.barrier_half = jnp.asarray(system.variable_torsion_barrier_half)
        self.periodicity = jnp.asarray(system.variable_torsion_periodicity)
        self.phase = jnp.asarray(system.variable_torsion_phase)

    def __call__(self, coordinates, beta: float = 1.0):
        del beta
        coordinates = jnp.asarray(coordinates)
        if coordinates.ndim != 3 or coordinates.shape[1] != 1:
            raise ValueError("FPT coordinates must have shape (replicas, 1, variables)")
        variables = coordinates[:, 0, :]
        cartesian = self.geometry.coordinates(variables)
        pair_and_cartesian_torsion = self.pair_potential(cartesian)
        variable_torsion = jnp.sum(
            self.barrier_half
            * (1.0 + jnp.cos(self.periodicity * variables - self.phase)),
            axis=-1,
            keepdims=True,
        )
        return pair_and_cartesian_torsion + variable_torsion


def coil_state(system: ECEPP3System):
    values = np.asarray(system.geometry.initial_variables).copy()
    for index, name in enumerate(system.geometry.variable_names):
        if name == "phi":
            values[index] = np.deg2rad(-150.0)
        elif name == "psi":
            values[index] = np.deg2rad(150.0)
    return jnp.asarray(values)


def alpha_unit_indicators(angles, variable_names):
    names = np.asarray(variable_names)
    degrees = np.rad2deg(np.asarray(angles))
    phi = degrees[:, names == "phi"]
    psi = degrees[:, names == "psi"]
    return (
        (phi >= -100.0)
        & (phi <= -30.0)
        & (psi >= -80.0)
        & (psi <= -5.0)
    )


def helicity(angles, variable_names):
    return alpha_unit_indicators(angles, variable_names).mean(axis=1)


def circular_rms_degrees(angles, reference, mask):
    delta = (angles[:, mask] - reference[mask] + np.pi) % (2.0 * np.pi) - np.pi
    return np.rad2deg(np.sqrt(np.mean(np.square(delta), axis=1)))


def alpha_backbone_hbond_pairs(
    system: ECEPP3System, reference_cutoff: float = 2.1
):
    """Identify the six Ala10 (i, i+4) backbone H--O pairs.

    The compact NPZ keeps ECEPP H-bond pairs but not residue labels. In the
    all-alpha reference, the characteristic backbone pairs are the only
    parameterized H-bond pairs below 2.1 Angstrom.
    """

    reference = np.asarray(
        system.geometry.coordinates(system.geometry.initial_variables)
    )
    candidates = np.asarray(system.potential.parameters.hydrogen_bond.pairs)
    distances = np.linalg.norm(
        reference[candidates[:, 0]] - reference[candidates[:, 1]], axis=-1
    )
    pairs = candidates[distances < reference_cutoff]
    if len(pairs) != 6:
        raise ValueError(
            "neutral Ala10 should have six alpha (i, i+4) backbone "
            f"H-bond pairs; found {len(pairs)}"
        )
    return pairs


def alpha_backbone_hbond_counts(
    system: ECEPP3System,
    angles,
    pairs,
    formation_cutoff: float = 2.5,
):
    """Count alpha H bonds with an explicit H--O distance cutoff."""

    cartesian = np.asarray(system.geometry.coordinates(angles))
    distances = np.linalg.norm(
        cartesian[:, pairs[:, 0]] - cartesian[:, pairs[:, 1]], axis=-1
    )
    return np.count_nonzero(distances < formation_cutoff, axis=1)


def evaluate(potential, system, coordinates, energy_batch_size: int):
    angles = np.asarray(coordinates)[:, 0, :]
    energies = []
    for start in range(0, len(angles), energy_batch_size):
        batch = jnp.asarray(angles[start : start + energy_batch_size])
        energies.append(np.asarray(potential(batch[:, None, :]))[:, 0])
    energies = np.concatenate(energies)
    helicity_values = helicity(angles, system.geometry.variable_names)
    state = np.where(
        helicity_values >= 0.7, 1, np.where(helicity_values <= 0.3, -1, 0)
    )
    return energies, helicity_values, state


def macro_state_masses(committed, particle_weights) -> dict[str, float]:
    """Normalized coil/intermediate/helix masses for one particle measure."""

    committed = np.asarray(committed)
    weights = np.asarray(particle_weights, dtype=np.float64)
    weights /= np.sum(weights)
    return {
        "coil": float(np.sum(weights[committed == -1])),
        "intermediate": float(np.sum(weights[committed == 0])),
        "helix": float(np.sum(weights[committed == 1])),
    }


def macro_mass_difference(after: dict, before: dict) -> dict[str, float]:
    return {state: float(after[state] - before[state]) for state in before}


def residue_block_structure(variable_names) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Group adjacent backbone psi/phi variables and keep cap torsions separate."""

    names = tuple(map(str, variable_names))
    block_ids = []
    labels = []
    variable_index = 0
    backbone_index = 1
    while variable_index < len(names):
        if (
            variable_index + 1 < len(names)
            and {names[variable_index], names[variable_index + 1]}
            == {"phi", "psi"}
        ):
            block_id = len(labels)
            block_ids.extend((block_id, block_id))
            labels.append(f"backbone_{backbone_index}:psi_phi")
            backbone_index += 1
            variable_index += 2
            continue
        block_ids.append(len(labels))
        labels.append(f"variable_{variable_index}:{names[variable_index]}")
        variable_index += 1
    return tuple(block_ids), tuple(labels)
