from __future__ import annotations

import jax.numpy as jnp

from .spec import ECEPPParameters


def _pair_distances(coordinates, pairs):
    delta = coordinates[:, pairs[:, 1], :] - coordinates[:, pairs[:, 0], :]
    return jnp.linalg.norm(delta, axis=-1)


def electrostatic_energy(coordinates, parameters: ECEPPParameters):
    term = parameters.electrostatic
    if term.pairs.shape[0] == 0:
        return jnp.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
    distances = _pair_distances(coordinates, term.pairs)
    qi = parameters.charges[term.pairs[:, 0]]
    qj = parameters.charges[term.pairs[:, 1]]
    prefactor = parameters.coulomb_constant / parameters.dielectric
    energy = prefactor * term.scales * qi * qj / distances
    return jnp.sum(energy, axis=-1, keepdims=True)


def lennard_jones_energy(coordinates, parameters: ECEPPParameters):
    term = parameters.lennard_jones
    if term.pairs.shape[0] == 0:
        return jnp.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
    distances = _pair_distances(coordinates, term.pairs)
    inverse_r6 = jnp.reciprocal(distances**6)
    energy = term.scales * (term.a * inverse_r6**2 - term.b * inverse_r6)
    return jnp.sum(energy, axis=-1, keepdims=True)


def hydrogen_bond_energy(coordinates, parameters: ECEPPParameters):
    term = parameters.hydrogen_bond
    if term.pairs.shape[0] == 0:
        return jnp.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
    distances = _pair_distances(coordinates, term.pairs)
    inverse_r2 = jnp.reciprocal(distances**2)
    inverse_r10 = inverse_r2**5
    energy = term.scales * (term.c * inverse_r10 * inverse_r2 - term.d * inverse_r10)
    return jnp.sum(energy, axis=-1, keepdims=True)


def dihedral_angles(coordinates, atoms):
    """Vectorized signed dihedrals for ``(batch, torsion, atom, xyz)``."""
    p0 = coordinates[:, atoms[:, 0], :]
    p1 = coordinates[:, atoms[:, 1], :]
    p2 = coordinates[:, atoms[:, 2], :]
    p3 = coordinates[:, atoms[:, 3], :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
    return jnp.arctan2(y, x)


def torsion_energy(coordinates, parameters: ECEPPParameters):
    term = parameters.torsion
    if term.atoms.shape[0] == 0:
        return jnp.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
    angles = dihedral_angles(coordinates, term.atoms)
    energy = term.barrier_half * (
        1.0 + jnp.cos(term.periodicity * angles - term.phase)
    )
    return jnp.sum(energy, axis=-1, keepdims=True)

