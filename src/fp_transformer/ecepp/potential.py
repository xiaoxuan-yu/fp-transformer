from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..potential.base import Potential
from .spec import ECEPPParameters
from .terms import (
    electrostatic_energy,
    hydrogen_bond_energy,
    lennard_jones_energy,
    torsion_energy,
)


def _batch_coordinates(coordinates, n_atoms: int):
    coordinates = jnp.asarray(coordinates)
    squeeze = False
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
        squeeze = True
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            "coordinates must have shape (batch, atoms, 3) or (atoms, 3), "
            f"got {coordinates.shape}"
        )
    if coordinates.shape[1] != n_atoms:
        raise ValueError(f"expected {n_atoms} atoms, got {coordinates.shape[1]}")
    if not jnp.issubdtype(coordinates.dtype, jnp.floating):
        coordinates = coordinates.astype(jnp.float32)
    return coordinates, squeeze


class ECEPPPotential(Potential):
    """Differentiable classic ECEPP potential for Cartesian coordinates.

    The evaluator is parameter-set agnostic. Supply ECEPP/2, ECEPP/3, or other
    compatible coefficients through :class:`ECEPPParameters`.
    """

    def __init__(self, parameters: ECEPPParameters):
        super().__init__()
        self.parameters = parameters
        self._energy_dict_jit = jax.jit(self._energy_dict_impl)
        self._fused_pairs = self._compile_fused_pairs()
        self._total_jit = jax.jit(self._total_impl)
        self._energy_and_gradient_jit = jax.jit(
            jax.value_and_grad(self._summed_total_with_energies, has_aux=True)
        )

    def _compile_fused_pairs(self):
        """Coalesce overlapping ECEPP terms into one distance evaluation."""
        coefficients: dict[tuple[int, int], np.ndarray] = {}

        def row(pair):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            return coefficients.setdefault(key, np.zeros(4, dtype=np.float64))

        electrostatic = self.parameters.electrostatic
        charges = np.asarray(self.parameters.charges)
        prefactor = self.parameters.coulomb_constant / self.parameters.dielectric
        for pair, scale in zip(
            np.asarray(electrostatic.pairs), np.asarray(electrostatic.scales)
        ):
            row(pair)[0] += prefactor * scale * charges[pair[0]] * charges[pair[1]]

        lennard_jones = self.parameters.lennard_jones
        for pair, a, b, scale in zip(
            np.asarray(lennard_jones.pairs),
            np.asarray(lennard_jones.a),
            np.asarray(lennard_jones.b),
            np.asarray(lennard_jones.scales),
        ):
            values = row(pair)
            values[1] += scale * a
            values[2] += scale * b

        hydrogen_bond = self.parameters.hydrogen_bond
        for pair, c, d, scale in zip(
            np.asarray(hydrogen_bond.pairs),
            np.asarray(hydrogen_bond.c),
            np.asarray(hydrogen_bond.d),
            np.asarray(hydrogen_bond.scales),
        ):
            values = row(pair)
            values[1] += scale * c
            values[3] += scale * d

        if not coefficients:
            return (
                jnp.empty((0, 2), dtype=jnp.int32),
                jnp.empty((0, 4), dtype=jnp.float32),
            )
        pairs = np.asarray(list(coefficients), dtype=np.int32)
        values = np.asarray(list(coefficients.values()), dtype=np.float32)
        return jnp.asarray(pairs), jnp.asarray(values)

    def _energy_dict_impl(self, coordinates):
        components = {
            "electrostatic": electrostatic_energy(coordinates, self.parameters),
            "lennard_jones": lennard_jones_energy(coordinates, self.parameters),
            "hydrogen_bond": hydrogen_bond_energy(coordinates, self.parameters),
            "torsion": torsion_energy(coordinates, self.parameters),
        }
        components["etotal"] = sum(components.values())
        return components

    def _total_impl(self, coordinates):
        pairs, coefficients = self._fused_pairs
        if pairs.shape[0] == 0:
            pair_total = jnp.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
        else:
            delta = coordinates[:, pairs[:, 1], :] - coordinates[:, pairs[:, 0], :]
            inverse_r2 = jnp.reciprocal(jnp.sum(delta * delta, axis=-1))
            inverse_r = jnp.sqrt(inverse_r2)
            inverse_r6 = inverse_r2**3
            inverse_r10 = inverse_r6 * inverse_r2**2
            inverse_r12 = inverse_r6**2
            pair_energy = (
                coefficients[:, 0] * inverse_r
                + coefficients[:, 1] * inverse_r12
                - coefficients[:, 2] * inverse_r6
                - coefficients[:, 3] * inverse_r10
            )
            pair_total = jnp.sum(pair_energy, axis=-1, keepdims=True)
        return pair_total + torsion_energy(coordinates, self.parameters)

    def _summed_total_with_energies(self, coordinates):
        energies = self._total_impl(coordinates)
        return jnp.sum(energies), energies

    def energy_dict(self, coordinates):
        batched, squeeze = _batch_coordinates(coordinates, self.parameters.n_atoms)
        components = self._energy_dict_jit(batched)
        if squeeze:
            return {name: value[0] for name, value in components.items()}
        return components

    def energy_and_force(self, coordinates):
        """Return fused total energies and forces from one cached reverse pass."""
        batched, squeeze = _batch_coordinates(coordinates, self.parameters.n_atoms)
        (_, energies), gradient = self._energy_and_gradient_jit(batched)
        forces = -gradient
        if squeeze:
            return energies[0], forces[0]
        return energies, forces

    def force(self, topology):
        coordinates = topology.coordinate
        if coordinates is None:
            raise ValueError("Topology coordinate is None")
        return self.energy_and_force(coordinates)[1]

    def __call__(self, coordinates, beta: float = 1.0, *, return_dict=False):
        del beta
        if return_dict:
            return self.energy_dict(coordinates)
        batched, squeeze = _batch_coordinates(coordinates, self.parameters.n_atoms)
        total = self._total_jit(batched)
        return total[0] if squeeze else total
