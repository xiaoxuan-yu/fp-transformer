from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from .ecepp3 import ECEPP3ParameterSet
from .internal_coordinates import EyringInternalCoordinates
from .potential import ECEPPPotential
from .spec import ECEPPParameters


@dataclass(frozen=True)
class ECEPP3System:
    """A compiled ECEPP/3 molecule plus its fixed-geometry coordinates."""

    geometry: EyringInternalCoordinates
    parameters: ECEPPParameters
    reference_coordinates: np.ndarray
    gold_components: dict[str, float]
    variable_torsion_barrier_half: np.ndarray
    variable_torsion_periodicity: np.ndarray
    variable_torsion_phase: np.ndarray

    @property
    def potential(self) -> ECEPPPotential:
        return ECEPPPotential(self.parameters)

    def energy_dict_variables(self, variables):
        """Evaluate the complete ECEPP/3 energy in internal coordinates."""
        values = jnp.asarray(variables)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.geometry.n_variables:
            raise ValueError(
                "variables must have shape (batch, variables) or (variables,)"
            )
        coordinates = self.geometry.coordinates(values)
        components = dict(self.potential.energy_dict(coordinates))
        torsion = jnp.sum(
            jnp.asarray(self.variable_torsion_barrier_half)
            * (
                1.0
                + jnp.cos(
                    jnp.asarray(self.variable_torsion_periodicity) * values
                    - jnp.asarray(self.variable_torsion_phase)
                )
            ),
            axis=-1,
            keepdims=True,
        )
        components["torsion"] = components["torsion"] + torsion
        components["etotal"] = components["etotal"] + torsion
        if squeeze:
            return {name: value[0] for name, value in components.items()}
        return components

    def energy_variables(self, variables):
        return self.energy_dict_variables(variables)["etotal"]

    @classmethod
    def from_npz(cls, path: str | Path) -> ECEPP3System:
        geometry = EyringInternalCoordinates.from_npz(path)
        with np.load(path, allow_pickle=False) as data:
            parameter_set = ECEPP3ParameterSet()
            parameters = parameter_set.compile(
                charges=data["charges"],
                atom_types=data["atom_types"],
                nonbonded_pairs=data["nonbonded_pairs"],
                one_four_pairs=data["one_four_pairs"],
            )
            if "variable_classes" in data:
                variable_classes = data["variable_classes"]
            else:
                # Compatibility with the first exported systems. These are
                # the class labels encoded in the SMMP ECEPP/3 residue library.
                class_by_name = {
                    "omg": 1,
                    "phi": 2,
                    "psi": 3,
                    "x1": 4,
                    "pht": 2,
                    "pst": 9,
                    "omt": 16,
                }
                try:
                    variable_classes = np.asarray(
                        [class_by_name[name] for name in geometry.variable_names]
                    )
                except KeyError as error:
                    raise ValueError(
                        "system NPZ lacks variable_classes and contains an "
                        f"unknown SMMP variable name: {error.args[0]}"
                    ) from error
            variable_torsion = parameter_set.variable_torsion_parameters(
                variable_classes
            )
            components = {
                name: float(data[f"gold_{name}"])
                for name in (
                    "electrostatic",
                    "lennard_jones",
                    "hydrogen_bond",
                    "torsion",
                    "etotal",
                )
            }
            return cls(
                geometry=geometry,
                parameters=parameters,
                reference_coordinates=np.asarray(data["reference_coordinates"]),
                gold_components=components,
                variable_torsion_barrier_half=variable_torsion[0],
                variable_torsion_periodicity=variable_torsion[1],
                variable_torsion_phase=variable_torsion[2],
            )
