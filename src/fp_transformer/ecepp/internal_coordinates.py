from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def wrap_angles(angles):
    """Wrap angles in radians to the half-open interval ``[-pi, pi)``."""
    return (angles + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@dataclass(frozen=True)
class EyringInternalCoordinates:
    """SMMP/ECEPP fixed-geometry Eyring internal-coordinate tree.

    ``variable_effects[i, k]`` is one when changing variable ``k`` shifts the
    stored torsion of atom ``i``. ECEPP rotates every direct child of a parent
    together, which is the detail that distinguishes this representation from
    an ordinary four-atom Z matrix.
    """

    parent: jnp.ndarray
    bond_length: jnp.ndarray
    bond_angle: jnp.ndarray
    base_torsion: jnp.ndarray
    variable_effects: jnp.ndarray
    initial_variables: jnp.ndarray
    root_position: jnp.ndarray
    root_x_axis: jnp.ndarray
    root_z_axis: jnp.ndarray
    variable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parent = np.asarray(self.parent, dtype=np.int32)
        n_atoms = len(parent)
        effects = np.asarray(self.variable_effects)
        n_variables = len(np.asarray(self.initial_variables))
        if parent.shape != (n_atoms,) or np.any(parent[1:] >= np.arange(1, n_atoms)):
            raise ValueError("parent must describe a topologically ordered tree")
        if effects.shape != (n_atoms, n_variables):
            raise ValueError("variable_effects must have shape (atoms, variables)")
        for name, value, shape in (
            ("bond_length", self.bond_length, (n_atoms,)),
            ("bond_angle", self.bond_angle, (n_atoms,)),
            ("base_torsion", self.base_torsion, (n_atoms,)),
            ("root_position", self.root_position, (3,)),
            ("root_x_axis", self.root_x_axis, (3,)),
            ("root_z_axis", self.root_z_axis, (3,)),
        ):
            if np.asarray(value).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.variable_names and len(self.variable_names) != n_variables:
            raise ValueError("variable_names and initial_variables must be aligned")

        depth = np.zeros(n_atoms, dtype=np.int32)
        for atom in range(1, n_atoms):
            depth[atom] = depth[parent[atom]] + 1
        tree_depth = int(depth.max(initial=0))
        level_width = max(
            (int(np.count_nonzero(depth == level)) for level in range(1, tree_depth + 1)),
            default=1,
        )
        level_atoms = np.zeros((tree_depth, level_width), dtype=np.int32)
        level_parents = np.zeros_like(level_atoms)
        level_mask = np.zeros((tree_depth, level_width), dtype=bool)
        for level in range(1, tree_depth + 1):
            atoms = np.flatnonzero(depth == level)
            level_atoms[level - 1, : len(atoms)] = atoms
            level_parents[level - 1, : len(atoms)] = parent[atoms]
            level_mask[level - 1, : len(atoms)] = True

        object.__setattr__(self, "parent", jnp.asarray(parent))
        object.__setattr__(self, "_tree_depth", tree_depth)
        object.__setattr__(self, "_level_atoms", jnp.asarray(level_atoms))
        object.__setattr__(self, "_level_parents", jnp.asarray(level_parents))
        object.__setattr__(self, "_level_mask", jnp.asarray(level_mask))
        for name in (
            "bond_length",
            "bond_angle",
            "base_torsion",
            "variable_effects",
            "initial_variables",
            "root_position",
            "root_x_axis",
            "root_z_axis",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))

    @property
    def n_atoms(self) -> int:
        return int(self.parent.shape[0])

    @property
    def n_variables(self) -> int:
        return int(self.initial_variables.shape[0])

    @classmethod
    def from_npz(cls, path: str | Path) -> EyringInternalCoordinates:
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(name) for name in data["variable_names"].tolist())
            return cls(
                parent=data["parent"],
                bond_length=data["bond_length"],
                bond_angle=data["bond_angle"],
                base_torsion=data["base_torsion"],
                variable_effects=data["variable_effects"],
                initial_variables=data["initial_variables"],
                root_position=data["root_position"],
                root_x_axis=data["root_x_axis"],
                root_z_axis=data["root_z_axis"],
                variable_names=names,
            )

    def coordinates(self, variables):
        """Build Cartesian coordinates from one or a batch of torsion vectors."""
        variables = jnp.asarray(variables)
        squeeze = variables.ndim == 1
        if squeeze:
            variables = variables[None, :]
        if variables.ndim != 2 or variables.shape[1] != self.n_variables:
            raise ValueError(
                f"variables must have shape ({self.n_variables},) or "
                f"(batch, {self.n_variables})"
            )
        result = self._coordinates_batch(variables)
        return result[0] if squeeze else result

    def _coordinates_one(self, variables):
        return self._coordinates_batch(variables[None, :])[0]

    def _coordinates_batch(self, variables):
        delta = wrap_angles(variables - self.initial_variables)
        torsion = self.base_torsion[None, :] + delta @ self.variable_effects.T
        batch_size = variables.shape[0]
        xyz = jnp.zeros((self.n_atoms, batch_size, 3), dtype=variables.dtype)
        x_axis = jnp.zeros_like(xyz)
        z_axis = jnp.zeros_like(xyz)
        xyz = xyz.at[0].set(jnp.broadcast_to(self.root_position, (batch_size, 3)))
        x_axis = x_axis.at[0].set(
            jnp.broadcast_to(self.root_x_axis, (batch_size, 3))
        )
        z_axis = z_axis.at[0].set(
            jnp.broadcast_to(self.root_z_axis, (batch_size, 3))
        )

        def place_level(level, state):
            coordinates, x_axes, z_axes = state
            atoms = self._level_atoms[level]
            owners = self._level_parents[level]
            active = self._level_mask[level, :, None, None]
            owner_x = x_axes[owners]
            owner_z = z_axes[owners]
            y_axis = jnp.cross(owner_z, owner_x)
            angle = self.bond_angle[atoms, None, None]
            dihedral = jnp.swapaxes(torsion[:, atoms], 0, 1)[..., None]
            new_x = (
                -jnp.cos(angle) * owner_x
                - jnp.sin(angle) * jnp.cos(dihedral) * y_axis
                - jnp.sin(angle) * jnp.sin(dihedral) * owner_z
            )
            new_x = new_x / jnp.linalg.norm(new_x, axis=-1, keepdims=True)
            new_z = -jnp.sin(dihedral) * y_axis + jnp.cos(dihedral) * owner_z
            new_z = new_z / jnp.linalg.norm(new_z, axis=-1, keepdims=True)
            new_coordinates = (
                coordinates[owners] + self.bond_length[atoms, None, None] * new_x
            )
            coordinates = coordinates.at[atoms].add(
                jnp.where(active, new_coordinates - coordinates[atoms], 0.0)
            )
            x_axes = x_axes.at[atoms].add(
                jnp.where(active, new_x - x_axes[atoms], 0.0)
            )
            z_axes = z_axes.at[atoms].add(
                jnp.where(active, new_z - z_axes[atoms], 0.0)
            )
            return (
                coordinates,
                x_axes,
                z_axes,
            )

        xyz, _, _ = jax.lax.fori_loop(
            0, self._tree_depth, place_level, (xyz, x_axis, z_axis)
        )
        return jnp.swapaxes(xyz, 0, 1)
