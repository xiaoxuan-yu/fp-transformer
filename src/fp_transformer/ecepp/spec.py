from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np


ArrayLike = Sequence[float] | Sequence[int] | np.ndarray | jnp.ndarray


def _indices(value: ArrayLike, width: int, name: str) -> jnp.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.size == 0:
        array = np.empty((0, width), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {array.shape}")
    return jnp.asarray(array)


def _values(value: ArrayLike, size: int, name: str) -> jnp.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return jnp.asarray(array)


def _validate_atom_indices(indices: jnp.ndarray, n_atoms: int, name: str) -> None:
    host = np.asarray(indices)
    if host.size and (np.any(host < 0) or np.any(host >= n_atoms)):
        raise ValueError(f"{name} contains an atom index outside [0, {n_atoms})")
    if host.shape[1] == 2 and np.any(host[:, 0] == host[:, 1]):
        raise ValueError(f"{name} cannot contain self-pairs")
    if host.shape[1] == 4 and any(len(set(row)) != 4 for row in host.tolist()):
        raise ValueError(f"{name} must contain four distinct atoms per torsion")


def _validate_unique_pairs(indices: jnp.ndarray, name: str) -> None:
    host = np.sort(np.asarray(indices), axis=1)
    if len(host) != len(np.unique(host, axis=0)):
        raise ValueError(f"{name} contains duplicate atom pairs")


@dataclass(frozen=True)
class ElectrostaticParameters:
    """ECEPP Coulomb pair list and per-pair scale factors."""

    pairs: jnp.ndarray
    scales: jnp.ndarray

    def __post_init__(self) -> None:
        pairs = _indices(self.pairs, 2, "electrostatic pairs")
        scales = _values(self.scales, len(pairs), "electrostatic scales")
        if np.any(np.asarray(scales) < 0):
            raise ValueError("electrostatic scales must be non-negative")
        _validate_unique_pairs(pairs, "electrostatic pairs")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def create(
        cls, pairs: ArrayLike = (), scales: ArrayLike | None = None
    ) -> ElectrostaticParameters:
        pair_array = _indices(pairs, 2, "electrostatic pairs")
        n_pairs = pair_array.shape[0]
        if scales is None:
            scales = np.ones(n_pairs, dtype=np.float32)
        return cls(pair_array, _values(scales, n_pairs, "electrostatic scales"))


@dataclass(frozen=True)
class LennardJonesParameters:
    """Pair coefficients for ``A / r^12 - B / r^6`` in kcal/mol."""

    pairs: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    scales: jnp.ndarray

    def __post_init__(self) -> None:
        pairs = _indices(self.pairs, 2, "Lennard-Jones pairs")
        n_pairs = len(pairs)
        scales = _values(self.scales, n_pairs, "Lennard-Jones scales")
        if np.any(np.asarray(scales) < 0):
            raise ValueError("Lennard-Jones scales must be non-negative")
        _validate_unique_pairs(pairs, "Lennard-Jones pairs")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "a", _values(self.a, n_pairs, "Lennard-Jones A"))
        object.__setattr__(self, "b", _values(self.b, n_pairs, "Lennard-Jones B"))
        object.__setattr__(self, "scales", scales)

    @classmethod
    def create(
        cls,
        pairs: ArrayLike = (),
        a: ArrayLike = (),
        b: ArrayLike = (),
        scales: ArrayLike | None = None,
    ) -> LennardJonesParameters:
        pair_array = _indices(pairs, 2, "Lennard-Jones pairs")
        n_pairs = pair_array.shape[0]
        if scales is None:
            scales = np.ones(n_pairs, dtype=np.float32)
        return cls(
            pair_array,
            _values(a, n_pairs, "Lennard-Jones A"),
            _values(b, n_pairs, "Lennard-Jones B"),
            _values(scales, n_pairs, "Lennard-Jones scales"),
        )


@dataclass(frozen=True)
class HydrogenBondParameters:
    """Pair coefficients for the ECEPP ``C / r^12 - D / r^10`` term."""

    pairs: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    scales: jnp.ndarray

    def __post_init__(self) -> None:
        pairs = _indices(self.pairs, 2, "hydrogen-bond pairs")
        n_pairs = len(pairs)
        scales = _values(self.scales, n_pairs, "hydrogen-bond scales")
        if np.any(np.asarray(scales) < 0):
            raise ValueError("hydrogen-bond scales must be non-negative")
        _validate_unique_pairs(pairs, "hydrogen-bond pairs")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "c", _values(self.c, n_pairs, "hydrogen-bond C"))
        object.__setattr__(self, "d", _values(self.d, n_pairs, "hydrogen-bond D"))
        object.__setattr__(self, "scales", scales)

    @classmethod
    def create(
        cls,
        pairs: ArrayLike = (),
        c: ArrayLike = (),
        d: ArrayLike = (),
        scales: ArrayLike | None = None,
    ) -> HydrogenBondParameters:
        pair_array = _indices(pairs, 2, "hydrogen-bond pairs")
        n_pairs = pair_array.shape[0]
        if scales is None:
            scales = np.ones(n_pairs, dtype=np.float32)
        return cls(
            pair_array,
            _values(c, n_pairs, "hydrogen-bond C"),
            _values(d, n_pairs, "hydrogen-bond D"),
            _values(scales, n_pairs, "hydrogen-bond scales"),
        )


@dataclass(frozen=True)
class TorsionParameters:
    """Parameters for ``U * (1 + cos(n * phi - phase))``."""

    atoms: jnp.ndarray
    barrier_half: jnp.ndarray
    periodicity: jnp.ndarray
    phase: jnp.ndarray

    def __post_init__(self) -> None:
        atoms = _indices(self.atoms, 4, "torsion atoms")
        n_torsions = len(atoms)
        periodicity = _values(self.periodicity, n_torsions, "torsion periodicity")
        if np.any(np.asarray(periodicity) <= 0):
            raise ValueError("torsion periodicity must be positive")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(
            self,
            "barrier_half",
            _values(self.barrier_half, n_torsions, "torsion barrier_half"),
        )
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(
            self, "phase", _values(self.phase, n_torsions, "torsion phase")
        )

    @classmethod
    def create(
        cls,
        atoms: ArrayLike = (),
        barrier_half: ArrayLike = (),
        periodicity: ArrayLike = (),
        phase: ArrayLike | None = None,
    ) -> TorsionParameters:
        atom_array = _indices(atoms, 4, "torsion atoms")
        n_torsions = atom_array.shape[0]
        if phase is None:
            phase = np.zeros(n_torsions, dtype=np.float32)
        periodicity_array = _values(periodicity, n_torsions, "torsion periodicity")
        if np.any(np.asarray(periodicity_array) <= 0):
            raise ValueError("torsion periodicity must be positive")
        return cls(
            atom_array,
            _values(barrier_half, n_torsions, "torsion barrier_half"),
            periodicity_array,
            _values(phase, n_torsions, "torsion phase"),
        )


@dataclass(frozen=True)
class ECEPPParameters:
    """Validated, evaluator-ready parameters for the classic ECEPP energy form.

    Coordinates are measured in Angstrom and energies in kcal/mol. ECEPP keeps
    bond lengths and bond angles fixed, so this specification intentionally has
    no bond-stretch or angle-bend term.
    """

    n_atoms: int
    charges: jnp.ndarray
    electrostatic: ElectrostaticParameters
    lennard_jones: LennardJonesParameters
    hydrogen_bond: HydrogenBondParameters
    torsion: TorsionParameters
    dielectric: float = 2.0
    coulomb_constant: float = 332.06371

    def __post_init__(self) -> None:
        if self.n_atoms <= 0:
            raise ValueError("n_atoms must be positive")
        charges = _values(self.charges, self.n_atoms, "charges")
        object.__setattr__(self, "charges", charges)
        if not np.isfinite(self.dielectric) or self.dielectric <= 0:
            raise ValueError("dielectric must be finite and positive")
        if not np.isfinite(self.coulomb_constant):
            raise ValueError("coulomb_constant must be finite")

        _validate_atom_indices(
            self.electrostatic.pairs, self.n_atoms, "electrostatic pairs"
        )
        _validate_atom_indices(
            self.lennard_jones.pairs, self.n_atoms, "Lennard-Jones pairs"
        )
        _validate_atom_indices(
            self.hydrogen_bond.pairs, self.n_atoms, "hydrogen-bond pairs"
        )
        _validate_atom_indices(self.torsion.atoms, self.n_atoms, "torsion atoms")
