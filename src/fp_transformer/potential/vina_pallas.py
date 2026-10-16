"""Pallas/Triton kernel for fused Vina interaction scoring and gradients."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

from ..type import JaxArray


VINA_INTRA_PAIR_BLOCK_SIZE = 256


def _vina_interaction_score_gradient_kernel(
    coordinate_ref,
    receptor_coordinate_ref,
    receptor_radii_ref,
    ligand_radii_ref,
    receptor_hydrophobic_ref,
    ligand_hydrophobic_ref,
    receptor_donor_ref,
    receptor_acceptor_ref,
    ligand_donor_ref,
    ligand_acceptor_ref,
    receptor_valid_ref,
    weights_ref,
    cutoff_ref,
    atom_score_ref,
    *coordinate_gradient_refs,
):
    """Accumulate all receptor pairs for one replica/ligand-atom program."""

    replica = pl.program_id(0)
    ligand_atom = pl.program_id(1)

    # Scalar xyz loads avoid Triton's power-of-two vector-width restriction.
    dx = (
        coordinate_ref[replica, ligand_atom, 0]
        - receptor_coordinate_ref[:, 0]
    )
    dy = (
        coordinate_ref[replica, ligand_atom, 1]
        - receptor_coordinate_ref[:, 1]
    )
    dz = (
        coordinate_ref[replica, ligand_atom, 2]
        - receptor_coordinate_ref[:, 2]
    )
    squared_distance = dx * dx + dy * dy + dz * dz
    tiny = jnp.finfo(coordinate_ref.dtype).tiny
    distance = jnp.sqrt(jnp.maximum(squared_distance, tiny))
    surface = distance - (
        receptor_radii_ref[:] + ligand_radii_ref[ligand_atom]
    )

    gauss1 = jnp.exp(-jnp.square(surface / 0.5))
    gauss2 = jnp.exp(-jnp.square((surface - 3.0) / 2.0))
    repulsion = jnp.square(jnp.minimum(surface, 0.0))
    hydrophobic = jnp.clip(1.5 - surface, 0.0, 1.0)
    hydrogen_bond = jnp.clip(-surface / 0.7, 0.0, 1.0)
    hydrophobic_pair = (
        receptor_hydrophobic_ref[:]
        & ligand_hydrophobic_ref[ligand_atom]
    )
    hydrogen_bond_pair = (
        receptor_donor_ref[:] & ligand_acceptor_ref[ligand_atom]
    ) | (
        receptor_acceptor_ref[:] & ligand_donor_ref[ligand_atom]
    )
    active = receptor_valid_ref[:] & (distance < cutoff_ref[0])

    pair_score = (
        weights_ref[0] * gauss1
        + weights_ref[1] * gauss2
        + weights_ref[2] * repulsion
        + weights_ref[3]
        * jnp.where(hydrophobic_pair, hydrophobic, 0.0)
        + weights_ref[4]
        * jnp.where(hydrogen_bond_pair, hydrogen_bond, 0.0)
    )
    atom_score_ref[replica, ligand_atom] = jnp.sum(
        jnp.where(active, pair_score, 0.0)
    )

    # A score-only call has no second output. This branch is static at trace
    # time, so MC does not execute or compile the derivative calculations.
    if not coordinate_gradient_refs:
        return
    coordinate_gradient_ref = coordinate_gradient_refs[0]

    # d(score)/d(surface distance), with zero derivative on flat branches.
    surface_derivative = (
        weights_ref[0] * gauss1 * (-8.0 * surface)
        + weights_ref[1] * gauss2 * (-(surface - 3.0) / 2.0)
        + weights_ref[2]
        * jnp.where(surface < 0.0, 2.0 * surface, 0.0)
        + weights_ref[3]
        * jnp.where(
            hydrophobic_pair & (surface > 0.5) & (surface < 1.5),
            -1.0,
            0.0,
        )
        + weights_ref[4]
        * jnp.where(
            hydrogen_bond_pair
            & (surface > -0.7)
            & (surface < 0.0),
            -1.0 / 0.7,
            0.0,
        )
    )
    radial_derivative = jnp.where(active, surface_derivative, 0.0) / distance
    coordinate_gradient_ref[replica, ligand_atom, 0] = jnp.sum(
        radial_derivative * dx
    )
    coordinate_gradient_ref[replica, ligand_atom, 1] = jnp.sum(
        radial_derivative * dy
    )
    coordinate_gradient_ref[replica, ligand_atom, 2] = jnp.sum(
        radial_derivative * dz
    )


def _pallas_score_and_coordinate_gradient(
    coordinate: JaxArray,
    receptor_coordinate: JaxArray,
    receptor_radii: JaxArray,
    ligand_radii: JaxArray,
    receptor_hydrophobic: JaxArray,
    ligand_hydrophobic: JaxArray,
    receptor_donor: JaxArray,
    receptor_acceptor: JaxArray,
    ligand_donor: JaxArray,
    ligand_acceptor: JaxArray,
    receptor_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
    *,
    interpret: bool,
) -> tuple[JaxArray, JaxArray]:
    out_shape = (
        ShapeDtypeStruct(coordinate.shape[:2], coordinate.dtype),
        ShapeDtypeStruct(coordinate.shape, coordinate.dtype),
    )
    atom_score, coordinate_gradient = pl.pallas_call(
        _vina_interaction_score_gradient_kernel,
        out_shape=out_shape,
        grid=coordinate.shape[:2],
        interpret=interpret,
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
        name="vina_interaction_score_gradient",
    )(
        coordinate,
        receptor_coordinate,
        receptor_radii,
        ligand_radii,
        receptor_hydrophobic,
        ligand_hydrophobic,
        receptor_donor,
        receptor_acceptor,
        ligand_donor,
        ligand_acceptor,
        receptor_valid,
        weights,
        cutoff,
    )
    return jnp.sum(atom_score, axis=1, keepdims=True), coordinate_gradient


def _pallas_score(
    coordinate: JaxArray,
    receptor_coordinate: JaxArray,
    receptor_radii: JaxArray,
    ligand_radii: JaxArray,
    receptor_hydrophobic: JaxArray,
    ligand_hydrophobic: JaxArray,
    receptor_donor: JaxArray,
    receptor_acceptor: JaxArray,
    ligand_donor: JaxArray,
    ligand_acceptor: JaxArray,
    receptor_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
    *,
    interpret: bool,
) -> JaxArray:
    atom_score = pl.pallas_call(
        _vina_interaction_score_gradient_kernel,
        out_shape=ShapeDtypeStruct(coordinate.shape[:2], coordinate.dtype),
        grid=coordinate.shape[:2],
        interpret=interpret,
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
        name="vina_interaction_score",
    )(
        coordinate,
        receptor_coordinate,
        receptor_radii,
        ligand_radii,
        receptor_hydrophobic,
        ligand_hydrophobic,
        receptor_donor,
        receptor_acceptor,
        ligand_donor,
        ligand_acceptor,
        receptor_valid,
        weights,
        cutoff,
    )
    return jnp.sum(atom_score, axis=1, keepdims=True)


@partial(jax.custom_vjp, nondiff_argnums=(13,))
def vina_interaction_energy(
    coordinate: JaxArray,
    receptor_coordinate: JaxArray,
    receptor_radii: JaxArray,
    ligand_radii: JaxArray,
    receptor_hydrophobic: JaxArray,
    ligand_hydrophobic: JaxArray,
    receptor_donor: JaxArray,
    receptor_acceptor: JaxArray,
    ligand_donor: JaxArray,
    ligand_acceptor: JaxArray,
    receptor_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
    interpret: bool,
) -> JaxArray:
    """Return weighted interaction energy using a fused analytic pullback."""

    return _pallas_score(
        coordinate,
        receptor_coordinate,
        receptor_radii,
        ligand_radii,
        receptor_hydrophobic,
        ligand_hydrophobic,
        receptor_donor,
        receptor_acceptor,
        ligand_donor,
        ligand_acceptor,
        receptor_valid,
        weights,
        cutoff,
        interpret=interpret,
    )


def _vina_interaction_energy_fwd(
    coordinate,
    receptor_coordinate,
    receptor_radii,
    ligand_radii,
    receptor_hydrophobic,
    ligand_hydrophobic,
    receptor_donor,
    receptor_acceptor,
    ligand_donor,
    ligand_acceptor,
    receptor_valid,
    weights,
    cutoff,
    interpret,
):
    score, coordinate_gradient = _pallas_score_and_coordinate_gradient(
        coordinate,
        receptor_coordinate,
        receptor_radii,
        ligand_radii,
        receptor_hydrophobic,
        ligand_hydrophobic,
        receptor_donor,
        receptor_acceptor,
        ligand_donor,
        ligand_acceptor,
        receptor_valid,
        weights,
        cutoff,
        interpret=interpret,
    )
    return score, coordinate_gradient


def _vina_interaction_energy_bwd(
    interpret, coordinate_gradient, score_cotangent
):
    del interpret
    gradient = coordinate_gradient * score_cotangent[:, None, :]
    return (gradient,) + (None,) * 12


vina_interaction_energy.defvjp(
    _vina_interaction_energy_fwd,
    _vina_interaction_energy_bwd,
)


def _vina_intra_score_gradient_kernel(
    coordinate_ref,
    pair_i_ref,
    pair_j_ref,
    pair_radius_ref,
    hydrophobic_pair_ref,
    hydrogen_bond_pair_ref,
    pair_valid_ref,
    weights_ref,
    cutoff_ref,
    coordinate_gradient_init_ref,
    block_score_ref,
    coordinate_gradient_ref,
):
    """Score a tile of intraligand pairs and atomically accumulate its VJP."""

    del coordinate_gradient_init_ref
    replica = pl.program_id(0)
    pair_block = pl.program_id(1)
    offsets = pair_block * VINA_INTRA_PAIR_BLOCK_SIZE + jnp.arange(
        VINA_INTRA_PAIR_BLOCK_SIZE
    )
    pair_i = pair_i_ref[offsets]
    pair_j = pair_j_ref[offsets]
    valid = pair_valid_ref[offsets]

    dx = coordinate_ref[replica, pair_i, 0] - coordinate_ref[replica, pair_j, 0]
    dy = coordinate_ref[replica, pair_i, 1] - coordinate_ref[replica, pair_j, 1]
    dz = coordinate_ref[replica, pair_i, 2] - coordinate_ref[replica, pair_j, 2]
    squared_distance = dx * dx + dy * dy + dz * dz
    tiny = jnp.finfo(coordinate_ref.dtype).tiny
    distance = jnp.sqrt(jnp.maximum(squared_distance, tiny))
    surface = distance - pair_radius_ref[offsets]

    gauss1 = jnp.exp(-jnp.square(surface / 0.5))
    gauss2 = jnp.exp(-jnp.square((surface - 3.0) / 2.0))
    repulsion = jnp.square(jnp.minimum(surface, 0.0))
    hydrophobic = jnp.clip(1.5 - surface, 0.0, 1.0)
    hydrogen_bond = jnp.clip(-surface / 0.7, 0.0, 1.0)
    hydrophobic_pair = hydrophobic_pair_ref[offsets]
    hydrogen_bond_pair = hydrogen_bond_pair_ref[offsets]
    active = valid & (distance < cutoff_ref[0])

    pair_score = (
        weights_ref[0] * gauss1
        + weights_ref[1] * gauss2
        + weights_ref[2] * repulsion
        + weights_ref[3]
        * jnp.where(hydrophobic_pair, hydrophobic, 0.0)
        + weights_ref[4]
        * jnp.where(hydrogen_bond_pair, hydrogen_bond, 0.0)
    )
    block_score_ref[replica, pair_block] = jnp.sum(
        jnp.where(active, pair_score, 0.0)
    )

    surface_derivative = (
        weights_ref[0] * gauss1 * (-8.0 * surface)
        + weights_ref[1] * gauss2 * (-(surface - 3.0) / 2.0)
        + weights_ref[2]
        * jnp.where(surface < 0.0, 2.0 * surface, 0.0)
        + weights_ref[3]
        * jnp.where(
            hydrophobic_pair & (surface > 0.5) & (surface < 1.5),
            -1.0,
            0.0,
        )
        + weights_ref[4]
        * jnp.where(
            hydrogen_bond_pair
            & (surface > -0.7)
            & (surface < 0.0),
            -1.0 / 0.7,
            0.0,
        )
    )
    radial_derivative = jnp.where(active, surface_derivative, 0.0) / distance
    gradient_x = radial_derivative * dx
    gradient_y = radial_derivative * dy
    gradient_z = radial_derivative * dz
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_i, 0),
        gradient_x,
    )
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_i, 1),
        gradient_y,
    )
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_i, 2),
        gradient_z,
    )
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_j, 0),
        -gradient_x,
    )
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_j, 1),
        -gradient_y,
    )
    pltriton.atomic_add(
        coordinate_gradient_ref,
        (replica, pair_j, 2),
        -gradient_z,
    )


def _jax_intra_score(
    coordinate: JaxArray,
    pair_i: JaxArray,
    pair_j: JaxArray,
    pair_radius: JaxArray,
    hydrophobic_pair: JaxArray,
    hydrogen_bond_pair: JaxArray,
    pair_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
) -> JaxArray:
    """Score-only primal; retaining XLA's fast non-gradient path helps MC."""

    displacement = coordinate[:, pair_i, :] - coordinate[:, pair_j, :]
    squared_distance = jnp.sum(jnp.square(displacement), axis=-1)
    distance = jnp.sqrt(
        jnp.maximum(squared_distance, jnp.finfo(coordinate.dtype).tiny)
    )
    surface = distance - pair_radius[None, :]
    gauss1 = jnp.exp(-jnp.square(surface / 0.5))
    gauss2 = jnp.exp(-jnp.square((surface - 3.0) / 2.0))
    repulsion = jnp.square(jnp.minimum(surface, 0.0))
    hydrophobic = jnp.clip(1.5 - surface, 0.0, 1.0)
    hydrogen_bond = jnp.clip(-surface / 0.7, 0.0, 1.0)
    active = pair_valid[None, :] & (distance < cutoff[0])
    score = (
        weights[0] * gauss1
        + weights[1] * gauss2
        + weights[2] * repulsion
        + weights[3]
        * jnp.where(hydrophobic_pair[None, :], hydrophobic, 0.0)
        + weights[4]
        * jnp.where(hydrogen_bond_pair[None, :], hydrogen_bond, 0.0)
    )
    return jnp.sum(jnp.where(active, score, 0.0), axis=1, keepdims=True)


def _pallas_intra_score_and_coordinate_gradient(
    coordinate: JaxArray,
    pair_i: JaxArray,
    pair_j: JaxArray,
    pair_radius: JaxArray,
    hydrophobic_pair: JaxArray,
    hydrogen_bond_pair: JaxArray,
    pair_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
    *,
    interpret: bool,
) -> tuple[JaxArray, JaxArray]:
    num_pair_blocks = pair_i.shape[0] // VINA_INTRA_PAIR_BLOCK_SIZE
    gradient_init = jnp.zeros_like(coordinate)
    block_score, coordinate_gradient = pl.pallas_call(
        _vina_intra_score_gradient_kernel,
        out_shape=(
            ShapeDtypeStruct(
                (coordinate.shape[0], num_pair_blocks), coordinate.dtype
            ),
            ShapeDtypeStruct(coordinate.shape, coordinate.dtype),
        ),
        grid=(coordinate.shape[0], num_pair_blocks),
        input_output_aliases={9: 1},
        interpret=interpret,
        compiler_params=pltriton.CompilerParams(num_warps=8, num_stages=2),
        name="vina_intra_score_gradient",
    )(
        coordinate,
        pair_i,
        pair_j,
        pair_radius,
        hydrophobic_pair,
        hydrogen_bond_pair,
        pair_valid,
        weights,
        cutoff,
        gradient_init,
    )
    return jnp.sum(block_score, axis=1, keepdims=True), coordinate_gradient


@partial(jax.custom_vjp, nondiff_argnums=(9,))
def vina_intra_energy(
    coordinate: JaxArray,
    pair_i: JaxArray,
    pair_j: JaxArray,
    pair_radius: JaxArray,
    hydrophobic_pair: JaxArray,
    hydrogen_bond_pair: JaxArray,
    pair_valid: JaxArray,
    weights: JaxArray,
    cutoff: JaxArray,
    interpret: bool,
) -> JaxArray:
    """Return intraligand energy with a fused pair-to-atom pullback."""

    return _jax_intra_score(
        coordinate,
        pair_i,
        pair_j,
        pair_radius,
        hydrophobic_pair,
        hydrogen_bond_pair,
        pair_valid,
        weights,
        cutoff,
    )


def _vina_intra_energy_fwd(
    coordinate,
    pair_i,
    pair_j,
    pair_radius,
    hydrophobic_pair,
    hydrogen_bond_pair,
    pair_valid,
    weights,
    cutoff,
    interpret,
):
    score, coordinate_gradient = _pallas_intra_score_and_coordinate_gradient(
        coordinate,
        pair_i,
        pair_j,
        pair_radius,
        hydrophobic_pair,
        hydrogen_bond_pair,
        pair_valid,
        weights,
        cutoff,
        interpret=interpret,
    )
    return score, coordinate_gradient


def _vina_intra_energy_bwd(
    interpret, coordinate_gradient, score_cotangent
):
    del interpret
    gradient = coordinate_gradient * score_cotangent[:, None, :]
    return (gradient,) + (None,) * 8


vina_intra_energy.defvjp(
    _vina_intra_energy_fwd,
    _vina_intra_energy_bwd,
)


__all__ = [
    "VINA_INTRA_PAIR_BLOCK_SIZE",
    "vina_interaction_energy",
    "vina_intra_energy",
]
