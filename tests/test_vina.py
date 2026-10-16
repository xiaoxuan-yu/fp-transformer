import jax
import jax.numpy as jnp
import pytest

from fp_transformer.potential import (
    VinaScore,
    VinaWeights,
    build_ligand_intra_pairs,
    vina_pairwise_terms,
)
from fp_transformer.potential.Vina import _build_torsion_coordinates


def test_pairwise_terms_match_vina_piecewise_definitions():
    d = jnp.array([-1.0, -0.7, 0.0, 0.5, 1.0, 1.5, 3.0])
    terms = vina_pairwise_terms(d)

    assert jnp.allclose(terms["gauss1"], jnp.exp(-jnp.square(d / 0.5)))
    assert jnp.allclose(terms["gauss2"], jnp.exp(-jnp.square((d - 3) / 2)))
    assert jnp.allclose(
        terms["repulsion"], jnp.array([1.0, 0.49, 0, 0, 0, 0, 0])
    )
    assert jnp.allclose(
        terms["hydrophobic"], jnp.array([1, 1, 1, 1, 0.5, 0, 0])
    )
    assert jnp.allclose(
        terms["hydrogen_bond"], jnp.array([1, 1, 0, 0, 0, 0, 0])
    )


def test_build_ligand_intra_pairs_excludes_12_13_14_and_rigid_pairs():
    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [3, 5]])
    rotation_masks = jnp.array([[False, False, True, True, True, True]])
    pairs = build_ligand_intra_pairs(6, bonds, rotation_masks)
    assert jnp.array_equal(pairs, jnp.array([[0, 4], [0, 5]]))


def test_score_applies_atom_features_and_weights():
    weights = VinaWeights(
        gauss1=1.0,
        gauss2=2.0,
        repulsion=3.0,
        hydrophobic=4.0,
        hydrogen_bond=5.0,
        rotatable_bond=0.5,
    )
    potential = VinaScore(
        receptor_coordinate=jnp.array([[0.0, 0.0, 0.0]]),
        receptor_vdw_radii=jnp.array([1.0]),
        ligand_vdw_radii=jnp.array([1.0]),
        ligand_reference_coordinate=jnp.array([[1.3, 0.0, 0.0]]),
        rotatable_bonds=jnp.empty((0, 2), dtype=int),
        rotation_masks=jnp.empty((0, 1), dtype=bool),
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
        receptor_hydrophobic=jnp.array([True]),
        ligand_hydrophobic=jnp.array([True]),
        receptor_hbond_donor=jnp.array([True]),
        ligand_hbond_acceptor=jnp.array([True]),
        weights=weights,
    )
    torsion = jnp.empty((2, 1, 0))
    components = potential.components(torsion)
    expected_terms = vina_pairwise_terms(jnp.array(-0.7))
    for name in expected_terms:
        assert jnp.allclose(components[name][0, 0], expected_terms[name])

    expected = sum(
        weight * expected_terms[name]
        for name, weight in {
            "gauss1": 1.0,
            "gauss2": 2.0,
            "repulsion": 3.0,
            "hydrophobic": 4.0,
            "hydrogen_bond": 5.0,
        }.items()
    )
    score = potential(torsion)
    assert score.shape == (2, 1)
    assert jnp.allclose(score[0, 0], expected)
    assert jnp.allclose(score[1, 0], expected)


def test_hbond_accepts_either_donor_acceptor_orientation():
    common = dict(
        receptor_coordinate=jnp.array([[0.0, 0.0, 0.0]]),
        receptor_vdw_radii=jnp.array([1.0]),
        ligand_vdw_radii=jnp.array([1.0]),
        ligand_reference_coordinate=jnp.array([[1.3, 0.0, 0.0]]),
        rotatable_bonds=jnp.empty((0, 2), dtype=int),
        rotation_masks=jnp.empty((0, 1), dtype=bool),
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
    )
    torsion = jnp.empty((1, 0))
    receptor_donor = VinaScore(
        **common,
        receptor_hbond_donor=[True],
        ligand_hbond_acceptor=[True],
    )
    ligand_donor = VinaScore(
        **common,
        receptor_hbond_acceptor=[True],
        ligand_hbond_donor=[True],
    )
    assert receptor_donor.components(torsion)["hydrogen_bond"][0, 0] == 1
    assert ligand_donor.components(torsion)["hydrogen_bond"][0, 0] == 1


def test_score_is_jittable_and_differentiable():
    potential = VinaScore(
        receptor_coordinate=jnp.array([[0.0, 1.0, 0.0]]),
        receptor_vdw_radii=jnp.array([1.5]),
        ligand_vdw_radii=jnp.array([1.5, 1.5, 1.5]),
        ligand_reference_coordinate=jnp.array(
            [[2.5, 0.0, 0.0], [3.5, 0.0, 0.0], [3.5, 1.0, 0.0]]
        ),
        rotatable_bonds=jnp.array([[0, 1]]),
        rotation_masks=jnp.array([[False, True, True]]),
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
    )
    torsion = jnp.array([[[0.2]]])

    assert jnp.allclose(jax.jit(potential)(torsion), potential(torsion))
    gradient = jax.grad(lambda x: jnp.sum(potential(x)))(torsion)
    assert gradient.shape == torsion.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert not jnp.allclose(gradient, 0)


def test_pallas_interaction_score_and_gradient_match_jax_backend():
    arguments = dict(
        receptor_coordinate=jnp.array(
            [
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.5],
                [4.0, 2.0, -0.5],
                [-2.0, 0.5, 1.0],
                [12.0, 0.0, 0.0],
            ]
        ),
        receptor_vdw_radii=[1.5, 1.7, 1.8, 1.6, 1.5],
        ligand_vdw_radii=[1.5, 1.6, 1.7],
        ligand_reference_coordinate=jnp.array(
            [[2.5, 0.0, 0.0], [3.5, 0.0, 0.0], [3.5, 1.0, 0.5]]
        ),
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False, True, True]],
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
        receptor_hydrophobic=[True, False, True, False, True],
        ligand_hydrophobic=[True, False, True],
        receptor_hbond_donor=[False, True, False, False, False],
        receptor_hbond_acceptor=[True, False, False, True, False],
        ligand_hbond_donor=[False, True, False],
        ligand_hbond_acceptor=[True, False, True],
    )
    reference = VinaScore(**arguments, interaction_backend="jax")
    fused = VinaScore(**arguments, interaction_backend="pallas")
    torsion = jnp.array([[[-1.1]], [[0.2]], [[2.4]]])

    reference_score = jax.jit(reference)(torsion)
    fused_score = jax.jit(fused)(torsion)
    reference_gradient = jax.jit(
        jax.grad(lambda value: jnp.sum(reference(value)))
    )(torsion)
    fused_gradient = jax.jit(
        jax.grad(lambda value: jnp.sum(fused(value)))
    )(torsion)

    assert jnp.allclose(fused_score, reference_score, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(
        fused_gradient, reference_gradient, rtol=3e-5, atol=3e-5
    )


def test_pallas_intraligand_gradient_matches_jax_across_pair_tiles():
    num_atoms = 24
    atom = jnp.arange(num_atoms, dtype=jnp.float32)
    coordinate = jnp.stack(
        (0.35 * atom, jnp.sin(0.4 * atom), jnp.cos(0.3 * atom)), axis=1
    )
    intra_pairs = jnp.asarray(
        [(i, j) for i in range(num_atoms) for j in range(i + 1, num_atoms)]
    )
    arguments = dict(
        receptor_coordinate=jnp.array([[100.0, 0.0, 0.0]]),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.5] * num_atoms,
        ligand_reference_coordinate=coordinate,
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False] + [True] * (num_atoms - 1)],
        ligand_intra_pairs=intra_pairs,
    )
    reference = VinaScore(**arguments, interaction_backend="jax")
    fused = VinaScore(**arguments, interaction_backend="pallas")
    torsion = jnp.array([[[-1.1]], [[0.2]], [[1.7]]])

    reference_score = jax.jit(reference)(torsion)
    fused_score = jax.jit(fused)(torsion)
    reference_gradient = jax.jit(
        jax.grad(lambda value: jnp.sum(reference(value)))
    )(torsion)
    fused_gradient = jax.jit(
        jax.grad(lambda value: jnp.sum(fused(value)))
    )(torsion)

    assert intra_pairs.shape[0] > 256
    assert jnp.allclose(fused_score, reference_score, rtol=5e-5, atol=2e-4)
    assert jnp.allclose(
        fused_gradient, reference_gradient, rtol=5e-5, atol=2e-4
    )


@pytest.mark.parametrize("interaction_backend", ["jax", "pallas"])
def test_total_binding_value_and_grad_share_results(interaction_backend):
    potential = VinaScore(
        receptor_coordinate=jnp.array(
            [[0.0, 1.0, 0.0], [2.0, -1.0, 0.5], [4.0, 2.0, -0.5]]
        ),
        receptor_vdw_radii=[1.5, 1.7, 1.8],
        ligand_vdw_radii=[1.5, 1.6, 1.7, 1.5],
        ligand_reference_coordinate=jnp.array(
            [
                [2.5, 0.0, 0.0],
                [3.5, 0.0, 0.0],
                [3.5, 1.0, 0.5],
                [2.5, 1.5, 0.3],
            ]
        ),
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False, True, True, True]],
        ligand_intra_pairs=[[0, 3]],
        receptor_hydrophobic=[True, False, True],
        ligand_hydrophobic=[True, False, True, False],
        interaction_backend=interaction_backend,
    )
    torsion = jnp.array([[[-1.1]], [[0.2]], [[2.4]]])

    total, binding = jax.jit(potential.total_and_binding_score)(torsion)
    (combined_total, combined_binding), combined_gradient = jax.jit(
        potential.value_and_grad
    )(torsion)
    expected_gradient = jax.jit(
        jax.grad(lambda value: jnp.sum(potential(value)))
    )(torsion)

    assert jnp.allclose(total, potential(torsion), rtol=2e-5, atol=2e-5)
    assert jnp.allclose(
        binding, potential.binding_score(torsion), rtol=2e-5, atol=2e-5
    )
    assert jnp.allclose(combined_total, total, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(combined_binding, binding, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(
        combined_gradient, expected_gradient, rtol=3e-5, atol=3e-5
    )


def test_torsion_coordinate_custom_vjp_matches_reference_autodiff():
    potential = VinaScore(
        receptor_coordinate=jnp.array([[8.0, 0.0, 0.0]]),
        receptor_vdw_radii=[1.5],
        ligand_vdw_radii=[1.5] * 5,
        ligand_reference_coordinate=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.5],
                [2.5, 1.5, 1.0],
            ]
        ),
        rotatable_bonds=[[0, 1], [1, 2]],
        rotation_masks=[
            [False, True, True, True, True],
            [False, False, True, True, True],
        ],
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
    )
    torsion = jax.random.uniform(
        jax.random.PRNGKey(31),
        shape=(4, 2),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(32), shape=(4, 5, 3))
    _, reference_pullback = jax.vjp(
        lambda value: _build_torsion_coordinates(
            value,
            potential.ligand_reference_coordinate,
            potential.rotatable_bonds,
            potential.rotation_masks,
        ),
        torsion,
    )
    _, analytic_pullback = jax.vjp(potential.coordinates, torsion)
    expected = reference_pullback(cotangent)[0]
    actual = analytic_pullback(cotangent)[0]
    assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_torsion_is_the_only_degree_of_freedom_and_uses_sampler_shape():
    potential = VinaScore(
        receptor_coordinate=jnp.array([[10.0, 0.0, 0.0]]),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.0, 1.0, 1.0],
        ligand_reference_coordinate=jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        ),
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False, True, True]],
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
    )

    coordinate = potential.coordinates(jnp.array([[[jnp.pi / 2]]]))
    expected = jnp.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]
    )
    assert jnp.allclose(coordinate, expected, atol=1e-6)


def test_rotatable_penalty_is_derived_from_number_of_torsions():
    weights = VinaWeights(
        gauss1=1.0,
        gauss2=0.0,
        repulsion=0.0,
        hydrophobic=0.0,
        hydrogen_bond=0.0,
        rotatable_bond=0.5,
    )
    potential = VinaScore(
        receptor_coordinate=jnp.array([[0.0, 1.0, 0.0]]),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.0, 1.0, 1.0],
        ligand_reference_coordinate=jnp.array(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]]
        ),
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False, True, True]],
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
        weights=weights,
    )
    torsion = jnp.array([[0.3]])
    raw_gauss = potential.components(torsion)["gauss1"]
    assert jnp.allclose(potential(torsion), raw_gauss)
    assert jnp.allclose(potential.binding_score(torsion), raw_gauss / 1.5)


def test_intramolecular_interaction_changes_with_torsion_and_backpropagates():
    weights = VinaWeights(
        gauss1=0.0,
        gauss2=0.0,
        repulsion=1.0,
        hydrophobic=0.0,
        hydrogen_bond=0.0,
        rotatable_bond=0.0,
    )
    potential = VinaScore(
        receptor_coordinate=jnp.array([[100.0, 0.0, 0.0]]),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.0, 1.0, 1.0, 1.0],
        ligand_reference_coordinate=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        rotatable_bonds=[[0, 1]],
        rotation_masks=[[False, True, True, False]],
        ligand_intra_pairs=[[2, 3]],
        weights=weights,
    )

    torsion = jnp.array([[0.3]])
    energies = potential.energy_components(torsion)
    assert energies["intermolecular"][0, 0] == 0
    assert energies["intramolecular"][0, 0] > 0
    assert jnp.allclose(energies["total"], potential(torsion))
    gradient = jax.grad(lambda value: jnp.sum(potential(value)))(torsion)
    assert jnp.all(jnp.isfinite(gradient))
    assert not jnp.allclose(gradient, 0)

    open_torsion = jnp.array([[jnp.pi]])
    assert potential(open_torsion)[0, 0] < potential(jnp.zeros((1, 1)))[0, 0]


def test_intramolecular_pair_features_use_ligand_atom_masks():
    potential = VinaScore(
        receptor_coordinate=jnp.array([[100.0, 0.0, 0.0]]),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.0, 1.0],
        ligand_reference_coordinate=jnp.array(
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]
        ),
        rotatable_bonds=jnp.empty((0, 2), dtype=int),
        rotation_masks=jnp.empty((0, 2), dtype=bool),
        ligand_intra_pairs=[[0, 1]],
        ligand_hydrophobic=[True, True],
        ligand_hbond_donor=[True, False],
        ligand_hbond_acceptor=[False, True],
    )
    terms = potential.intramolecular_components(jnp.empty((1, 0)))
    assert terms["hydrophobic"][0, 0] == 1
    assert terms["hydrogen_bond"][0, 0] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"receptor_vdw_radii": [1.0, 1.0]}, "receptor_vdw_radii"),
        ({"ligand_hydrophobic": [True, False]}, "ligand_hydrophobic"),
        ({"rotation_masks": [[True]]}, "rotation_masks"),
        ({"ligand_intra_pairs": [[0, 0]]}, "i < j"),
        ({"cutoff": 0.0}, "positive"),
        ({"interaction_backend": "cuda"}, "interaction_backend"),
    ],
)
def test_input_validation(kwargs, message):
    arguments = dict(
        receptor_coordinate=jnp.zeros((1, 3)),
        receptor_vdw_radii=[1.0],
        ligand_vdw_radii=[1.0],
        ligand_reference_coordinate=jnp.zeros((1, 3)),
        rotatable_bonds=jnp.empty((0, 2), dtype=int),
        rotation_masks=jnp.empty((0, 1), dtype=bool),
        ligand_intra_pairs=jnp.empty((0, 2), dtype=int),
    )
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        VinaScore(**arguments)
