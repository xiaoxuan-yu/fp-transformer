import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fp_transformer import ECEPPPotential
from fp_transformer.ecepp import (
    ECEPPParameters,
    ElectrostaticParameters,
    HydrogenBondParameters,
    LennardJonesParameters,
    TorsionParameters,
    ECEPP3Library,
    ECEPP3ParameterSet,
    dihedral_angles,
)
from fp_transformer.potential import ECEPPPotential as PotentialFromNamespace


def _empty_parameters(n_atoms, **overrides):
    values = {
        "n_atoms": n_atoms,
        "charges": jnp.zeros(n_atoms),
        "electrostatic": ElectrostaticParameters.create(),
        "lennard_jones": LennardJonesParameters.create(),
        "hydrogen_bond": HydrogenBondParameters.create(),
        "torsion": TorsionParameters.create(),
    }
    values.update(overrides)
    return ECEPPParameters(**values)


def test_public_imports_resolve_to_same_class():
    assert ECEPPPotential is PotentialFromNamespace


def test_pair_terms_match_classic_ecepp_formula():
    parameters = _empty_parameters(
        2,
        charges=jnp.array([1.0, -1.0]),
        electrostatic=ElectrostaticParameters.create([[0, 1]], [0.5]),
        lennard_jones=LennardJonesParameters.create(
            [[0, 1]], a=[8192.0], b=[64.0]
        ),
        hydrogen_bond=HydrogenBondParameters.create(
            [[0, 1]], c=[8192.0], d=[2048.0]
        ),
        dielectric=2.0,
    )
    components = ECEPPPotential(parameters)(
        jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), return_dict=True
    )

    np.testing.assert_allclose(
        components["electrostatic"], [-332.06371 / 8.0], rtol=2e-6
    )
    np.testing.assert_allclose(components["lennard_jones"], [1.0], rtol=2e-6)
    np.testing.assert_allclose(components["hydrogen_bond"], [0.0], atol=2e-6)
    expected = sum(
        np.asarray(components[name])
        for name in ("electrostatic", "lennard_jones", "hydrogen_bond", "torsion")
    )
    np.testing.assert_allclose(components["etotal"], expected)
    np.testing.assert_allclose(
        ECEPPPotential(parameters)(
            jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        ),
        components["etotal"],
        rtol=2e-6,
    )


@pytest.mark.parametrize("angle", [-2.1, -0.4, 0.0, 0.7, 2.5])
def test_signed_dihedral_and_torsion_formula(angle):
    coordinates = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, np.cos(angle), np.sin(angle)],
        ]
    )
    atoms = jnp.array([[0, 1, 2, 3]])
    np.testing.assert_allclose(
        dihedral_angles(coordinates[None], atoms), [[angle]], atol=2e-6
    )

    torsion = TorsionParameters.create(
        atoms, barrier_half=[0.7], periodicity=[3], phase=[0.25]
    )
    potential = ECEPPPotential(_empty_parameters(4, torsion=torsion))
    expected = 0.7 * (1.0 + np.cos(3.0 * angle - 0.25))
    np.testing.assert_allclose(potential(coordinates), [expected], rtol=2e-6)


def test_batch_jit_and_single_coordinate_shapes():
    parameters = _empty_parameters(
        2,
        charges=jnp.array([0.2, -0.3]),
        electrostatic=ElectrostaticParameters.create([[0, 1]]),
    )
    potential = ECEPPPotential(parameters)
    single = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    batch = jnp.stack([single, single * 2.0])

    assert potential(single).shape == (1,)
    assert potential(batch).shape == (2, 1)
    np.testing.assert_allclose(
        jax.jit(potential)(batch), potential(batch), rtol=2e-6
    )
    np.testing.assert_allclose(potential(batch)[0], potential(single), rtol=2e-6)


def test_energy_is_translation_and_rotation_invariant():
    parameters = _empty_parameters(
        4,
        charges=jnp.array([0.1, -0.2, 0.3, -0.4]),
        electrostatic=ElectrostaticParameters.create([[0, 2], [1, 3]]),
        lennard_jones=LennardJonesParameters.create(
            [[0, 3]], a=[7.0], b=[2.0]
        ),
        hydrogen_bond=HydrogenBondParameters.create(
            [[0, 2]], c=[3.0], d=[1.0]
        ),
        torsion=TorsionParameters.create(
            [[0, 1, 2, 3]], barrier_half=[0.8], periodicity=[2], phase=[np.pi]
        ),
    )
    potential = ECEPPPotential(parameters)
    coordinates = jnp.array(
        [[0.2, 0.4, 0.1], [1.0, 0.0, 0.0], [1.6, 0.8, 0.2], [2.0, 1.1, 1.0]]
    )
    rotation = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    reference = potential(coordinates)
    transformed = coordinates @ rotation.T + jnp.array([3.0, -2.0, 0.7])
    np.testing.assert_allclose(potential(transformed), reference, rtol=2e-5, atol=2e-5)


def test_autodiff_agrees_with_central_difference():
    parameters = _empty_parameters(
        2,
        charges=jnp.array([0.25, -0.4]),
        electrostatic=ElectrostaticParameters.create([[0, 1]]),
        lennard_jones=LennardJonesParameters.create(
            [[0, 1]], a=[8.0], b=[3.0]
        ),
    )
    potential = ECEPPPotential(parameters)
    coordinates = jnp.array([[0.0, 0.0, 0.0], [1.7, 0.2, -0.1]])
    gradient = jax.grad(lambda x: jnp.sum(potential(x)))(coordinates)

    epsilon = 2e-3
    plus = coordinates.at[1, 0].add(epsilon)
    minus = coordinates.at[1, 0].add(-epsilon)
    finite_difference = (potential(plus)[0] - potential(minus)[0]) / (2 * epsilon)
    np.testing.assert_allclose(gradient[1, 0], finite_difference, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), jnp.zeros(3), atol=2e-5)

    topology = type("Topology", (), {"coordinate": coordinates})()
    np.testing.assert_allclose(potential.force(topology), -gradient, rtol=2e-6)
    energy, force = potential.energy_and_force(coordinates)
    np.testing.assert_allclose(energy, potential(coordinates), rtol=2e-6)
    np.testing.assert_allclose(force, -gradient, rtol=2e-6)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ElectrostaticParameters.create([[0, 2]]), "outside"),
        (
            lambda: LennardJonesParameters.create([[0, 1]], a=[1.0, 2.0], b=[1.0]),
            "shape",
        ),
        (
            lambda: TorsionParameters.create(
                [[0, 1, 2, 3]], barrier_half=[1.0], periodicity=[0]
            ),
            "positive",
        ),
        (
            lambda: ElectrostaticParameters.create([[0, 1], [1, 0]]),
            "duplicate",
        ),
    ],
)
def test_parameter_validation(factory, match):
    if match == "outside":
        with pytest.raises(ValueError, match=match):
            _empty_parameters(2, electrostatic=factory())
    else:
        with pytest.raises(ValueError, match=match):
            factory()


def test_coordinate_validation():
    potential = ECEPPPotential(_empty_parameters(2))
    with pytest.raises(ValueError, match="expected 2 atoms"):
        potential(jnp.zeros((1, 3, 3)))
    with pytest.raises(ValueError, match="coordinates must have shape"):
        potential(jnp.zeros((2, 2)))


def test_smmp_ecepp3_library_parser():
    library = ECEPP3Library.from_smmp_text(
        """\
#ala 4 3
n 1.325 115.0 3*psi 180.0 -.356 13 0 2 3 0
h 1.0 124.0 0.0 .176 4 1 0 0 0
ca 1.453 121.0 1*omg 180.0 .064 7 1 4 0 0
c 1.53 109.3 2*phi 180.0 .45 8 3 -4 0 0
""",
        source="test-lib.sh3",
    )
    alanine = library.residues["ala"]
    assert library.source == "test-lib.sh3"
    assert alanine.declared_atom_count == 4
    assert alanine.atoms[0].variable_class == 3
    assert alanine.atoms[0].variable_name == "psi"
    assert alanine.atoms[1].variable_class is None
    assert alanine.atoms[2].atom_type == 7
    assert alanine.atoms[3].connectivity == (3, -4, 0, 0)


def test_ecepp3_parameter_list_compiles_lj_hbond_one_four_and_torsion():
    parameter_set = ECEPP3ParameterSet()
    parameters = parameter_set.compile(
        charges=[0.1, 0.2, -0.3, 0.4],
        atom_types=[7, 7, 3, 11],
        nonbonded_pairs=[[0, 1], [2, 3]],
        one_four_pairs=[[0, 3]],
        torsion_atoms=[[0, 1, 2, 3], [3, 2, 1, 0]],
        torsion_classes=[4, 2],
    )

    assert parameters.coulomb_constant == 332.0
    assert parameters.dielectric == 2.0
    np.testing.assert_array_equal(parameters.lennard_jones.pairs, [[0, 1], [0, 3]])
    np.testing.assert_array_equal(parameters.hydrogen_bond.pairs, [[2, 3]])
    np.testing.assert_allclose(parameters.lennard_jones.a[0], 906100.0)
    np.testing.assert_allclose(
        parameters.lennard_jones.a[1],
        0.5 * parameter_set.a12[6, 10],
        rtol=2e-6,
    )
    np.testing.assert_allclose(parameters.hydrogen_bond.c, [12040.0])
    np.testing.assert_allclose(parameters.hydrogen_bond.d, [4014.0])
    assert parameters.torsion.atoms.shape == (1, 4)
    np.testing.assert_allclose(parameters.torsion.barrier_half, [1.35])
    np.testing.assert_allclose(parameters.torsion.periodicity, [3.0])


def test_fused_total_matches_component_path_for_ecepp3():
    parameter_set = ECEPP3ParameterSet()
    parameters = parameter_set.compile(
        charges=[-0.2, 0.15, 0.25, -0.3, 0.1],
        atom_types=[7, 8, 3, 11, 9],
        nonbonded_pairs=[[0, 2], [0, 4], [1, 3], [2, 3]],
        one_four_pairs=[[1, 4]],
        torsion_atoms=[[0, 1, 2, 3]],
        torsion_classes=[4],
    )
    potential = ECEPPPotential(parameters)
    coordinates = jnp.array(
        [
            [0.0, 0.2, 0.1],
            [1.2, 0.0, -0.1],
            [1.8, 0.9, 0.2],
            [2.5, 1.1, 1.0],
            [0.4, 1.8, -0.5],
        ]
    )
    component_total = potential(coordinates, return_dict=True)["etotal"]
    np.testing.assert_allclose(potential(coordinates), component_total, rtol=2e-5)

    fused_gradient = jax.grad(lambda x: jnp.sum(potential(x)))(coordinates)
    component_gradient = jax.grad(
        lambda x: jnp.sum(potential(x, return_dict=True)["etotal"])
    )(coordinates)
    np.testing.assert_allclose(fused_gradient, component_gradient, rtol=2e-5, atol=2e-5)
