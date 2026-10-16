import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.ecepp import EyringInternalCoordinates, wrap_angles


def _sibling_geometry():
    return EyringInternalCoordinates(
        parent=jnp.array([0, 0, 0]),
        bond_length=jnp.array([0.0, 1.0, 2.0]),
        bond_angle=jnp.array([0.0, np.pi / 2, np.pi / 2]),
        base_torsion=jnp.array([0.0, 0.0, np.pi / 2]),
        variable_effects=jnp.array([[0.0], [1.0], [1.0]]),
        initial_variables=jnp.array([0.0]),
        root_position=jnp.array([1.0, 2.0, 3.0]),
        root_x_axis=jnp.array([1.0, 0.0, 0.0]),
        root_z_axis=jnp.array([0.0, 0.0, 1.0]),
        variable_names=("phi",),
    )


def _serial_coordinates(geometry, variables):
    """Reference implementation matching the original atom-ordered loop."""

    delta = wrap_angles(variables - geometry.initial_variables)
    torsion = geometry.base_torsion + geometry.variable_effects @ delta
    xyz = jnp.zeros((geometry.n_atoms, 3), dtype=variables.dtype)
    x_axis = jnp.zeros_like(xyz)
    z_axis = jnp.zeros_like(xyz)
    xyz = xyz.at[0].set(geometry.root_position)
    x_axis = x_axis.at[0].set(geometry.root_x_axis)
    z_axis = z_axis.at[0].set(geometry.root_z_axis)

    for atom in range(1, geometry.n_atoms):
        owner = int(geometry.parent[atom])
        y_axis = jnp.cross(z_axis[owner], x_axis[owner])
        angle = geometry.bond_angle[atom]
        dihedral = torsion[atom]
        new_x = (
            -jnp.cos(angle) * x_axis[owner]
            - jnp.sin(angle) * jnp.cos(dihedral) * y_axis
            - jnp.sin(angle) * jnp.sin(dihedral) * z_axis[owner]
        )
        new_x = new_x / jnp.linalg.norm(new_x)
        new_z = -jnp.sin(dihedral) * y_axis + jnp.cos(dihedral) * z_axis[owner]
        new_z = new_z / jnp.linalg.norm(new_z)
        xyz = xyz.at[atom].set(xyz[owner] + geometry.bond_length[atom] * new_x)
        x_axis = x_axis.at[atom].set(new_x)
        z_axis = z_axis.at[atom].set(new_z)
    return xyz


def test_eyring_geometry_rotates_direct_siblings_together():
    geometry = _sibling_geometry()
    reference = geometry.coordinates(jnp.array([0.0]))
    rotated = geometry.coordinates(jnp.array([np.pi / 2]))

    np.testing.assert_allclose(reference[1], [1.0, 1.0, 3.0], atol=2e-6)
    np.testing.assert_allclose(reference[2], [1.0, 2.0, 1.0], atol=2e-6)
    np.testing.assert_allclose(rotated[1], [1.0, 2.0, 2.0], atol=2e-6)
    np.testing.assert_allclose(rotated[2], [1.0, 4.0, 3.0], atol=2e-6)


def test_eyring_geometry_batches_and_differentiates():
    geometry = _sibling_geometry()
    variables = jnp.array([[0.0], [0.3], [-0.7]])
    assert geometry.coordinates(variables).shape == (3, 3, 3)
    gradient = jax.grad(lambda angle: jnp.sum(geometry.coordinates(angle) ** 2))(
        jnp.array([0.2])
    )
    assert np.all(np.isfinite(gradient))


def test_depth_batched_geometry_matches_serial_tree_and_gradient():
    geometry = EyringInternalCoordinates(
        parent=jnp.array([0, 0, 0, 1, 1, 2, 3]),
        bond_length=jnp.array([0.0, 1.1, 1.2, 1.3, 0.9, 1.0, 1.4]),
        bond_angle=jnp.array([0.0, 1.2, 1.0, 1.4, 1.1, 1.3, 0.8]),
        base_torsion=jnp.array([0.0, 0.2, -0.7, 1.1, -1.3, 0.4, 2.0]),
        variable_effects=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, -0.2],
                [-0.3, 0.7],
                [0.2, 0.4],
                [0.8, -0.1],
            ]
        ),
        initial_variables=jnp.array([0.1, -0.2]),
        root_position=jnp.array([0.3, -0.4, 0.7]),
        root_x_axis=jnp.array([1.0, 0.0, 0.0]),
        root_z_axis=jnp.array([0.0, 0.0, 1.0]),
        variable_names=("phi", "psi"),
    )
    variables = jnp.array([[0.2, -0.5], [-1.1, 0.8], [2.4, -2.2]])
    reference = jax.vmap(lambda value: _serial_coordinates(geometry, value))(variables)
    np.testing.assert_allclose(geometry.coordinates(variables), reference, atol=2e-6)

    value = variables[0]
    depth_batched_gradient = jax.grad(
        lambda current: jnp.sum(geometry.coordinates(current) ** 2)
    )(value)
    serial_gradient = jax.grad(
        lambda current: jnp.sum(_serial_coordinates(geometry, current) ** 2)
    )(value)
    np.testing.assert_allclose(
        depth_batched_gradient, serial_gradient, rtol=2e-5, atol=2e-5
    )


def test_wrap_angles_has_half_open_periodic_range():
    wrapped = wrap_angles(jnp.array([-3 * np.pi, -np.pi, np.pi, 3 * np.pi]))
    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi, -np.pi, -np.pi], atol=2e-6)
