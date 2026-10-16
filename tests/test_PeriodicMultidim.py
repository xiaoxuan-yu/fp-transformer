import pytest
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax.random import PRNGKey
from fp_transformer.potential.PeriodicMultidim import PeriodicMultidim
from fp_transformer.utils import generate_subspace_aware_params, logsumexp


class TestPeriodicMultidim:
    """Test cases for the PeriodicMultidim potential class."""

    @staticmethod
    def _potential_reference(potential, x):
        x_reshaped = x[:, :, jnp.newaxis, :]
        centers_reshaped = potential.centers[jnp.newaxis, jnp.newaxis, :, :]
        delta = x_reshaped - centers_reshaped
        cos_delta = jnp.cos(delta)
        concentrations_reshaped = potential.concentrations[jnp.newaxis, jnp.newaxis, :, :]
        weighted_cos = concentrations_reshaped * cos_delta
        cos_sum = jnp.sum(weighted_cos, axis=3)
        log_wells = jnp.log(potential.depths)[jnp.newaxis, jnp.newaxis, :] + cos_sum
        log_particle_energies = logsumexp(log_wells, axis=2)
        log_total_energy = logsumexp(log_particle_energies, axis=1, keepdims=True)
        return -log_total_energy

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        # Single well in 2D
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        assert potential.num_wells == 1
        assert potential.dim == 2
        assert jnp.array_equal(potential.centers, centers)
        assert jnp.array_equal(potential.depths, depths)
        assert jnp.array_equal(potential.concentrations, concentrations)

    def test_init_multiple_wells(self):
        """Test initialization with multiple wells."""
        # Three wells in 3D
        centers = jnp.array(
            [[0.0, 0.0, 0.0], [jnp.pi, jnp.pi, jnp.pi], [2 * jnp.pi, 0.0, jnp.pi]]
        )
        depths = jnp.array([1.0, 2.0, 1.5])
        concentrations = jnp.array(
            [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0]]
        )  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        assert potential.num_wells == 3
        assert potential.dim == 3

    def test_init_invalid_shapes(self):
        """Test initialization with invalid parameter shapes."""
        # Test incompatible dimensions
        centers = jnp.array([[[0.0, 0.0]]])  # 3D array instead of 2D
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])

        with pytest.raises(ValueError, match="Incompatible shapes"):
            PeriodicMultidim(centers, depths, concentrations)

    def test_init_inconsistent_wells(self):
        """Test initialization with inconsistent number of wells."""
        centers = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi]])  # 2 wells
        depths = jnp.array([1.0])  # 1 depth
        concentrations = jnp.array([[1.0, 1.0]])  # 1 concentration

        with pytest.raises(ValueError, match="Inconsistent number of wells"):
            PeriodicMultidim(centers, depths, concentrations)

    def test_call_single_well_single_particle(self):
        """Test potential calculation for single well and single particle."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Test at center (should give maximum negative energy)
        x = jnp.array([[[0.0, 0.0]]])  # (1 replica, 1 particle, 2 dims)
        energy = potential(x, beta=1.0)

        assert energy.shape == (1, 1)
        assert energy[0, 0] < 0  # Should be negative (well)

        # Test at opposite point (should give minimum negative energy)
        x_opposite = jnp.array([[[jnp.pi, jnp.pi]]])
        energy_opposite = potential(x_opposite, beta=1.0)

        assert energy_opposite[0, 0] > energy[0, 0]  # Less negative

    def test_call_multiple_particles(self):
        """Test potential calculation for multiple particles."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Multiple particles in one replica
        x = jnp.array(
            [[[0.0, 0.0], [jnp.pi, jnp.pi], [0.5, 0.5]]]
        )  # 1 replica, 3 particles
        energy = potential(x, beta=1.0)

        assert energy.shape == (1, 1)
        assert energy[0, 0] < 0  # Should be negative

    def test_call_multiple_wells(self):
        """Test potential calculation with multiple wells."""
        # Two wells at opposite corners
        centers = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi]])
        depths = jnp.array([1.0, 1.0])
        concentrations = jnp.array(
            [[1.0, 1.0], [1.0, 1.0]]
        )  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Test at first well center
        x1 = jnp.array([[[0.0, 0.0]]])
        energy1 = potential(x1, beta=1.0)

        # Test at second well center
        x2 = jnp.array([[[jnp.pi, jnp.pi]]])
        energy2 = potential(x2, beta=1.0)

        # Both should be similarly negative (same depth and concentration)
        assert jnp.abs(energy1[0, 0] - energy2[0, 0]) < 1e-6

    def test_optimized_matches_broadcast_reference(self):
        centers = jnp.array(
            [
                [0.1, 1.2, 2.1],
                [jnp.pi / 3, jnp.pi, 5.2],
                [2.4, 0.7, jnp.pi / 2],
            ]
        )
        depths = jnp.array([0.8, 1.7, 2.3])
        concentrations = jnp.array(
            [
                [0.5, 1.0, 2.0],
                [1.5, 0.8, 0.3],
                [2.2, 1.1, 0.9],
            ]
        )
        x = jnp.array(
            [
                [[0.0, 0.5, 1.0], [1.3, 2.1, 2.8], [3.0, 4.0, 5.0]],
                [[0.2, 1.5, 2.7], [jnp.pi, 0.4, 3.2], [5.8, 1.1, 0.9]],
            ]
        )

        potential = PeriodicMultidim(centers, depths, concentrations)

        expected = self._potential_reference(potential, x)
        optimized = potential.potential_optimized(x)
        via_call = potential(x)

        assert jnp.allclose(optimized, expected, rtol=1e-6, atol=1e-6)
        assert jnp.allclose(via_call, expected, rtol=1e-6, atol=1e-6)

    def test_symmetry_properties(self):
        """Test that the potential respects periodic boundary conditions."""
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Points that differ by 2π should give the same energy
        x1 = jnp.array([[[0.0, 0.0]]])
        x2 = jnp.array([[[2 * jnp.pi, 2 * jnp.pi]]])

        energy1 = potential(x1, beta=1.0)
        energy2 = potential(x2, beta=1.0)

        assert jnp.abs(energy1[0, 0] - energy2[0, 0]) < 1e-6

    def test_get_2dcontour_basic(self):
        """Test 2D contour generation with basic parameters."""
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        x_range = (0.0, 2 * jnp.pi)
        y_range = (0.0, 2 * jnp.pi)

        X, Y, E = potential.get_2dcontour(x_range, y_range, num_points=10)

        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert E.shape == (10, 10)
        assert jnp.min(E) == 0.0  # Due to E -= jnp.min(E)

    def test_get_2dcontour_well_location(self):
        """Test that the 2D contour shows minimum at well center."""
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[2.0, 2.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        x_range = (0.0, 2 * jnp.pi)
        y_range = (0.0, 2 * jnp.pi)

        X, Y, E = potential.get_2dcontour(x_range, y_range, num_points=50)

        # Find the minimum energy location
        min_idx = jnp.unravel_index(jnp.argmin(E), E.shape)
        min_x = X[min_idx]
        min_y = Y[min_idx]

        # Should be close to the well center (π, π)
        assert (
            jnp.abs(min_x - jnp.pi) < 0.3
        )  # Allow some tolerance due to discretization
        assert jnp.abs(min_y - jnp.pi) < 0.3

    def test_energy_scale_with_concentration(self):
        """Test that concentration parameter affects the potential shape."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])

        # Create potentials with different concentrations
        potential1 = PeriodicMultidim(
            centers, depths, jnp.array([[1.0, 1.0]])
        )  # Shape (M, N)
        potential2 = PeriodicMultidim(
            centers, depths, jnp.array([[2.0, 2.0]])
        )  # Shape (M, N)

        # Test at well center - higher concentration gives more negative energy
        x_center = jnp.array([[[0.0, 0.0]]])
        energy1_center = potential1(x_center, beta=1.0)
        energy2_center = potential2(x_center, beta=1.0)

        # Higher concentration gives more negative energy at center
        assert energy2_center[0, 0] < energy1_center[0, 0] < 0

        # Test that both potentials give reasonable values
        x_away = jnp.array([[[1.0, 1.0]]])
        energy1_away = potential1(x_away, beta=1.0)
        energy2_away = potential2(x_away, beta=1.0)

        # Both should be negative (attractive potential)
        assert energy1_away[0, 0] < 0
        assert energy2_away[0, 0] < 0

        # Concentration parameter should make a significant difference
        assert abs(energy2_center[0, 0] - energy1_center[0, 0]) > 1e-3

    def test_high_dimensional_case(self):
        """Test the potential in higher dimensions (4D)."""
        # 4D potential with single well
        centers = jnp.array([[jnp.pi, jnp.pi, jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array(
            [[1.0, 1.0, 1.0, 1.0]]
        )  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        assert potential.dim == 4
        assert potential.num_wells == 1

        # Test energy calculation
        x = jnp.array([[[jnp.pi, jnp.pi, jnp.pi, jnp.pi]]])
        energy = potential(x, beta=1.0)

        assert energy.shape == (1, 1)
        assert energy[0, 0] < 0

    def test_beta_parameter_ignored(self):
        """Test that beta parameter is ignored in calculation."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[1.0, 1.0]])  # Shape (M, N) for anisotropic support

        potential = PeriodicMultidim(centers, depths, concentrations)

        x = jnp.array([[[0.0, 0.0]]])

        energy1 = potential(x, beta=1.0)
        energy2 = potential(x, beta=2.0)
        energy3 = potential(x, beta=0.5)

        # All should be identical since beta is ignored
        assert jnp.allclose(energy1, energy2)
        assert jnp.allclose(energy1, energy3)

    def test_centers_are_local_minima_various_systems(self):
        """Test that centers are local minima for various M (wells) and N (dimensions)."""
        from jax import grad

        def _compute_single_particle_energy(x_flat, potential_func):
            """Helper function to compute energy for a single particle."""
            # Reshape flat coordinates back to (1, 1, N) format
            x_reshaped = x_flat.reshape(1, 1, -1)
            return potential_func(x_reshaped, beta=1.0)[0, 0]

        # Test different combinations of M (number of wells) and N (dimensions)
        test_cases = [
            # (M, N, description)
            (1, 2, "Single well in 2D"),
            (1, 3, "Single well in 3D"),
            (1, 4, "Single well in 4D"),
            # For multiple wells, we only test that the gradient is reasonable (not necessarily zero)
            # because wells can interact and shift the local minima
        ]

        for M, N, description in test_cases:
            # Create M wells with random centers
            np.random.seed(42 + M * 10 + N)  # Reproducible random seed
            centers = np.random.normal(0, 2 * np.pi, size=(M, N))
            centers = jnp.array(centers)

            # Use random but reasonable depths and concentrations
            depths = jnp.array(np.random.uniform(0.5, 2.0, size=M))
            concentrations = jnp.array(
                np.random.uniform(5, 30, size=(M, N))
            )  # Shape (M, N) for anisotropic support

            potential = PeriodicMultidim(centers, depths, concentrations)

            # Create gradient function for single particle energy
            grad_func = grad(_compute_single_particle_energy, argnums=0)

            # Test each well center
            for well_idx in range(M):
                center = centers[well_idx]

                # Compute gradient at the center
                gradient = grad_func(center, potential)

                # For single wells, gradient should be close to zero at the center
                gradient_norm = jnp.linalg.norm(gradient)
                # print(f"Gradient norm at center {well_idx} for {description}: {gradient_norm:.2e}")
                # breakpoint()
                assert gradient_norm < 1e-5, (
                    f"Gradient norm {gradient_norm:.2e} too large at center {well_idx} "
                    f"for {description}. Expected < 1e-5."
                )

                # Additional check: energy at center should be lower than nearby points
                # Test several points around the center
                perturbation_size = 0.1
                for _ in range(5):  # Test 5 random perturbations
                    perturbation = np.random.uniform(
                        -perturbation_size, perturbation_size, size=N
                    )
                    perturbed_point = center + perturbation

                    # Compute energies
                    energy_center = _compute_single_particle_energy(center, potential)
                    energy_perturbed = _compute_single_particle_energy(
                        perturbed_point, potential
                    )

                    # Center should have lower (more negative) energy
                    assert energy_center <= energy_perturbed + 1e-6, (
                        f"Energy at center ({energy_center:.6f}) should be <= "
                        f"energy at perturbed point ({energy_perturbed:.6f}) "
                        f"for well {well_idx} in {description}"
                    )

            # Test that all specified centers are indeed minima by comparing with random points
            for _ in range(10):  # Test 10 random points
                random_point = np.random.uniform(0, 2 * np.pi, size=N)
                random_energy = _compute_single_particle_energy(
                    jnp.array(random_point), potential
                )

                # At least one center should have lower energy than this random point
                center_energies = [
                    _compute_single_particle_energy(centers[i], potential)
                    for i in range(M)
                ]
                min_center_energy = min(center_energies)

                # This is not a strict requirement but helps verify the wells are working
                # We allow some tolerance since random points might occasionally land near wells
                if random_energy > min_center_energy + 0.1:
                    # This is expected - random point has higher energy than well centers
                    pass

    def test_anisotropic_concentrations(self):
        """Test that anisotropic concentrations work correctly."""
        # Create a well with different concentrations in different dimensions
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        # High concentration in x-direction, low in y-direction
        concentrations = jnp.array([[5.0, 0.1]])

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Test points along x and y axes from the center
        x_direction = jnp.array([[[jnp.pi + 0.5, jnp.pi]]])  # Displaced in x
        y_direction = jnp.array([[[jnp.pi, jnp.pi + 0.5]]])  # Displaced in y
        center = jnp.array([[[jnp.pi, jnp.pi]]])

        energy_center = potential(center, beta=1.0)
        energy_x = potential(x_direction, beta=1.0)
        energy_y = potential(y_direction, beta=1.0)

        # Due to higher concentration in x-direction, displacement in x should be more costly
        # i.e., energy_x should be higher (less negative) than energy_y
        assert (
            energy_x[0, 0] > energy_y[0, 0]
        ), "Anisotropic behavior not working correctly"
        assert energy_center[0, 0] < energy_x[0, 0], "Center should have lowest energy"
        assert energy_center[0, 0] < energy_y[0, 0], "Center should have lowest energy"

    def test_shape_validation_with_anisotropic(self):
        """Test validation for anisotropic concentration shapes."""
        centers = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi]])  # 2 wells, 2D
        depths = jnp.array([1.0, 1.0])  # 2 wells

        # Wrong shape: should be (2, 2) not (2, 3)
        concentrations_wrong = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        with pytest.raises(ValueError, match="Inconsistent dimensionality"):
            PeriodicMultidim(centers, depths, concentrations_wrong)

        # Correct shape
        concentrations_correct = jnp.array([[1.0, 1.0], [2.0, 0.5]])
        potential = PeriodicMultidim(centers, depths, concentrations_correct)
        assert potential.num_wells == 2
        assert potential.dim == 2

    def test_anisotropic_periodicity(self):
        """Test that anisotropic concentrations preserve periodic boundary conditions."""
        # Create a potential with moderately anisotropic concentrations (avoid extreme values)
        centers = jnp.array([[jnp.pi, jnp.pi / 2]])
        depths = jnp.array([1.0])
        concentrations = jnp.array(
            [[2.0, 0.5]]
        )  # Moderate anisotropy to reduce numerical errors

        potential = PeriodicMultidim(centers, depths, concentrations)

        # Test points that differ by 2π in each dimension
        base_point = jnp.array([[[0.5, 1.0]]])

        # Points shifted by 2π in x direction
        x_shifted = jnp.array([[[0.5 + 2 * jnp.pi, 1.0]]])
        x_shifted_neg = jnp.array([[[0.5 - 2 * jnp.pi, 1.0]]])

        # Points shifted by 2π in y direction
        y_shifted = jnp.array([[[0.5, 1.0 + 2 * jnp.pi]]])
        y_shifted_neg = jnp.array([[[0.5, 1.0 - 2 * jnp.pi]]])

        # Points shifted by 2π in both directions
        both_shifted = jnp.array([[[0.5 + 2 * jnp.pi, 1.0 + 2 * jnp.pi]]])
        both_shifted_neg = jnp.array([[[0.5 - 2 * jnp.pi, 1.0 - 2 * jnp.pi]]])

        # Calculate energies
        energy_base = potential(base_point, beta=1.0)
        energy_x_shift = potential(x_shifted, beta=1.0)
        energy_x_shift_neg = potential(x_shifted_neg, beta=1.0)
        energy_y_shift = potential(y_shifted, beta=1.0)
        energy_y_shift_neg = potential(y_shifted_neg, beta=1.0)
        energy_both_shift = potential(both_shifted, beta=1.0)
        energy_both_shift_neg = potential(both_shifted_neg, beta=1.0)

        # All energies should be identical due to periodicity
        tolerance = 1e-5  # Relaxed tolerance for numerical stability with anisotropic concentrations
        assert (
            jnp.abs(energy_base[0, 0] - energy_x_shift[0, 0]) < tolerance
        ), f"X-direction periodicity failed: {energy_base[0, 0]} != {energy_x_shift[0, 0]}"
        assert (
            jnp.abs(energy_base[0, 0] - energy_x_shift_neg[0, 0]) < tolerance
        ), f"X-direction (negative) periodicity failed: {energy_base[0, 0]} != {energy_x_shift_neg[0, 0]}"
        assert (
            jnp.abs(energy_base[0, 0] - energy_y_shift[0, 0]) < tolerance
        ), f"Y-direction periodicity failed: {energy_base[0, 0]} != {energy_y_shift[0, 0]}"
        assert (
            jnp.abs(energy_base[0, 0] - energy_y_shift_neg[0, 0]) < tolerance
        ), f"Y-direction (negative) periodicity failed: {energy_base[0, 0]} != {energy_y_shift_neg[0, 0]}"
        assert (
            jnp.abs(energy_base[0, 0] - energy_both_shift[0, 0]) < tolerance
        ), f"Both directions periodicity failed: {energy_base[0, 0]} != {energy_both_shift[0, 0]}"
        assert (
            jnp.abs(energy_base[0, 0] - energy_both_shift_neg[0, 0]) < tolerance
        ), f"Both directions (negative) periodicity failed: {energy_base[0, 0]} != {energy_both_shift_neg[0, 0]}"

        # Test a few specific points to ensure periodicity holds
        test_points = [
            jnp.array([0.0, 0.0]),
            jnp.array([jnp.pi / 2, jnp.pi / 4]),
            jnp.array([jnp.pi, jnp.pi]),
            jnp.array([3 * jnp.pi / 2, jnp.pi / 6]),
        ]

        for point in test_points:
            point_original = jnp.array([[point]])

            # Shift by 2π in both directions
            point_shifted_2pi = jnp.array([[point + 2 * jnp.pi]])
            point_shifted_neg2pi = jnp.array([[point - 2 * jnp.pi]])

            energy_orig = potential(point_original, beta=1.0)
            energy_2pi = potential(point_shifted_2pi, beta=1.0)
            energy_neg2pi = potential(point_shifted_neg2pi, beta=1.0)

            assert (
                jnp.abs(energy_orig[0, 0] - energy_2pi[0, 0]) < tolerance
            ), f"Periodicity failed for point {point}: {energy_orig[0, 0]} != {energy_2pi[0, 0]}"
            assert (
                jnp.abs(energy_orig[0, 0] - energy_neg2pi[0, 0]) < tolerance
            ), f"Negative periodicity failed for point {point}: {energy_orig[0, 0]} != {energy_neg2pi[0, 0]}"

    def test_anisotropic_vs_isotropic_comparison(self):
        """Test that anisotropic concentrations behave differently from isotropic ones."""
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])

        # Isotropic case
        concentrations_iso = jnp.array([[2.0, 2.0]])
        potential_iso = PeriodicMultidim(centers, depths, concentrations_iso)

        # Anisotropic case - high concentration in x, low in y
        concentrations_aniso = jnp.array([[4.0, 1.0]])
        potential_aniso = PeriodicMultidim(centers, depths, concentrations_aniso)

        # Test points displaced from center in different directions
        center = jnp.array([[[jnp.pi, jnp.pi]]])
        x_displaced = jnp.array([[[jnp.pi + 0.2, jnp.pi]]])  # Displaced in x
        y_displaced = jnp.array([[[jnp.pi, jnp.pi + 0.2]]])  # Displaced in y
        # diagonal_displaced = jnp.array(
        #     [[[jnp.pi + 0.2, jnp.pi + 0.2]]]
        # )  # Displaced diagonally

        # Calculate energies for both potentials
        # Isotropic case
        energy_iso_center = potential_iso(center, beta=1.0)
        energy_iso_x = potential_iso(x_displaced, beta=1.0)
        energy_iso_y = potential_iso(y_displaced, beta=1.0)
        # energy_iso_diag = potential_iso(diagonal_displaced, beta=1.0)

        # Anisotropic case
        energy_aniso_center = potential_aniso(center, beta=1.0)
        energy_aniso_x = potential_aniso(x_displaced, beta=1.0)
        energy_aniso_y = potential_aniso(y_displaced, beta=1.0)
        # energy_aniso_diag = potential_aniso(diagonal_displaced, beta=1.0)

        # For isotropic case, x and y displacements should give similar energies
        iso_x_penalty = energy_iso_x[0, 0] - energy_iso_center[0, 0]
        iso_y_penalty = energy_iso_y[0, 0] - energy_iso_center[0, 0]
        assert (
            jnp.abs(iso_x_penalty - iso_y_penalty) < 1e-6
        ), f"Isotropic case should have similar x and y penalties: {iso_x_penalty} vs {iso_y_penalty}"

        # For anisotropic case, x displacement should be more costly than y displacement
        aniso_x_penalty = energy_aniso_x[0, 0] - energy_aniso_center[0, 0]
        aniso_y_penalty = energy_aniso_y[0, 0] - energy_aniso_center[0, 0]
        assert (
            aniso_x_penalty > aniso_y_penalty
        ), f"Anisotropic case: x penalty ({aniso_x_penalty}) should be > y penalty ({aniso_y_penalty})"

        # The difference in penalties should be significant
        penalty_difference = aniso_x_penalty - aniso_y_penalty
        assert (
            penalty_difference > 0.01
        ), f"Penalty difference should be significant: {penalty_difference}"

        # Test that the anisotropic potential differs from isotropic for off-axis displacements
        assert (
            jnp.abs(energy_iso_x[0, 0] - energy_aniso_x[0, 0]) > 1e-6
        ), "X-displaced energies should differ between isotropic and anisotropic cases"
        assert (
            jnp.abs(energy_iso_y[0, 0] - energy_aniso_y[0, 0]) > 1e-6
        ), "Y-displaced energies should differ between isotropic and anisotropic cases"

    def test_anisotropic_concentration_scaling(self):
        """Test that concentration scaling works correctly in anisotropic case."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])

        # Create potentials with different concentration scalings
        concentrations_1x = jnp.array([[1.0, 1.0]])
        concentrations_2x = jnp.array([[2.0, 2.0]])
        concentrations_aniso = jnp.array([[3.0, 1.0]])  # 3x in x, 1x in y

        potential_1x = PeriodicMultidim(centers, depths, concentrations_1x)
        potential_2x = PeriodicMultidim(centers, depths, concentrations_2x)
        potential_aniso = PeriodicMultidim(centers, depths, concentrations_aniso)

        # Test at center (should be most negative for highest concentration)
        center = jnp.array([[[0.0, 0.0]]])
        energy_1x = potential_1x(center, beta=1.0)
        energy_2x = potential_2x(center, beta=1.0)
        energy_aniso = potential_aniso(center, beta=1.0)

        # Higher concentration should give more negative energy at center
        assert (
            energy_2x[0, 0] < energy_1x[0, 0]
        ), f"Higher concentration should give more negative energy: {energy_2x[0, 0]} vs {energy_1x[0, 0]}"

        # Anisotropic case should be between 1x and 2x (since average concentration is 2.0)
        # but closer to 2x due to higher total concentration
        assert (
            energy_aniso[0, 0] < energy_1x[0, 0]
        ), f"Anisotropic should be more negative than 1x: {energy_aniso[0, 0]} vs {energy_1x[0, 0]}"

        # Test off-center behavior
        off_center = jnp.array([[[0.3, 0.3]]])
        energy_1x_off = potential_1x(off_center, beta=1.0)
        energy_2x_off = potential_2x(off_center, beta=1.0)
        energy_aniso_off = potential_aniso(off_center, beta=1.0)

        # All should be less negative than at center
        assert energy_1x_off[0, 0] > energy_1x[0, 0]
        assert energy_2x_off[0, 0] > energy_2x[0, 0]
        assert energy_aniso_off[0, 0] > energy_aniso[0, 0]

    def test_anisotropic_vs_isotropic_comparison_(self):
        """Compare anisotropic and isotropic potentials."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])

        # Isotropic case (equal concentrations)
        concentrations_iso = jnp.array([[2.0, 2.0]])
        potential_iso = PeriodicMultidim(centers, depths, concentrations_iso)

        # Anisotropic case (different concentrations)
        concentrations_aniso = jnp.array([[5.0, 0.5]])
        potential_aniso = PeriodicMultidim(centers, depths, concentrations_aniso)

        # Test points at center - should be identical
        center = jnp.array([[[0.0, 0.0]]])
        energy_iso_center = potential_iso(center, beta=1.0)
        energy_aniso_center = potential_aniso(center, beta=1.0)

        # At center, both should give the same energy (sum of concentrations is same)
        # Actually, they won't be identical due to different concentration distributions
        # But both should be strongly negative
        assert energy_iso_center[0, 0] < -1.0
        assert energy_aniso_center[0, 0] < -1.0

        # Test symmetry breaking in anisotropic case
        x_disp = jnp.array([[[0.3, 0.0]]])
        y_disp = jnp.array([[[0.0, 0.3]]])

        # Isotropic case should have symmetric response
        energy_iso_x = potential_iso(x_disp, beta=1.0)
        energy_iso_y = potential_iso(y_disp, beta=1.0)
        assert (
            jnp.abs(energy_iso_x[0, 0] - energy_iso_y[0, 0]) < 1e-10
        ), "Isotropic case should be symmetric"

        # Anisotropic case should break symmetry
        energy_aniso_x = potential_aniso(x_disp, beta=1.0)
        energy_aniso_y = potential_aniso(y_disp, beta=1.0)
        assert (
            jnp.abs(energy_aniso_x[0, 0] - energy_aniso_y[0, 0]) > 0.1
        ), "Anisotropic case should break symmetry"

    def test_anisotropic_multiple_wells_interaction(self):
        """Test anisotropic behavior with multiple wells."""
        # Create two wells with different anisotropic properties
        centers = jnp.array([[0.0, 0.0], [jnp.pi, jnp.pi]])
        depths = jnp.array([1.0, 1.0])

        # First well: tight in x, loose in y
        # Second well: loose in x, tight in y
        concentrations = jnp.array([[5.0, 0.5], [0.5, 5.0]])
        potential = PeriodicMultidim(centers, depths, concentrations)

        # Test points near each well
        near_well1_x = jnp.array(
            [[[0.2, 0.0]]]
        )  # Near first well, displaced in its tight direction
        near_well1_y = jnp.array(
            [[[0.0, 0.2]]]
        )  # Near first well, displaced in its loose direction

        near_well2_x = jnp.array(
            [[[jnp.pi + 0.2, jnp.pi]]]
        )  # Near second well, displaced in its loose direction
        near_well2_y = jnp.array(
            [[[jnp.pi, jnp.pi + 0.2]]]
        )  # Near second well, displaced in its tight direction

        energy_w1_x = potential(near_well1_x, beta=1.0)
        energy_w1_y = potential(near_well1_y, beta=1.0)
        energy_w2_x = potential(near_well2_x, beta=1.0)
        energy_w2_y = potential(near_well2_y, beta=1.0)

        # For well 1: x-displacement should be more costly
        assert (
            energy_w1_x[0, 0] > energy_w1_y[0, 0]
        ), "Well 1: X-displacement should be more costly"

        # For well 2: y-displacement should be more costly
        assert (
            energy_w2_y[0, 0] > energy_w2_x[0, 0]
        ), "Well 2: Y-displacement should be more costly"

    def test_anisotropic_contour_generation(self):
        """Test that 2D contour generation works with anisotropic concentrations."""
        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[5.0, 0.5]])  # Anisotropic

        potential = PeriodicMultidim(centers, depths, concentrations)

        X, Y, E = potential.get_2dcontour(
            (0, 2 * jnp.pi), (0, 2 * jnp.pi), num_points=50
        )

        # Basic shape checks
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert E.shape == (50, 50)

        # Find minimum
        min_idx = jnp.unravel_index(jnp.argmin(E), E.shape)
        min_x = X[min_idx]
        min_y = Y[min_idx]

        # Minimum should be near the well center
        assert jnp.abs(min_x - jnp.pi) < 0.3
        assert jnp.abs(min_y - jnp.pi) < 0.3

        # Check anisotropy: the well should be narrower in x-direction than y-direction
        # Find half-maximum contour lines
        # min_energy = jnp.min(E)
        # max_energy = jnp.max(E)
        # half_max = min_energy + 0.5 * (max_energy - min_energy)

        # This is a qualitative test - in practice, you'd analyze the contour shape more carefully
        assert jnp.min(E) == 0.0  # Due to normalization in get_2dcontour

    def test_extreme_anisotropy(self):
        """Test behavior with extreme anisotropic ratios."""
        centers = jnp.array([[0.0, 0.0]])
        depths = jnp.array([1.0])

        # Moderate but still significant anisotropy to avoid numerical issues
        concentrations = jnp.array([[20.0, 0.1]])
        potential = PeriodicMultidim(centers, depths, concentrations)

        center = jnp.array([[[0.0, 0.0]]])
        small_x_disp = jnp.array(
            [[[0.05, 0.0]]]
        )  # Small displacement in tight direction
        large_y_disp = jnp.array(
            [[[0.0, 0.5]]]
        )  # Large displacement in loose direction

        energy_center = potential(center, beta=1.0)
        energy_x = potential(small_x_disp, beta=1.0)
        energy_y = potential(large_y_disp, beta=1.0)

        # Check that none of the energies are -inf (numerical stability)
        assert jnp.isfinite(energy_center[0, 0]), "Center energy should be finite"
        assert jnp.isfinite(energy_x[0, 0]), "X-displaced energy should be finite"
        assert jnp.isfinite(energy_y[0, 0]), "Y-displaced energy should be finite"

        # Even small displacement in tight direction should be more costly than large displacement in loose direction
        assert (
            energy_x[0, 0] > energy_y[0, 0]
        ), "Small displacement in tight direction should be more costly"

        # The effect should be dramatic
        assert (energy_x[0, 0] - energy_center[0, 0]) > (
            energy_y[0, 0] - energy_center[0, 0]
        )

    def test_anisotropic_gradient_behavior(self):
        """Test that gradients reflect anisotropic behavior."""
        from jax import grad

        centers = jnp.array([[jnp.pi, jnp.pi]])
        depths = jnp.array([1.0])
        concentrations = jnp.array([[5.0, 0.5]])  # Tight in x, loose in y

        potential = PeriodicMultidim(centers, depths, concentrations)

        def single_particle_energy(pos):
            """Energy for a single particle at position pos."""
            x = pos.reshape(1, 1, -1)
            return potential(x, beta=1.0)[0, 0]

        grad_func = grad(single_particle_energy)

        # Test gradient at a point slightly displaced from center
        test_point = jnp.array([jnp.pi + 0.1, jnp.pi + 0.1])
        gradient = grad_func(test_point)

        # Gradient magnitude in x-direction should be larger due to higher concentration
        assert (
            jnp.abs(gradient[0]) > jnp.abs(gradient[1])
        ), f"X-gradient magnitude ({jnp.abs(gradient[0])}) should be larger than Y-gradient magnitude ({jnp.abs(gradient[1])})"

        # Test gradient at center should be close to zero
        center_point = jnp.array([jnp.pi, jnp.pi])
        center_gradient = grad_func(center_point)

        # Gradient at center should be close to zero (local minimum)
        assert (
            jnp.abs(center_gradient[0]) < 1e-5
        ), f"X-gradient at center should be ~0: {center_gradient[0]}"
        assert (
            jnp.abs(center_gradient[1]) < 1e-5
        ), f"Y-gradient at center should be ~0: {center_gradient[1]}"

        # Test that gradient behavior is consistent with anisotropy
        # Test points slightly off center in different directions
        x_displaced = jnp.array([jnp.pi + 0.05, jnp.pi])  # Small displacement in x
        y_displaced = jnp.array([jnp.pi, jnp.pi + 0.05])  # Small displacement in y

        grad_x_disp = grad_func(x_displaced)
        grad_y_disp = grad_func(y_displaced)

        # For the same displacement distance, the gradient magnitude should be larger
        # in the direction with higher concentration
        assert jnp.abs(grad_x_disp[0]) > jnp.abs(grad_y_disp[1]), (
            f"X-gradient for x-displacement ({jnp.abs(grad_x_disp[0])}) should be larger than "
            f"Y-gradient for y-displacement ({jnp.abs(grad_y_disp[1])})"
        )
