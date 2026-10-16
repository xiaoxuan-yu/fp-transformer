import pytest
import sys

# Add the project root to the Python path
sys.path.append(".")

from src.fp_transformer import (
    Sampler,
    EnergySystem,
    CoordTopology,
    kernel,
    DynamicITSSampler,
)
from src.fp_transformer import potential
from src.fp_transformer.utils import (
    torus_pairwise_distance,
)
from jax import numpy as jnp
from jax import random
import optax


class Test2DSampling:
    """Test cases for 2D sampling with different sampler types."""

    @pytest.fixture
    def setup_system(self):
        """Set up the 2D periodic system for testing."""
        # Fixed parameters for testing
        num_replicas = 32  # Reduced for faster testing
        key = random.PRNGKey(0)

        # Initialize coordinates
        coord = (
            random.normal(key, (num_replicas, 1, 2)) * 0.1
            + jnp.array([[-2, 3]])[None, :, :]
        )

        # Calculate initial bandwidth
        init_dist2 = torus_pairwise_distance(coord)
        bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
            (num_replicas - 1) * (num_replicas - 2) // 2 // 2
        ]

        # Create kernel
        k = kernel.Gaussian(
            num_replicas,
            update_bw_method="rescale",
            bandwidth=bw_init,
            distance_func=torus_pairwise_distance,
        )

        # Create potential wells
        centers = jnp.array([[2, 1], [-2, 3]])
        depths = jnp.array([20.0, 20.0])
        concentrations = jnp.array([[10.0, 1.0], [1.5, 10.0]])

        # Create topology and energy system
        top = CoordTopology(coord)
        energy_ = potential.PeriodicMultidim(centers, depths, concentrations)
        sys = EnergySystem(top, energy_)

        # Create optimizer
        opt = optax.adagrad(0.1, 0.1)

        return {
            "system": sys,
            "kernel": k,
            "optimizer": opt,
            "coord": coord,
            "centers": centers,
            "key": key,
        }

    def test_sampler_initialization(self, setup_system):
        """Test that all samplers can be initialized correctly."""
        sys = setup_system["system"]
        k = setup_system["kernel"]
        opt = setup_system["optimizer"]

        # Test basic sampler
        basic_sampler = Sampler(system=sys, kernel=k, optimizer=opt, temperature=1)
        basic_sampler.set_beta(1.0)
        assert basic_sampler is not None
        assert basic_sampler.beta == 1.0

        # Test ITS sampler
        its_sampler = DynamicITSSampler(
            system=sys,
            kernel=k,
            optimizer=opt,
            beta=2,
            anneal_factor=0.3,
            max_num_temperatures=60,
            kl_div_threshold=1e-3,
            max_steps=5000,
            grad_norm_threshold_mixing=1e-2,
            grad_norm_threshold_annealing=1e-3,
        )
        assert its_sampler is not None
        assert its_sampler.beta == 2.0

    def test_basic_sampler_run(self, setup_system):
        """Test basic sampler functionality."""
        sys = setup_system["system"]
        k = setup_system["kernel"]
        opt = setup_system["optimizer"]
        coord = setup_system["coord"]

        # Create and run basic sampler
        sampler = Sampler(system=sys, kernel=k, optimizer=opt, temperature=1)

        # Set beta and run for a small number of steps
        sampler.beta = 2
        sampler.run(100, verbose=False)  # Reduced steps for testing

        # Check that coordinates have changed
        final_coord = sampler.system.coordinate
        assert final_coord is not None
        assert not jnp.array_equal(final_coord, coord)

        # Check that coordinates are within reasonable bounds
        assert jnp.all(jnp.isfinite(final_coord))

        # Transform coordinates to periodic space
        transformed_coords = jnp.sign(jnp.sin(final_coord)) * jnp.arccos(
            jnp.cos(final_coord)
        )
        assert jnp.all(jnp.isfinite(transformed_coords))

    def test_its_sampler_run(self, setup_system):
        """Test ITS (Iterative Temperature Scaling) sampler functionality."""
        sys = setup_system["system"]
        k = setup_system["kernel"]
        opt = setup_system["optimizer"]
        coord = setup_system["coord"]

        # Create and run ITS sampler
        sampler = DynamicITSSampler(
            system=sys,
            kernel=k,
            optimizer=opt,
            beta=2,
            anneal_factor=0.3,
            max_num_temperatures=3,  # Reduced for testing
            kl_div_threshold=1e-3,
            max_steps=100,  # Reduced for testing
            grad_norm_threshold_mixing=1e-2,
            grad_norm_threshold_annealing=1e-3,
        )

        # Run ITS iterations
        sampler.run(num_its_iterations=1)

        # Check that coordinates have changed
        final_coord = sampler.system.coordinate
        assert final_coord is not None
        assert not jnp.array_equal(final_coord, coord)

        # Check that coordinates are within reasonable bounds
        assert jnp.all(jnp.isfinite(final_coord))

        # Transform coordinates to periodic space
        transformed_coords = jnp.sign(jnp.sin(final_coord)) * jnp.arccos(
            jnp.cos(final_coord)
        )
        assert jnp.all(jnp.isfinite(transformed_coords))

        # Check that total steps is positive
        assert sampler.total_steps > 0

    def test_annealing_sampler_run(self, setup_system):
        """Test annealing sampler functionality."""
        sys = setup_system["system"]
        k = setup_system["kernel"]
        opt = setup_system["optimizer"]
        coord = setup_system["coord"]

        # Create sampler for annealing
        sampler = Sampler(system=sys, kernel=k, optimizer=opt, temperature=1)

        # Test different annealing factors
        afs = [0.3, 0.5, 0.7]
        initial_coord = coord.copy()

        for af in afs:
            # Set beta based on annealing factor
            sampler.beta = af * 2

            # Run for a small number of steps
            sampler.run(50, verbose=False)  # Reduced steps for testing

            # Check that coordinates have changed
            final_coord = sampler.system.coordinate
            assert final_coord is not None
            assert not jnp.array_equal(final_coord, initial_coord)

            # Check that coordinates are within reasonable bounds
            assert jnp.all(jnp.isfinite(final_coord))

            # Transform coordinates to periodic space
            transformed_coords = jnp.sign(jnp.sin(final_coord)) * jnp.arccos(
                jnp.cos(final_coord)
            )
            assert jnp.all(jnp.isfinite(transformed_coords))

            # Update initial coord for next iteration
            initial_coord = final_coord.copy()

    def test_energy_calculation(self, setup_system):
        """Test that energy calculations work correctly."""
        sys = setup_system["system"]
        coord = setup_system["coord"]

        # Calculate energy at initial coordinates
        initial_energy = sys.potential()

        # Check that energy is finite
        assert jnp.all(jnp.isfinite(initial_energy))

        # Check that energy shape matches number of replicas
        assert initial_energy.shape == (coord.shape[0], 1)

        # Test energy at different coordinates
        test_coord = jnp.array([[[0.0, 0.0]], [[jnp.pi, jnp.pi]]])
        sys.coordinate = test_coord
        test_energy = sys.potential()

        assert jnp.all(jnp.isfinite(test_energy))
        assert test_energy.shape == (test_coord.shape[0], 1)

    def test_kernel_functionality(self, setup_system):
        """Test that kernel functionality works correctly."""
        k = setup_system["kernel"]
        coord = setup_system["coord"]

        # Update kernel with coordinates
        k.update(coord)

        # Check that bandwidth is updated and finite
        assert jnp.all(jnp.isfinite(k.bandwidth))
        assert jnp.all(k.bandwidth > 0)

        # Test kernel calculation
        kernel_matrix = k.kernel(coord, k.bandwidth)

        # Check kernel matrix properties
        assert kernel_matrix.shape == (coord.shape[0], coord.shape[1], 1)
        assert jnp.all(jnp.isfinite(kernel_matrix))
        assert jnp.all(jnp.exp(kernel_matrix) > 0)  # log of a positive leave-one-out density

    def test_coordinate_transformation(self, setup_system):
        """Test coordinate transformation to periodic space."""
        coord = setup_system["coord"]

        # Test coordinate transformation
        transformed_coords = jnp.sign(jnp.sin(coord)) * jnp.arccos(jnp.cos(coord))

        # Check that transformed coordinates are within [-π, π]
        assert jnp.all(transformed_coords >= -jnp.pi)
        assert jnp.all(transformed_coords <= jnp.pi)

        # Check that transformation is consistent
        # Points differing by 2π should map to the same transformed coordinates
        test_point = jnp.array([[[0.0, 0.0]]])
        test_point_shifted = jnp.array([[[2 * jnp.pi, 2 * jnp.pi]]])

        transformed_test = jnp.sign(jnp.sin(test_point)) * jnp.arccos(
            jnp.cos(test_point)
        )
        transformed_shifted = jnp.sign(jnp.sin(test_point_shifted)) * jnp.arccos(
            jnp.cos(test_point_shifted)
        )

        # They should be very close (allowing for numerical precision)
        assert jnp.allclose(transformed_test, transformed_shifted, atol=1e-10)

    def test_system_coordinate_updates(self, setup_system):
        """Test that system coordinate updates work correctly."""
        sys = setup_system["system"]
        coord = setup_system["coord"]

        # Store initial coordinates
        initial_coord = sys.coordinate.copy()

        # Update coordinates
        new_coord = random.normal(random.PRNGKey(42), coord.shape)
        sys.coordinate = new_coord

        # Check that coordinates were updated
        assert jnp.array_equal(sys.coordinate, new_coord)
        assert not jnp.array_equal(sys.coordinate, initial_coord)

        # Check that topology coordinates are also updated
        assert jnp.array_equal(sys.topology.coordinate, new_coord)
