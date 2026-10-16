import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform

# add src to path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fp_transformer.utils import init_params_by_nearest_ci
from src.fp_transformer.potential.PeriodicMultidim import PeriodicMultidim


def _periodic_distance(a: jnp.ndarray, b: jnp.ndarray) -> float:
    """Compute periodic Euclidean distance on N-torus between two points a,b in [0,2pi)."""
    diff = jnp.abs(a - b)
    periodic_diff = jnp.minimum(diff, 2.0 * jnp.pi - diff)
    return jnp.linalg.norm(periodic_diff)


def test_init_params_by_nearest_ci_minima():
    key = PRNGKey(0)
    N = 4  # dimensionality
    M = 5  # number of wells
    depth_range = (1.0, 2.0)
    alpha = 0.95
    margin = 0.95

    params = init_params_by_nearest_ci(
        key, N, M, depth_range, alpha=alpha, margin=margin, verbose=False
    )

    assert jnp.all(jnp.isfinite(params.centers))
    assert jnp.all(jnp.isfinite(params.depths))
    assert jnp.all(jnp.isfinite(params.concentrations))

    centers = params.centers  # (M,N)
    depths = params.depths
    concentrations = params.concentrations

    # Construct potential
    potential = PeriodicMultidim(centers, depths, concentrations)

    # For each center, verify it is a local minimum relative to small perturbations.
    for i in range(M):
        center = centers[i]
        # Compute nearest neighbor distance
        dists = jnp.array(
            [
                _periodic_distance(center, centers[j]) if j != i else jnp.inf
                for j in range(M)
            ]
        )
        nearest_dist = jnp.min(dists)
        # target radius used in construction
        target_radius = margin * nearest_dist
        # small perturbation scale (fraction of target radius)
        eps = target_radius * 0.2
        # energy at the exact center (shape handling: (1,1,N))
        E_center = potential(center[None, None, :])[0, 0]

        # sample K random perturbations inside hypercube [-eps, eps]^N, wrap modulo 2pi
        K = 16
        subkeys = split(key, K + 1)
        key = subkeys[0]
        deltas = jnp.stack(
            [
                uniform(subkeys[k + 1], shape=(N,), minval=-eps, maxval=eps)
                for k in range(K)
            ]
        )
        perturbed = (center + deltas) % (2.0 * jnp.pi)
        # shape (K,1,N) for evaluation (num_replicas=K, num_particle=1)
        perturbed_batch = perturbed[:, None, :]
        E_pert = potential(perturbed_batch)[:, 0]  # (K,)

        # Assert center energy is <= energies of perturbations (local minimum)
        assert E_center <= jnp.min(E_pert) + 1e-6, (
            f"Center {i} energy not minimal within perturbation neighborhood:"
            f" E_center={float(E_center)}, min_pert={float(jnp.min(E_pert))}"
        )
        # Ensure at least one perturbation strictly higher (avoid trivial flat case)
        assert jnp.any(E_pert > E_center + 1e-8)

        # Gradient magnitude check at center (should be near stationary point)
        def energy_scalar(x_flat):
            # x_flat shape (N,)
            return potential(x_flat[None, None, :])[0, 0]

        grad_val = jax.grad(energy_scalar)(center)
        assert jnp.linalg.norm(grad_val) < 1e-2, (
            f"Gradient too large at center {i}: norm={float(jnp.linalg.norm(grad_val))}"
        )


def test_init_params_by_nearest_ci_shapes():
    key = PRNGKey(42)
    N = 3
    M = 7
    params = init_params_by_nearest_ci(key, N, M, (0.5, 1.5))
    assert params.depths.shape == (M,)
    assert params.concentrations.shape == (M, N)
    assert params.centers.shape == (M, N)
    assert jnp.all(params.depths > 0)
    assert jnp.all(params.concentrations > 0)
    # Centers in range [0, 2pi)
    assert jnp.all(params.centers >= 0) and jnp.all(params.centers < 2.0 * jnp.pi)


def test_init_params_by_nearest_ci_largeN_moderate():
    """Basic sanity for a moderately large dimension (N=16)."""
    key = PRNGKey(2024)
    N = 16
    M = 10
    depth_range = (0.1, 2.5)
    alpha = 0.95
    margin = 0.95
    params = init_params_by_nearest_ci(
        key, N, M, depth_range, alpha=alpha, margin=margin
    )
    assert params.centers.shape == (M, N)
    assert params.concentrations.shape == (M, N)
    assert params.depths.shape == (M,)
    # Value ranges
    assert jnp.all(jnp.isfinite(params.centers))
    assert jnp.all(jnp.isfinite(params.concentrations))
    assert jnp.all(jnp.isfinite(params.depths))
    assert jnp.all((params.centers >= 0) & (params.centers < 2.0 * jnp.pi))
    assert jnp.all(params.depths >= depth_range[0]) and jnp.all(
        params.depths <= depth_range[1]
    )
    # κ 正值并且不至于过小
    assert jnp.all(params.concentrations > 0)
    # 选几个随机 mode 检查与其他中心的多维周期距离非零（避免完全重合）
    for i in range(min(3, M)):
        center = params.centers[i]
        dists = jnp.array(
            [
                _periodic_distance(center, params.centers[j]) if j != i else jnp.inf
                for j in range(M)
            ]
        )
        assert jnp.min(dists) > 0.0

    # 检查局部极小：在邻域内中心能量最小（避免center发生偏移）
    potential = PeriodicMultidim(params.centers, params.depths, params.concentrations)
    for i in range(min(2, M)):
        center = params.centers[i]
        # 最近邻距离与目标半径
        dists = jnp.array(
            [
                _periodic_distance(center, params.centers[j]) if j != i else jnp.inf
                for j in range(M)
            ]
        )
        nearest_dist = jnp.min(dists)
        target_radius = margin * nearest_dist
        eps = target_radius * 0.2

        E_center = potential(center[None, None, :])[0, 0]
        # 在 [-eps, eps]^N 内随机扰动
        K = 12
        subkeys = split(key, K + 1)
        key = subkeys[0]
        deltas = jnp.stack(
            [
                uniform(subkeys[k + 1], shape=(N,), minval=-eps, maxval=eps)
                for k in range(K)
            ]
        )
        perturbed = (center + deltas) % (2.0 * jnp.pi)
        E_pert = potential(perturbed[:, None, :])[:, 0]

        assert E_center <= jnp.min(E_pert) + 1e-6

        # 梯度在中心应接近0（soft-min 会带来微小偏差，因此阈值稍宽松）
        def energy_scalar(x_flat):
            return potential(x_flat[None, None, :])[0, 0]

        grad_val = jax.grad(energy_scalar)(center)
        assert jnp.linalg.norm(grad_val) < 5e-2


def test_init_params_by_nearest_ci_largeN_high():
    """Stress a higher dimension (N=32) with fewer modes to remain fast."""
    key = PRNGKey(2025)
    N = 32
    M = 6  # keep small for speed
    depth_range = (0.2, 1.8)
    alpha = 0.95
    margin = 0.95
    params = init_params_by_nearest_ci(
        key, N, M, depth_range, alpha=alpha, margin=margin
    )
    # Shapes
    assert params.centers.shape == (M, N)
    assert params.concentrations.shape == (M, N)
    assert params.depths.shape == (M,)
    # All finite & ranges
    assert jnp.all(jnp.isfinite(params.centers))
    assert jnp.all(jnp.isfinite(params.concentrations))
    assert jnp.all(jnp.isfinite(params.depths))
    assert jnp.all((params.centers >= 0) & (params.centers < 2.0 * jnp.pi))
    assert jnp.all(params.concentrations > 0)
    # Depth statistics roughly inside range
    assert float(params.depths.min()) >= depth_range[0] - 1e-6
    assert float(params.depths.max()) <= depth_range[1] + 1e-6
    # Pairwise minimal periodic distance per dimension不为0（忽略极小数值误差）
    # 这里我们对每个维度检查中心是否存在完全重合
    eps = 1e-9
    for dim in range(N):
        col = params.centers[:, dim]
        # pairwise distances on circle
        for i in range(M):
            for j in range(i + 1, M):
                diff = jnp.abs(col[i] - col[j])
                circ = jnp.minimum(diff, 2.0 * jnp.pi - diff)
                assert circ > eps

    # 局部极小性检查（取少量mode 以控制时长）
    potential = PeriodicMultidim(params.centers, params.depths, params.concentrations)
    for i in range(min(2, M)):
        center = params.centers[i]
        dists = jnp.array(
            [
                _periodic_distance(center, params.centers[j]) if j != i else jnp.inf
                for j in range(M)
            ]
        )
        nearest_dist = jnp.min(dists)
        target_radius = margin * nearest_dist
        eps = target_radius * 0.15

        E_center = potential(center[None, None, :])[0, 0]
        K = 10
        subkeys = split(key, K + 1)
        key = subkeys[0]
        deltas = jnp.stack(
            [
                uniform(subkeys[k + 1], shape=(N,), minval=-eps, maxval=eps)
                for k in range(K)
            ]
        )
        perturbed = (center + deltas) % (2.0 * jnp.pi)
        E_pert = potential(perturbed[:, None, :])[:, 0]

        assert E_center <= jnp.min(E_pert) + 1e-6

        # 梯度阈值略放宽
        def energy_scalar(x_flat):
            return potential(x_flat[None, None, :])[0, 0]

        grad_val = jax.grad(energy_scalar)(center)
        assert jnp.linalg.norm(grad_val) < 7.5e-2
