import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fp_transformer import CoordTopology, EnergySystem, Sampler
from fp_transformer.kernel import Gaussian
from fp_transformer.kernel.fused import torus_row_score, torus_row_score_reference
from fp_transformer.potential import PeriodicSin4
from fp_transformer.utils import pairwise_distance, torus_pairwise_distance



@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _coords(n=37, atoms=2, dim=3, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, atoms, dim), minval=-2.0, maxval=2.0)


def _bandwidths(n, seed=1):
    return jax.random.uniform(jax.random.PRNGKey(seed), (n,), minval=0.5, maxval=1.5)


def _pair_inverse_squared(bw, n, mode):
    bw = jnp.broadcast_to(jnp.asarray(bw).reshape(-1), (n,))
    if mode == "symmetric":
        return 1.0 / (0.5 * (bw[:, None] ** 2 + bw[None, :] ** 2))
    return 1.0 / (bw[:, None] ** 2) * jnp.ones((1, n))


def _analytic_torus_score(x, bw, mode, include_self):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    m = flat.shape[1]
    inv = _pair_inverse_squared(bw, n, mode)
    logits = -0.5 * torus_pairwise_distance(x) * inv
    if not include_self:
        logits = jnp.where(jnp.eye(n, dtype=bool), -jnp.inf, logits)
    w = jax.nn.softmax(logits, axis=1)
    delta = jnp.sin(flat[None, :, :] - flat[:, None, :])  # sin(x_j - x_i)
    return (jnp.sum((w * inv)[:, :, None] * delta, axis=1) / m).reshape(x.shape)


def _analytic_euclidean_score(x, bw, mode, include_self):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    m = flat.shape[1]
    inv = _pair_inverse_squared(bw, n, mode)
    logits = -0.5 * pairwise_distance(x) * inv
    if not include_self:
        logits = jnp.where(jnp.eye(n, dtype=bool), -jnp.inf, logits)
    w = jax.nn.softmax(logits, axis=1)
    delta = flat[None, :, :] - flat[:, None, :]
    return (jnp.sum((w * inv)[:, :, None] * delta, axis=1) / m).reshape(x.shape)


@pytest.mark.parametrize("mode", ["row", "symmetric"])
@pytest.mark.parametrize("per_replica", [False, True])
@pytest.mark.parametrize(
    "distance, analytic",
    [
        (torus_pairwise_distance, _analytic_torus_score),
        (pairwise_distance, _analytic_euclidean_score),
    ],
)
def test_default_score_is_pointwise_leave_one_out(mode, per_replica, distance, analytic):
    x = _coords()
    bw = _bandwidths(x.shape[0]) if per_replica else jnp.asarray(0.8)
    kernel = Gaussian(x.shape[0], distance_func=distance, pair_bandwidth=mode)
    np.testing.assert_allclose(
        kernel.grad_jit(x, bw), analytic(x, bw, mode, False), rtol=1e-10, atol=1e-12
    )


def test_include_self_and_retained_centers_reproduce_full_objective():
    x = _coords()
    bw = jnp.asarray(0.8)
    kernel = Gaussian(
        x.shape[0],
        distance_func=torus_pairwise_distance,
        include_self=True,
        center_gradient="retain",
    )

    def objective(value):
        logits = -0.5 * torus_pairwise_distance(value) / bw**2
        return jnp.sum(jax.nn.logsumexp(logits, axis=1))

    np.testing.assert_allclose(
        kernel.grad_jit(x, bw), jax.grad(objective)(x), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        Gaussian(x.shape[0], distance_func=torus_pairwise_distance, include_self=True).grad_jit(
            x, bw
        ),
        _analytic_torus_score(x, bw, "row", True),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("include_self", [False, True])
def test_row_sum_excludes_self_by_default(include_self):
    x = _coords()
    bw = _bandwidths(x.shape[0])
    kernel = Gaussian(
        x.shape[0], distance_func=torus_pairwise_distance, include_self=include_self
    )
    full = jnp.exp(-0.5 * torus_pairwise_distance(x) / bw[:, None] ** 2)
    expected = jnp.sum(full, axis=1) - (0.0 if include_self else 1.0)
    np.testing.assert_allclose(
        kernel.compute_kernel_row_sum_(x, bw), expected, rtol=1e-10
    )


def test_fixed_bandwidth_refresh_is_identity():
    x = _coords()
    kernel = Gaussian(
        x.shape[0], bandwidth=0.4, distance_func=torus_pairwise_distance, update_bw_method="fixed"
    )
    state = kernel.get_state()
    assert kernel.update_with_state(x, state) is state


def test_score_fn_receives_particle_weights():
    x = _coords()
    weights = jnp.linspace(0.5, 1.5, x.shape[0])
    seen = {}

    def score_fn(coord, bandwidth, particle_weights):
        seen["weights"] = particle_weights
        return jnp.zeros_like(coord)

    kernel = Gaussian(x.shape[0], distance_func=torus_pairwise_distance, score_fn=score_fn)
    state = kernel.get_state()
    state = type(state)(**{**state.__dict__, "particle_weights": weights})
    grad, _ = kernel.kernel_grad_with_state(x, state)
    np.testing.assert_array_equal(grad, jnp.zeros_like(x))
    np.testing.assert_array_equal(seen["weights"], weights)


def test_particle_weights_without_score_fn_are_rejected():
    x = _coords()
    kernel = Gaussian(x.shape[0], distance_func=torus_pairwise_distance)
    state = kernel.get_state()
    state = type(state)(**{**state.__dict__, "particle_weights": jnp.ones(x.shape[0])})
    with pytest.raises(ValueError, match="score_fn"):
        kernel.kernel_grad_with_state(x, state)


@pytest.mark.parametrize("per_replica", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_fused_reference_matches_analytic_score(per_replica, include_self):
    x = _coords(n=53, atoms=1, dim=7)
    bw = _bandwidths(x.shape[0]) if per_replica else jnp.asarray(0.9)
    score, log_mass = torus_row_score_reference(x, bw, include_self=include_self)
    np.testing.assert_allclose(
        score, _analytic_torus_score(x, bw, "row", include_self), rtol=1e-10, atol=1e-12
    )
    full = jnp.exp(
        -0.5 * torus_pairwise_distance(x) * _pair_inverse_squared(bw, x.shape[0], "row")
    )
    mass = jnp.sum(full, axis=1) - (0.0 if include_self else 1.0)
    np.testing.assert_allclose(jnp.exp(log_mass), mass, rtol=1e-10)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="Pallas Triton kernel needs a GPU")
@pytest.mark.parametrize("per_replica", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_fused_kernel_matches_reference(per_replica, include_self):
    x = _coords(n=203, atoms=1, dim=20).astype(jnp.float32)
    bw = (_bandwidths(x.shape[0]) if per_replica else jnp.asarray(0.9)).astype(jnp.float32)
    score, log_mass = torus_row_score(x, bw, include_self=include_self)
    ref_score, ref_log_mass = torus_row_score_reference(
        x.astype(jnp.float64), bw.astype(jnp.float64), include_self=include_self
    )
    np.testing.assert_allclose(score, ref_score, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(log_mass, ref_log_mass, atol=1e-4)
    kernel = Gaussian(
        x.shape[0],
        distance_func=torus_pairwise_distance,
        include_self=include_self,
        score_backend="fused",
    )
    np.testing.assert_allclose(kernel.grad_jit(x, bw), score, rtol=1e-6, atol=1e-7)


def test_fused_backend_rejects_unsupported_configurations():
    with pytest.raises(ValueError):
        Gaussian(8, distance_func=pairwise_distance, score_backend="fused")
    with pytest.raises(ValueError):
        Gaussian(
            8,
            distance_func=torus_pairwise_distance,
            center_gradient="retain",
            score_backend="fused",
        )


def test_observe_fn_sees_every_update():
    x = _coords(n=16, atoms=1, dim=3)
    system = EnergySystem(CoordTopology(x), PeriodicSin4(c=3.0, p=1.0))
    kernel = Gaussian(16, bandwidth=0.5, distance_func=torus_pairwise_distance)
    sampler = Sampler(
        system=system,
        kernel=kernel,
        optimizer=optax.rmsprop(1e-2),
        temperature=1.0,
        gamma=1.0,
        update_pace=5,
    )
    steps = []
    sampler.run(7, verbose=False, observe_fn=lambda step, state: steps.append(step))
    assert steps == list(range(1, 8))


@pytest.mark.parametrize("dim", [3, 20])
def test_row_mass_score_hits_target_and_matches_gaussian(dim):
    from fp_transformer.kernel.row_mass import torus_row_mass_score

    theta = jax.random.uniform(jax.random.PRNGKey(3), (64, dim), minval=-np.pi, maxval=np.pi)
    score, h, mass = torus_row_mass_score(theta, 8.0, log_lower=float(np.log(1e-4)), iterations=60)
    np.testing.assert_allclose(mass, 8.0, rtol=1e-8)
    kernel = Gaussian(64, distance_func=torus_pairwise_distance)
    expected = kernel.grad_jit(theta[:, None, :], h / np.sqrt(dim))[:, 0, :]
    np.testing.assert_allclose(score, expected, rtol=1e-8, atol=1e-10)


def test_hybrid_row_mass_solver_hits_target_and_flags_saturation():
    from fp_transformer.kernel.row_mass import solve_row_mass_bandwidth_hybrid

    x = jax.random.normal(jax.random.PRNGKey(4), (200, 2))
    d2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    h, mass, low, high = solve_row_mass_bandwidth_hybrid(
        d2, base_bandwidth=0.1, selector_factor_min=0.03, selector_factor_max=100.0,
        target_kernel_mass=16.0, fit_iterations=16,
    )
    assert not bool(jnp.any(low | high))
    np.testing.assert_allclose(mass, 16.0, rtol=1e-6)
    _, _, low, high = solve_row_mass_bandwidth_hybrid(
        d2, base_bandwidth=0.1, selector_factor_min=0.03, selector_factor_max=0.05,
        target_kernel_mass=16.0, fit_iterations=16,
    )
    assert bool(jnp.all(high))
