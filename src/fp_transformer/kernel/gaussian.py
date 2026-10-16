# This file contains the implementation of the Gaussian kernel.
import warnings
from dataclasses import dataclass, replace

import jax
from jax import grad, jit, lax, random
from jax import numpy as jnp

from ..type import (
    JaxArray,
    Union,
)
from ..utils import (
    get_jax_array,
    logsumexp,
    pairwise_distance,
    pairwise_distance_between,
    torus_pairwise_distance,
    torus_pairwise_distance_between,
)
from .base import Kernel
from .fused import torus_row_score
from .rff_utils import (
    compute_rff_features,
    create_torus_bessel_pmf_dist,
    create_torus_bessel_logits,
    rff_feature_map_torus,
    rff_logsum,
    rff_power_row_sum,
    rff_row_sum,
    sample_rff_params,
    sample_rff_params_torus,
    sample_rff_params_torus_from_logits,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianKernelState:
    """Runtime state for Gaussian kernel. Keep mutable arrays out of object captures."""

    bandwidth: JaxArray
    rff_key: JaxArray
    gaussian_exp: JaxArray | None = None
    torus_logits: JaxArray | None = None
    torus_log_gamma: JaxArray | None = None
    particle_weights: JaxArray | None = None

    def tree_flatten(self):
        return (
            self.bandwidth,
            self.rff_key,
            self.gaussian_exp,
            self.torus_logits,
            self.torus_log_gamma,
            self.particle_weights,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            bandwidth,
            rff_key,
            gaussian_exp,
            torus_logits,
            torus_log_gamma,
            particle_weights,
        ) = children
        return cls(
            bandwidth=bandwidth,
            rff_key=rff_key,
            gaussian_exp=gaussian_exp,
            torus_logits=torus_logits,
            torus_log_gamma=torus_log_gamma,
            particle_weights=particle_weights,
        )


@dataclass(frozen=True)
class PerReplicaRescaleConfig:
    """Configuration surface for per-replica bandwidth rescaling.

    Notes:
      - This dataclass is intentionally broader than the current implementation.
      - Current logic uses:
          * enabled
          * fixed_row_sum_min / fixed_row_sum_max (fallback to kernel_min/max)
          * bw_min / bw_max (reserved; enforcement to be added in strategy migration)
      - Remaining fields define the upcoming adaptive strategy interface.
    """

    enabled: bool = False
    target_mode: str = "fixed"  # "fixed" | "adaptive_quantile"
    fixed_row_sum_min: float | None = None
    fixed_row_sum_max: float | None = None
    adaptive_ref_quantile: float = 0.5
    adaptive_band_low: float = 0.8
    adaptive_band_high: float = 1.25
    bw_min: float = 1e-3
    bw_max: float = 1.0
    log_step_size: float = 0.05
    log_error_clip: float = 0.5
    global_pull: float = 0.0


class Gaussian(Kernel):
    """
    Gaussian kernel.

    Args:
        num_replicas: The number of replicas.
        bandwidth: The bandwidth of the Gaussian kernel.
        distance_func: The distance function to use. Supported: pairwise_distance, torus_pairwise_distance.
                      When using torus_pairwise_distance with rff=True, will automatically use torus-specific RFF.
        kernel_min: The minimum value of the kernel.
        kernel_max: The maximum value of the kernel.
        kernel_scale: The scale of the kernel.
        update_bw_method: Method for updating bandwidth. Options: "fixed", "rescale", "trick", "EM".
                         "fixed" keeps the configured bandwidth and makes runtime refresh a no-op.
                         Note: When rff=True, only "EM" and "rescale" methods are supported.
                         The "trick" method requires full distance matrix and is not compatible with RFF.
        rescale_per_replica: Whether to independently rescale bandwidth per replica in
                            non-RFF `update_bw_method="rescale"` path.
        rescale_per_replica_config: Extended configuration for per-replica rescaling.
                            Legacy `rescale_per_replica=True` forces config.enabled=True.
        include_self: Whether each replica's own kernel term enters its density,
                            row sums, and bandwidth control. The default is the
                            leave-one-out estimator.
        center_gradient: "freeze" differentiates only the evaluation point of each
                            log-density row (pointwise KDE score with the kernel
                            centers held fixed); "retain" also differentiates the
                            kernel centers.
        score_backend: "jax" (dense autodiff) or "fused" (Pallas row pass of
                            ``kernel.fused.torus_row_score``; torus distance,
                            non-RFF, and center_gradient="freeze" only).
        fused_allow_tf32: TF32 tensor-core dots in the fused row pass.
        pair_bandwidth: How per-replica bandwidths enter the pair kernel:
                            "row" uses h_i in row i; "symmetric" uses
                            h_ij^2 = (h_i^2 + h_j^2) / 2. Identical for a shared bandwidth.
        score_fn: Optional replacement for the built-in score, called as
                            ``score_fn(coord, bandwidth, particle_weights)`` with
                            ``particle_weights`` taken from the kernel state (or None).
        rff: Whether to use Random Fourier Features for memory efficiency.
        rff_dim: Dimension of Random Fourier Features.
        torus_m_cutoff: Cutoff parameter for torus RFF Bessel function approximation (only used with torus distance).
    """

    def _warn_rff_scalar_bw_once(self, api_name: str) -> None:
        warned = getattr(self, "_rff_bw_warned_apis", None)
        if warned is None:
            warned = set()
            self._rff_bw_warned_apis = warned
        if api_name in warned:
            return
        warnings.warn(
            f"`{api_name}` received per-replica bandwidth in RFF mode. "
            "RFF mode does not support per-replica bandwidth; using scalar mean(bw).",
            UserWarning,
            stacklevel=2,
        )
        warned.add(api_name)

    def _resolve_per_replica_rescale_config(
        self,
        *,
        rescale_per_replica: bool,
        cfg: PerReplicaRescaleConfig | None,
        kernel_min: float,
        kernel_max: float,
    ) -> PerReplicaRescaleConfig:
        cfg = cfg if cfg is not None else PerReplicaRescaleConfig()
        if rescale_per_replica and not cfg.enabled:
            cfg = replace(cfg, enabled=True)

        if cfg.target_mode not in ("fixed", "adaptive_quantile"):
            raise ValueError(
                "PerReplicaRescaleConfig.target_mode must be 'fixed' or "
                f"'adaptive_quantile', got {cfg.target_mode!r}."
            )
        if not (0.0 < cfg.adaptive_ref_quantile < 1.0):
            raise ValueError(
                "PerReplicaRescaleConfig.adaptive_ref_quantile must be in (0, 1)."
            )
        if not (cfg.adaptive_band_low > 0.0 and cfg.adaptive_band_high > 0.0):
            raise ValueError(
                "PerReplicaRescaleConfig adaptive band values must be > 0."
            )
        if cfg.adaptive_band_low > cfg.adaptive_band_high:
            raise ValueError(
                "PerReplicaRescaleConfig.adaptive_band_low must be <= "
                "adaptive_band_high."
            )
        if cfg.bw_min <= 0.0 or cfg.bw_max <= 0.0 or cfg.bw_min > cfg.bw_max:
            raise ValueError(
                "PerReplicaRescaleConfig requires 0 < bw_min <= bw_max."
            )
        if cfg.log_step_size <= 0.0:
            raise ValueError("PerReplicaRescaleConfig.log_step_size must be > 0.")
        if cfg.log_error_clip <= 0.0:
            raise ValueError("PerReplicaRescaleConfig.log_error_clip must be > 0.")
        if not (0.0 <= cfg.global_pull <= 1.0):
            raise ValueError("PerReplicaRescaleConfig.global_pull must be in [0, 1].")

        row_min = (
            kernel_min if cfg.fixed_row_sum_min is None else float(cfg.fixed_row_sum_min)
        )
        row_max = (
            kernel_max if cfg.fixed_row_sum_max is None else float(cfg.fixed_row_sum_max)
        )
        if row_min <= 0.0 or row_max <= 0.0 or row_min > row_max:
            raise ValueError(
                "PerReplicaRescaleConfig fixed row-sum bounds require "
                "0 < fixed_row_sum_min <= fixed_row_sum_max."
            )
        return replace(cfg, fixed_row_sum_min=row_min, fixed_row_sum_max=row_max)

    def _normalize_bw_init(self, bandwidth: Union[float, JaxArray]) -> JaxArray:
        bw_arr = jnp.asarray(bandwidth)
        if bw_arr.ndim == 0:
            bw_vec = jnp.full((self.num_replicas,), bw_arr)
        elif bw_arr.ndim == 1:
            if bw_arr.shape[0] == self.num_replicas:
                bw_vec = bw_arr
            elif bw_arr.shape[0] == 1:
                bw_vec = jnp.full((self.num_replicas,), bw_arr[0])
            else:
                raise ValueError(
                    f"bandwidth shape mismatch: expected scalar, (1,), or "
                    f"({self.num_replicas},), got {bw_arr.shape}."
                )
        else:
            raise ValueError(
                f"bandwidth rank must be 0 or 1, got shape {bw_arr.shape}."
            )
        return get_jax_array(bw_vec, sharding=self.in_sharding)

    def _normalize_bw_runtime(
        self, bw: JaxArray, bw_template: JaxArray
    ) -> tuple[JaxArray, bool]:
        """Normalize runtime bandwidth to shape (num_replicas,)."""
        bw_arr = jnp.asarray(bw)
        if bw_arr.ndim == 0:
            return jnp.ones_like(bw_template) * bw_arr, False
        if bw_arr.ndim == 1:
            if bw_arr.shape[0] == self.num_replicas:
                return bw_arr, True
            if bw_arr.shape[0] == 1:
                return jnp.ones_like(bw_template) * bw_arr[0], False
            raise ValueError(
                f"bandwidth shape mismatch: expected scalar, (1,), or "
                f"({self.num_replicas},), got {bw_arr.shape}."
            )
        raise ValueError(f"bandwidth rank must be 0 or 1, got shape {bw_arr.shape}.")

    def _rff_scalar_bw(
        self, bw_vec: JaxArray, *, explicit_per_replica: bool, api_name: str
    ) -> JaxArray:
        if explicit_per_replica:
            self._warn_rff_scalar_bw_once(api_name)
        return jnp.mean(bw_vec)

    def _rff_key_replicated_sharding(self):
        """Return a replicated sharding on the active mesh, if available."""
        cached = getattr(self, "_cached_replicated_sharding", None)
        if cached is not None:
            return cached
        for sh in (self.in_sharding, self.in_sharding_, self.out_sharding):
            mesh = getattr(sh, "mesh", None)
            if mesh is None:
                continue
            try:
                sharding = jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec()
                )
                self._cached_replicated_sharding = sharding
                return sharding
            except Exception:
                return None
        return None

    def _normalize_rff_key_sharding(self, key: JaxArray) -> JaxArray:
        """Keep RFF keys on a stable replicated sharding to avoid JIT cache splits."""
        if key is None:
            return key
        target_sharding = self._rff_key_replicated_sharding()
        if target_sharding is None:
            return key
        cur_sharding = getattr(key, "sharding", None)
        if cur_sharding == target_sharding:
            return key
        try:
            return jax.device_put(key, target_sharding)
        except Exception:
            return key

    def _normalize_torus_logits_sharding(self, logits: JaxArray) -> JaxArray:
        if logits is None:
            return logits
        target_sharding = self._rff_key_replicated_sharding()
        if target_sharding is None:
            return logits
        cur_sharding = getattr(logits, "sharding", None)
        if cur_sharding == target_sharding:
            return logits
        try:
            return jax.device_put(logits, target_sharding)
        except Exception:
            return logits

    def _normalize_torus_log_gamma_sharding(self, log_gamma: JaxArray) -> JaxArray:
        if log_gamma is None:
            return log_gamma
        target_sharding = self._rff_key_replicated_sharding()
        if target_sharding is None:
            return log_gamma
        cur_sharding = getattr(log_gamma, "sharding", None)
        if cur_sharding == target_sharding:
            return log_gamma
        try:
            return jax.device_put(log_gamma, target_sharding)
        except Exception:
            return log_gamma

    def _torus_cache_arrays_from_bw(self, bw_scalar: JaxArray, n_dim: int):
        gamma = 1.0 / (2.0 * jnp.square(bw_scalar))
        gamma = jnp.maximum(gamma, jnp.asarray(1e-30, dtype=jnp.asarray(gamma).dtype))
        gamma_eff = gamma / jnp.asarray(n_dim, dtype=jnp.asarray(gamma).dtype)
        gamma_eff = jnp.maximum(
            gamma_eff, jnp.asarray(1e-30, dtype=jnp.asarray(gamma_eff).dtype)
        )
        logits, _ = create_torus_bessel_logits(gamma, self.torus_m_cutoff, n_dim)
        log_gamma = jnp.log(gamma_eff)
        logits = self._normalize_torus_logits_sharding(logits)
        log_gamma = self._normalize_torus_log_gamma_sharding(log_gamma)
        return logits, log_gamma

    def _torus_cache_state_for_bw(
        self, coord: JaxArray, state: GaussianKernelState, bw_scalar: JaxArray
    ) -> GaussianKernelState:
        if not (self.rff and self.is_torus_distance):
            return state
        logits = state.torus_logits
        log_gamma = state.torus_log_gamma
        if logits is None or log_gamma is None:
            logits, log_gamma = self._torus_cache_arrays_from_bw(
                bw_scalar, coord.shape[-1]
            )
            return replace(state, torus_logits=logits, torus_log_gamma=log_gamma)

        gamma = 1.0 / (2.0 * jnp.square(bw_scalar))
        gamma = jnp.maximum(gamma, jnp.asarray(1e-30, dtype=jnp.asarray(gamma).dtype))
        gamma_eff = gamma / jnp.asarray(coord.shape[-1], dtype=jnp.asarray(gamma).dtype)
        gamma_eff = jnp.maximum(
            gamma_eff, jnp.asarray(1e-30, dtype=jnp.asarray(gamma_eff).dtype)
        )
        target_log_gamma = jnp.log(gamma_eff)
        tol = jnp.asarray(1e-6, dtype=target_log_gamma.dtype)
        target_sharding = self._rff_key_replicated_sharding()

        def _reuse(_):
            return logits, log_gamma

        def _recompute(_):
            # JIT path: keep everything in JAX and preserve replicated sharding.
            new_logits, _ = create_torus_bessel_logits(
                gamma, self.torus_m_cutoff, coord.shape[-1]
            )
            if target_sharding is not None:
                new_logits = lax.with_sharding_constraint(new_logits, target_sharding)
                new_log_gamma = lax.with_sharding_constraint(
                    target_log_gamma, target_sharding
                )
            else:
                new_log_gamma = target_log_gamma
            return new_logits, new_log_gamma

        logits, log_gamma = lax.cond(
            jnp.abs(target_log_gamma - log_gamma) <= tol,
            _reuse,
            _recompute,
            operand=None,
        )
        return replace(state, torus_logits=logits, torus_log_gamma=log_gamma)

    def __init__(
        self,
        num_replicas: int,
        bandwidth: Union[float, JaxArray] = 0.1,
        distance_func: callable = pairwise_distance,
        kernel_min: float = 2.0,
        kernel_max: float = 4.0,
        kernel_scale: float = 0.1,
        update_bw_method="rescale",
        rescale_per_replica: bool = False,
        rescale_per_replica_config: PerReplicaRescaleConfig | None = None,
        include_self: bool = False,
        center_gradient: str = "freeze",
        score_backend: str = "jax",
        fused_allow_tf32: bool = False,
        pair_bandwidth: str = "row",
        score_fn=None,
        rff=False,
        rff_dim=512,
        rff_seed=0,
        torus_m_cutoff=30,
        max_rescale_iters: int = 128,
        in_sharding=None,
        in_sharding_=None,
        out_sharding=None,
    ):
        super().__init__(num_replicas=num_replicas)

        if center_gradient not in ("freeze", "retain"):
            raise ValueError(
                f"center_gradient must be 'freeze' or 'retain' but got {center_gradient!r}"
            )
        if pair_bandwidth not in ("row", "symmetric"):
            raise ValueError(
                f"pair_bandwidth must be 'row' or 'symmetric' but got {pair_bandwidth!r}"
            )
        self.include_self = bool(include_self)
        self.center_gradient = center_gradient
        self.pair_bandwidth = pair_bandwidth
        self.score_fn = score_fn
        self.particle_weights = None

        # 验证距离函数
        supported_distance_funcs = {pairwise_distance, torus_pairwise_distance}
        if distance_func not in supported_distance_funcs:
            raise ValueError(
                f"Unsupported distance function: {distance_func}. "
                f"Supported functions are: {[func.__name__ for func in supported_distance_funcs]}"
            )

        if in_sharding is not None:
            if in_sharding_ is None:
                in_sharding_ = in_sharding
            if out_sharding is None:
                out_sharding = in_sharding
            self.in_sharding = in_sharding
            self.in_sharding_ = in_sharding_
            self.out_sharding = out_sharding
            self.distance_func = jax.jit(
                distance_func,
                in_shardings=self.in_sharding,
                out_shardings=self.out_sharding,
            )
        else:
            self.in_sharding = None
            self.out_sharding = None
            self.in_sharding_ = None
            self.distance_func = jax.jit(distance_func)

        self.kernel_min = get_jax_array(kernel_min, sharding=self.in_sharding)
        self.kernel_max = get_jax_array(kernel_max, sharding=self.in_sharding)
        self.kernel_mid = (self.kernel_max + self.kernel_min) / 2

        self.kernel_scale = get_jax_array(
            kernel_scale, requires_grad=False, sharding=self.in_sharding
        )
        self.bandwidth = self._normalize_bw_init(bandwidth)
        # may need sharding here

        self.bw_init = self.bandwidth

        self.update_bw_method = update_bw_method
        self.rescale_per_replica_config = self._resolve_per_replica_rescale_config(
            rescale_per_replica=rescale_per_replica,
            cfg=rescale_per_replica_config,
            kernel_min=kernel_min,
            kernel_max=kernel_max,
        )
        self.rescale_per_replica = bool(self.rescale_per_replica_config.enabled)
        # 添加rescale kernel所需的参数
        self.ksum_max = (
            self.rescale_per_replica_config.fixed_row_sum_max
            if self.rescale_per_replica_config.fixed_row_sum_max is not None
            else kernel_max
        )
        self.ksum_min = (
            self.rescale_per_replica_config.fixed_row_sum_min
            if self.rescale_per_replica_config.fixed_row_sum_min is not None
            else kernel_min
        )
        self.ds = jnp.power(2.0, 0.1)  # 1 + eps
        self.max_rescale_iters = int(max_rescale_iters)

        self.m_min = self.kernel_min / (self.num_replicas - 1)
        self.m_max = self.kernel_max / (self.num_replicas - 1)

        self.rff = rff
        self.rff_dim = rff_dim
        self.rff_key = random.PRNGKey(rff_seed)
        if self.rff:
            self.rff_key = self._normalize_rff_key_sharding(self.rff_key)
        self.rff_counter = 0  # 添加计数器确保每次调用都使用不同的随机数

        # 环面RFF相关参数
        self.torus_m_cutoff = torus_m_cutoff
        self.is_torus_distance = distance_func == torus_pairwise_distance
        self.cross_distance_func = jax.jit(
            torus_pairwise_distance_between
            if self.is_torus_distance
            else pairwise_distance_between
        )

        # 只在非RFF模式或使用trick方法时初始化gaussian_exp矩阵
        if not self.rff or self.update_bw_method == "trick":
            self.gaussian_exp = get_jax_array(
                jnp.zeros(
                    (self.num_replicas, self.num_replicas), device=self.out_sharding
                ),
                requires_grad=False,
                sharding=self.out_sharding,
            )
        else:
            # RFF模式下不需要完整的距离矩阵，设置为None节省内存
            self.gaussian_exp = None

        self.torus_k_range = None
        self.torus_logits = None
        self.torus_log_gamma = None
        if self.rff and self.is_torus_distance:
            _, k_range = create_torus_bessel_logits(
                jnp.asarray(1.0, dtype=jnp.float32), self.torus_m_cutoff, 1
            )
            self.torus_k_range = self._normalize_torus_logits_sharding(k_range)
            k_size = int(2 * self.torus_m_cutoff + 1)
            logits0 = jnp.zeros((k_size,), dtype=jnp.float32)
            log_gamma0 = jnp.asarray(jnp.inf, dtype=jnp.float32)
            self.torus_logits = self._normalize_torus_logits_sharding(logits0)
            self.torus_log_gamma = self._normalize_torus_log_gamma_sharding(log_gamma0)

        # JIT 编译核心函数
        self.kernel_jit = jax.jit(self.kernel_)
        self.compute_kernel_row_sum_jit = jax.jit(self.compute_kernel_row_sum_)
        self.compute_kernel_power_row_sum_jit = jax.jit(
            self.compute_kernel_power_row_sum_
        )
        self.rescale_bandwidth_by_row_sum_jit = jax.jit(
            self.rescale_bandwidth_by_row_sum_
        )
        # self.kernel_grad = jax.jit(self.kernel_grad_)
        self.kernel_grad = self.kernel_grad_
        self.get_bandwidth = jax.jit(self.get_bandwidth_)
        self.rescale_kernel = jax.jit(self.rescale_kernel_)
        self.update = self.update_
        self._compute_rff_features = jit(self._compute_rff_features_)
        self._compute_rff_features_torus = jit(self._compute_rff_features_torus_)
        self.sample_rff_params = jit(
            sample_rff_params,
            static_argnames=["input_dim", "n_features"],
            out_shardings=self.in_sharding_,
        )
        # self.update = jax.jit(self.update_)

        if self.update_bw_method == "trick":
            # trick方法需要完整的距离矩阵，与RFF不兼容
            if self.rff:
                raise ValueError(
                    "Update method 'trick' requires full distance matrix and is not compatible with RFF mode. "
                    "Please use 'EM' or 'rescale' method instead."
                )
            self.update_bw_jit = self.get_bandwidth
            self.eye = jnp.eye(
                self.num_replicas, self.num_replicas, device=self.out_sharding
            )
            self.eye = get_jax_array(
                self.eye, sharding=self.out_sharding, requires_grad=False
            )

        # jitted grad
        if self.rff:
            self.grad_jit = jit(grad(self._grad_jit_func_rff, 0))
            self.grad_jit_torus_cached = jit(
                grad(self._grad_jit_func_rff_torus_cached, 0)
            )
        else:
            self.grad_jit = jit(grad(self._grad_jit_func, 0))

        if score_backend not in ("jax", "fused"):
            raise ValueError(
                f"score_backend must be 'jax' or 'fused' but got {score_backend!r}"
            )
        self.score_backend = score_backend
        if score_backend == "fused":
            if self.rff or not self.is_torus_distance or center_gradient != "freeze":
                raise ValueError(
                    "score_backend='fused' requires the non-RFF torus kernel with "
                    "center_gradient='freeze'"
                )
            include_self_ = self.include_self
            allow_tf32 = bool(fused_allow_tf32)
            if pair_bandwidth != "row":
                raise ValueError("score_backend='fused' requires pair_bandwidth='row'")
            self.grad_jit = jit(
                lambda x, bw: torus_row_score(
                    x, bw, include_self=include_self_, allow_tf32=allow_tf32
                )[0]
            )

    def _grad_jit_func(self, x, bw):
        return jnp.sum(self._log_density(x, bw))

    def _grad_jit_func_rff(self, x, bw, subkey):
        z = self._compute_rff_features(x, bw, subkey)
        return jnp.sum(self._rff_log_density(z))

    def _grad_jit_func_rff_torus_cached(self, x, bw, subkey, torus_logits):
        del bw
        z = self._compute_rff_features_torus_cached_(x, subkey, torus_logits)
        return jnp.sum(self._rff_log_density(z))

    def _log_density(self, coord: JaxArray, bw: JaxArray) -> JaxArray:
        """Per-replica log kernel density (B,) for the score objective.

        With ``center_gradient="freeze"`` the kernel centers enter through
        ``stop_gradient``, so the gradient of the summed log density is the
        pointwise KDE score at every evaluation point.
        """
        if self.center_gradient == "freeze":
            distance_mat = self.cross_distance_func(coord, lax.stop_gradient(coord))
        else:
            distance_mat = self.distance_func(coord)
        kernel_exp = -0.5 * distance_mat * self._pair_inverse_squared(bw)
        if not self.include_self:
            kernel_exp = jnp.where(
                jnp.eye(coord.shape[0], dtype=jnp.bool_), -jnp.inf, kernel_exp
            )
        return logsumexp(kernel_exp, axis=1)

    def _rff_log_density(self, z: JaxArray) -> JaxArray:
        """RFF counterpart of ``_log_density`` from feature rows z (B, F)."""
        if self.center_gradient == "retain":
            return rff_logsum(z, exclude_self=not self.include_self)[:, 0]
        centers = lax.stop_gradient(z)
        row_sum = z @ jnp.sum(centers, axis=0)
        if not self.include_self:
            row_sum = row_sum - jnp.sum(z * centers, axis=-1)
        return jnp.log(jnp.maximum(row_sum, 1e-10))

    def _pair_inverse_squared(self, bw: JaxArray) -> JaxArray:
        """Inverse squared pair bandwidth, broadcastable to (B, B)."""
        bw = jnp.asarray(bw)
        if bw.ndim == 0:
            return jnp.reciprocal(jnp.square(bw))
        bw_squared = jnp.square(bw.reshape(-1))
        if self.pair_bandwidth == "symmetric":
            return jnp.reciprocal(0.5 * (bw_squared[:, None] + bw_squared[None, :]))
        return jnp.reciprocal(bw_squared)[:, None]

    def _without_self(self, kernel: JaxArray) -> JaxArray:
        """Zero the diagonal of a square replica kernel matrix (leave-one-out)."""
        if self.include_self:
            return kernel
        return kernel - jnp.diag(jnp.diag(kernel))

    def get_bandwidth_(
        self, gaussian_exp: JaxArray, bandwidth: JaxArray = None
    ) -> JaxArray:
        """
        Calculate optimized bandwidth.

        Args:
            gaussian_exp: The exponential term of Gaussian kernel matrix.
            bandwidth: The bandwidth of the Gaussian kernel.
        Returns:
            bandwidth: The optimized bandwidth.
        """

        def _mean(gaussian_exp_: JaxArray, bandwidth_: JaxArray) -> JaxArray:
            k = jnp.exp(gaussian_exp_ * jnp.reciprocal(jnp.square(bandwidth_)))
            k = self._without_self(k)
            return jnp.mean(jnp.sum(k, axis=-1))

        if bandwidth is None:
            k_mean = _mean(gaussian_exp, bandwidth)
            if self.kernel_min <= k_mean <= self.kernel_max:
                return jnp.copy(bandwidth)

        # 初始化参数
        k_min = jnp.min(gaussian_exp)
        self_mask = jnp.eye(gaussian_exp.shape[0], dtype=jnp.bool_)
        k_max = jnp.max(jnp.where(self_mask, k_min, gaussian_exp))

        s_min = jnp.sqrt(k_max / jnp.log(self.m_min))
        s_max = jnp.sqrt(k_min / jnp.log(self.m_max))

        # repeat s_min and s_max

        # 调整 s_min 和 s_max
        def adjust_s(operand):
            s_min, s_max, bandwidth = operand
            k_mean = _mean(gaussian_exp, bandwidth)
            s_max = jnp.where(
                (k_mean > self.kernel_max) & (s_max > bandwidth),
                bandwidth,
                s_max,
            )
            s_min = jnp.where(
                (k_mean < self.kernel_min) & (s_min < bandwidth),
                bandwidth,
                s_min,
            )
            return s_min, s_max

        def repeat_s(operand):
            s_min, s_max, bandwidth = operand
            # return s_min, s_max
            return jnp.ones_like(bandwidth) * s_min, jnp.ones_like(bandwidth) * s_max

        s_min, s_max = lax.cond(
            bandwidth is not None,
            adjust_s,
            repeat_s,
            operand=(s_min, s_max, bandwidth),
        )

        # 初始化带宽
        bandwidth_init = (s_max + s_min) / 2

        # 定义条件函数和循环体
        def cond_fun(loop_carry):
            s_min, s_max, bandwidth = loop_carry
            k_mean = _mean(gaussian_exp, bandwidth)
            return ~jnp.all((self.kernel_min <= k_mean) & (k_mean <= self.kernel_max))

        def body_fun(loop_carry):
            s_min, s_max, bandwidth = loop_carry
            k_mean = _mean(gaussian_exp, bandwidth)
            s_max = jnp.where(k_mean > self.kernel_max, bandwidth, s_max)
            s_min = jnp.where(k_mean < self.kernel_min, bandwidth, s_min)
            bandwidth = (s_max + s_min) / 2
            return s_min, s_max, bandwidth

        # 使用 jax.lax.while_loop 替换 while 循环
        s_min, s_max, bandwidth = lax.while_loop(
            cond_fun, body_fun, (s_min, s_max, bandwidth_init)
        )

        return bandwidth

    def rescale_kernel_(
        self, kernel: JaxArray, scale: JaxArray
    ) -> tuple[JaxArray, JaxArray]:
        """重新调整kernel的scale以确保kernel matrix的行和在合适范围内 (JIT 兼容版本)

        Args:
            kernel: kernel matrix
            scale: 当前scale值

        Returns:
            tuple: (新的scale值, 新的kernel row sum)
        """
        kernel = self._without_self(kernel)
        k = kernel
        s = scale

        # Accept scalar/(1,)/vector scale and normalize to per-replica vector
        # so lax.while_loop carry shape stays invariant.
        scale_arr = jnp.asarray(scale)
        if scale_arr.ndim == 0:
            scale_vec = jnp.full((k.shape[0],), scale_arr)
        elif scale_arr.ndim == 1:
            if scale_arr.shape[0] == k.shape[0]:
                scale_vec = scale_arr
            elif scale_arr.shape[0] == 1:
                scale_vec = jnp.full((k.shape[0],), scale_arr[0])
            else:
                raise ValueError(
                    f"scale shape mismatch: expected scalar, (1,), or ({k.shape[0]},), "
                    f"got {scale_arr.shape}."
                )
        else:
            raise ValueError(
                f"scale rank must be 0 or 1, got shape {scale_arr.shape}."
            )
        s = scale_vec

        if self.rescale_per_replica:
            cfg = self.rescale_per_replica_config
            mode = cfg.target_mode
            bw_min = jnp.asarray(cfg.bw_min, dtype=scale_vec.dtype)
            bw_max = jnp.asarray(cfg.bw_max, dtype=scale_vec.dtype)
            log_bw_min = jnp.log(bw_min)
            log_bw_max = jnp.log(bw_max)
            row_floor = jnp.asarray(cfg.fixed_row_sum_min, dtype=scale_vec.dtype)
            row_ceil = jnp.asarray(cfg.fixed_row_sum_max, dtype=scale_vec.dtype)
            eta = jnp.asarray(cfg.log_step_size, dtype=scale_vec.dtype)
            err_clip = jnp.asarray(cfg.log_error_clip, dtype=scale_vec.dtype)
            global_pull = jnp.asarray(cfg.global_pull, dtype=scale_vec.dtype)
            eps = jnp.asarray(1e-12, dtype=scale_vec.dtype)

            def _fixed_bounds(row_sum_c):
                low = jnp.ones_like(row_sum_c) * row_floor
                high = jnp.ones_like(row_sum_c) * row_ceil
                return low, high

            if mode == "adaptive_quantile":
                q = float(cfg.adaptive_ref_quantile)
                band_low = jnp.asarray(cfg.adaptive_band_low, dtype=scale_vec.dtype)
                band_high = jnp.asarray(cfg.adaptive_band_high, dtype=scale_vec.dtype)

                def _bounds(row_sum_c):
                    ref = jnp.quantile(row_sum_c, q)
                    low = jnp.clip(ref * band_low, row_floor, row_ceil)
                    high = jnp.clip(ref * band_high, row_floor, row_ceil)
                    return (
                        jnp.ones_like(row_sum_c) * low,
                        jnp.ones_like(row_sum_c) * high,
                    )
            else:

                def _bounds(row_sum_c):
                    return _fixed_bounds(row_sum_c)

            s = jnp.clip(s, bw_min, bw_max)
            row_sum_init = jnp.sum(k, axis=-1)

            def cond_fn(c):
                _, row_sum_c, iters_c = c
                low_c, high_c = _bounds(row_sum_c)
                finite = jnp.isfinite(row_sum_c)
                hi = finite & (row_sum_c > high_c)
                lo = finite & (row_sum_c < low_c)
                return jnp.any(hi | lo) & (iters_c < self.max_rescale_iters)

            def body_fn(c):
                s_c, row_sum_c, iters_c = c
                low_c, high_c = _bounds(row_sum_c)
                finite = jnp.isfinite(row_sum_c)
                row_safe = jnp.maximum(row_sum_c, eps)
                hi_err = jnp.log(jnp.maximum(high_c, eps)) - jnp.log(row_safe)
                lo_err = jnp.log(jnp.maximum(low_c, eps)) - jnp.log(row_safe)
                err = jnp.where(
                    row_sum_c > high_c, hi_err, jnp.where(row_sum_c < low_c, lo_err, 0.0)
                )
                err = jnp.where(finite, err, 0.0)
                dz = eta * jnp.clip(err, -err_clip, err_clip)
                z = jnp.log(jnp.maximum(s_c, bw_min)) + dz
                z = jnp.clip(z, log_bw_min, log_bw_max)
                z = (1.0 - global_pull) * z + global_pull * jnp.mean(z)
                z = jnp.clip(z, log_bw_min, log_bw_max)
                s_new = jnp.exp(z)
                exponent = self._pair_inverse_squared(
                    s_new
                ) / self._pair_inverse_squared(scale_vec)
                k_new = jnp.power(kernel, exponent)
                row_sum_new = jnp.sum(k_new, axis=-1)
                return s_new, row_sum_new, iters_c + 1

            s_final, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, (s, row_sum_init, jnp.int32(0))
            )
            return s_final, jnp.mean(row_sum_final)

        row_sum_init = jnp.mean(jnp.sum(k, axis=-1))

        # replica = self.num_replicas

        # --- 核心修改部分 ---

        # 为 lax.cond 定义三个分支函数。
        # 它们必须具有相同的输入和输出签名。
        # 输入: carry -> (scale, kernel, row_sum)
        # 输出: result -> (final_scale, final_row_sum)

        def scale_down_loop(carry):
            """执行 'scale down' 的 while 循环并返回最终结果"""
            s_current, k_current, row_sum_current = carry

            def cond_fn(c):
                # 当行和大于目标值时继续循环；加入最大迭代保护防止超长循环
                _, _, row_sum_c, iters_c = c
                return (
                    jnp.isfinite(row_sum_c)
                    & (row_sum_c > self.ksum_max)
                    & (iters_c < self.max_rescale_iters)
                )

            def body_fn(c):
                s_c, _, _, iters_c = c
                s_new = s_c / self.ds
                # 为了数值稳定性，从原始 kernel 计算，而不是迭代地 power
                k_new = jnp.power(kernel, (scale / s_new) ** 2)
                row_sum_new = jnp.mean(jnp.sum(k_new, axis=-1))
                return s_new, k_new, row_sum_new, iters_c + 1

            s_final, _, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, (s_current, k_current, row_sum_current, jnp.int32(0))
            )
            return s_final, row_sum_final

        def scale_up_loop(carry):
            """执行 'scale up' 的 while 循环并返回最终结果"""
            s_current, k_current, row_sum_current = carry

            def cond_fn(c):
                # 当行和小于目标值时继续循环；加入最大迭代保护防止超长循环
                _, _, row_sum_c, iters_c = c
                return (
                    jnp.isfinite(row_sum_c)
                    & (row_sum_c < self.ksum_min)
                    & (iters_c < self.max_rescale_iters)
                )

            def body_fn(c):
                s_c, _, _, iters_c = c
                s_new = s_c * self.ds
                k_new = jnp.power(kernel, (scale / s_new) ** 2)
                row_sum_new = jnp.mean(jnp.sum(k_new, axis=-1))
                return s_new, k_new, row_sum_new, iters_c + 1

            s_final, _, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, (s_current, k_current, row_sum_current, jnp.int32(0))
            )
            return s_final, row_sum_final

        def do_nothing(carry):
            """在无需调整的情况下直接返回初始值"""
            s_current, _, row_sum_current = carry
            return s_current, row_sum_current

        # 初始状态，将作为 operand 传递给 lax.cond
        initial_state = (s, k, row_sum_init)

        # 使用嵌套的 lax.cond 实现三路分支
        # 外层 cond 判断是否需要 scale down
        # 内层 cond 判断是需要 scale up 还是 do nothing
        return lax.cond(
            row_sum_init > self.ksum_max,
            # True 分支: 执行 scale down
            scale_down_loop,
            # False 分支: 进入下一个条件判断
            lambda carry: lax.cond(
                row_sum_init < self.ksum_min,
                # True 分支: 执行 scale up
                scale_up_loop,
                # False 分支: 什么也不做
                do_nothing,
                operand=carry,  # 将 carry 传递给内层 cond
            ),
            operand=initial_state,  # 传递给外层 cond 的初始状态
        )

    def update_(self, coord) -> None:
        state = self.update_with_state(coord, self.get_state())
        self._set_state_fast(state)

    def kernel_(
        self, coord: JaxArray, bw: JaxArray, rff_key: JaxArray = None
    ) -> JaxArray:
        """
        Compute the Gaussian kernel matrix for the given coordinates.

        Args:
            coord: The coordinates of the replicas.
            bandwidth: The bandwidth of the Gaussian kernel.
            rff_key: Random key for RFF sampling (required when rff=True).
        Returns:
            Per-replica log kernel density (leave-one-out unless include_self).
        """
        # (B,A,D) -> (B,B)
        if not self.rff:
            distance_mat = self.distance_func(coord)
            # (B,B)
            kernel_exp = -0.5 * distance_mat * self._pair_inverse_squared(bw)
            if not self.include_self:
                kernel_exp = jnp.where(
                    jnp.eye(coord.shape[0], dtype=jnp.bool_), -jnp.inf, kernel_exp
                )
            # (B,B) -> (B,1)
            kernel = logsumexp(kernel_exp, axis=1, keepdims=True)
        else:
            # TODO: rff when bw is ArrayLike
            z = self._compute_rff_features(coord, bw, rff_key)
            kernel = rff_logsum(z, exclude_self=not self.include_self)

        # 确保返回形状为(B, 1, 1)以便于广播
        return kernel[:, None, :]

    def kernel_grad_(self, coord: JaxArray, bw=None) -> JaxArray:
        """
        Compute the gradient of the Gaussian kernel.
        """
        grad_val, state = self.kernel_grad_with_state(coord, self.get_state(), bw=bw)
        self._set_state_fast(state)
        return grad_val

    def bw_reset(self) -> None:
        """Reset the bandwidth to a default value."""
        self._set_state_fast(self.bw_reset_with_state(self.get_state()))

    def bw_grad(self, coord: JaxArray) -> JaxArray:
        """
        Compute the gradient of the bandwidth of the Gaussian kernel.

        Args:
            coord: The coordinates of the replicas.
        Returns:
            The gradient of the bandwidth.
        """
        grads = grad(lambda x, bw: jnp.sum(self.kernel(x, bw)), 1)(
            coord, self.bandwidth
        )
        return jnp.clip(grads, -1, 1)

    def compute_kernel_row_sum_(
        self, coord: JaxArray, bw: JaxArray = None, rff_key: JaxArray = None
    ) -> JaxArray:
        """
        计算排除对角项后kernel矩阵的行和。

        Args:
            coord: The coordinates of the replicas.
            bw: The bandwidth of the Gaussian kernel. If None, use current bandwidth.
            rff_key: Random key for RFF sampling (required when rff=True).
        Returns:
            row_sum: The row sum of the kernel matrix (B,)
        """

        if bw is None:
            bw = self.bandwidth

        if not self.rff:
            distance_mat = self.distance_func(coord)
            kernel = jnp.exp(-0.5 * distance_mat * self._pair_inverse_squared(bw))
            row_sum = jnp.sum(self._without_self(kernel), axis=-1)
        else:
            # RFF模式：使用RFF特征计算行和
            z = self._compute_rff_features(coord, bw, rff_key)
            row_sum = rff_row_sum(z, exclude_self=not self.include_self)

        return row_sum

    def compute_kernel_power_row_sum_(
        self,
        coord: JaxArray,
        bw: JaxArray,
        power: Union[float, JaxArray],
        rff_key: JaxArray = None,
    ):
        if bw is None:
            bw = self.bandwidth

        if not self.rff:
            distance_mat = self.distance_func(coord)
            kernel = jnp.exp(-0.5 * distance_mat * self._pair_inverse_squared(bw))
            kernel = self._without_self(kernel)
            row_sum_power = jnp.sum(jnp.power(kernel, power), axis=-1)
        else:
            # RFF模式：使用RFF特征计算行和
            z = self._compute_rff_features(coord, bw, rff_key)
            # print("z.shape:", z.shape)
            # print("power.shape:", power.shape)
            row_sum_power = rff_power_row_sum(z, power, exclude_self=not self.include_self)

        return row_sum_power

    def rescale_bandwidth_by_row_sum_(
        self, coord: JaxArray, bw: JaxArray, rff_key: JaxArray = None
    ) -> tuple[JaxArray, JaxArray]:
        """
        通过kernel矩阵的行和来重新调整bandwidth，支持RFF模式

        Args:
            coord: The coordinates of the replicas.
            bw: 当前bandwidth值
            rff_key: Random key for RFF sampling (required when rff=True).

        Returns:
            tuple: (新的bandwidth值, 新的kernel row sum)
        """

        # replicas = self.num_replicas

        if self.rff:
            # RFF path: use log-space secant updates on scalar bandwidth and
            # evaluate only mean row-sum (no power/eigh in the loop).
            key_eval = rff_key

            def compute_mean_row_sum_bw(bw_scalar: JaxArray) -> JaxArray:
                row_sum = self.compute_kernel_row_sum_jit(coord, bw_scalar, key_eval)
                return jnp.mean(row_sum)

            eps_bw = jnp.asarray(1e-12, dtype=jnp.asarray(bw).dtype)
            eps_m = jnp.asarray(1e-12, dtype=jnp.asarray(bw).dtype)
            eps_f = jnp.asarray(1e-8, dtype=jnp.asarray(bw).dtype)
            log_ds = jnp.log(self.ds)
            target = jnp.maximum(jnp.mean((self.ksum_max + self.ksum_min) / 2), eps_m)
            log_target = jnp.log(target)

            # Keep log(bw) in a finite range to avoid exp overflow/underflow.
            log_bw_min = jnp.log(eps_bw)
            log_bw_max = jnp.asarray(30.0, dtype=jnp.asarray(bw).dtype)

            def safe_residual(m_val: JaxArray) -> JaxArray:
                m_safe = jnp.where(
                    jnp.isfinite(m_val), jnp.maximum(m_val, eps_m), eps_m
                )
                return jnp.log(m_safe) - log_target

            bw0 = jnp.maximum(jnp.asarray(bw), eps_bw)
            x0 = jnp.clip(jnp.log(bw0), log_bw_min, log_bw_max)
            m0 = compute_mean_row_sum_bw(bw0)
            f0 = safe_residual(m0)

            # warm start with bounded fixed step in log-space
            x1 = x0 - jnp.sign(f0) * log_ds
            x1 = jnp.clip(x1, log_bw_min, log_bw_max)
            bw1 = jnp.maximum(jnp.exp(x1), eps_bw)
            m1 = compute_mean_row_sum_bw(bw1)
            f1 = safe_residual(m1)

            def cond_fn(carry):
                _, _, _, _, _, m_cur, iters = carry
                return (
                    jnp.isfinite(m_cur)
                    & ((m_cur > self.ksum_max) | (m_cur < self.ksum_min))
                    & (iters < self.max_rescale_iters)
                )

            def body_fn(carry):
                x_prev, f_prev, x_cur, f_cur, _, m_cur, iters = carry
                den = f_cur - f_prev
                use_secant = jnp.isfinite(den) & (jnp.abs(den) > eps_f)
                den_safe = jnp.where(use_secant, den, jnp.ones_like(den))
                x_sec = x_cur - f_cur * (x_cur - x_prev) / den_safe
                x_fb = x_cur - jnp.sign(f_cur) * log_ds
                x_prop = jnp.where(use_secant, x_sec, x_fb)
                dx = jnp.clip(x_prop - x_cur, -log_ds, log_ds)
                x_next = jnp.clip(x_cur + dx, log_bw_min, log_bw_max)

                bw_next = jnp.maximum(jnp.exp(x_next), eps_bw)
                m_next = compute_mean_row_sum_bw(bw_next)
                f_next = safe_residual(m_next)
                return x_cur, f_cur, x_next, f_next, bw_next, m_next, iters + 1

            init_carry = (x0, f0, x1, f1, bw1, m1, jnp.int32(0))
            _, _, _, _, bw_final, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, init_carry
            )
            ok = jnp.isfinite(row_sum_final)
            bw_final = jnp.where(ok, bw_final, bw0)
            row_sum_final = jnp.where(ok, row_sum_final, m0)
            return bw_final, row_sum_final
        else:
            if self.rescale_per_replica:
                gaussian_exp = -0.5 * self.distance_func(coord)
                kernel = jnp.exp(gaussian_exp * self._pair_inverse_squared(bw))
                return self.rescale_kernel(kernel, bw)

            # Non-RFF path keeps the previous compute-kernel-row-sum behavior.
            def compute_mean_row_sum(coord_: JaxArray, bw_: JaxArray) -> JaxArray:
                row_sum = self.compute_kernel_row_sum_jit(coord_, bw_, None)
                return jnp.mean(row_sum)

            def compute_mean_row_sum_power(
                coord_: JaxArray, bw_: JaxArray, power: Union[float, JaxArray]
            ) -> JaxArray:
                row_sum_power = self.compute_kernel_power_row_sum_jit(
                    coord_, bw_, power, None
                )
                return jnp.mean(row_sum_power)

            row_sum_init = compute_mean_row_sum(coord, bw)

        # 为 lax.cond 定义三个分支函数
        def scale_down_loop(carry):
            """执行 'scale down' 的 while 循环并返回最终结果"""
            bw_current, row_sum_current = carry

            def cond_fn(c):
                # 当行和大于目标值时继续循环
                _, row_sum_c, iters_c = c
                return (
                    jnp.isfinite(row_sum_c)
                    & (row_sum_c > self.ksum_max)
                    & (iters_c < self.max_rescale_iters)
                )

            def body_fn(c):
                bw_c, _, iters_c = c
                bw_new = bw_c / self.ds
                power = (bw / bw_new) ** 2
                row_sum_new = compute_mean_row_sum_power(coord, bw_new, power)
                return bw_new, row_sum_new, iters_c + 1

            bw_final, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, (bw_current, row_sum_current, jnp.int32(0))
            )
            return bw_final, row_sum_final

        def scale_up_loop(carry):
            """执行 'scale up' 的 while 循环并返回最终结果"""
            bw_current, row_sum_current = carry

            def cond_fn(c):
                # 当行和小于目标值时继续循环
                _, row_sum_c, iters_c = c
                return (
                    jnp.isfinite(row_sum_c)
                    & (row_sum_c < self.ksum_min)
                    & (iters_c < self.max_rescale_iters)
                )

            def body_fn(c):
                bw_c, _, iters_c = c
                bw_new = bw_c * self.ds
                power = (bw / bw_new) ** 2
                row_sum_new = compute_mean_row_sum_power(coord, bw_new, power)
                return bw_new, row_sum_new, iters_c + 1

            bw_final, row_sum_final, _ = lax.while_loop(
                cond_fn, body_fn, (bw_current, row_sum_current, jnp.int32(0))
            )
            return bw_final, row_sum_final

        def do_nothing(carry):
            """在无需调整的情况下直接返回初始值"""
            bw_current, row_sum_current = carry
            return bw_current, row_sum_current

        # 初始状态
        initial_state = (bw, row_sum_init)

        # 使用嵌套的 lax.cond 实现三路分支
        return lax.cond(
            row_sum_init > self.ksum_max,
            # True 分支: 执行 scale down
            scale_down_loop,
            # False 分支: 进入下一个条件判断
            lambda carry: lax.cond(
                row_sum_init < self.ksum_min,
                # True 分支: 执行 scale up
                scale_up_loop,
                # False 分支: 什么也不做
                do_nothing,
                operand=carry,  # 将 carry 传递给内层 cond
            ),
            operand=initial_state,  # 传递给外层 cond 的初始状态
        )

    def _compute_rff_features_(
        self, coord: JaxArray, bw: JaxArray, rff_key: JaxArray
    ) -> JaxArray:
        """
        计算RFF特征的通用方法

        Args:
            coord: The coordinates of the replicas.
            bw: The bandwidth of the Gaussian kernel.
            rff_key: Random key for RFF sampling.
        Returns:
            z: RFF features
        """
        # 根据距离函数类型自动选择RFF方法
        if self.is_torus_distance:
            # 使用环面距离时自动使用环面RFF
            return self._compute_rff_features_torus_(coord, bw, rff_key)
        else:
            # 使用标准欧几里得距离时使用标准高斯RFF
            w, b = self.sample_rff_params(
                rff_key,
                coord.shape[-1] * coord.shape[-2],
                n_features=self.rff_dim,
                bandwidth=jnp.square(bw),
            )
            z = compute_rff_features(coord, w, b)
            del w, b
            return z

    def _compute_rff_features_torus_(
        self, coord: JaxArray, bw: JaxArray, rff_key: JaxArray
    ) -> JaxArray:
        """
        计算环面RFF特征的方法

        Args:
            coord: The coordinates of the replicas.
            bw: The bandwidth of the Gaussian kernel.
            rff_key: Random key for RFF sampling.
        Returns:
            z: Torus RFF features
        """
        # 计算 gamma 参数
        gamma = 1.0 / (2.0 * jnp.square(bw))
        gamma = jnp.mean(gamma)  # 确保 gamma 是标量

        # 创建环面分布并采样
        n_dim = coord.shape[-1]
        dist, k_range = create_torus_bessel_pmf_dist(gamma, self.torus_m_cutoff, n_dim)

        input_dim = coord.shape[-1] * coord.shape[-2]
        k_samples = sample_rff_params_torus(
            rff_key, dist, k_range, input_dim, self.rff_dim
        )

        z = rff_feature_map_torus(coord, k_samples)
        return z

    def _compute_rff_features_torus_cached_(
        self, coord: JaxArray, rff_key: JaxArray, torus_logits: JaxArray
    ) -> JaxArray:
        input_dim = coord.shape[-1] * coord.shape[-2]
        k_samples = sample_rff_params_torus_from_logits(
            rff_key, torus_logits, self.torus_k_range, input_dim, self.rff_dim
        )
        return rff_feature_map_torus(coord, k_samples)

    def kernel(self, coord: JaxArray, bw: JaxArray = None) -> JaxArray:
        """
        Compute the Gaussian kernel matrix for the given coordinates.
        This is a wrapper function that handles random key management for RFF mode.

        Args:
            coord: The coordinates of the replicas.
            bw: The bandwidth of the Gaussian kernel. If None, use current bandwidth.
        Returns:
            The Gaussian kernel matrix.
        """
        out, state = self.kernel_with_state(coord, self.get_state(), bw=bw)
        self._set_state_fast(state)
        return out

    def compute_kernel_row_sum(self, coord: JaxArray, bw: JaxArray = None) -> JaxArray:
        """
        计算kernel矩阵的行和，这是一个包装函数，用于处理RFF模式的随机数密钥管理

        Args:
            coord: The coordinates of the replicas.
            bw: The bandwidth of the Gaussian kernel. If None, use current bandwidth.
        Returns:
            row_sum: The row sum of the kernel matrix (B,)
        """
        out, state = self.compute_kernel_row_sum_with_state(
            coord, self.get_state(), bw=bw
        )
        self._set_state_fast(state)
        return out

    def rescale_bandwidth_by_row_sum(
        self, coord: JaxArray, bw: JaxArray
    ) -> tuple[JaxArray, JaxArray]:
        """
        通过kernel矩阵的行和来重新调整bandwidth，这是一个包装函数，用于处理RFF模式的随机数密钥管理

        Args:
            coord: The coordinates of the replicas.
            bw: 当前bandwidth值

        Returns:
            tuple: (新的bandwidth值, 新的kernel row sum)
        """
        (new_bw, row_sum), state = self.rescale_bandwidth_by_row_sum_with_state(
            coord, self.get_state(), bw
        )
        self._set_state_fast(state)
        return new_bw, row_sum

    # -----------------------------
    # State API (Phase 1-3)
    # -----------------------------
    def get_state(self) -> GaussianKernelState:
        return GaussianKernelState(
            bandwidth=self.bandwidth,
            rff_key=self.rff_key,
            gaussian_exp=self.gaussian_exp,
            torus_logits=self.torus_logits,
            torus_log_gamma=self.torus_log_gamma,
            particle_weights=self.particle_weights,
        )

    def _set_state_fast(self, state: GaussianKernelState) -> None:
        """Internal fast path: assume state invariants (shape/sharding) already hold."""
        self.bandwidth = state.bandwidth
        self.rff_key = state.rff_key
        self.gaussian_exp = state.gaussian_exp
        self.torus_logits = state.torus_logits
        self.torus_log_gamma = state.torus_log_gamma
        self.particle_weights = state.particle_weights

    def set_state(self, state: GaussianKernelState) -> None:
        self.bandwidth = state.bandwidth
        self.rff_key = self._normalize_rff_key_sharding(state.rff_key)
        self.gaussian_exp = state.gaussian_exp
        self.torus_logits = self._normalize_torus_logits_sharding(state.torus_logits)
        self.torus_log_gamma = self._normalize_torus_log_gamma_sharding(
            state.torus_log_gamma
        )
        self.particle_weights = state.particle_weights

    def kernel_with_state(
        self, coord: JaxArray, state: GaussianKernelState, bw: JaxArray = None
    ) -> tuple[JaxArray, GaussianKernelState]:
        bw_in = state.bandwidth if bw is None else bw
        bw_vec, explicit_per_replica = self._normalize_bw_runtime(
            bw_in, state.bandwidth
        )
        if self.rff:
            next_key, subkey = random.split(state.rff_key)
            bw_scalar = self._rff_scalar_bw(
                bw_vec,
                explicit_per_replica=(bw is not None and explicit_per_replica),
                api_name="Gaussian.kernel_with_state",
            )
            out = self.kernel_jit(coord, bw_scalar, subkey)
            return out, replace(state, rff_key=next_key)
        return self.kernel_jit(coord, bw_vec), state

    def kernel_grad_with_state(
        self, coord: JaxArray, state: GaussianKernelState, bw: JaxArray = None
    ) -> tuple[JaxArray, GaussianKernelState]:
        bw_in = state.bandwidth if bw is None else bw
        bw_vec, explicit_per_replica = self._normalize_bw_runtime(
            bw_in, state.bandwidth
        )
        if self.score_fn is not None:
            return self.score_fn(coord, bw_vec, state.particle_weights), state
        if state.particle_weights is not None:
            raise ValueError(
                "particle-weighted scores require a score_fn on the Gaussian kernel"
            )
        if self.rff:
            next_key, subkey = random.split(state.rff_key)
            bw_scalar = self._rff_scalar_bw(
                bw_vec,
                explicit_per_replica=(bw is not None and explicit_per_replica),
                api_name="Gaussian.kernel_grad_with_state",
            )
            if self.is_torus_distance and state.torus_logits is not None:
                cached_state = self._torus_cache_state_for_bw(coord, state, bw_scalar)
                out = self.grad_jit_torus_cached(
                    coord, bw_scalar, subkey, cached_state.torus_logits
                )
                return out, replace(cached_state, rff_key=next_key)
            out = self.grad_jit(coord, bw_scalar, subkey)
            return out, replace(state, rff_key=next_key)
        return self.grad_jit(coord, bw_vec), state

    def compute_kernel_row_sum_with_state(
        self, coord: JaxArray, state: GaussianKernelState, bw: JaxArray = None
    ) -> tuple[JaxArray, GaussianKernelState]:
        bw_in = state.bandwidth if bw is None else bw
        bw_vec, explicit_per_replica = self._normalize_bw_runtime(
            bw_in, state.bandwidth
        )
        if self.rff:
            next_key, subkey = random.split(state.rff_key)
            bw_scalar = self._rff_scalar_bw(
                bw_vec,
                explicit_per_replica=(bw is not None and explicit_per_replica),
                api_name="Gaussian.compute_kernel_row_sum_with_state",
            )
            out = self.compute_kernel_row_sum_jit(coord, bw_scalar, subkey)
            return out, replace(state, rff_key=next_key)
        return self.compute_kernel_row_sum_jit(coord, bw_vec), state

    def rescale_bandwidth_by_row_sum_with_state(
        self, coord: JaxArray, state: GaussianKernelState, bw: JaxArray
    ) -> tuple[tuple[JaxArray, JaxArray], GaussianKernelState]:
        bw_vec, explicit_per_replica = self._normalize_bw_runtime(bw, state.bandwidth)
        if self.rff:
            next_key, subkey = random.split(state.rff_key)
            warn_per_replica = explicit_per_replica and (bw is not state.bandwidth)
            bw_scalar = self._rff_scalar_bw(
                bw_vec,
                explicit_per_replica=warn_per_replica,
                api_name="Gaussian.rescale_bandwidth_by_row_sum_with_state",
            )
            (new_bw, row_sum) = self.rescale_bandwidth_by_row_sum_jit(
                coord, bw_scalar, subkey
            )
            new_bw_vec = jnp.ones_like(state.bandwidth) * new_bw
            return (new_bw_vec, row_sum), replace(state, rff_key=next_key)

        gaussian_exp = state.gaussian_exp
        if gaussian_exp is None:
            gaussian_exp = -0.5 * self.distance_func(coord)
        kernel = jnp.exp(gaussian_exp * self._pair_inverse_squared(bw_vec))
        out = self.rescale_kernel(kernel, bw_vec)
        return out, replace(state, gaussian_exp=gaussian_exp)

    def update_with_state(
        self, coord: JaxArray, state: GaussianKernelState
    ) -> GaussianKernelState:
        """Functional update path for future scan/jit migration."""
        if self.update_bw_method == "fixed":
            return state
        bw = state.bandwidth
        gaussian_exp = state.gaussian_exp
        rff_key = state.rff_key

        if self.rff:
            if self.update_bw_method == "EM":
                next_key, subkey = random.split(rff_key)
                # d/d bw sum(kernel_jit(coord, bw, subkey))
                bw_grad = grad(
                    lambda bw_: jnp.sum(self.kernel_jit(coord, bw_, subkey))
                )(bw)
                bw = bw - 1e-3 * jnp.clip(bw_grad, -100, 100)
                rff_key = next_key
            elif self.update_bw_method == "rescale":
                (bw, _), new_state = self.rescale_bandwidth_by_row_sum_with_state(
                    coord, replace(state, bandwidth=bw, rff_key=rff_key), bw
                )
                rff_key = new_state.rff_key
            elif self.update_bw_method == "trick":
                raise ValueError(
                    "Update method 'trick' requires full distance matrix and is not compatible with RFF mode."
                )
            else:
                raise NotImplementedError(
                    f"Update method {self.update_bw_method} is not implemented."
                )
            return replace(state, bandwidth=bw, rff_key=rff_key)

        gaussian_exp = -0.5 * self.distance_func(coord)
        if self.update_bw_method == "trick":
            bw = self.get_bandwidth(gaussian_exp, bw)
        elif self.update_bw_method == "EM":
            bw_grad = grad(lambda bw_: jnp.sum(self.kernel_jit(coord, bw_)))(bw)
            bw = bw - 1e-3 * jnp.clip(bw_grad, -100, 100)
        elif self.update_bw_method == "rescale":
            kernel = jnp.exp(gaussian_exp * self._pair_inverse_squared(bw))
            bw, _ = self.rescale_kernel(kernel, bw)
        else:
            raise NotImplementedError(
                f"Update method {self.update_bw_method} is not implemented."
            )
        return replace(state, bandwidth=bw, gaussian_exp=gaussian_exp)

    def bw_reset_with_state(self, state: GaussianKernelState) -> GaussianKernelState:
        """Reset bandwidth in state without any kernel/gradient computation."""
        return replace(state, bandwidth=self.bw_init)
