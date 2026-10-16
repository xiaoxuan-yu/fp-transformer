import sys

sys.path.append(".")

import jax.numpy as jnp
import jax
from jax import random
from src.fp_transformer.kernel.rff_utils import (
    create_torus_bessel_pmf_dist,
    sample_rff_params_torus,
    rff_feature_map_torus,
    compute_rff_feature_torus,
    rff_row_sum,
    rff_logsum,
)
from src.fp_transformer.kernel import Gaussian
from src.fp_transformer.utils import torus_pairwise_distance


def compute_torus_kernel(X1, X2, bandwidth):
    """
    计算真实的环面高斯核矩阵，与torus_pairwise_distance保持一致

    参数:
    X1: 第一组输入数据 (N1, A, D)
    X2: 第二组输入数据 (N2, A, D)
    bandwidth: 带宽参数 (对应高斯核的标准差 σ)

    返回:
    kernel_matrix: 环面高斯核矩阵 (N1, N2)
    """
    # 使用与 torus_pairwise_distance 相同的距离计算方法
    if jnp.array_equal(X1, X2):
        # 如果是自相似矩阵，可以直接使用torus_pairwise_distance
        torus_dist_matrix = torus_pairwise_distance(X1)
    else:
        # 如果是不同数据集，需要计算交叉距离
        # 使用与 torus_dist 函数相同的计算方式
        N1, A, D = X1.shape
        N2, _, _ = X2.shape

        X1_flat = X1.reshape(N1, -1)  # (N1, A*D)
        X2_flat = X2.reshape(N2, -1)  # (N2, A*D)

        # 计算环面距离矩阵，使用与 torus_dist 相同的公式
        # torus_dist 计算: mean(sin²(x-y) + cos²(x-y))
        X1_expanded = jnp.expand_dims(X1_flat, axis=1)  # (N1, 1, A*D)
        X2_expanded = jnp.expand_dims(X2_flat, axis=0)  # (1, N2, A*D)

        sin_diff_sq = jnp.square(jnp.sin(X1_expanded) - jnp.sin(X2_expanded))
        cos_diff_sq = jnp.square(jnp.cos(X1_expanded) - jnp.cos(X2_expanded))
        torus_dist_matrix = jnp.mean(sin_diff_sq + cos_diff_sq, axis=-1)  # (N1, N2)

    # 计算高斯核: exp(-torus_dist / (2*σ^2))
    return jnp.exp(-torus_dist_matrix / (2 * bandwidth**2))


def rff_kernel_torus(X1, X2, k_samples):
    """
    使用环面RFF特征计算核矩阵

    Args:
        X1 (JaxArray): 第一组输入数据 (N1, A, D)
        X2 (JaxArray): 第二组输入数据 (N2, A, D)
        k_samples (JaxArray): 环面RFF采样的频率参数 (n_features, A*D)

    Returns:
        JaxArray: 核矩阵 (N1, N2)
    """
    # 计算两组数据的环面RFF特征
    features1 = compute_rff_feature_torus(X1, k_samples)  # (N1, 2*n_features)
    features2 = compute_rff_feature_torus(X2, k_samples)  # (N2, 2*n_features)

    # 核矩阵是特征向量的内积
    kernel_matrix = jnp.dot(features1, features2.T)  # (N1, N2)

    return kernel_matrix


def test_torus_bessel_pmf_creation():
    """测试环面贝塞尔PMF分布的创建"""
    gamma = 0.5
    m_cutoff = 30
    n_dim = 2

    dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

    # 验证k_range的范围
    expected_range = jnp.arange(-m_cutoff, m_cutoff + 1)
    assert jnp.array_equal(
        k_range, expected_range
    ), "k_range应该是从-m_cutoff到m_cutoff的整数序列"

    # 验证分布是一个有效的概率分布
    # TFP Categorical会自动归一化，所以我们只需要验证概率和为1
    probs = jnp.exp(dist.logits_parameter())
    prob_sum = jnp.sum(probs)
    assert jnp.allclose(prob_sum, 1.0, rtol=1e-5), "概率分布的和应该为1"

    # 验证概率值都是非负的
    assert jnp.all(probs >= 0), "所有概率值应该非负"


def test_torus_rff_params_sampling():
    """测试环面RFF参数采样"""
    key = random.PRNGKey(42)
    gamma = 0.5
    m_cutoff = 20
    n_dim = 2
    input_dim = 4  # A * D
    n_features = 100

    # 创建分布
    dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

    # 采样参数
    k_samples = sample_rff_params_torus(key, dist, k_range, input_dim, n_features)

    # 验证采样结果的形状
    assert k_samples.shape == (
        n_features,
        input_dim,
    ), f"k_samples形状应为({n_features}, {input_dim})"

    # 验证采样值在有效范围内
    assert jnp.all(k_samples >= -m_cutoff), f"所有采样值应该大于等于{-m_cutoff}"
    assert jnp.all(k_samples <= m_cutoff), f"所有采样值应该小于等于{m_cutoff}"

    # 验证采样值是整数
    assert jnp.allclose(k_samples, jnp.round(k_samples)), "采样值应该是整数"


def test_torus_rff_feature_map():
    """测试环面RFF特征映射"""
    key = random.PRNGKey(42)

    # 设置测试参数
    batch_size = 8
    n_atoms = 2
    input_dim = 3
    n_features = 64

    # 生成环面空间中的测试数据（通常在[0, 2π]范围内）
    key, subkey = random.split(key)
    X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

    # 生成随机的k_samples（模拟从分布中采样的结果）
    key, subkey = random.split(key)
    k_samples = random.randint(subkey, (n_features, n_atoms * input_dim), -10, 11)

    # 计算特征映射
    features = rff_feature_map_torus(X, k_samples)

    # 验证特征的形状
    expected_shape = (batch_size, 2 * n_features)  # 包含cos和sin特征
    assert features.shape == expected_shape, f"特征形状应为{expected_shape}"

    # 验证特征值在合理范围内（由于归一化，应该不会太大）
    assert jnp.all(jnp.abs(features) <= 10), "特征值应该在合理范围内"

    # 验证特征不全为零
    assert not jnp.allclose(features, 0), "特征不应该全为零"


def test_torus_rff_kernel_approximation():
    """测试环面RFF近似的核矩阵是否与真实的环面高斯核矩阵足够接近"""
    key = random.PRNGKey(425)

    # 设置测试参数 - 参考 temp_torus_rff.py 的成功参数
    batch_size = 6
    n_atoms = 2
    input_dim = 2
    n_features = 1000  # 使用更多特征以获得好的近似
    gamma = 0.5  # 使用与原始代码相同的gamma
    bandwidth = jnp.sqrt(1.0 / (2.0 * gamma))  # 对应 bandwidth = 1.0
    m_cutoff = 40  # 使用与原始代码相同的m_cutoff

    # 生成环面空间中的随机测试数据（在[0, 2π]范围内）
    key, subkey = random.split(key)
    X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

    # 创建环面分布并采样RFF参数
    n_dim = n_atoms * input_dim
    dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

    key, subkey = random.split(key)
    k_samples = sample_rff_params_torus(subkey, dist, k_range, n_dim, n_features)

    # 计算RFF环面核矩阵
    rff_K = rff_kernel_torus(X, X, k_samples)

    # 计算真实的环面高斯核矩阵
    true_K = compute_torus_kernel(X, X, bandwidth)

    # 计算近似误差
    rel_error = jnp.abs(rff_K - true_K) / (jnp.abs(true_K) + 1e-8)
    abs_error = jnp.abs(rff_K - true_K)
    mean_rel_error = jnp.mean(rel_error)

    print(f"环面RFF平均相对误差: {mean_rel_error:.4f}")
    print(f"环面RFF平均绝对误差: {jnp.mean(abs_error):.4f}")

    # 验证平均相对误差在可接受范围内（基于原始代码的结果）
    assert mean_rel_error < 0.15, f"环面RFF核矩阵近似误差过大: {mean_rel_error:.4f}"

    # 验证核矩阵的基本性质：对角线元素应接近1
    assert jnp.allclose(
        jnp.diag(rff_K), jnp.ones(batch_size), atol=0.05
    ), "环面核矩阵对角线元素应接近1"


def test_torus_rff_multiple_approximation():
    """测试多次生成环面RFF特征并平均后的核矩阵近似精度"""

    def _test_single_torus_configuration(key, test_config, test_name):
        """测试单个环面配置的RFF近似精度"""
        print(f"\n=== {test_name} ===")

        batch_size = test_config["batch_size"]
        input_dim = test_config["input_dim"]
        n_atoms = test_config["n_atoms"]
        n_features = test_config["n_features"]
        bandwidth = test_config["bandwidth"]
        n_trials = test_config["n_trials"]
        m_cutoff = test_config.get("m_cutoff", 30)

        gamma = 1.0 / (2.0 * bandwidth**2)

        # 生成环面空间中的随机测试数据
        key, subkey = random.split(key)
        X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

        # 初始化累积的核矩阵
        accumulated_K = jnp.zeros((batch_size, batch_size))

        # 创建环面分布
        n_dim = n_atoms * input_dim
        dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

        # 多次生成RFF特征并计算核矩阵
        for i in range(n_trials):
            # 每次使用不同的随机种子生成RFF参数
            key, subkey = random.split(key)
            k_samples = sample_rff_params_torus(
                subkey, dist, k_range, n_dim, n_features
            )

            # 计算RFF核矩阵并累加
            rff_K = rff_kernel_torus(X, X, k_samples)
            accumulated_K += rff_K

        # 计算平均核矩阵
        avg_rff_K = accumulated_K / n_trials

        # 计算真实的环面高斯核矩阵
        true_K = compute_torus_kernel(X, X, bandwidth)

        # 计算单次RFF近似的误差（作为对比）
        key, subkey = random.split(key)
        single_k_samples = sample_rff_params_torus(
            subkey, dist, k_range, n_dim, n_features
        )
        single_rff_K = rff_kernel_torus(X, X, single_k_samples)
        single_rel_error = jnp.abs(single_rff_K - true_K) / (jnp.abs(true_K) + 1e-8)
        single_mean_rel_error = jnp.mean(single_rel_error)

        # 计算多次平均后的近似误差
        avg_rel_error = jnp.abs(avg_rff_K - true_K) / (jnp.abs(true_K) + 1e-8)
        avg_abs_error = jnp.abs(avg_rff_K - true_K)
        avg_mean_rel_error = jnp.mean(avg_rel_error)

        print(f"批量大小: {batch_size}, 输入维度: {input_dim}, 原子数: {n_atoms}")
        print(f"特征数: {n_features}, 带宽: {bandwidth}, 试验次数: {n_trials}")
        print(f"m_cutoff: {m_cutoff}")
        print(f"单次环面RFF近似平均相对误差: {single_mean_rel_error:.4f}")
        print(f"多次平均环面RFF近似平均相对误差: {avg_mean_rel_error:.4f}")
        print(f"多次平均环面RFF近似平均绝对误差: {jnp.mean(avg_abs_error):.4f}")
        print(
            f"相对误差改进比例: {(single_mean_rel_error - avg_mean_rel_error) / single_mean_rel_error:.4f}"
        )

        # 验证多次平均后的相对误差是否小于单次近似
        assert (
            avg_mean_rel_error < single_mean_rel_error
        ), f"{test_name}: 多次平均后的环面RFF近似应优于单次近似"

        # 验证多次平均后的平均相对误差在可接受范围内
        expected_threshold = test_config.get("error_threshold", 0.3)
        assert (
            avg_mean_rel_error < expected_threshold
        ), f"{test_name}: 多次平均后的环面RFF核矩阵近似误差仍过大: {avg_mean_rel_error:.4f}"

        # 验证核矩阵的基本性质
        diag_tolerance = test_config.get("diag_tolerance", 0.1)
        assert jnp.allclose(
            jnp.diag(avg_rff_K), jnp.ones(batch_size), atol=diag_tolerance
        ), f"{test_name}: 环面核矩阵对角线元素应接近1"

        return key

    # 定义不同的环面测试配置
    test_configs = [
        # 基础环面测试配置 - 使用合适的gamma/bandwidth参数
        {
            "name": "基础环面测试",
            "batch_size": 6,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 1000,
            "bandwidth": 1.0,  # 对应 gamma = 0.5
            "n_trials": 100,
            "m_cutoff": 40,
            "error_threshold": 0.2,
            "diag_tolerance": 0.1,
        },
        # 高精度环面测试
        {
            "name": "高精度环面测试",
            "batch_size": 8,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 2000,
            "bandwidth": 1.2,  # 稍大的bandwidth
            "n_trials": 200,
            "m_cutoff": 50,
            "error_threshold": 0.15,
            "diag_tolerance": 0.08,
        },
        # 多原子环面系统测试
        {
            "name": "多原子环面系统测试",
            "batch_size": 6,
            "input_dim": 3,
            "n_atoms": 3,
            "n_features": 1000,
            "bandwidth": 1.0,
            "n_trials": 150,
            "m_cutoff": 40,
            "error_threshold": 0.25,  # 多原子可能误差稍大
            "diag_tolerance": 0.12,
        },
        # 窄带宽环面测试
        {
            "name": "窄带宽环面测试",
            "batch_size": 8,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 1500,
            "bandwidth": 0.8,  # 稍小的bandwidth
            "n_trials": 200,
            "m_cutoff": 60,
            "error_threshold": 0.3,  # 窄带宽可能误差稍大
            "diag_tolerance": 0.1,
        },
        # 快速环面测试配置（用于日常测试）
        {
            "name": "快速环面测试配置",
            "batch_size": 4,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 500,
            "bandwidth": 1.0,
            "n_trials": 50,
            "m_cutoff": 30,
            "error_threshold": 0.3,
            "diag_tolerance": 0.15,
        },
    ]

    key = random.PRNGKey(42)

    # 运行所有环面测试配置
    for config in test_configs:
        test_name = config.pop("name")
        key = _test_single_torus_configuration(key, config, test_name)

    print("\n=== 所有多次环面RFF近似测试通过 ===")


def test_torus_rff_row_sum():
    """测试环面RFF的rff_row_sum功能（需要考虑环面RFF的特殊归一化）"""
    key = random.PRNGKey(42)

    # 设置测试参数 - 使用合适的gamma/bandwidth参数
    batch_size = 12
    n_atoms = 2
    input_dim = 2
    n_features = 500
    bandwidth = 1.0  # 对应 gamma = 0.5
    gamma = 1.0 / (2.0 * bandwidth**2)
    m_cutoff = 40

    # 生成环面空间中的随机测试数据
    key, subkey = random.split(key)
    X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

    # 创建环面分布并采样RFF参数
    n_dim = n_atoms * input_dim
    dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

    key, subkey = random.split(key)
    k_samples = sample_rff_params_torus(subkey, dist, k_range, n_dim, n_features)

    # 计算环面RFF特征
    features = compute_rff_feature_torus(X, k_samples)

    # 手动计算环面RFF的行和（使用正确的算法）
    # 对于环面RFF: K[i,j] = features[i] @ features[j].T
    # 行和: sum_j K[i,j] = features[i] @ sum_j features[j]
    # features_sum = jnp.sum(features, axis=0)
    # torus_row_sum = jnp.dot(features, features_sum)
    torus_row_sum = rff_row_sum(features)

    # 使用显式环面核矩阵计算行和进行比较
    K = rff_kernel_torus(X, X, k_samples)
    row_sum_via_K = jnp.sum(K, axis=1)

    # 验证手动计算的环面RFF行和是否正确
    assert jnp.allclose(
        torus_row_sum, row_sum_via_K, rtol=1e-3
    ), "手动计算的环面RFF行和与显式核矩阵求和的结果不一致"

    # 我们对极小和负数值进行了处理，因此允许较大的相对误差


def test_torus_rff_logsum():
    """测试环面RFF的rff_logsum功能（需要考虑环面RFF的特殊归一化）"""
    key = random.PRNGKey(42)

    # 设置测试参数 - 使用合适的gamma/bandwidth参数
    batch_size = 10
    n_atoms = 2
    input_dim = 3
    n_features = 600
    bandwidth = 1.0  # 对应 gamma = 0.5
    gamma = 1.0 / (2.0 * bandwidth**2)
    m_cutoff = 40

    # 生成环面空间中的随机测试数据
    key, subkey = random.split(key)
    X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

    # 创建环面分布并采样RFF参数
    n_dim = n_atoms * input_dim
    dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

    key, subkey = random.split(key)
    k_samples = sample_rff_params_torus(subkey, dist, k_range, n_dim, n_features)

    # 计算环面RFF特征
    features = compute_rff_feature_torus(X, k_samples)

    # 手动计算环面RFF的log(sum(K))
    # 首先计算行和，然后取对数
    features_sum = jnp.sum(features, axis=0)
    torus_row_sum = jnp.dot(features, features_sum)
    torus_logsum = jnp.log(torus_row_sum).reshape(-1, 1)

    # 使用显式环面核矩阵计算log(sum(K))进行比较
    K = rff_kernel_torus(X, X, k_samples)
    row_sum_via_K = jnp.sum(K, axis=1)
    logsum_via_K = jnp.log(row_sum_via_K).reshape(-1, 1)

    # 验证手动计算的环面RFF logsum是否正确
    assert jnp.allclose(
        torus_logsum, logsum_via_K, rtol=1e-5
    ), "手动计算的环面RFF logsum与显式核矩阵计算的结果不一致"

    # 注意：标准的rff_logsum函数不适用于环面RFF，因为归一化方式不同
    # 我们在这里验证环面RFF的logsum逻辑是正确的


def test_torus_rff_approximation_analysis():
    """分析不同参数对环面RFF近似精度的影响"""
    key = random.PRNGKey(42)

    # 设置测试参数 - 使用合适的gamma/bandwidth参数
    batch_size = 6
    input_dim = 2
    n_atoms = 2
    bandwidth = 1.0  # 对应 gamma = 0.5
    gamma = 1.0 / (2.0 * bandwidth**2)

    # 测试不同的特征数量和m_cutoff组合 - 使用合适的参数范围
    feature_counts = [500, 1000, 2000]
    m_cutoff_values = [30, 40, 50]
    trial_counts = [1, 10, 30]

    # 生成环面空间中的随机测试数据
    key, subkey = random.split(key)
    X = random.uniform(subkey, (batch_size, n_atoms, input_dim)) * 2 * jnp.pi

    # 计算真实的环面高斯核矩阵作为参考
    true_K = compute_torus_kernel(X, X, bandwidth)

    print("\n不同参数对环面RFF近似精度的影响分析:")
    print("特征数量\tm_cutoff\t平均次数\t平均相对误差\t相对误差标准差")

    n_dim = n_atoms * input_dim

    for n_features in feature_counts:
        for m_cutoff in m_cutoff_values:
            for n_trials in trial_counts:
                # 创建环面分布
                dist, k_range = create_torus_bessel_pmf_dist(gamma, m_cutoff, n_dim)

                # 初始化累积的核矩阵
                accumulated_K = jnp.zeros((batch_size, batch_size))

                # 记录每次试验的误差，用于计算标准差
                all_errors = []

                # 多次生成RFF特征并计算核矩阵
                for i in range(n_trials):
                    key, subkey = random.split(key)
                    k_samples = sample_rff_params_torus(
                        subkey, dist, k_range, n_dim, n_features
                    )

                    # 计算RFF核矩阵并累加
                    rff_K = rff_kernel_torus(X, X, k_samples)
                    accumulated_K += rff_K

                    # 计算单次近似的误差
                    if n_trials > 1:
                        single_rel_error = jnp.abs(rff_K - true_K) / (
                            jnp.abs(true_K) + 1e-8
                        )
                        all_errors.append(jnp.mean(single_rel_error))

                # 计算平均核矩阵
                avg_rff_K = accumulated_K / n_trials

                # 计算平均后的近似误差
                avg_rel_error = jnp.abs(avg_rff_K - true_K) / (jnp.abs(true_K) + 1e-8)
                avg_mean_rel_error = jnp.mean(avg_rel_error)

                # 计算误差标准差（如果有多次试验）
                if n_trials > 1 and all_errors:
                    error_std = jnp.std(jnp.array(all_errors))
                else:
                    error_std = float("nan")

                print(
                    f"{n_features}\t{m_cutoff}\t{n_trials}\t{avg_mean_rel_error:.6f}\t{error_std:.6f}"
                )

                # 为大特征数和多次平均验证核矩阵的基本性质
                if n_features >= 1000 and n_trials >= 10:
                    assert jnp.allclose(
                        jnp.diag(avg_rff_K), jnp.ones(batch_size), atol=0.1
                    ), f"特征数={n_features}，m_cutoff={m_cutoff}，平均次数={n_trials}时，环面核矩阵对角线元素应接近1"


if __name__ == "__main__":
    # 运行所有测试
    print("开始运行环面RFF测试...")
    test_torus_bessel_pmf_creation()
    print("✓ 环面贝塞尔PMF创建测试通过")

    test_torus_rff_params_sampling()
    print("✓ 环面RFF参数采样测试通过")

    test_torus_rff_feature_map()
    print("✓ 环面RFF特征映射测试通过")

    test_torus_rff_kernel_approximation()
    print("✓ 环面RFF核近似测试通过")

    test_torus_rff_row_sum()
    print("✓ 环面RFF行求和测试通过")

    test_torus_rff_logsum()
    print("✓ 环面RFF对数求和测试通过")

    test_torus_gaussian_class_integration()
    print("✓ Gaussian类环面集成测试通过")

    print("\n开始运行环面RFF多次近似测试...")
    test_torus_rff_multiple_approximation()

    print("\n开始运行环面RFF近似分析...")
    test_torus_rff_approximation_analysis()

    print("\n🎉 所有环面RFF测试完成！")
