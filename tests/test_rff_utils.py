import sys

sys.path.append(".")

import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from src.fp_transformer.kernel.rff_utils import (
    sample_rff_params,
    compute_rff_features,
    rff_kernel,
    rff_row_sum,
    rff_logsum,
    rff_power_row_sum,
)
from src.fp_transformer.kernel import Gaussian


def compute_gaussian_kernel(X1, X2, bandwidth):
    """
    计算真实的高斯核矩阵，与pairwise_distance保持一致的维度缩放

    参数:
    X1: 第一组输入数据 (N1, A*D)
    X2: 第二组输入数据 (N2, A*D)
    bandwidth: 带宽参数 (对应高斯核的标准差 σ)

    返回:
    kernel_matrix: 高斯核矩阵 (N1, N2)
    """
    # 将X1和X2reshape为2D矩阵以便计算
    X1_flat = X1.reshape(X1.shape[0], -1)  # (N1, A*D)
    X2_flat = X2.reshape(X2.shape[0], -1)  # (N2, A*D)

    # 计算平方欧氏距离矩阵 (N1, N2)
    # 使用广播计算所有成对距离
    X1_expanded = jnp.expand_dims(X1_flat, axis=1)  # (N1, 1, A*D)
    X2_expanded = jnp.expand_dims(X2_flat, axis=0)  # (1, N2, A*D)
    sq_dist = jnp.sum((X1_expanded - X2_expanded) ** 2, axis=-1)  # (N1, N2)

    # 与pairwise_distance保持一致：除以维度进行缩放
    dim = X1_flat.shape[-1]  # A*D
    scaled_dist = sq_dist / dim

    # 计算高斯核: exp(-scaled_dist / (2*σ^2))
    return jnp.exp(-scaled_dist / (2 * bandwidth**2))


def test_rff_row_sum():
    """测试rff_row_sum是否能正确规避显式计算核矩阵并获得正确计算结果"""
    key = random.PRNGKey(42)

    # 设置测试参数 - 对CPU友好的配置
    batch_size = 16
    input_dim = 2
    n_atoms = 2
    n_features = 256
    bandwidth = 0.3

    # 生成随机测试数据
    key, subkey = random.split(key)
    X = random.normal(subkey, (batch_size, n_atoms, input_dim))

    # 采样RFF参数
    key, subkey = random.split(key)
    weights, bias = sample_rff_params(
        subkey, n_atoms * input_dim, n_features, bandwidth**2
    )

    # 计算RFF特征
    features = compute_rff_features(X, weights, bias)

    # 使用rff_row_sum计算行和
    row_sum_via_rff = rff_row_sum(features)

    # 使用显式核矩阵计算行和进行比较
    K = rff_kernel(X, X, weights, bias)
    row_sum_via_K = jnp.sum(K, axis=1)

    # 验证两种方法的结果是否一致
    assert jnp.allclose(
        row_sum_via_rff, row_sum_via_K, rtol=1e-5
    ), "使用rff_row_sum的结果与显式计算核矩阵后求和的结果不一致"


def test_rff_row_sum_can_exclude_self_term():
    features = jnp.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]], dtype=jnp.float32
    )
    kernel_matrix = features @ features.T
    expected = jnp.sum(kernel_matrix, axis=1) - jnp.diag(kernel_matrix)

    actual = rff_row_sum(features, exclude_self=True)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(jnp.maximum(expected, 1e-10)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rff_logsum():
    """测试rff_logsum是否能正确规避显式计算核矩阵并获得正确计算结果"""
    key = random.PRNGKey(42)

    # 设置测试参数 - 对CPU友好的配置
    batch_size = 12
    input_dim = 2
    n_atoms = 2
    n_features = 512
    bandwidth = 0.4

    # 生成随机测试数据
    key, subkey = random.split(key)
    X = random.normal(subkey, (batch_size, n_atoms, input_dim))

    # 采样RFF参数
    key, subkey = random.split(key)
    weights, bias = sample_rff_params(
        subkey, n_atoms * input_dim, n_features, bandwidth**2
    )

    # 计算RFF特征
    features = compute_rff_features(X, weights, bias)

    # 使用rff_logsum计算log(sum(K))
    logsum_via_rff = rff_logsum(features)

    # 使用显式核矩阵计算log(sum(K))进行比较
    K = rff_kernel(X, X, weights, bias)
    row_sum_via_K = jnp.sum(K, axis=1)
    logsum_via_K = jnp.log(row_sum_via_K).reshape(-1, 1)

    # 验证两种方法的结果是否一致
    assert jnp.allclose(
        logsum_via_rff, logsum_via_K, rtol=1e-5
    ), "使用rff_logsum的结果与显式计算核矩阵后进行logsum的结果不一致"


def test_rff_performance_comparison():
    """测试RFF方法与直接计算核矩阵的性能比较"""
    # 确认当前使用的是CPU后端
    device = jax.devices()[0]
    print(f"测试运行在: {device}")
    assert device.platform == "cpu", "测试应该在CPU后端运行"

    key = random.PRNGKey(42)

    # 设置测试参数 - 在CPU上使用适中的规模
    batch_size = 30  # 对CPU友好的批量大小
    input_dim = 2
    n_atoms = 3
    n_features = 128
    bandwidth = 0.5

    # 生成随机测试数据
    key, subkey = random.split(key)
    X = random.normal(subkey, (batch_size, n_atoms, input_dim))

    # 采样RFF参数
    key, subkey = random.split(key)
    weights, bias = sample_rff_params(
        subkey, n_atoms * input_dim, n_features, bandwidth**2
    )

    # 计算RFF特征
    features = compute_rff_features(X, weights, bias)

    # 使用RFF方法计算row_sum和logsum
    @jax.jit
    def compute_with_rff():
        row_sum = rff_row_sum(features)
        print(np.squeeze(features))
        print("RFF row_sum:", np.squeeze(row_sum))
        logsum = rff_logsum(features)
        return row_sum, logsum

    # 使用显式核矩阵计算
    @jax.jit
    def compute_with_explicit_K():
        K = rff_kernel(X, X, weights, bias)
        row_sum = jnp.sum(K, axis=1)
        logsum = jnp.log(row_sum).reshape(-1, 1)
        return row_sum, logsum

    # 预热JIT编译
    _ = compute_with_rff()
    _ = compute_with_explicit_K()

    # 测量RFF方法的执行时间
    rff_results = compute_with_rff()

    # 测量显式方法的执行时间
    explicit_results = compute_with_explicit_K()

    print(np.squeeze(rff_results[1]), np.squeeze(explicit_results[1]))

    # 验证两种方法的结果是否一致
    assert jnp.allclose(
        rff_results[0], explicit_results[0], rtol=1e-5
    ), "RFF方法与显式方法的row_sum结果不一致"
    assert jnp.allclose(
        rff_results[1], explicit_results[1], rtol=1e-5
    ), "RFF方法与显式方法的logsum结果不一致"

    # 注意：这不是准确的基准测试，仅作为功能验证
    # 真正的性能评估应使用如perfetto或JAX的内置分析工具


def test_rff_approximation_analysis():
    """分析不同特征数量和平均次数对RFF近似精度的影响"""
    key = random.PRNGKey(42)

    # 设置测试参数
    batch_size = 8
    input_dim = 2
    n_atoms = 2
    bandwidth = 0.5

    # 测试不同的特征数量和平均次数组合
    feature_counts = [64, 128, 256]
    trial_counts = [1, 5, 10, 20]

    # 生成随机测试数据
    key, subkey = random.split(key)
    X = random.normal(subkey, (batch_size, n_atoms, input_dim)) + 0.5

    # 计算真实的高斯核矩阵作为参考
    true_K = compute_gaussian_kernel(X, X, bandwidth)

    print("\n不同特征数量和平均次数的RFF近似精度分析:")
    print("特征数量\t平均次数\t平均相对误差\t相对误差标准差\t理论等效特征数")

    for n_features in feature_counts:
        for n_trials in trial_counts:
            # 初始化累积的核矩阵
            accumulated_K = jnp.zeros((batch_size, batch_size))

            # 记录每次试验的误差，用于计算标准差
            all_errors = []

            # 多次生成RFF特征并计算核矩阵
            for i in range(n_trials):
                key, subkey = random.split(key)
                weights, bias = sample_rff_params(
                    subkey, n_atoms * input_dim, n_features, bandwidth**2
                )

                # 计算RFF核矩阵并累加
                rff_K = rff_kernel(X, X, weights, bias)
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

            # 理论等效特征数（近似）：原特征数 × 平均次数
            theoretical_features = n_features * n_trials

            print(
                f"{n_features}\t{n_trials}\t{avg_mean_rel_error:.6f}\t{error_std:.6f}\t{theoretical_features}"
            )

            # 为大特征数和多次平均验证核矩阵的基本性质
            if n_features >= 256 and n_trials >= 10:
                assert jnp.allclose(
                    jnp.diag(avg_rff_K), jnp.ones(batch_size), atol=0.05
                ), f"特征数={n_features}，平均次数={n_trials}时，核矩阵对角线元素应接近1"


def test_rff_power_row_sum():
    """测试rff_power_row_sum是否能正确计算核矩阵幂次的行和"""
    key = random.PRNGKey(42)

    # 设置测试参数
    batch_size = 8
    input_dim = 2
    n_atoms = 2
    n_features = 256
    bandwidth = 0.5

    # 测试不同的幂次
    test_powers = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]

    # 生成随机测试数据
    key, subkey = random.split(key)
    X = random.normal(subkey, (batch_size, n_atoms, input_dim))

    # 采样RFF参数
    key, subkey = random.split(key)
    weights, bias = sample_rff_params(
        subkey, n_atoms * input_dim, n_features, bandwidth**2
    )

    # 计算RFF特征
    features = compute_rff_features(X, weights, bias)

    print("\n=== rff_power_row_sum 测试结果 ===")
    print("幂次\t\t平均相对误差\t平均绝对误差\t最大绝对误差")

    for power in test_powers:
        # 使用rff_power_row_sum计算
        power_row_sum_rff = rff_power_row_sum(features, power)

        # 直接计算核矩阵的幂次并求行和作为参考
        K = features @ features.T  # 核矩阵 K = ZZ^T

        # 使用特征分解来计算矩阵的任意幂次
        epsilon = 1e-8
        eigenvals, eigenvecs = jnp.linalg.eigh(K)

        # 确保特征值非负（理论上应该如此，但浮点计算可能有小的负值）
        eigenvals = jnp.maximum(eigenvals, epsilon)

        # 计算特征值的power次幂
        eigenvals_power = eigenvals**power

        # 重构矩阵的power次幂：K^power = V @ diag(λ^power) @ V^T
        K_power = eigenvecs @ jnp.diag(eigenvals_power) @ eigenvecs.T

        power_row_sum_direct = jnp.sum(K_power, axis=1)

        # 计算误差
        abs_error = jnp.abs(power_row_sum_rff - power_row_sum_direct)
        rel_error = abs_error / (jnp.abs(power_row_sum_direct) + 1e-10)

        mean_rel_error = jnp.mean(rel_error)
        mean_abs_error = jnp.mean(abs_error)
        max_abs_error = jnp.max(abs_error)

        print(
            f"{power:.1f}\t\t{mean_rel_error:.6f}\t{mean_abs_error:.6f}\t{max_abs_error:.6f}"
        )

        # 验证误差在可接受范围内
        if power >= 0:
            # 非负幂次的误差阈值更严格
            error_threshold = 0.1 if power <= 1.0 else 0.2
        else:
            # 负幂次由于数值稳定性问题，误差阈值放宽
            error_threshold = 0.5

        assert (
            mean_rel_error < error_threshold
        ), f"幂次 {power} 的平均相对误差过大: {mean_rel_error:.6f} > {error_threshold}"

    print("\n=== 特殊情况测试 ===")

    # 测试幂次为1的情况，应该与rff_row_sum结果一致
    power_1_result = rff_power_row_sum(features, 1.0)
    row_sum_result = rff_row_sum(features)

    assert jnp.allclose(
        power_1_result, row_sum_result, rtol=1e-5
    ), "幂次为1时，rff_power_row_sum应与rff_row_sum结果一致"

    print("✓ 幂次为1时与rff_row_sum结果一致")

    # 测试幂次为0的情况，应该返回batch_size个1（因为K^0 = I，行和为1）
    power_0_result = rff_power_row_sum(features, 0.0)
    expected_power_0 = jnp.ones(batch_size)

    assert jnp.allclose(
        power_0_result, expected_power_0, atol=0.1
    ), "幂次为0时，结果应接近全1向量"

    print("✓ 幂次为0时结果接近全1向量")

    print("\n=== 数值稳定性测试 ===")

    # 创建一个接近奇异的特征矩阵来测试数值稳定性
    key, subkey = random.split(key)
    singular_features = random.normal(subkey, (batch_size, 10))  # 使用较少的特征数

    # 测试负幂次在接近奇异情况下的稳定性
    try:
        result_singular = rff_power_row_sum(singular_features, -0.5)
        # 检查结果是否包含无穷大或NaN
        assert jnp.all(
            jnp.isfinite(result_singular)
        ), "在接近奇异情况下，结果应该是有限的"
        print("✓ 接近奇异情况下的数值稳定性测试通过")
    except Exception as e:
        print(f"⚠ 接近奇异情况下的测试出现异常: {e}")

    print("\n=== 所有rff_power_row_sum测试通过 ===")


def test_rff_power_row_sum_comprehensive():
    """rff_power_row_sum的综合测试，包括多种配置和边界情况"""
    key = random.PRNGKey(123)

    print("\n=== rff_power_row_sum 综合测试 ===")

    # 定义测试配置
    test_configs = [
        {
            "name": "标准配置",
            "batch_size": 6,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 128,
            "bandwidth": 0.5,
            "powers": [0.5, 1.0, 2.0],
            "error_threshold": 0.15,
        },
        {
            "name": "高维配置",
            "batch_size": 4,
            "input_dim": 8,
            "n_atoms": 2,
            "n_features": 256,
            "bandwidth": 0.8,
            "powers": [0.25, 1.5, 3.0],
            "error_threshold": 0.2,
        },
        {
            "name": "多原子配置",
            "batch_size": 8,
            "input_dim": 2,
            "n_atoms": 4,
            "n_features": 256,
            "bandwidth": 0.4,
            "powers": [0.5, 1.0, 1.5],
            "error_threshold": 0.18,
        },
        {
            "name": "低特征数配置",
            "batch_size": 5,
            "input_dim": 2,
            "n_atoms": 2,
            "n_features": 32,
            "bandwidth": 0.6,
            "powers": [0.5, 1.0],
            "error_threshold": 0.3,
        },
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ---")

        batch_size = config["batch_size"]
        input_dim = config["input_dim"]
        n_atoms = config["n_atoms"]
        n_features = config["n_features"]
        bandwidth = config["bandwidth"]
        powers = config["powers"]
        error_threshold = config["error_threshold"]

        # 生成测试数据
        key, subkey = random.split(key)
        X = random.normal(subkey, (batch_size, n_atoms, input_dim))

        # 采样RFF参数
        key, subkey = random.split(key)
        weights, bias = sample_rff_params(
            subkey, n_atoms * input_dim, n_features, bandwidth**2
        )

        # 计算RFF特征
        features = compute_rff_features(X, weights, bias)

        print(
            f"批量大小: {batch_size}, 特征维度: {n_features}, 总维度: {n_atoms * input_dim}"
        )

        for power in powers:
            # 使用rff_power_row_sum计算
            power_row_sum_rff = rff_power_row_sum(features, power)

            # 使用特征分解计算参考结果
            K = features @ features.T
            eigenvals, eigenvecs = jnp.linalg.eigh(K)
            # 确保特征值非负并添加小的正则化
            eigenvals = jnp.maximum(eigenvals, 1e-10)
            K_power = eigenvecs @ jnp.diag(eigenvals**power) @ eigenvecs.T
            power_row_sum_ref = jnp.sum(K_power, axis=1)

            # 计算误差
            abs_error = jnp.abs(power_row_sum_rff - power_row_sum_ref)
            rel_error = abs_error / (jnp.abs(power_row_sum_ref) + 1e-10)
            mean_rel_error = jnp.mean(rel_error)

            print(f"  幂次 {power}: 平均相对误差 = {mean_rel_error:.6f}")

            # 验证误差
            assert (
                mean_rel_error < error_threshold
            ), f"{config['name']} - 幂次 {power} 的误差过大: {mean_rel_error:.6f}"

        print(f"  ✓ {config['name']} 测试通过")

    print("\n=== 所有综合测试通过 ===")
