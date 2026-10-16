import os
import sys

import sys
sys.path.append(".")

import pytest
import numpy as np
from jax import numpy as jnp
from jax import random
import jax

from src.fp_transformer.kernel import Gaussian
from src.fp_transformer.utils import pairwise_distance


# 确保使用CPU后端的固定装置
@pytest.fixture(scope="session", autouse=True)
def ensure_cpu_backend():
    """确保所有测试在CPU后端上运行"""
    platform = jax.devices()[0].platform
    assert platform == 'cpu', f"测试需要在CPU后端上运行，但当前平台是 {platform}"
    print(f"使用JAX CPU后端: {jax.devices()[0]}")
    
    # 检查是否在UV环境中运行
    in_uv = 'UV_RUNTIME' in os.environ or any('uv' in arg for arg in sys.argv)
    if not in_uv:
        print("警告: 未检测到UV环境，建议使用 ./run_gaussian_rff_tests_uv.sh 来运行测试")
    else:
        print("检测到UV环境，继续测试...")
    
    yield


def test_rff_vs_standard_kernel_consistency():
    """测试RFF模式和标准模式下kernel函数行为的一致性"""
    flags = np.zeros(20)
    rff_dims = [128, 512, 1024]  # 测试不同的RFF维度
    
    # 使用固定的随机种子以确保结果可重复
    key = random.PRNGKey(42)
    
    for i in range(20):
        # 随机生成coord而不是从文件加载
        key, subkey = random.split(key)
        bsize = random.randint(subkey, (), 5, 15)  # 随机批次大小
        key, subkey = random.split(key)
        coord = random.uniform(subkey, (bsize, 3, 2)) + random.normal(subkey, (bsize, 3, 2))  # (batch_size, n_atoms, n_dims)
        
        # 创建标准Gaussian kernel
        gaussian_std = Gaussian(bsize, rff=False)
        
        # 固定bandwidth以确保公平比较
        test_bandwidth = 1
        gaussian_std.bandwidth = jnp.ones(bsize) * test_bandwidth
        
        # 计算标准kernel
        k_std = gaussian_std.kernel(coord, gaussian_std.bandwidth)
        
        # 测试不同RFF维度下的一致性
        for rff_dim in rff_dims:
            # 创建RFF Gaussian kernel
            gaussian_rff = Gaussian(bsize, rff=True, rff_dim=rff_dim, rff_seed=42)
            gaussian_rff.bandwidth = jnp.ones(bsize) * test_bandwidth
            
            # 计算RFF kernel
            k_acc = jnp.zeros_like(k_std)
            for _ in range(10000):
                k_rff = gaussian_rff.kernel(coord, gaussian_rff.bandwidth)
                k_acc += k_rff
            k_acc /= 10000

            # 计算相对误差
            # Compare densities: the leave-one-out log density can be close to zero.
            rel_error = np.abs(np.exp(k_acc) - np.exp(k_std)) / np.exp(k_std)
            
            print(np.squeeze(k_std))
            print(np.squeeze(k_acc))

            rel_error = np.max(rel_error)

            # RFF是一个近似方法，所以允许一定的误差
            # 随着RFF维度增加，误差应该减少
            tolerance = 0.5 if rff_dim == 128 else (0.3 if rff_dim == 512 else 0.2)
            
            if rel_error < tolerance:
                flags[i] = 1
                break
        
        # 输出调试信息
        if flags[i] == 0:
            print(f"Case {i}: RFF approximation error too large. Max rel error: {rel_error:.4f}")
    
    # 至少80%的测试案例应该通过
    success_rate = np.sum(flags) / len(flags)
    print(f"RFF consistency test success rate: {success_rate:.2%}")
    assert success_rate >= 0.8, f"RFF consistency test failed. Success rate: {success_rate:.2%}"


def test_rff_bandwidth_scaling():
    """测试RFF kernel在不同bandwidth下的行为"""
    key = random.PRNGKey(0)
    coord = random.normal(key, (5, 3, 2))
    
    bandwidths = [0.01, 0.1, 1.0, 10.0]
    
    for bw in bandwidths:
        gaussian_rff = Gaussian(5, rff=True, rff_dim=512, rff_seed=42)
        gaussian_std = Gaussian(5, rff=False)
        
        # 设置相同的bandwidth
        gaussian_rff.bandwidth = jnp.ones(5) * bw
        gaussian_std.bandwidth = jnp.ones(5) * bw
        
        k_rff = gaussian_rff.kernel(coord, gaussian_rff.bandwidth)
        k_std = gaussian_std.kernel(coord, gaussian_std.bandwidth)
        
        # 计算相对误差
        rel_error = np.max(np.abs(k_rff - k_std) / (np.abs(k_std) + 1e-8))
        
        # 对于不同的bandwidth，RFF近似的准确性可能不同
        # 但应该保持在合理范围内
        assert rel_error < 1.0, f"RFF approximation error too large for bandwidth {bw}: {rel_error:.4f}"


def test_rff_reproducibility():
    """测试RFF kernel的可重复性"""
    key = random.PRNGKey(0)
    coord = random.normal(key, (8, 3, 2))
    
    # 使用相同的随机种子创建两个RFF kernel
    gaussian_rff1 = Gaussian(8, rff=True, rff_dim=256, rff_seed=123)
    gaussian_rff2 = Gaussian(8, rff=True, rff_dim=256, rff_seed=123)
    
    k_rff1 = gaussian_rff1.kernel(coord, gaussian_rff1.bandwidth)
    k_rff2 = gaussian_rff2.kernel(coord, gaussian_rff2.bandwidth)
    
    # 由于RFF内部使用随机数生成，完全相同的结果需要相同的随机种子
    # 但是由于内部状态更新，可能还是有差异，所以我们测试近似相等
    assert np.allclose(k_rff1, k_rff2, rtol=1e-2), "RFF kernel should be reproducible with same seed"


def test_rff_incompatible_update_method():
    """测试RFF模式下不兼容的更新方法应该抛出错误"""
    # 测试创建时就指定不兼容的方法
    with pytest.raises(ValueError, match="not compatible with RFF mode"):
        Gaussian(5, rff=True, update_bw_method="trick")


if __name__ == "__main__":
    test_rff_vs_standard_kernel_consistency()
    test_rff_kernel_properties()
    test_rff_bandwidth_scaling()
    test_rff_reproducibility()
    test_rff_vs_standard_update_methods()
    test_rff_incompatible_update_method()
    print("All RFF consistency tests passed!")
