from fp_transformer.kernel import Gaussian
from fp_transformer.utils import pairwise_distance
import jax
import numpy as np
from jax import numpy as jnp

data_path = "./tests/test-data/gaussian/"


def test_distance():
    flags = np.zeros(20)
    for i in range(20):
        coord = np.load(data_path + "coord_{}.npy".format(i), allow_pickle=True)
        ref_mat = np.load(data_path + "gauss_{}.npy".format(i), allow_pickle=True)
        # The package defaults to TF32 matmuls; compare at full precision.
        with jax.default_matmul_precision("highest"):
            mat = pairwise_distance(coord)
        mat = -0.5 * mat
        if np.allclose(mat, ref_mat, atol=1e-6):
            flags[i] = 1
    assert np.all(flags) == 1


def test_bw():
    for i in range(20):
        coord = np.load(data_path + "coord_{}.npy".format(i), allow_pickle=True)
        mat = pairwise_distance(coord)
        mat = -0.5 * mat
        bsize = mat.shape[0]
        gaussian = Gaussian(bsize)
        bw = gaussian.get_bandwidth(mat, jnp.ones(bsize) * 0.1)
        kernel_mat = np.exp(np.asarray(mat) / np.square(np.asarray(bw)[:, None]))
        np.fill_diagonal(kernel_mat, 0.0)
        mean_row_sum = np.mean(np.sum(kernel_mat, axis=1))
        kernel_min = np.asarray(gaussian.kernel_min).item()
        kernel_max = np.asarray(gaussian.kernel_max).item()
        assert kernel_min <= mean_row_sum <= kernel_max


def test_kernel():
    for i in range(20):
        coord = np.load(data_path + "coord_{}.npy".format(i), allow_pickle=True)
        bsize = coord.shape[0]
        gaussian = Gaussian(bsize)

        gaussian.update(coord)
        k = gaussian.kernel(coord, gaussian.bandwidth)
        distance_mat = pairwise_distance(coord)
        kernel_mat = np.exp(
            -0.5
            * np.asarray(distance_mat)
            / np.square(np.asarray(gaussian.bandwidth)[:, None])
        )
        np.fill_diagonal(kernel_mat, 0.0)
        expected = np.log(np.sum(kernel_mat, axis=1))[:, None, None]
        np.testing.assert_allclose(np.asarray(k), expected, rtol=1e-5, atol=1e-5)


"""def test_rff_kernel():
    flags = np.zeros(20)
    for i in range(20):
        coord = np.load(data_path + "coord_{}.npy".format(i), allow_pickle=True)
        bsize = coord.shape[0]
        gaussian_rff = Gaussian(bsize, rff=True)
        gaussian_rff.bandwidth = 0.1

        k_rff = gaussian_rff.kernel(coord, gaussian_rff.bandwidth)

        gaussian = Gaussian(bsize)
        gaussian.bandwidth = 0.1

        k = gaussian.kernel(coord, gaussian.bandwidth)

        print(np.max(np.abs(k_rff - k) / k))

        if np.allclose(k_rff, k, rtol=1e-1):
            flags[i] = 1
    assert np.all(flags) == 1"""
