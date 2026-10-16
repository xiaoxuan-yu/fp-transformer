import numpy as np

from fp_transformer.potential import TiwaryBerne

CENTERS = np.array([(-0.5, 0.5), (0.8, 1.2), (0.5, -0.3)])
DEPTHS = np.array([16.0, 18.0, 16.0])


def tb_reference(x, y):
    """Tiwary and Berne, J. Chem. Phys. 147, 152701 (2017), Eq. (10)."""
    xy = np.stack([x, y], axis=-1)[..., None, :]
    wells = np.sum(DEPTHS * np.exp(-2.0 * np.sum((xy - CENTERS) ** 2, axis=-1)), axis=-1)
    return 20.0 - wells + 0.5 * (x**6 + y**6)


def test_tb():
    tb = TiwaryBerne()
    xy = np.random.default_rng(0).uniform(-1.5, 1.5, size=(64, 2))
    u = np.asarray(tb(xy[:, None, :]))[:, 0]
    assert np.allclose(u, tb_reference(xy[:, 0], xy[:, 1]), rtol=1e-5, atol=1e-4)
    for vrange in [(-1, 1.5), (-2, 2)]:
        x, y, z = tb.get_2dcontour(vrange, vrange, 100)
        z_ref = tb_reference(np.asarray(x), np.asarray(y))
        z_ref -= z_ref.min()  # get_2dcontour reports energies relative to the grid minimum
        assert np.allclose(np.asarray(z), z_ref, rtol=1e-5, atol=1e-4)
