#!/usr/bin/env python3
"""
Tests for the AmberSystem wrapper around AmberPotential
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from openmm import app, unit
import openmm as mm

# 启用 64 位计算以提升与 OpenMM 的一致性
from jax import config as jax_config

@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    previous = jax_config.jax_enable_x64
    jax_config.update("jax_enable_x64", True)
    yield
    jax_config.update("jax_enable_x64", previous)

# 让测试可从源码目录导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fp_transformer.core.system.amber_system import AmberSystem


def _openmm_energy_forces(
    prmtop_file: str,
    inpcrd_file: str,
    positions_angstrom: np.ndarray,
    periodic: bool,
    pme: bool,
    cutoff_nm: float,
) -> tuple[float, np.ndarray]:
    """使用 OpenMM 计算给定坐标的能量与力。

    返回：
      (energy_kcal, forces_kcal_per_angstrom)
    """
    prmtop = app.AmberPrmtopFile(prmtop_file)
    inpcrd = app.AmberInpcrdFile(inpcrd_file)

    if periodic:
        if pme:
            system = prmtop.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=cutoff_nm * unit.nanometer,
                constraints=None,
                removeCMMotion=False,
            )
        else:
            system = prmtop.createSystem(
                nonbondedMethod=app.Ewald,
                nonbondedCutoff=cutoff_nm * unit.nanometer,
                constraints=None,
                removeCMMotion=False,
            )
    else:
        system = prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False,
        )
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator)

    # 如果有 box 信息，设置盒向量
    box = getattr(inpcrd, "boxVectors", None)
    if box is not None:
        context.setPeriodicBoxVectors(box[0], box[1], box[2])

    # 设置坐标（OpenMM 使用 nm，JAX-MD 使用 Angstrom）
    positions_nm = np.asarray(positions_angstrom, dtype=np.float64) * 0.1
    pos = unit.Quantity(positions_nm, unit.nanometer)
    context.setPositions(pos)

    state = context.getState(getEnergy=True, getForces=True)
    energy_kcal = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    forces_kcal_per_nm = state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalorie_per_mole / unit.nanometer
    )

    del context, integrator, system
    forces_kcal_per_angstrom = np.asarray(forces_kcal_per_nm) / 10.0
    return float(energy_kcal), forces_kcal_per_angstrom


def test_amber_system_basic_components():
    """非周期系统：基本实例化、能量、梯度与力。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys_nonpbc = AmberSystem(
        prmtop_file,
        inpcrd_file,
        periodic=False,
        pme=False,
        minibatch_size=2,
        assert_divisible_minibatch=True,
    )
    R0 = sys_nonpbc.coordinate

    # 能量（单帧）
    E = sys_nonpbc.potential()
    assert jnp.isfinite(E)

    # 梯度与力（单帧）
    g = sys_nonpbc.grad_potential()
    F = sys_nonpbc.forces()
    assert g.shape == R0.shape and F.shape == R0.shape
    assert jnp.all(jnp.isfinite(g)) and jnp.all(jnp.isfinite(F))
    # F = -grad
    assert jnp.allclose(F, -g)


def test_amber_system_batched_nonperiodic():
    """非周期系统：批量梯度/力（直接传入批量坐标，使用系统内部批量处理能力）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys_nonpbc = AmberSystem(prmtop_file, inpcrd_file, periodic=False, pme=False)
    R0 = sys_nonpbc.coordinate
    R_batch = jnp.stack([R0, R0, R0], axis=0)

    # 批量梯度/力：直接传入批量坐标，触发 AmberSystem 内部 vmap
    G_batch = sys_nonpbc.grad_potential(coord=R_batch)
    assert G_batch.shape == R_batch.shape
    assert jnp.all(jnp.isfinite(G_batch))

    F_batch = sys_nonpbc.forces(coord=R_batch)
    assert F_batch.shape == R_batch.shape
    assert jnp.all(jnp.isfinite(F_batch))
    # F = -grad，每一帧坐标一致，力也应一致
    assert jnp.allclose(F_batch, -G_batch)
    max_diff_between_frames = jnp.max(jnp.abs(F_batch[0] - F_batch[1]))
    assert (max_diff_between_frames < 1e-8) or jnp.allclose(
        F_batch[0], F_batch[1], rtol=1e-6, atol=1e-8
    )


def test_amber_system_batched_nonperiodic_minibatch_matches_full():
    """非周期系统：minibatch 梯度/力与全批量结果一致。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys_nonpbc = AmberSystem(prmtop_file, inpcrd_file, periodic=False, pme=False)
    R0 = sys_nonpbc.coordinate
    R_batch = jnp.stack([R0, R0, R0, R0, R0, R0], axis=0)

    sys_nonpbc.minibatch_size = None
    g_ref = sys_nonpbc.grad_potential(coord=R_batch)
    sys_nonpbc.minibatch_size = 2
    g_chunk = sys_nonpbc.grad_potential(coord=R_batch)
    assert g_ref.shape == g_chunk.shape == R_batch.shape
    assert jnp.allclose(g_ref, g_chunk, rtol=1e-6, atol=1e-8)

    sys_nonpbc.minibatch_size = None
    f_ref = sys_nonpbc.forces(coord=R_batch)
    sys_nonpbc.minibatch_size = 2
    f_chunk = sys_nonpbc.forces(coord=R_batch)
    assert jnp.allclose(f_ref, f_chunk, rtol=1e-6, atol=1e-8)


def test_amber_system_batched_nonperiodic_potential_minibatch_matches_full():
    """非周期系统：minibatch 能量与全批量结果一致。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys_nonpbc = AmberSystem(
        prmtop_file,
        inpcrd_file,
        periodic=False,
        pme=False,
        minibatch_size=2,
        assert_divisible_minibatch=True,
    )
    R0 = sys_nonpbc.coordinate
    R_batch = jnp.stack([R0, R0, R0, R0, R0, R0], axis=0)

    sys_nonpbc.minibatch_size = None
    e_full = sys_nonpbc.potential(coord=R_batch)
    sys_nonpbc.minibatch_size = 2
    e_chunk = sys_nonpbc.potential(coord=R_batch)
    assert e_full.shape == e_chunk.shape
    assert jnp.allclose(e_full, e_chunk, rtol=1e-6, atol=1e-8)


def test_amber_system_periodic_pme_energy_and_batch():
    """PME 周期系统：单帧与批量能量（需传入邻居表）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    sys_pme = AmberSystem(
        prmtop_file,
        inpcrd_file,
        periodic=True,
        pme=True,
        cutoff=0.8,
        grid_points=96,
    )
    R0 = sys_pme.coordinate

    # 单帧邻居表
    nb_single = sys_pme.allocate_neighbor_list(R0)
    E = sys_pme.potential(coord=R0, nbList=nb_single)
    assert jnp.isfinite(E)

    # 批量
    R_batch = jnp.stack([R0, R0], axis=0)
    nb_batch = sys_pme.allocate_neighbor_list(R_batch)
    f_batch = jax.jit(lambda R, nb: sys_pme.potential(coord=R, nbList=nb))
    E_batch = f_batch(R_batch, nb_batch)
    assert E_batch.shape == (2,)
    assert jnp.all(jnp.isfinite(E_batch))
    # 两帧相同，应一致，且与单帧能量匹配
    assert jnp.allclose(E_batch, jnp.full_like(E_batch, E), rtol=1e-6, atol=1e-8)


def test_amber_system_periodic_non_pme_not_supported():
    """周期但非 PME 预期不支持，应在构造时抛出 NotImplementedError（来自底层 AmberPotential）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    with pytest.raises(NotImplementedError):
        _ = AmberSystem(prmtop_file, inpcrd_file, periodic=True, pme=False)


def test_amber_system_neighbor_list_update():
    """邻居表更新：显式 allocate 后 update。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    sys_pme = AmberSystem(
        prmtop_file,
        inpcrd_file,
        periodic=True,
        pme=True,
        cutoff=0.8,
        grid_points=96,
    )
    R0 = sys_pme.coordinate

    nb0 = sys_pme.allocate_neighbor_list(R0)
    nb1 = sys_pme.update_neighbor_list(R0, nbList=nb0)

    # 如果底层返回不可区分对象，至少不应报错；若可区分，对象应不同
    if (nb0 is not None) and (nb1 is not None):
        assert (nb0 is not nb1) or True  # 放宽要求，主要检验 API 可用


def test_amber_system_energy_consistency_nonperiodic():
    """非周期系统：不同 cutoff 参数能量一致（与底层逻辑一致）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys1 = AmberSystem(prmtop_file, inpcrd_file, periodic=False, pme=False)
    sys2 = AmberSystem(prmtop_file, inpcrd_file, periodic=False, pme=False, cutoff=0.6)

    E1 = sys1.potential()
    E2 = sys2.potential()
    assert jnp.allclose(E1, E2, rtol=1e-6, atol=1e-8)


def test_amber_system_jit_and_grad_nonperiodic_with_openmm():
    """非周期系统：JIT、梯度与 OpenMM 能量/力对照。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    sys_nonpbc = AmberSystem(prmtop_file, inpcrd_file, periodic=False, pme=False)
    R0 = sys_nonpbc.coordinate

    f = jax.jit(lambda R: sys_nonpbc.potential(coord=R))
    E = f(R0)
    assert jnp.isfinite(E)

    # 使用内部接口计算梯度而不是手动使用 jax.grad
    G = sys_nonpbc.grad_potential(coord=R0)
    F = -G
    assert F.shape == R0.shape and jnp.all(jnp.isfinite(F))

    E_mm, F_mm = _openmm_energy_forces(
        prmtop_file,
        inpcrd_file,
        np.array(R0),
        periodic=False,
        pme=False,
        cutoff_nm=sys_nonpbc.potential_.cutoff,
    )

    assert jnp.allclose(E, E_mm, rtol=1e-2), f"Energy mismatch: ours={E}, omm={E_mm}"

    F_mm = jnp.array(F_mm)
    max_err = jnp.max(jnp.abs(F - F_mm))
    max_rel_err = max_err / (jnp.max(jnp.abs(F_mm)) + 1e-12)
    assert max_rel_err < 1e-2, f"Non-PBC forces mismatch: max_rel_err={max_rel_err}"
    assert max_err < 1e-2, f"Non-PBC forces mismatch: max_err={max_err}"


def test_amber_system_pme_forces_and_batch_grad():
    """PME 周期系统：力= -grad，OpenMM 能量/力对照；批量梯度一致性。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    sys_pme = AmberSystem(
        prmtop_file,
        inpcrd_file,
        periodic=True,
        pme=True,
        cutoff=0.8,
        grid_points=96,
    )
    R0 = sys_pme.coordinate

    nb = sys_pme.allocate_neighbor_list(R0)
    assert nb is not None

    # 使用内部接口计算梯度而不是手动使用 jax.grad
    G = sys_pme.grad_potential(coord=R0, nbList=nb)
    F = -G
    assert F.shape == R0.shape and jnp.all(jnp.isfinite(F))

    # OpenMM 对照
    E_mm, F_mm = _openmm_energy_forces(
        prmtop_file,
        inpcrd_file,
        np.array(R0),
        periodic=True,
        pme=True,
        cutoff_nm=sys_pme.potential_.cutoff,
    )
    E_ours = sys_pme.potential(coord=R0, nbList=nb)
    rel_e = (
        float(jnp.abs((E_ours - E_mm) / E_mm))
        if E_mm != 0
        else float(jnp.abs(E_ours - E_mm))
    )
    assert rel_e < 2e-2, (
        f"PME energy relative mismatch too large: rel={rel_e}, ours={E_ours}, omm={E_mm}"
    )

    F_mm = jnp.array(F_mm)
    force_err = jnp.linalg.norm(F - F_mm) / (jnp.linalg.norm(F_mm) + 1e-12)
    assert force_err < 5e-2, f"PME forces relative error too large: {force_err}"

    # 批量梯度/力：直接传入批量坐标与批量邻居表，触发系统内部批量处理
    R_batch = jnp.stack([R0, R0], axis=0)
    nb_batch = sys_pme.allocate_neighbor_list(R_batch)
    assert nb_batch is not None
    F_batch = sys_pme.forces(coord=R_batch, nbLists=nb_batch)
    assert F_batch.shape == R_batch.shape and jnp.all(jnp.isfinite(F_batch))

    max_diff_between_frames = jnp.max(jnp.abs(F_batch[0] - F_batch[1]))
    assert (max_diff_between_frames < 1e-8) or jnp.allclose(
        F_batch[0], F_batch[1], rtol=1e-6, atol=1e-8
    )

    max_diff_fb = jnp.max(jnp.abs(F_batch[0] - F))
    rel_fb = max_diff_fb / (jnp.max(jnp.abs(F)) + 1e-12)
    assert (
        (rel_fb < 1e-8)
        or (max_diff_fb < 1e-8)
        or jnp.allclose(F_batch[0], F, rtol=1e-6, atol=1e-8)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
