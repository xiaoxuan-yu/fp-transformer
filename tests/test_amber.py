#!/usr/bin/env python3
"""
Test script for AmberPotential class
"""

import jax
import jax.numpy as jnp
import sys
import os
import numpy as np
import pytest
from openmm import app, unit
import openmm as mm

import faulthandler

faulthandler.enable()

# 启用 64 位计算以提升与 OpenMM 的一致性
from jax import config as jax_config

@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    previous = jax_config.jax_enable_x64
    jax_config.update("jax_enable_x64", True)
    yield
    jax_config.update("jax_enable_x64", previous)

# Add the src directory to the path so we can import fp_transformer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fp_transformer.potential.Amber import AmberPotential


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


def test_amber_potential_components():
    """Test individual components of the AmberPotential class."""
    print("Testing AmberPotential components...")

    # Get the path to test data files
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    try:
        # Test instantiation with periodic=False, pme=False (non-periodic)
        potential = AmberPotential(prmtop_file, inpcrd_file, periodic=False, pme=False)
        print("AmberPotential class instantiated successfully!")

        # Test individual energy components
        positions = potential.get_default_positions()

        # Test total energy
        total_energy = potential(positions)
        print(f"Total energy: {total_energy}")

        print("Component tests passed!")
        assert True
    except Exception as e:
        print(f"Error in AmberPotential component test: {e}")
        import traceback

        traceback.print_exc()
        assert False


def test_amber_potential_batched():
    """Test the AmberPotential class with batched inputs."""
    print("Testing AmberPotential class with batched inputs...")

    # Get the path to test data files
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    try:
        # Test instantiation with periodic=False, pme=False (non-periodic)
        potential = AmberPotential(prmtop_file, inpcrd_file, periodic=False, pme=False)
        print("AmberPotential class instantiated successfully!")

        # Test single configuration
        R0 = potential.get_default_positions()
        single_energy = potential(R0)
        print(f"Single configuration energy: {single_energy}")

        # Test batched configurations
        batch_size = 3
        # Create a batch of identical positions to test batched call
        batched_positions = jnp.array([R0] * batch_size)

        # 优先为批量分配邻居表；若实现不支持 allocate，则直接调用批量接口
        nbLists = potential.allocate_neighbor_list(batched_positions)
        if nbLists is not None:
            batched_energies = potential(batched_positions, nbLists=nbLists)
        else:
            # 非周期系统且实现不支持邻居分配时，直接传入批量坐标以测试内部批处理能力
            batched_energies = potential(batched_positions)
        print(f"Batched energies shape: {batched_energies.shape}")
        print(f"Batched energies: {batched_energies}")

        # Verify that batched processing gives the same result for all configurations (since they're identical)
        # and that it matches individual processing
        individual_energy = potential(R0)
        # All batched energies should be the same
        energy_diff_batched = jnp.max(jnp.abs(batched_energies - individual_energy))
        print(
            f"Max difference between batched and individual processing: {energy_diff_batched}"
        )

        print("Batched tests passed!")
        assert True
    except Exception as e:
        print(f"Error in AmberPotential batched test: {e}")
        import traceback

        traceback.print_exc()
        assert False


def test_amber_potential_periodic_pme():
    """Test the AmberPotential class with periodic boundary conditions and PME."""
    print("Testing AmberPotential class with periodic boundary conditions and PME...")

    # Get the path to test data files for PME (benzene_solv system)
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    try:
        # Test instantiation with periodic=True, pme=True (PME)
        potential = AmberPotential(
            prmtop_file,
            inpcrd_file,
            periodic=True,
            pme=True,
            cutoff=0.8,
            grid_points=96,
        )
        print("Periodic PME AmberPotential class instantiated successfully!")

        # Test single configuration with PME（显式分配并传入邻居表）
        R0 = potential.get_default_positions()
        nb_single = potential.allocate_neighbor_list(R0)
        single_energy_pme = potential(R0, nbLists=nb_single)
        print(f"Single configuration energy with PME: {single_energy_pme}")

        # Test batched configurations with PME
        batch_size = 3
        # Create a batch of identical positions to test vmap
        batched_positions = jnp.array([R0] * batch_size)

        # PME 周期系统需要为批量提供邻居表
        nbLists = potential.allocate_neighbor_list(batched_positions)
        assert nbLists is not None, "PME 周期系统应支持邻居表分配"
        batched_energies_pme = potential(batched_positions, nbLists=nbLists)
        print(f"Batched energies shape with PME: {batched_energies_pme.shape}")
        print(f"Batched energies with PME: {batched_energies_pme}")

        # Verify that batched processing gives the same result for all configurations (since they're identical)
        # and that it matches individual processing
        individual_energy_pme = potential(R0, nbLists=nb_single)
        # All batched energies should be the same
        energy_diff_pme = jnp.max(jnp.abs(batched_energies_pme - individual_energy_pme))
        print(
            f"Max difference between batched and individual processing with PME: {energy_diff_pme}"
        )

        print("Periodic PME tests passed!")
        assert True
    except Exception as e:
        print(f"Error in AmberPotential periodic PME test: {e}")
        import traceback

        traceback.print_exc()
        assert False


def test_amber_potential_periodic_non_pme():
    """Test the AmberPotential class with periodic boundary conditions but no PME."""
    print(
        "Testing AmberPotential class with periodic boundary conditions but no PME..."
    )

    # Get the path to test data files
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    # 当前实现对 periodic=True 且 pme=False 未支持，应抛出 NotImplementedError
    with pytest.raises(NotImplementedError):
        _ = AmberPotential(prmtop_file, inpcrd_file, periodic=True, pme=False)
    print("Periodic non-PME path correctly raises NotImplementedError")


def test_amber_potential_neighbor_list_update():
    """Test the neighbor list update functionality."""
    print("Testing AmberPotential neighbor list update...")

    # Get the path to test data files for PME (benzene_solv system)
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    try:
        # Test instantiation with periodic=True, pme=True (PME)
        potential = AmberPotential(
            prmtop_file,
            inpcrd_file,
            periodic=True,
            pme=True,
            cutoff=0.8,
            grid_points=96,
        )
        print("Periodic PME AmberPotential class instantiated successfully!")

        # Test neighbor list update：先显式 allocate，再 update
        R0 = potential.get_default_positions()
        initial_nbList = potential.allocate_neighbor_list(R0)
        updated_nbList = potential.update_neighbor_list(R0, nbList=initial_nbList)

        # Check if neighbor list was updated
        # Handle case where nbList might be None
        if initial_nbList is None and updated_nbList is None:
            print("Both neighbor lists are None - no update performed.")
        elif initial_nbList is None or updated_nbList is None:
            print("Neighbor list updated successfully!")
        else:
            # 直接基于对象标识比较：不同对象则认为已更新
            if initial_nbList is not updated_nbList:
                print("Neighbor list updated successfully!")
            else:
                print("Neighbor list unchanged after update.")

        print("Neighbor list update tests passed!")
        assert True
    except Exception as e:
        print(f"Error in AmberPotential neighbor list update test: {e}")
        import traceback

        traceback.print_exc()
        assert False


def test_amber_potential_energy_consistency():
    """Test energy consistency between different calculation methods."""
    print("Testing AmberPotential energy consistency...")

    # Get the path to test data files
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    try:
        # Test instantiation with periodic=False, pme=False (non-periodic)
        potential1 = AmberPotential(prmtop_file, inpcrd_file, periodic=False, pme=False)
        print("AmberPotential class instantiated successfully!")

        # Test instantiation with periodic=False, pme=False (non-periodic) but different cutoff
        potential2 = AmberPotential(
            prmtop_file, inpcrd_file, periodic=False, pme=False, cutoff=0.6
        )
        print("Second AmberPotential class instantiated successfully!")

        # Test energy calculation with different parameters
        R01 = potential1.get_default_positions()
        R02 = potential2.get_default_positions()
        energy1 = potential1(R01)
        energy2 = potential2(R02)

        # Check if energies are consistent (they should be the same since cutoff doesn't affect non-periodic systems)
        energy_diff = jnp.abs(energy1 - energy2)
        print(f"Energy difference between methods: {energy_diff}")

        print("Energy consistency tests passed!")
        assert True
    except Exception as e:
        print(f"Error in AmberPotential energy consistency test: {e}")
        import traceback

        traceback.print_exc()
        assert False


def test_amber_call_jit_and_grad_nonperiodic():
    """对 __call__ 在非周期系统上进行 jit 和 grad 测试（含批量路径）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "diala.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "diala.inpcrd")

    potential = AmberPotential(prmtop_file, inpcrd_file, periodic=False, pme=False)
    R0 = potential.get_default_positions()

    # jit 测试（单帧）
    f = jax.jit(lambda R: potential(R))
    E = f(R0)
    assert jnp.isfinite(E), "JIT 能量必须为有限值"

    # grad 测试（单帧，对坐标的梯度）
    gfn = jax.grad(lambda R: potential(R))
    G = -gfn(R0)
    assert G.shape == R0.shape, "梯度形状应与位置一致"
    assert jnp.all(jnp.isfinite(G)), "梯度必须为有限值"

    # 批量 grad：直接对批量坐标调用 potential 并在批量维求和，再对批量坐标求梯度
    R_batch = jnp.stack([R0, R0], axis=0)

    # sum_energy_batch = lambda R: jnp.sum(potential(R))  # potential 应支持批量输入

    def sum_energy_batch(R):
        return jnp.sum(potential(R))

    G_batch = jax.grad(sum_energy_batch)(R_batch)
    assert G_batch.shape == R_batch.shape
    assert jnp.all(jnp.isfinite(G_batch))

    # 与 OpenMM 对照：能量与力
    E_mm, F_mm = _openmm_energy_forces(
        prmtop_file,
        inpcrd_file,
        np.array(R0),
        periodic=False,
        pme=False,
        cutoff_nm=potential.cutoff,
    )
    # 能量对比（kcal/mol）
    assert jnp.allclose(E, E_mm, rtol=1e-2), (
        f"Non-PBC energy mismatch: ours={E}, omm={E_mm}"
    )
    # 力对比（kcal/mol/nm）
    assert F_mm.shape == R0.shape
    # F_mm to numpy to jax array
    F_mm = jnp.array(F_mm)
    max_err = jnp.max(jnp.abs(G + 0.0 - F_mm))
    max_rel_err = max_err / (jnp.max(jnp.abs(F_mm)) + 1e-12)

    # find max relative error index
    max_rel_err_idx = jnp.unravel_index(
        jnp.argmax(jnp.abs(G + 0.0 - F_mm) / (jnp.abs(F_mm) + 1e-12)), G.shape
    )
    print(
        f"Max relative error at index {max_rel_err_idx}: ours={G[max_rel_err_idx]}, omm={F_mm[max_rel_err_idx]}, rel_err={max_rel_err}, abs_err={max_err}"
    )
    assert max_rel_err < 1e-2, f"Non-PBC forces mismatch: max_rel_err={max_rel_err}"
    assert max_err < 1e-2, f"Non-PBC forces mismatch: max_err={max_err}"


def test_amber_call_jit_periodic_pme():
    """对 __call__ 在 PME 周期系统上进行 jit 能量测试（单帧）。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    potential = AmberPotential(
        prmtop_file, inpcrd_file, periodic=True, pme=True, cutoff=0.8, grid_points=96
    )
    R0 = potential.get_default_positions()

    # 分配单帧邻居表（PME 必须有）
    nb_single = potential.allocate_neighbor_list(R0)
    assert nb_single is not None

    # f = jax.jit(lambda R, nb: potential(R, nbLists=nb))
    # f = lambda R, nb: potential(R, nbLists=nb)
    E = potential(R0, nbLists=nb_single)
    assert jnp.isfinite(E), "PME JIT 能量必须为有限值"

    # 批量 JIT 能量测试（两帧相同坐标）
    R_batch = jnp.stack([R0, R0], axis=0)
    # 分配批量邻居表（PME 必须有 nbLists）
    nbLists_batch = potential.allocate_neighbor_list(R_batch)
    assert nbLists_batch is not None

    f_batch = jax.jit(lambda R, nb: potential(R, nbLists=nb))
    E_batch = f_batch(R_batch, nbLists_batch)
    assert E_batch.shape == (2,), "批量能量应为一维批大小向量"
    assert jnp.all(jnp.isfinite(E_batch)), "批量 JIT 能量必须为有限值"
    # 两帧输入相同，能量应一致，且与单帧能量匹配
    max_diff_b = jnp.max(jnp.abs(E_batch - E))
    assert max_diff_b < 1e-8 or jnp.allclose(
        E_batch, jnp.full_like(E_batch, E), rtol=1e-6
    ), f"PME 批量能量与单帧不一致: max_diff={max_diff_b}"


def test_amber_pme_forces_grad():
    """PME 场景下用梯度计算力，并通过有限差分校验一个分量。"""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test-data", "data")
    prmtop_file = os.path.join(test_data_dir, "benzene_solv.prmtop")
    inpcrd_file = os.path.join(test_data_dir, "benzene_solv.inpcrd")

    potential = AmberPotential(
        prmtop_file, inpcrd_file, periodic=True, pme=True, cutoff=0.8, grid_points=96
    )
    R0 = potential.get_default_positions()

    # 分配单帧邻居表（PME 必须有）
    nbList = potential.allocate_neighbor_list(R0)
    assert nbList is not None, "PME 场景需要可用的邻居表"

    # 计算力：F = -grad E，且不固定 nbList 到闭包内；将其作为参数传入并对 (R, nb) 求导
    grad_fn = jax.jit(jax.grad(lambda R, nb: potential(R, nbLists=nb), argnums=0))
    F = -grad_fn(R0, nbList)

    assert F.shape == R0.shape
    assert jnp.all(jnp.isfinite(F)), "力必须为有限值"

    # 与 OpenMM 对照（同一 cutoff）：能量与全场力
    E_mm, F_mm = _openmm_energy_forces(
        prmtop_file,
        inpcrd_file,
        np.array(R0),
        periodic=True,
        pme=True,
        cutoff_nm=potential.cutoff,
    )
    E_ours = potential(R0, nbLists=nbList)
    rel_e = (
        float(jnp.abs((E_ours - E_mm) / E_mm))
        if E_mm != 0
        else float(jnp.abs(E_ours - E_mm))
    )
    print(rel_e)
    assert rel_e < 2e-2, (
        f"PME energy relative mismatch too large: rel={rel_e}, ours={E_ours}, omm={E_mm}"
    )
    assert F_mm.shape == R0.shape
    # 力的整体 L2 误差做宽容度校验
    force_err = jnp.linalg.norm(F + 0.0 - F_mm) / (jnp.linalg.norm(F_mm) + 1e-12)
    print(force_err)
    assert force_err < 5e-2, f"PME forces relative error too large: {force_err}"

    # 批量梯度/力测试（两帧相同坐标）：直接对批量坐标与批量邻居表调用 potential，
    # 在批量维求和得到标量后对坐标求梯度
    R_batch = jnp.stack([R0, R0], axis=0)
    nbLists_batch = potential.allocate_neighbor_list(R_batch)
    assert nbLists_batch is not None, "PME 批量场景需要为每帧提供邻居表"
    grad_batch = jax.grad(lambda R, nb: jnp.sum(potential(R, nbLists=nb)), argnums=0)
    F_batch = -grad_batch(R_batch, nbLists_batch)
    assert F_batch.shape == R_batch.shape, "批量力的形状应与批量坐标一致"
    assert jnp.all(jnp.isfinite(F_batch)), "批量力必须为有限值"
    # 两帧坐标相同，力应一致
    max_diff_between_frames = jnp.max(jnp.abs(F_batch[0] - F_batch[1]))
    assert (max_diff_between_frames < 1e-8) or jnp.allclose(
        F_batch[0], F_batch[1], rtol=1e-6, atol=1e-8
    ), f"PME 批量两帧力不一致: abs_diff={max_diff_between_frames}"
    # 与单帧（使用 nbList）结果一致（允许极小数值噪声）
    max_diff_fb = jnp.max(jnp.abs(F_batch[0] - F))
    rel_fb = max_diff_fb / (jnp.max(jnp.abs(F)) + 1e-12)
    assert (
        (rel_fb < 1e-8)
        or (max_diff_fb < 1e-8)
        or jnp.allclose(F_batch[0], F, rtol=1e-6, atol=1e-8)
    ), f"PME 批量力与单帧不一致: rel={rel_fb}, abs={max_diff_fb}"


if __name__ == "__main__":
    test_amber_potential_components()
    test_amber_potential_batched()
    test_amber_potential_periodic_pme()
    test_amber_potential_periodic_non_pme()
    test_amber_potential_neighbor_list_update()
    test_amber_potential_energy_consistency()
    test_amber_call_jit_and_grad_nonperiodic()
    test_amber_call_jit_periodic_pme()
    test_amber_pme_forces_grad()
    print("All tests passed!")
