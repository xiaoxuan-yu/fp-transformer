import os
from typing import Optional

import jax.numpy as jnp

from ..type import JaxArray
from .base import Potential
from .gaff_utils import (
    GAFFForceFieldEnergy,
    convert_sponge_input_to_dict,
    # FF_TARGET_SHAPE,
)


class GAFF(Potential):
    """
    GAFF (General Amber Force Field) 势能计算类。

    该类继承自 Potential 基类，用于计算分子系统的 GAFF 力场能量。
    支持从 SPONGE 格式文件加载力场参数，并进行批量能量计算。

    Parameters
    ----------
    sponge_prefix : str
        SPONGE 参数文件前缀，用于加载力场参数文件
    n_atoms : int, optional
        原子数，如果未提供则从参数文件中自动推断

    Attributes
    ----------
    n_atoms : int
        系统中的原子数
    ff_params : dict
        力场参数字典，包含所有 GAFF 力场参数
    energy_calculator : GAFFForceFieldEnergy
        能量计算器实例
    """

    def __init__(self, sponge_prefix: str, n_atoms: Optional[int] = None):
        """
        初始化 GAFF 势能计算器。

        Parameters
        ----------
        sponge_prefix : str
            SPONGE 参数文件前缀
        n_atoms : int, optional
            原子数，如果未提供则从参数文件中自动推断

        Raises
        ------
        FileNotFoundError
            当指定的 SPONGE 参数文件不存在时
        ValueError
            当参数文件格式不正确或缺少必要参数时
        """
        super().__init__()

        # 验证参数文件是否存在
        required_files = [
            f"{sponge_prefix}_charge.txt",
            f"{sponge_prefix}_LJ.txt",
            f"{sponge_prefix}_bond.txt",
            f"{sponge_prefix}_angle.txt",
        ]

        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required parameter file not found: {file}")

        # 加载力场参数
        try:
            self.ff_params = convert_sponge_input_to_dict(sponge_prefix)
        except Exception as e:
            raise ValueError(
                f"Failed to load force field parameters from {sponge_prefix}: {str(e)}"
            )

        # 确定原子数
        if n_atoms is not None:
            self.n_atoms = n_atoms
        else:
            # 从电荷数组长度推断原子数
            self.n_atoms = len(self.ff_params["C"])

        # 验证原子数一致性
        if self.n_atoms <= 0:
            raise ValueError(f"Invalid number of atoms: {self.n_atoms}")

        # 初始化能量计算器 - 使用最大原子数以兼容批量参数格式
        from .gaff_utils import NMAX_ATOMS

        self.energy_calculator = GAFFForceFieldEnergy(n_atom=NMAX_ATOMS)

        # 预处理参数以适应批量计算
        self._prepare_batch_parameters()

    def _prepare_batch_parameters(self):
        """
        预处理力场参数以适应批量计算格式。

        将所有参数扩展到目标形状，以便支持批量计算。
        """
        from .gaff_utils import FF_TARGET_SHAPE

        self.batch_params = {}

        for key, target_shape in FF_TARGET_SHAPE.items():
            if key in self.ff_params:
                param = self.ff_params[key]

                # 确保参数是 numpy 数组
                if not isinstance(param, (jnp.ndarray, jnp.ndarray)):
                    param = jnp.array(param)

                # 扩展参数到目标形状
                if param.shape != target_shape:
                    # 对于一维参数，进行零填充
                    if len(target_shape) == 1:
                        if len(param.shape) == 1:
                            padded_param = jnp.zeros(target_shape, dtype=param.dtype)
                            padded_param = padded_param.at[: param.shape[0]].set(param)
                            param = padded_param
                        else:
                            raise ValueError(
                                f"Parameter {key} has incompatible shape {param.shape} for target {target_shape}"
                            )
                    # 对于二维参数，进行零填充
                    elif len(target_shape) == 2:
                        if len(param.shape) == 2:
                            padded_param = jnp.zeros(target_shape, dtype=param.dtype)
                            padded_param = padded_param.at[
                                : param.shape[0], : param.shape[1]
                            ].set(param)
                            param = padded_param
                        else:
                            raise ValueError(
                                f"Parameter {key} has incompatible shape {param.shape} for target {target_shape}"
                            )

                self.batch_params[key] = param
            else:
                # 如果参数不存在，用零填充
                self.batch_params[key] = jnp.zeros(target_shape, dtype=jnp.float32)

    def __call__(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        """
        计算给定坐标的 GAFF 势能。

        Parameters
        ----------
        x : JaxArray
            坐标数组，形状为 (num_replicas, num_particles, 3)
        beta : float, optional
            逆温度参数，默认为 1.0

        Returns
        -------
        JaxArray
            势能数组，形状为 (num_replicas, 1)

        Raises
        ------
        ValueError
            当输入坐标形状不正确时
        """
        # 验证输入形状
        if len(x.shape) != 3:
            raise ValueError(
                f"Input coordinates must have shape (num_replicas, num_particles, 3), got {x.shape}"
            )

        if x.shape[1] != self.n_atoms:
            raise ValueError(f"Expected {self.n_atoms} atoms, got {x.shape[1]}")

        if x.shape[2] != 3:
            raise ValueError(
                f"Each coordinate must have 3 dimensions, got {x.shape[2]}"
            )

        num_replicas = x.shape[0]

        # 扩展参数以匹配批量大小
        batch_ff_inputs = {}
        for key, param in self.batch_params.items():
            # 对于每个参数，复制 num_replicas 次
            if len(param.shape) == 1:
                batch_ff_inputs[key] = jnp.repeat(param[None, :], num_replicas, axis=0)
            elif len(param.shape) == 2:
                batch_ff_inputs[key] = jnp.repeat(
                    param[None, :, :], num_replicas, axis=0
                )
            else:
                batch_ff_inputs[key] = param

        # 需要将坐标扩展到最大原子数以匹配参数维度
        from .gaff_utils import NMAX_ATOMS

        # 创建填充后的坐标数组
        padded_crds = jnp.zeros((num_replicas, NMAX_ATOMS, 3), dtype=x.dtype)
        padded_crds = padded_crds.at[:, : self.n_atoms, :].set(x)

        # 提取参数以匹配 GAFFForceFieldEnergy.__call__ 的接口
        crds = padded_crds
        charge = batch_ff_inputs["C"]
        lj_epsilon = batch_ff_inputs["E"]
        lj_sigma = batch_ff_inputs["S"]
        exclude = batch_ff_inputs["Ex"]

        # Bond parameters
        bond_a = batch_ff_inputs["Ba"]
        bond_b = batch_ff_inputs["Bb"]
        bond_k = batch_ff_inputs["Bk"]
        bond_r0 = batch_ff_inputs["Br"]

        # Angle parameters
        angle_a = batch_ff_inputs["Aa"]
        angle_b = batch_ff_inputs["Ab"]
        angle_c = batch_ff_inputs["Ac"]
        angle_k = batch_ff_inputs["Ak"]
        angle_theta0 = batch_ff_inputs["At"]

        # Dihedral parameters
        dih_a = batch_ff_inputs["Da"]
        dih_b = batch_ff_inputs["Db"]
        dih_c = batch_ff_inputs["Dc"]
        dih_d = batch_ff_inputs["Dd"]
        dih_n = batch_ff_inputs["Dn"]
        dih_k = batch_ff_inputs["Dk"]
        dih_phi0 = batch_ff_inputs["Dp"]

        # NB14 parameters
        nb14_a = batch_ff_inputs["Na"]
        nb14_b = batch_ff_inputs["Nb"]
        nb14_lj_f = batch_ff_inputs["Nlf"]
        nb14_q_f = batch_ff_inputs["Nqf"]

        # 计算能量
        try:
            energy = self.energy_calculator(
                crds,
                charge,
                lj_epsilon,
                lj_sigma,
                exclude,
                bond_a,
                bond_b,
                bond_k,
                bond_r0,
                angle_a,
                angle_b,
                angle_c,
                angle_k,
                angle_theta0,
                dih_a,
                dih_b,
                dih_c,
                dih_d,
                dih_n,
                dih_k,
                dih_phi0,
                nb14_a,
                nb14_b,
                nb14_lj_f,
                nb14_q_f,
            )

            # 乘以 beta 参数并调整形状
            energy = energy  # * beta
            return energy.reshape(-1, 1)

        except Exception as e:
            raise RuntimeError(f"Failed to compute GAFF energy: {str(e)}")
