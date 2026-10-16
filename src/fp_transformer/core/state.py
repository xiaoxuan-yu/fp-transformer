from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax

from ..type import JaxArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KernelState:
    """Generic kernel runtime state for jit/scan friendly flows."""

    bandwidth: JaxArray
    rff_key: Optional[JaxArray] = None
    aux: Optional[Any] = None

    def tree_flatten(self):
        return (self.bandwidth, self.rff_key, self.aux), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        bandwidth, rff_key, aux = children
        return cls(bandwidth=bandwidth, rff_key=rff_key, aux=aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SamplerState:
    """Common sampler runtime state."""

    coordinate: JaxArray
    opt_state: Any
    rng_key: JaxArray
    kernel_state: Optional[Any] = None
    nb_list: Optional[Any] = None

    def tree_flatten(self):
        return (
            self.coordinate,
            self.opt_state,
            self.rng_key,
            self.kernel_state,
            self.nb_list,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        coordinate, opt_state, rng_key, kernel_state, nb_list = children
        return cls(
            coordinate=coordinate,
            opt_state=opt_state,
            rng_key=rng_key,
            kernel_state=kernel_state,
            nb_list=nb_list,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ITSState:
    """ITS-specific adaptive state."""

    beta_array: JaxArray
    f_array: JaxArray
    mask: JaxArray
    anneal_factor: Optional[JaxArray] = None
    total_steps: Optional[JaxArray] = None

    def tree_flatten(self):
        return (
            self.beta_array,
            self.f_array,
            self.mask,
            self.anneal_factor,
            self.total_steps,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        beta_array, f_array, mask, anneal_factor, total_steps = children
        return cls(
            beta_array=beta_array,
            f_array=f_array,
            mask=mask,
            anneal_factor=anneal_factor,
            total_steps=total_steps,
        )
