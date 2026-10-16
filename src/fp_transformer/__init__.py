import jax

from .core import (
    BaseSystem,
    CoordTopology,
    DynamicITSSampler,
    EnergySystem,
    ForceSystem,
    ITSSampler,
    Sampler,
    Topology,
)


def jax_config():
    # use tf32 for matmul and convolutions on compatible devices
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    # enable 64-bit float as default
    # jax.config.update("jax_enable_x64", True)
    return True


jax_config()

del jax_config

__all__ = [
    "Sampler",
    "DynamicITSSampler",
    "ITSSampler",
    "BaseSystem",
    "EnergySystem",
    "ForceSystem",
    "ECEPPPotential",
    "Topology",
    "CoordTopology",
]


def __getattr__(name):
    if name == "ECEPPPotential":
        from .ecepp import ECEPPPotential as _ECEPPPotential

        return _ECEPPPotential
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
