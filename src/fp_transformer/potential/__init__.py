from .base import Potential
from .TiwaryBerne import TiwaryBerne
from .BereSzabo import BerezhkovskiiSzabo
from .Periodic import Periodic
from .PeriodicMultidim import PeriodicMultidim
from .hidim_periodic import HiDimPeriodic
from .periodic_sin4 import PeriodicSin4
from .GAFF import GAFF
from .GMMBias import GMMBiasPotential
from .SumPotential import SumPotential
from .Vina import (
    VinaScore,
    VinaWeights,
    build_ligand_intra_pairs,
    vina_pairwise_terms,
)

__all__ = [
    "Potential",
    "TiwaryBerne",
    "BerezhkovskiiSzabo",
    "Periodic",
    "PeriodicMultidim",
    "HiDimPeriodic",
    "PeriodicSin4",
    "GAFF",
    "AmberPotential",
    "ECEPPPotential",
    "GMMBiasPotential",
    "SumPotential",
    "VinaScore",
    "VinaWeights",
    "build_ligand_intra_pairs",
    "vina_pairwise_terms",
]


def __getattr__(name):
    if name == "AmberPotential":
        # Lazy import avoids loading OpenMM/CUDA plugins unless AmberPotential is used.
        from .Amber import AmberPotential as _AmberPotential

        return _AmberPotential
    if name == "ECEPPPotential":
        from ..ecepp import ECEPPPotential as _ECEPPPotential

        return _ECEPPPotential
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
