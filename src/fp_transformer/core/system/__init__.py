from .system import BaseSystem, EnergySystem, ForceSystem

__all__ = ["BaseSystem", "EnergySystem", "ForceSystem", "AmberSystem"]


def __getattr__(name):
    if name == "AmberSystem":
        from .amber_system import AmberSystem as _AmberSystem

        return _AmberSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
