from .base import Potential


class RingPolymer(Potential):
    """
    A Ring polymer numerical system.
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
