from typing import Optional
from ..type import JaxArray


class Topology:
    def __init__(self):
        pass

    @property
    def coordinate(self) -> Optional[JaxArray]:
        """Get the coordinate array. Should be implemented by subclasses."""
        return None

    @coordinate.setter
    def coordinate(self, coordinate: JaxArray) -> None:
        """Set the coordinate array. Should be implemented by subclasses."""
        pass


class CoordTopology(Topology):
    def __init__(self, coordinate: JaxArray):
        super().__init__()
        self._coordinate = coordinate

    @property
    def coordinate(self) -> JaxArray:
        """Get the coordinate array."""
        return self._coordinate

    @coordinate.setter
    def coordinate(self, coordinate: JaxArray) -> None:
        """Set the coordinate array."""
        self._coordinate = coordinate
