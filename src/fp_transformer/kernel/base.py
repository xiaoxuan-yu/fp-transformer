# The base class for all kernels.

from ..type import JaxArray


class Kernel:
    def __init__(
        self,
        num_replicas: int,
    ):
        """
        Base class for all kernels.

        Args:
            num_replicas: The number of replicas (batch size).
        """
        self.num_replicas = num_replicas

    def update(self, *args, **kwargs):
        """
        update parameters of kernels
        """
        raise NotImplementedError

    def kernel(self, coordinate: JaxArray):
        """
        Calculate the potential.

        Args:
            coordinate: The input coordinate. The shape is (num_replicas, num_atoms, dimension).
        Returns:
            potential: The potential energy. The shape is (num_replicas, 1).
        """
        raise NotImplementedError

    def kernel_grad(self, coordinate: JaxArray):
        """
        Calculate the gradient of the potential.

        Args:
            coordinate: The input coordinate. The shape is (num_replicas, num_atoms, dimension).
        Returns:
            grad: The gradient of potential energy. The shape is (num_replicas, num_atoms, dimension).
        """
        raise NotImplementedError
