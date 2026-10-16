import typing
import numpy as np
from optax import GradientTransformation

from jax import Array, numpy

JaxArray = Array
KeyArray = Array
Any = typing.Any
Callable = typing.Callable
List = typing.List
NamedTuple = typing.NamedTuple
Optional = typing.Optional
Sequence = typing.Sequence
Tuple = typing.Tuple
Union = typing.Union
Array = np.ndarray
Optimizer = GradientTransformation
FloatLike = Union[
    float,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    np.float16,
    np.float32,
    np.float64,
]

del typing
