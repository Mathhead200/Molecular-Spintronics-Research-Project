from __future__ import annotations
from numpy.typing import NDArray
from typing import Annotated, Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec

type numpy_vec = Annotated[NDArray[np.float64], (3,)]
type numpy_list = Annotated[NDArray[np.float64], ("N",)]
type numpy_mat = Annotated[NDArray[np.float64], ("N", 3)]
type Node = Any    # type parameter
type Region = Any  # type parameter
type Edge = tuple[Node, Node]
type ERegion = tuple[Region, Region]
type Parameter = str


VEC_ZERO = np.zeros(3, dtype=float)
VEC_I = np.array([1.0, 0.0, 0.0])
VEC_J = np.array([0.0, 1.0, 0.0])
VEC_K = np.array([0.0, 0.0, 10])


def simvec(v: vec|numpy_vec|None) -> numpy_vec:
	""" Convert a Runtime tuple vec to a Simulation numpy ndarray. """
	if v is None:
		return VEC_ZERO
	return np.asarray(v, dtype=float)

def rtvec(v: vec|numpy_vec|None) -> vec:
	""" Converts a Simulation numpy ndarray to a Runtime tuple vec. """
	return tuple(simvec(v).astype(float).tolist())

def simscal(x: float|None) -> float:
	""" Convert a Runtime float|None to a Simulation float. """
	if x is None:
		return 0.0
	return x

def rtscal(x: float|None) -> float:
	""" Convert a Simulation float to a Runtime float. """
	if x is None:
		return 0.0
	return x
