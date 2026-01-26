from __future__ import annotations
from .util import Numeric, ReadOnlyDict
from collections.abc import Mapping
from numpy.typing import NDArray
from typing import Annotated, Any, TYPE_CHECKING, override
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation_proxies import Proxy

type numpy_vec = Annotated[NDArray[np.float64], (3,)]
type numpy_list = Annotated[NDArray[np.float64], ("N",)]
type numpy_mat = Annotated[NDArray[np.float64], ("N", 3)]
type numpy_sq = Annotated[NDArray[np.float64], (3, 3)]
type numpy_col = Annotated[NDArray[np.float64], ("N", 1)]
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


def _dot(a: NDArray, b: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	"""
	Typical case: Compute the rowwise dot product between two matricies of row-vectors.
	The result is the (1D) transpose of a column vector with each dot product in the corresponding row.
	Precondition: All parameters should be the same shape.
	Postcondition: temp can be one of the source operands, but clobbers temp either way!
	"""
	d = a.ndim - 1
	if temp is None:
		return np.sum(a * b, axis=d)
	else:
		np.multiply(a, b, out=temp)
		return np.sum(temp, axis=d)

def _norm_sq(array: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	"""
	Typical case: compute the norm squared of a matrix of row-vectors.
	The result is the (1D) transpose of a column vector with each squre in the corresponding row.
	"""
	return _dot(array, array, temp=temp)

def _norm(array: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	"""
	Typical case: compute the norm of a matrix of row-vectors.
	The result is the (1D) column vector with each norm in the corresponding row.
	"""
	temp = _norm_sq(array, temp=temp)
	if isinstance(temp, np.ndarray):
		return np.sqrt(temp, out=temp)
	else:
		return np.sqrt(temp)

def _mean(a: NDArray) -> NDArray:
	return np.mean(a, axis=0)

def _cov(a: NDArray, b: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	mean_a = _mean(a)
	mean_b = _mean(b)
	dotp = _dot(a, b, temp=temp)
	return _mean(dotp) - dot(mean_a, mean_b, temp=temp)

def _var(a: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	mean = _mean(a)
	dotp = _dot(a, a, temp=temp)
	return _mean(dotp) - dot(mean, mean, temp=temp)

def _std(a: NDArray, temp: NDArray=None) -> NDArray|np.float64:
	return np.sqrt(_var(a, temp=temp))

def _asarray(temp: NDArray, *potential_proxies: Proxy|NDArray) -> tuple[NDArray, ...]:
	for i, p in enumerate(potential_proxies):
		if not isinstance(p, np.ndarray):
			p = p.history.values(None)  # allocate new buffer
			potential_proxies[i] = p
			temp = p  # we can use this newly allocated buffer as temp in case temp was None
	return *potential_proxies, temp

def _asfloat(value: NDArray|np.float64) -> NDArray|float:
	if isinstance(value, np.ndarray):
		return value
	return float(value)

def dot(a: Proxy|NDArray, b: Proxy|NDArray, temp: NDArray=None) -> NDArray|float:
	return _asfloat(_dot(*_asarray(temp, a, b)))

def norm_sq(proxy: Proxy|NDArray, temp: NDArray=None) -> NDArray|float:
	""" Returns the historical rowwise norm squared of each vector/value in this proxies history. """
	return _asfloat(_norm_sq(*_asarray(temp, proxy)))

def norm(proxy: Proxy|NDArray, temp: NDArray=None) -> NDArray|float:
	""" Returns the historical rowwise norm of each vector/value in this proxies history. """
	return _asfloat(_norm(*_asarray(temp, proxy)))

def mean(proxy: Proxy|NDArray) -> Any:
	""" The arithmetic mean of a Simulation variable (over time). """
	a, _ = _asarray(None, proxy)
	return _asfloat(_mean(a))

def cov(a: Proxy|NDArray, b: Proxy|NDArray, temp: NDArray=None) -> NDArray|np.float64:
	""" The (scalar) covariance between two simulation variables (over time). """
	return _asfloat(_cov(*_asarray(temp, a, b)))

def var(proxy: Proxy|NDArray, temp: NDArray=None) -> NDArray|np.float64:
	""" The variance of a Simulation variable (over time). """
	return _asfloat(_var(*_asarray(temp, proxy)))

def std(proxy: Proxy|NDArray, temp: NDArray=None) -> NDArray|np.float64:
	""" The standard deviation of a Simulation variable (over time). """
	return _asfloat(_std(*_asarray(temp, proxy)))


# [interface] Behaves as an numpy ndarray
class Array(Numeric[NDArray]):
	def __array__(self, dtype=None) -> NDArray:
		return self.value
	
	def __matmul__(self, other):   return self.value @ other
	def __rmatmul__(self, other):  return other @ self.value
	def __imatmul__(self, other):  self.value @= other;  return self

	# numpy specific stuff:
	#	Should we prefer this object's ufunc __array_ufunc__ and __array_function__
	#	over ndarray's (ndarray.__array_priority__ == 0). This allows our methods
	#	to first unwrap all Numeric objects, then propogate to ndarray (or any other "ducks").
	__array_priority__ = 1000.0  # VERY HIGH
	
	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		inputs = tuple(x.value if isinstance(x, Numeric) else x for x in inputs)
		return self.value.__array_ufunc__(ufunc, method, *inputs, **kwargs)
	
	# def __array_function__(self, func, types, args, kwargs):
	# 	print(str(func), *(type(arg) for arg in args))
	# 	args = tuple(x.value if isinstance(x, Numeric) else x for x in args)
	# 	print(str(func), *(type(arg) for arg in args))
	# 	return self.value.__array_function__(func, types, args, kwargs)

type Vector = Array  # except __array__ -> numpy_vec
Vector = Array  # So we can use Vector at runtime as parent type, etc.

# [interface] Behaves as a float
class Scalar(Numeric[float]):
	def __float__(self) -> float:
		return self.value


# [interface]
class Arrangeable:
	@property
	def __dtype__(self) -> type:
		return None  # default impl.
	
	@property
	def values(self, out: NDArray=None) -> NDArray:
		raise NotImplementedError("abstract")

# [interface]
class NumericArrangeable(Array, Arrangeable):
	@override
	@property
	def value(self) -> NDArray:
		return self.values(None)

# TODO: remove
# [interface]
class ArrangeableMapping[K, V](Arrangeable, Mapping[K, Numeric[V]]):
	@override
	def array(self, out: NDArray=None) -> NDArray:
		values = [self[key].value for key in self]
		if out is None:
			return np.array(values, dtype=self.__dtype__)
		else:
			out[:] = values  # use existing output buffer
			return out

# Wrapper for dict of Numerics to interop as ndarray.
# Used by Proxy.history
class ArrangeableDict[K, V](ReadOnlyDict[K, V], NumericArrangeable):
	def __init__(self, d: Mapping, dtype=None):
		super().__init__(d)
		if dtype is not None:
			setattr(self, "__dtype__", dtype)

	@override
	def array(self, out: NDArray=None) -> NDArray:
		values = [*self.values()]
		if out is None:
			return np.array(values, dtype=self.__dtype__)
		else:
			out[:] = values  # use existing output buffer
			return out
