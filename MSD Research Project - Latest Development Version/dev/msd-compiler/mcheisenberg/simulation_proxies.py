from __future__ import annotations
from .util import AbstractReadableDict, ReadOnlyCollection, ReadOnlyList
from collections.abc import Callable, Collection, Iterator, Sequence
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation import *


__NODES__ = "__NODES__"  # enum
__EDGES__ = "__EDGES__"  # enum


class Proxy:
	def __init__(self, simulation, sim_name: str, elements: Collection, subscripts: Sequence=[]):
		self._sim = simulation
		self._name = sim_name
		self._elements = elements
		self._subscripts = subscripts

	def filter(self, key: Any) -> Collection:
		""" Parse the filter, and returns a collection of candidate elements. """
		raise NotImplementedError()  # abstract

	def __getitem__(self, key: Any) -> Proxy:
		elements = [x for x in self.filter(key) if x in self._elements]  # itersection of new elements and self._elements
		if len(elements) == 0:
			raise ValueError(f"{key} is valid by itself, but disjoint with previous subscripts: {self._subscripts}")
		return self.__class__(self._sim, self._name, elements, self._subscripts + [key])  # new Proxy of same type (i.e. subclass)

	@property
	def name(self) -> str:
		return self._name
	
	@property
	def elements(self):
		return ReadOnlyCollection(self._elements)

	@property
	def subscripts(self):
		return ReadOnlyList(self._subscripts)


# Interface for a numerical/mathamatical object, e.g. numpy ndarray, or float.
#	Any object which can be used in mathamatical expressions, namely
#	parameters proxies and result proxies can inherit from this class.
class Numeric:
	@property
	def value(self) -> Any:  # return somthing "numerical"
		raise NotImplementedError()  # abstract

	def __add__(self, addend):       return self.value + addend
	def __sub__(self, subtrahend):   return self.value - subtrahend
	def __mul__(self, multiplier):   return self.value * multiplier
	def __truediv__(self, divisor):  return self.value / divisor
	def __floordiv__(self, divisor): return self.value // divisor
	def __mod__(self, modulus):      return self.value % modulus
	def __pow__(self, exponent):     return self.value ** exponent
	def __neg__(self):               return -self.value
	def __pos__(self):               return +self.value
	def __abs__(self):               return abs(self.value)

	def __radd__(self, augend):         return augend + self.value
	def __rsub__(self, minuend):        return minuend - self.value
	def __rmul__(self, multiplicand):   return multiplicand * self.value
	def __rtruediv__(self, dividend):   return dividend / self.value
	def __rfloordiv__(self, dividend):  return dividend // self.value
	def __rmod__(self, dividend):       return dividend % self.value
	def __rpow__(self, base):           return base ** self.value

	def __iadd__(self, addend):        self.value = self + addend;     return self
	def __isub__(self, subtrahend):    self.value = self - subtrahend; return self
	def __imul__(self, multiplier):    self.value = self * multiplier; return self
	def __itruediv__(self, divisor):   self.value = self / divisor;    return self
	def __ifloordiv__(self, divisor):  self.value = self // divisor;   return self
	def __imod__(self, modulus):       self.value = self % modulus;    return self
	def __ipow__(self, exponent):      self.value = self ** exponent;  return self

	def __eq__(self, other):  return self.value == other
	def __ne__(self, other):  return self.value != other
	def __lt__(self, other):  return self.value < other
	def __le__(self, other):  return self.value <= other
	def __gt__(self, other):  return self.value > other
	def __ge__(self, other):  return self.value >= other

	def __str__(self) -> str:  return str(self.value)
	def __hash__(self):  return hash(self.value)

# Behaves as an numpy ndarray.
class Vector(Numeric):
	def __array__(self, dtype=None) -> numpy_vec:
		return np.asarray(self.value, dtype=dtype)
	
	def __len__(self):
		assert len(self.value) == 3  # DEBUG
		return 3
	
	def __iter__(self):
		return iter(self.value)

	# numpy specific stuff
	__array_priority__ = -1000.0  # VERY LOW: Let other numpy arrray coerce this object

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		return self.value.__array_ufunc__(ufunc, method, *inputs, **kwargs)
	
	def __array_function__(self, func, types, args, kwargs):
		return self.value.__array_function__(func, types, args, kwargs)

# Behaves as a float.
class Scalar(Numeric):
	def __float__(self) -> float:
		return self.value

class Int(Numeric):
	def __int__(self) -> int:    return self.value
	def __index__(self) -> int:  return self.value

	def __or__(self, other):      return self.value | other
	def __xor__(self, other):     return self.value ^ other
	def __and__(self, other):     return self.value & other
	def __lshift__(self, shift):  return self.value << shift
	def __rshift__(self, shift):  return self.value >> shift
	def __invert__(self):         return ~self.value

	def __ror__(self, other):      return other | self.value
	def __rxor__(self, other):     return other ^ self.value
	def __rand__(self, other):     return other & self.value
	def __rlshift__(self, value):  return value << self.value
	def __rrshift__(self, value):  return value << self.value

class NumericProxy(Numeric, Proxy):  pass

# Function
def values(proxy: NumericProxy) -> NDArray:
	return np.asarray([ proxy[x].value for x in proxy._elements ])


# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"results", e.g. s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy(Numeric, Proxy):
	def __init__(self,
		simulation: Simulation,
		param: str,
		elements: Sequence[Node|Edge],
		to_sim: Callable[[vec|float], numpy_vec|float],  # for getting simulation value from runtime
		to_rt: Callable[[numpy_vec|float], vec|float]    # for updating runtime value from simualtion
	):
		super().__init__(simulation, param, elements)
		self._runtime_proxy = getattr(simulation.rt, param)  # Returned proxy should be const. Get once and store.
		self._to_sim = to_sim
		self._to_rt = to_rt
	
	@property
	def key(self) -> Node|Edge|Region|ERegion:
		return self._subscripts[-1]  # only the last subscript is relevant for prarameter proxies
	
	@property
	def _runtime_value(self) -> vec|float:
		if len(self._subscripts) == 0:
			return self._runtime_proxy.value
		else:
			return self._runtime_proxy[self.key]
	
	@_runtime_value.setter
	def _runtime_value(self, value: vec|float) -> None:
		if len(self._subscripts) == 0:
			self._runtime_proxy.value = value
		else:
			self._runtime_proxy[self.key] = value
	
	@property
	def value(self) -> numpy_vec:
		return self._to_sim(self._runtime_value)

	@value.setter
	def value(self, value: numpy_vec) -> None:
		self._runtime_value = self._to_rt(value)
	
	def __setitem__(self, key: Node|Edge|Region|ERegion, value: numpy_vec|float) -> None:
		self[key].value = value

class NodeParameterProxy(ParameterProxy):
	def __init__(self, simulation: Simulation, param: str, to_sim: Callable[[vec|float], numpy_vec|float], to_rt: Callable[[numpy_vec|float], vec|float]):
		super().__init__(simulation, param, simulation.nodes, to_sim, to_rt)
	
	def filter(self, key: Node|Region) -> Collection[Node]:
		config = self._sim.rt.config
		if key in config.nodes:
			# key is interpreted a (local) Node
			elements = [key]
		elif key in config.regions:
			# key is interpreted as a Region
			elements = [config.regions[key]]
		else:
			raise KeyError(f"Subscript [{key}] is undefined for parameter {self.name}: not a node nor region in Config")
		return elements

class EdgeParameterProxy(ParameterProxy):
	def __init__(self, simulation: Simulation, param: str, to_sim: Callable[[vec|float], numpy_vec|float], to_rt: Callable[[numpy_vec|float], vec|float]):
		super().__init__(simulation, param, simulation.edges, to_sim, to_rt)
	
	def filter(self, key: Edge|ERegion) -> Collection[Edge]:
		config = self._sim.rt.config
		elements = None
		if key in config.edges:
			# key is interpreted as a (local) Edge
			elements = [key]  
		elif key in config.regionCombos:
			# key is interpreted as an ERegion
			R = (config.regions[key[i]] for i in range(2))  # nodes in both regions
			elements = [e for e in config.edges if e[0] in R[0] and e[1] in R[1]]
		else:
			raise KeyError(f"Subscript [{key}] is undefined for parameter {self.name}: not an edge nor edge-region in Config")
		return elements

class VectorNodeParameterProxy(Vector, NodeParameterProxy):
	def __init__(self, simulation: Simulation, param: str):
		super().__init__(simulation, param, to_sim=simvec, to_rt=rtvec)

class ScalarNodeParameterProxy(Scalar, NodeParameterProxy):
	def __init__(self, simulation: Simulation, param: str):
		super().__init__(simulation, param, to_sim=simscal, to_rt=rtscal)

class VectorEdgeParameterProxy(Vector, EdgeParameterProxy):
	def __init__(self, simulation: Simulation, param: str):
		super().__init__(simulation, param, to_sim=simvec, to_rt=rtvec)

class ScalarEdgeParameterProxy(Scalar, EdgeParameterProxy):
	def __init__(self, simulation: Simulation, param: str):
		super().__init__(simulation, param, to_sim=simscal, to_rt=rtscal)


# Number of nodes in proxy
class NProxy(Scalar, Proxy):
	def __init__(self, simulation: Simulation, sim_name: str="n"):
		super().__init__(simulation, sim_name, simulation.nodes + simulation.edges)

	@property
	def value(self) -> int:



# TODO: REMOVE -- vv OLD STUFF vv
class ResultProxy(NumericProxy):
	def __init__(self, simulation: Simulation, sim_name: str):
		self._sim = simulation
		self._name = sim_name
		self._elements: list[Node|Edge] = None  # set by subclass

	@property
	def name(self) -> str:
		return self._name
	
class VectorResultProxy(ResultProxy):
	def __init__(self, simulation: Simulation, sim_name: str):
		super().

class VectorStateProxy(AbstractReadableDict):
	def __init__(self, simulation: Simulation, runtime_name: str, sim_name: str):
		self._sim = simulation
		self._runtime = simulation.rt
		self._prop = runtime_name
		self._name = sim_name

	@property
	def name(self):
		return self._name

	def __iter__(self) -> Iterator[Node]:  return iter(self._runtime.nodes)	
	def __len__(self) -> int:              return len(self._runtime.nodes)
	def __contains__(self, node) -> bool:  return node in self._runtime.nodes

	# Override: allows getting all nodes as a shape=(N, 3) numpy matrix for
	#	vector states, or shape=(N,) numpy vector for scalar states.
	def values(self) -> numpy_mat:
		return np.asarray([ *super().values() ])

class RuntimeVectorStateProxy(VectorStateProxy):
	@property
	def runtime_name(self):
		return self._prop
	
	def __getitem__(self, node: Node) -> numpy_vec:
		return _simvec(getattr(self._runtime.node[node], self.runtime_name))

	def __setitem__(self, node: Node, value: numpy_vec) -> None:
		setattr(self._runtime.node[node], self.runtime_name, _rtvec(value))

class LocalMagnetizationProxy(VectorStateProxy):
	def __getitem__(self, node: Node) -> numpy_vec:
		sim = self._sim
		return sim.s[node] + sim.f[node]
	
	def __setitem__(self, node, value):
		raise NotImplementedError(f"{self.name} is a read-only value. To set, use sim.s and/or sim.f. (m = s + f)")
