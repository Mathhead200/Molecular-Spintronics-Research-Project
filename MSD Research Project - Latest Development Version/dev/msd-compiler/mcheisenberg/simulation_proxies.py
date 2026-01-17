from __future__ import annotations
from .simulation_util import rtscal, rtvec, simscal, simvec, PARAMETERS
from .util import ReadOnlyCollection, ReadOnlyList
from collections.abc import Callable, Collection, Sequence
from copy import copy
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation import Simulation
	from .simulation_util import numpy_vec


type Filter = Callable[[Simulation, str, Any], Collection]  # (Proxy.simulation, Proxy.name, key) -> candidate_elements

class Proxy:
	def __init__(self, simulation, sim_name: str, elements: Collection, filter: Filter, subscripts: Sequence=[]):
		self._sim: Simulation = simulation
		self._name = sim_name
		self._elements: Collection = elements
		self._filter = filter
		self._subscripts: Sequence = subscripts

	def __getitem__(self, key: Any) -> Proxy:
		proxy = copy(self)
		candidate_elements = self._filter(self._sim, self._name, key)
		proxy._elements = { x: None for x in candidate_elements if x in self._elements }  # (set) itersection of candidate_elements and self._elements
		if len(proxy._elements) == 0:
			raise ValueError(f"{key} is valid by itself, but disjoint with previous subscripts: {self._subscripts}")
		proxy._subscripts += [key]
		return proxy

	@property
	def name(self) -> str:
		return self._name

	@property
	def elements(self) -> Collection:
		return ReadOnlyCollection(self._elements)

	@property
	def subscripts(self) -> Sequence:
		return ReadOnlyList(self._subscripts)

def _node_filter(sim: Simulation, param: str, key: Node|Region) -> Collection[Node]:
	if key in sim.nodes:    return [key]  # key is interpreted a (local) node
	if key in sim.regions:  return sim.regions[key]  # key is interpreted as a (node) region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not a node nor region in Config")
	
def _edge_filter(sim: Simulation, param: str, key: Edge|ERegion) -> Collection[Edge]:
	if key in sim.edges:     return [key]              # key is interpreted as a (local) edge
	if key in sim.eregions:  return sim.eregions[key]  # key is interpreted as an edge-region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not an edge nor edge-region in Config")

def _full_filter(sim: Simulation, param: str, key: Node|Edge|Region|ERegion|str) -> Collection[Node|Edge]:
	if key == __NODES__:       return sim.nodes  # interpreted as the set of all nodes
	if key == __EDGES__:       return sim.edges  # interpreted as the set of all edges
	if key in sim.nodes:       return [key]      # interpreted as a (local) node
	if key in sim.edges:       return [key]      # interpreted as a (local) edge
	if key in sim.regions:     return sim.regions[key]     # interpreted as a (node) region
	if key in sim.eregions:    return sim.eregions[key]    # interpreted as an edge-region
	if key in sim.parameters:  return sim.parameters[key]  # interpreted as a (node or edge) parameter
	raise KeyError(f"For parameter {param}, undefined subscript: {key}")


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

# Behaves as an numpy ndarray
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

# Behaves as a float
class Scalar(Numeric):
	def __float__(self) -> float:
		return self.value

# Behaves as int
class Int(Numeric):
	def __int__(self) -> int:    return self.value
	def __index__(self) -> int:  return self.value
	def __bool__(self) -> bool:  return bool(self.value)

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


class NumericProxy(Proxy, Numeric):
	def values(self) -> NDArray:
		return np.array([ self[key] for key in self._elements ])
	
	def items(self) -> Collection[tuple[Any, Any]]:
		return ReadOnlyDict([ (key, self[key]) for key in self._elements ])


# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"states", e.g. n, s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy(NumericProxy):
	def __init__(self,
		simulation: Simulation,
		param: str,
		elements: Sequence[Node|Edge],
		filter: Filter,
		to_sim: Callable[[vec|float], numpy_vec|float]=None,  # for getting simulation value from runtime
		to_rt: Callable[[numpy_vec|float], vec|float]=None    # for updating runtime value from simualtion
	):
		super().__init__(simulation, param, elements, filter)
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

class VectorNodeParameterProxy(ParameterProxy, Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simvec, to_rt=rtvec)

class ScalarNodeParameterProxy(ParameterProxy, Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simscal, to_rt=rtscal)

class VectorEdgeParameterProxy(ParameterProxy, Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simvec, to_rt=rtvec)

class ScalarEdgeParameterProxy(ParameterProxy, Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simscal, to_rt=rtscal)


# For spin and flux
class StateProxy(NumericProxy, Vector):
	def __init__(self, sim: Simulation, rt_attr: str, sim_name: str):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)
		self._runtime_proxy = getattr(sim.rt, rt_attr)
	
	@property
	def key(self) -> Node:
		return self._subscripts[-1]  # only the last subscript is relevant for "state" proxies

	@property
	def value(self) -> numpy_vec:
		return np.sum(self.values(), axis=0)  # add all the row (i.e. axis=0) vectors
	
	@value.setter
	def value(self, value: numpy_vec) -> None:
		if len(self._elements) != 1:
			raise ValueError(f"Can't directly assign to {self._name} for a non-local selection. (Subscript(s): {self._subscripts}.) Instead, select nodes individually and set them one at a time.")
		self._runtime_proxy[self.key] = rtvec(value)

	def __setitem__(self, key: Node|Edge|Region|ERegion, value: numpy_vec|float) -> None:
		self[key].value = value

# Number of nodes in selection
class NProxy(NumericProxy, Int):
	def __init__(self, sim: Simulation, sim_name: str="n"):
		super().__init__(sim, sim_name, elements=(sim.nodes | sim.edges), filter=_full_filter)
	
	@property
	def value(self) -> int:
		return len(self._elements)

# Internal Energy in selection
class UProxy(NumericProxy, Scalar):
	def __init__(self, sim: Simulation, sim_name="u"):
		super().__init__(sim, sim_name, elements=(sim.nodes | sim.edges), filter=_full_filter)
	
	@property
	def value(self) -> float:
		return 0.0  # TODO: (stub)

