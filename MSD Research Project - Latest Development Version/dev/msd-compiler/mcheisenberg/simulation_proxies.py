from __future__ import annotations
from .util import AbstractReadableDict
from collections.abc import Callable, Iterator
from typing import Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation import *

class Proxy:
	def __init__(self, simulation, sim_name: str, elements: list):
		self._sim = simulation
		self._name = sim_name
		self._elements = elements
	
	@property
	def name(self) -> str:
		return self._name
	
	@property
	def values(self) -> numpy_mat|numpy_list:
		return np.asarray()  # TODO: stub

	def __getitem__(self, key: Node|Edge|Region|ERegion) -> Proxy:
		runtime = self._sim.rt
		config = runtime.config
		if param in config.localNodeParameters.get(key, {}):
			self._elements = 
			return self

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

# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"results", e.g. s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy(Numeric):
	def __init__(self,
		simulation: Simulation,
		param: str,
		to_sim: Callable[[vec|float], numpy_vec|float],  # for getting simulation value from runtime
		to_rt: Callable[[numpy_vec|float], vec|float]    # for updating runtime value from simualtion
	):
		self._runtime_proxy = getattr(simulation.rt, param)  # Returned proxy should be const. Get once and store.
		self._param = param
		self._to_sim = to_sim
		self._to_rt = to_rt
		self._nodes = simulation.nodes  # only node_list_proxy. No copy is made.
	
	@property
	def _runtime_value(self) -> vec|float:
		return self._runtime_proxy.value
	
	@_runtime_value.setter
	def _runtime_value(self, value: vec|float) -> None:
		self._runtime_proxy.value = value

	def get_runtime_value(self, key: Node|Edge|Region|ERegion):
		return self._runtime_proxy[key]
	
	def set_runtime_value(self, key: Node|Edge|Region|ERegion, value: vec|float) -> None:
		self._runtime_proxy[key] = value
	
	@property
	def name(self) -> str:
		return self._param
	
	@property
	def value(self) -> numpy_vec:
		return self._to_sim(self._runtime_proxy.value)

	@value.setter
	def value(self, value: numpy_vec) -> None:
		self._runtime_proxy.value = self._to_rt(value)
	
	def __getitem__(self, key: Node|Edge|Region|ERegion) -> numpy_vec:
		return _simvec(self.get_runtime_value(key))
	
	def __setitem__(self, key: Node|Edge|Region|ERegion, value: numpy_vec) -> None:
		self.set_runtime_value(key, _rtvec(value))
	
	def values(self) -> numpy_mat|numpy_list:
		return np.asarray([ self[node] for node in self._nodes ])

class VectorParameterProxy(VectorProxy, ParameterProxy):  pass
class ScalarParameterProxy(ScalarProxy, ParameterProxy):  pass

# Keeps an internal list, _elements, of either nodes, or edges, (or both, e.g.
#	in the case of energy, u). Provides a subscript method which 
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
