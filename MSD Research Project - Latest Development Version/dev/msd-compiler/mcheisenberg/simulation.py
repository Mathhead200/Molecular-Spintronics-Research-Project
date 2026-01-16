from __future__ import annotations
from .runtime import Runtime
from .util import AbstractReadableDict
from numpy import ndarray
from typing import Any, Mapping, Sequence
import numpy as np

def _simvec(v: tuple|None) -> ndarray:
	""" Convert a Runtime tuple vec to a Simulation numpy ndarray. """
	if v is not None:
		return np.asarray(v, dtype=float)
	return Simulation.VEC_ZERO

def _rtvec(v: ndarray) -> tuple:
	""" Converts a Simulation numpy ndarray to a Runtime tuple vec. """
	return tuple(v.astype(float).tolist())

class StateProxy(AbstractReadableDict):
	def __init__(self, simulation: Simulation, runtime_name: str, sim_name: str):
		self._sim = simulation
		self._runtime = simulation.rt
		self._prop = runtime_name
		self._name = sim_name

	@property
	def name(self):
		return self._name

	def __iter__(self):                    return iter(self._runtime.nodes)	
	def __len__(self) -> int:              return len(self._runtime.nodes)
	def __contains__(self, node) -> bool:  return node in self._runtime.nodes

class RuntimeStateProxy(StateProxy):
	@property
	def runtime_name(self):
		return self._prop
	
	def __getitem__(self, node) -> ndarray:
		return _simvec(getattr(self._runtime.node[node], self.runtime_name))

	def __setitem__(self, node, value: ndarray) -> None:
		setattr(self._runtime.node[node], self.runtime_name, _rtvec(value))

class LocalMagnetizationProxy(StateProxy):
	def __getitem__(self, node) -> ndarray:
		sim = self._sim
		return sim.s[node] + sim.f[node]
	
	def __setitem__(self, node, value):
		raise NotImplementedError(f"{self.name} is a read-only value. To set, use sim.s and/or sim.f. (m = s + f)")

class ParameterProxy:
	def __init__(self, simulation: Simulation, param: str):
		self._runtime_proxy = getattr(simulation.rt, param)  # Returned proxy should be const. Get once and store.
		self._param = param
	
	@property
	def _runtime_value(self):
		return self._runtime_proxy.value
	
	@_runtime_value.setter
	def _runtime_value(self, value) -> None:
		self._runtime_proxy.value = value

	def __getitem__(self, key):
		return self._runtime_proxy[key]
	
	def __setitem__(self, key, value) -> None:
		self._runtime_proxy[key] = value
	
	@property
	def name(self):
		return self._param
	
	@property
	def value(self):
		raise NotImplementedError()  # abstract
	
	@value.setter
	def value(self, v) -> None:
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

class VectorParameterProxy(ParameterProxy):
	def __array__(self, dtype=None) -> ndarray:
		return np.asarray(self._runtime_value, dtype=dtype)

	@property
	def value(self) -> ndarray:
		return _simvec(self._runtime_value)
	
	@value.setter
	def value(self, value: ndarray):
		self._runtime_value = tuple(value.astype(float).tolist())

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
	
class ScalarParameterProxy(ParameterProxy):
	def __float__(self) -> float:
		return self._runtime_value

	@property
	def value(self) -> float:
		return float(self)
	
	@value.setter
	def value(self, value: float) -> None:
		self._runtime_value = value

# The full state of the Simulation at some simulation time, t.
class Snapshot:
	def __init__(self, sim: Simulation):
		self.t = sim.t
		self.spins = np.array(sim.spins)
		self.fluxes = np.array(sim.fluxes)  # TODO: if fluxes?
		# TODO: copy mutatable parameters
		# TODO: copy other state information: spin/flux, parameters (if they vary), energy, and (maybe) macros like overall magnetization, etc.

	# TODO ...

# Runtime wrapper which converts everything to Numpy arrays.ndarray and adds
#	simulation logic like recording samples, aggregates (e.g. mag, M, MS, MF, etc.), etc.
class Simulation:
	VEC_ZERO = _simvec(Runtime.VEC_ZERO)
	VEC_I    = _simvec(Runtime.VEC_I)
	VEC_J    = _simvec(Runtime.VEC_J)
	VEC_K    = _simvec(Runtime.VEC_K)

	PARAMETERS = {"A", "B", "S", "F", "kT", "Je0", "J", "Je1", "Jee", "b", "D"}
	STATES = {"s", "f", "m"}  # TODO M, MS, MF, U, etc...

	def __init__(self, rt: Runtime):
		self.rt = rt
		self.t = 0  # current simulation time since last restart (i.e. seed, reinitialization, or randomization)
		self.samples = []
		self._spin_proxy = RuntimeStateProxy(self, "spin", "s")
		self._flux_proxy = RuntimeStateProxy(self, "flux", "f")
		self._local_magnetization_proxy = LocalMagnetizationProxy(self, None, "m")
		for param in ["A", "B", "D"]:
			setattr(self, f"_{param}_proxy", VectorParameterProxy(self, param))
		for param in ["S", "F", "kT", "Je0", "J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarParameterProxy(self, param))

	def _recalclate_state(self):
		pass  # TODO: U, M, etc.

	def seed(self, *seed: int) -> None:
		self.rt.seed()
		self.t = 0
		self.samples = []
	
	def reinitialize(self, initSpin: ndarray=VEC_J, initFlux: ndarray=VEC_ZERO) -> None:
		self.rt.reinitialize(tuple(initSpin.tolist()), tuple(initFlux.tolist()))
		self.t = 0
		self.samples = []
		# TODO: recalculate energy?
	
	def randomize(self, *seed: int) -> None:
		self.rt.randomize(*seed)
		self.t = 0
		self.samples = []
		# TODO: recalculate energy?

	def metropolis(self, iterations: int, freq: int=0, bookend: bool=True):
		"""
		Run the metropolis algorithm on this model for the given number of iterations.
		May specify a recording/sampling period (freq), i.e. record a snapshot every freq iterations.
		If bookend, a final snapshot will be recorded after all iterations are completed. E.g.
			sim.metropolis(100, freq=10, bookend=True) will generate 11 snapshopts at times, t=0, 10, 20, ..., 100.
			sim.metropolis(100, freq=10, bookend=False) will generate 10 snapshots at times, t=0, 10, 20, ..., 90.
				The algorithm will still run for 100 iterations.
			sim.metropolis(100, freq=11, bookend=True) will generate 11 snapshots at times, t=0, 11, 22, ..., 99, 100.
		"""
		if not freq:
			self.rt.metropolis(iterations)
			self.t += iterations
		else:
			self.record()
			while iterations > freq:
				self.rt.metropolis(freq)
				self.t += freq
				self.record()
				iterations -= freq
			if iterations != 0:
				self.rt.metropolis(iterations)
				self.t += iterations
			if bookend:
				self.record()
	
	def record(self):
		self.samples.append(Snapshot(self))
	
	@property
	def nodes(self) -> Sequence:
		return self.rt.nodes
	
	@property
	def edges(self) -> Sequence[tuple]:
		return self.rt.edges
	
	@property
	def regions(self) -> Sequence:
		return self.rt.regions
	
	@property
	def eregions(self) -> Sequence[tuple]:
		return self.rt.eregions

	@property
	def s(self) -> Mapping[Any, ndarray]:
		""" spin """
		return self._spin_proxy
	
	@property
	def f(self) -> Mapping[Any, ndarray]:
		""" flux """
		return self._flux_proxy
	
	@property
	def m(self) -> Mapping[Any, ndarray]:
		""" (local) magnetization """
		return self._local_magnetization_proxy
	
	@property
	def parameters(self) -> Sequence[str]:
		return self.rt.config.allKeys

	def __getitem__(self, attr: str):
		if attr not in Simulation.PARAMETERS | Simulation.STATES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in Simulation.PARAMETERS:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
