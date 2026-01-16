from __future__ import annotations
from .runtime import Runtime
from collections.abc import Sequence, Mapping
from numpy import ndarray
import numpy as np

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
		return self.__array__(dtype=float)
	
	@value.setter
	def value(self, value: ndarray):
		self._runtime_value = tuple(value.astype(float).tolist())

	def __len__(self):   return len(self._runtime_value)
	def __iter__(self):  return iter(self.value)

	# numpy specific stuff
	__array_priority__ = -1000.0  # VERY LOW: Let other numpy arrray coerce this object

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		return getattr(ufunc, method)(self.value, *inputs, **kwargs)
	
	def __array_function__(self, func, types, args, kwargs):
		return func(self.value, *args, **kwargs)
	
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
		self.spins = np.array([s.copy() for s in sim.spins])
		self.fluxes = np.array([f.copy() for f in sim.fluxes])  # TODO: if fluxes?
		# TODO: copy mutatable parameters
		# TODO: copy other state information: spin/flux, parameters (if they vary), energy, and (maybe) macros like overall magnetization, etc.

	# TODO ...

# Runtime wrapper which converts everything to Numpy arrays.ndarray and adds
#	simulation logic like recording samples, aggregates (e.g. mag, M, MS, MF, etc.), etc.
class Simulation:
	VEC_ZERO = np.array([0.0, 0.0, 0.0])
	VEC_I    = np.array([1.0, 0.0, 0.0])
	VEC_J    = np.array([0.0, 1.0, 0.0])
	VEC_K    = np.array([0.0, 0.0, 1.0])

	def __init__(self, rt: Runtime):
		self.rt = rt
		self.t = 0  # current simulation time since last restart (i.e. seed, reinitialization, or randomization)
		self.samples = []
		for param in ["A", "B", "D"]:
			setattr(self, f"_{param}_proxy", VectorParameterProxy(self, param))
		for param in ["S", "F", "kT", "Je0", "J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarParameterProxy(self, param))

	def seed(self, *seed: int) -> None:
		self.rt.seed()
		self.t = 0
		self.samples = []
	
	def reinitialize(self, initSpin: ndarray=VEC_I, initFlux: ndarray=VEC_ZERO) -> None:
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
				self.record()
				iterations -= freq
			if iterations != 0:
				self.rt.metropolis(iterations)
			if bookend:
				self.record()
	
	def record(self):
		self.samples.append(Snapshot(self))
	
	@property
	def spins(self) -> ndarray:
		""" shape: (n, 3) """
		return np.array([np.array(s) for s in self.rt.spins])
	
	@property
	def fluxes(self) -> ndarray:
		""" shape: (n, 3) """
		return np.array([np.array(f) if f is not None else Simulation.VEC_ZERO for f in self.rt.fluxes])

	@property
	def magnetizations(self) -> ndarray:
		""" shape: (n, 3) """
		return self.spins + self.fluxes
	
	# TODO ...

for param in ["A", "B", "S", "F", "kT", "Je0", "J", "Je1", "Jee", "b", "D"]:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: 
	))