from __future__ import annotations
from .runtime import Runtime
from collections.abc import Sequence, Mapping
from numpy import ndarray
import numpy as np

class NodeVectorParameterProxy:
	def __init__(self)

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
