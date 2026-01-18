from __future__ import annotations
from .runtime import Runtime
from .simulation_proxies import *
from .simulation_util import EDGE_PARAMETERS, NODE_PARAMETERS, PARAMETERS, STATES, VEC_J, VEC_ZERO
from .util import ReadOnlyDict
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from collections.abc import Sequence


# The full state of the Simulation at some simulation time, t.
class Snapshot:
	def __init__(self, sim: Simulation):
		self.t = sim.t
		self.spins = np.array(sim.spins)
		self.fluxes = np.array(sim.fluxes)  # TODO: if fluxes?
		# TODO: copy mutatable parameters
		# TODO: copy other state information: spin/flux, parameters (if they vary), energy, and (maybe) macros like overall magnetization, etc.

	# TODO ...


# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording samples, aggregates (e.g. m, U, n, etc.), etc.
class Simulation:
	def __init__(self, rt: Runtime):
		self.rt: Runtime = rt

		self.t = 0  # current simulation time since last restart (i.e. seed, reinitialization, or randomization)
		self.samples: list[Snapshot] = []
		
		for param in ["A", "B"]:
			setattr(self, f"_{param}_proxy", VectorNodeParameterProxy(self, param))
		for param in ["S", "F", "kT", "Je0"]:
			setattr(self, f"_{param}_proxy", ScalarNodeParameterProxy(self, param))
		for param in ["J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarEdgeParameterProxy(self, param))
		for param in ["D"]:
			setattr(self, f"_{param}_proxy", VectorEdgeParameterProxy(self, param))
		self._n_proxy = NProxy(self)
		self._s_proxy = StateProxy(self, "spin", "s")
		self._f_proxy = StateProxy(self, "flux", "f")
		self._m_proxy = MProxy(self)
		self._u_proxy = UProxy(self)
		self._c_proxy = None  # TODO: (stub)
		self._x_proxy = None  # TODO: (stub)

		# set of defined parameters, preserving order defined in Config:
		config = self.rt.config
		node_p =    [ p for params in config.localNodeParameters.values()  for p in params ]
		edge_p =    [ p for params in config.localEdgeParameters.values()  for p in params ]
		region_p =  [ p for params in config.regionNodeParameters.values() for p in params ]
		eregion_p = [ p for params in config.regionEdgeParameters.values() for p in params ]
		global_p =  [ p for p in config.globalParameters.keys() ]
		self._parameters = ordered_set(global_p + region_p + eregion_p + node_p + edge_p)

	def seed(self, *seed: int) -> None:
		self.rt.seed()
		self.t = 0
		self.samples = []
	
	def reinitialize(self, initSpin: numpy_vec=VEC_J, initFlux: numpy_vec=VEC_ZERO) -> None:
		self.rt.reinitialize(tuple(initSpin.tolist()), tuple(initFlux.tolist()))
		self.t = 0
		self.samples = []
	
	def randomize(self, *seed: int) -> None:
		self.rt.randomize(*seed)
		self.t = 0
		self.samples = []

	def metropolis(self, iterations: int, freq: int=0, bookend: bool=True) -> None:
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
	
	def record(self) -> Snapshot:
		sample = Snapshot(self)
		self.samples.append(sample)
		return sample
	
	@property
	def nodes(self) -> ReadOnlyDict[Node, None]:  # Note: dict: Node -> None acting as ordered set
		return ReadOnlyDict({ node: None for node in self.rt.config.nodes })
	
	@property
	def edges(self) -> ReadOnlyDict[Edge, None]:  # Note: dict: Edge -> None acting as ordered set
		return ReadOnlyDict({ edge: None for edge in self.rt.config.edges })
	
	@property
	def regions(self) -> ReadOnlyDict[Region, Collection[Node]]:
		return ReadOnlyDict(self.rt.config.regions)
	
	@property
	def eregions(self) -> ReadOnlyDict[ERegion, Collection[Edge]]:
		edges = self.rt.config.edges    # list[Edge]
		nodes = self.rt.config.regions  # Region -> list[Node]
		return ReadOnlyDict({
			eregion: [
				edge
				for edge in edges
				if edge[0] in nodes[eregion[0]] and edge[1] in nodes[eregion[1]]
			] for eregion in self.rt.config.regionEdgeParameters.keys()
		})
	
	@property
	def parameters(self) -> ReadOnlyDict[str, Sequence[Node]|Sequence[Edge]]:
		config = self.rt.config
		parameters = {}
		for p in self._parameters:
			if p in NODE_PARAMETERS:
				parameters[p] = [node for node in config.nodes if config.hasNodeParameter(node, p)]
			else:
				assert p in EDGE_PARAMETERS
				parameters[p] = [edge for edge in config.edges if config.hasEdgeParameter(edge, p)]
		return ReadOnlyDict(parameters)

	def __getitem__(self, attr: str):
		if attr not in PARAMETERS | STATES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in PARAMETERS | STATES:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
