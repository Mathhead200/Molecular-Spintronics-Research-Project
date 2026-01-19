from __future__ import annotations
from .constants import EDGE_PARAMETERS, NODE_PARAMETERS, PARAMETERS
from .runtime import Runtime
from .simulation_proxies import MProxy, NProxy, ScalarEdgeParameterProxy, ScalarNodeParameterProxy, StateProxy, UProxy, VectorEdgeParameterProxy, VectorNodeParameterProxy
from .simulation_util import VEC_J, VEC_ZERO
from .util import ReadOnlyDict, ReadOnlyOrderedSet, ordered_set
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .simulation_util import Edge, ERegion, Node, Region, numpy_vec
	from collections.abc import Collection, Sequence


# The full state of the Simulation at some simulation time, t.
class Snapshot:
	def __init__(self, sim: Simulation):
		self.t = sim.t
		self.s = np.array(sim.s.items())
		self.f = np.array(sim.f.items())
		# TODO: copy mutatable parameters
		# TODO: copy other state information: spin/flux, parameters (if they vary), energy, and (maybe) macros like overall magnetization, etc.

	# TODO ...


# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording samples, aggregates (e.g. m, U, n, etc.), etc.
class Simulation:
	NODE_PARAMETERS = ordered_set(NODE_PARAMETERS)
	EDGE_PARAMETERS = ordered_set(EDGE_PARAMETERS)
	STATES = ordered_set(["n", "s", "f", "m", "u", "c", "x"])
	ALL_PROXIES = ordered_set(chain(PARAMETERS, STATES))

	def __init__(self, rt: Runtime):
		self.rt: Runtime = rt

		# set of defined parameters, preserving order defined in Config:
		config = self.rt.config
		node_p =    [ p for params in config.localNodeParameters.values()  for p in params ]
		edge_p =    [ p for params in config.localEdgeParameters.values()  for p in params ]
		region_p =  [ p for params in config.regionNodeParameters.values() for p in params ]
		eregion_p = [ p for params in config.regionEdgeParameters.values() for p in params ]
		global_p =  [ p for p in config.globalParameters.keys() ]
		self._parameters = ordered_set(global_p + region_p + eregion_p + node_p + edge_p)

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
		return ReadOnlyOrderedSet(ordered_set(self.rt.config.nodes))
	
	@property
	def edges(self) -> ReadOnlyDict[Edge, None]:  # Note: dict: Edge -> None acting as ordered set
		return ReadOnlyOrderedSet(ordered_set(self.rt.config.edges))
	
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
			if p in Simulation.NODE_PARAMETERS:
				parameters[p] = [node for node in config.nodes if config.hasNodeParameter(node, p)]
			else:
				assert p in EDGE_PARAMETERS
				parameters[p] = [edge for edge in config.edges if config.hasEdgeParameter(edge, p)]
		return ReadOnlyDict(parameters)

	def __getitem__(self, attr: str):
		if attr not in Simulation.ALL_PROXIES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in Simulation.ALL_PROXIES:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
