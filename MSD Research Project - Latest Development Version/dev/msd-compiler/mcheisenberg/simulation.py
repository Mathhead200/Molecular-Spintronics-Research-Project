from __future__ import annotations
from .constants import EDGE_PARAMETERS, NODE_PARAMETERS, PARAMETERS
from .runtime import Runtime
from .simulation_proxies import *
from .simulation_util import VEC_J, VEC_ZERO, simscal, simvec
from .util import *
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .simulation_util import *
	from collections.abc import Sequence


# The full state of the Simulation at some simulation time, t.
class Snapshot:
	SAVES = [ *NODE_PARAMETERS, *EDGE_PARAMETERS, "s", "f", "m" ]  # "u"

	def __init__(self, sim: Simulation):
		self.t = sim.t
		rt = sim.rt
		nodes = sim.nodes
		edges = sim.edges
		self.B   = { i:  simvec(rt.B[i])   for i in nodes }
		self.A   = { i:  simvec(rt.A[i])   for i in nodes }
		self.S   = { i:  simvec(rt.S[i])   for i in nodes }
		self.F   = { i: simscal(rt.F[i])   for i in nodes }
		self.kT  = { i: simscal(rt.kT[i])  for i in nodes }
		self.Je0 = { i: simscal(rt.Je0[i]) for i in nodes }

		self.J   = { e: simscal(rt.J[e])   for e in edges }
		self.Je1 = { e: simscal(rt.Je1[e]) for e in edges }
		self.Jee = { e: simscal(rt.Jee[e]) for e in edges }
		self.b   = { e: simscal(rt.b[e])   for e in edges }
		self.D   = { e:  simvec(rt.D[e])   for e in edges }
		
		self.s = { e: simvec(rt.spin[e])    for e in nodes }
		self.f = { e: simvec(rt.flux[e])    for e in nodes }
		self.m = { e: self.s[e] + self.f[e] for e in nodes }

		u = sim.u.values()  # compute energies in paralell with numpy
		u_keys = chain(nodes, edges)
		self.u = { key: float(u[idx]) for idx, key in enumerate(u_keys) }

# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording snapshots, aggregates (e.g. m, U, n, etc.), etc.
class Simulation:
	NODE_PARAMETERS = ordered_set(NODE_PARAMETERS)
	EDGE_PARAMETERS = ordered_set(EDGE_PARAMETERS)
	STATES = ordered_set(["n", "s", "f", "m", "u", "x", "c"])
	ALL_PROXIES = ordered_set(chain(PARAMETERS, STATES))

	def __init__(self, rt: Runtime):
		self.rt: Runtime = rt
		self.t: int = 0  # current simulation time since last restart (i.e. reinitialization, or randomization)
		self.history: Sequence[Snapshot] = []

		config = rt.config
		# Just need to configure nodes, edges, regions, and eregions once and cache
		edges = config.edges    # list[Edge]
		rnodes = config.regions  # Region -> list[Node]
		self._nodes = ReadOnlyOrderedSet(ordered_set(config.nodes))
		self._edges = ReadOnlyOrderedSet(ordered_set(edges))
		self._regions = ReadOnlyDict({
			region: ReadOnlyOrderedSet(nodes)
			for region, nodes in rnodes.items()
		})
		self._eregions = ReadOnlyDict({
			eregion: ReadOnlyOrderedSet([
				edge
				for edge in edges
				if edge[0] in rnodes[eregion[0]] and edge[1] in rnodes[eregion[1]]
			]) for eregion in config.regionEdgeParameters.keys()
		})
		# set of defined parameters, preserving order defined in Config:
		node_p =    [ p for params in config.localNodeParameters.values()  for p in params ]
		edge_p =    [ p for params in config.localEdgeParameters.values()  for p in params ]
		region_p =  [ p for params in config.regionNodeParameters.values() for p in params ]
		eregion_p = [ p for params in config.regionEdgeParameters.values() for p in params ]
		global_p =  [ p for p in config.globalParameters.keys() ]
		parameters = {}
		for p in ordered_set(global_p + region_p + eregion_p + node_p + edge_p):
			if p in Simulation.NODE_PARAMETERS:
				parameters[p] = ReadOnlyOrderedSet(ordered_set(node for node in config.nodes if config.hasNodeParameter(node, p)))
			else:
				assert p in EDGE_PARAMETERS
				parameters[p] = ReadOnlyOrderedSet(ordered_set(edge for edge in config.edges if config.hasEdgeParameter(edge, p)))
		self._parameters = ReadOnlyDict(parameters)
		
		# proxies
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
		self._x_proxy = ChiProxy(self)
		self._c_proxy = CProxy(self)

	def seed(self, *seed: int) -> None:
		self.rt.seed(seed)
	
	def reinitialize(self, initSpin: numpy_vec=VEC_J, initFlux: numpy_vec=VEC_ZERO) -> None:
		self.rt.reinitialize(rtvec(initSpin), rtvec(initFlux))
		self.clear_history()
	
	def randomize(self, *seed: int) -> None:
		self.rt.randomize(*seed)
		self.clear_history()

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
		self.history.append(sample)
		return sample

	def clear_history(self) -> None:
		self.t = 0
		self.history = []
	
	@property
	def nodes(self) -> ReadOnlyOrderedSet[Node]:
		return self._nodes
	
	@property
	def edges(self) -> ReadOnlyOrderedSet[Edge]:
		return self._edges
	
	@property
	def regions(self) -> ReadOnlyDict[Region, ReadOnlyOrderedSet[Node]]:
		return self._regions
	
	@property
	def eregions(self) -> ReadOnlyDict[ERegion, ReadOnlyOrderedSet[Edge]]:
		return self._eregions
	
	@property
	def parameters(self) -> ReadOnlyDict[Parameter, ReadOnlyOrderedSet[Node]|ReadOnlyOrderedSet[Edge]]:
		return self._parameters

	# @properties added below: Simulation.s, .f, .m, .u, .n, .J, .B, etc.

	def __getitem__(self, attr: str):
		if attr not in Simulation.ALL_PROXIES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in Simulation.ALL_PROXIES:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
