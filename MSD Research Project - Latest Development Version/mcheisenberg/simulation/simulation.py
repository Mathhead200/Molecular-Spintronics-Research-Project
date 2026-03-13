from __future__ import annotations
from ..runtime import Runtime
from ..util import NODE_PARAMETERS, EDGE_PARAMETERS, PARAMETERS, ordered_set, ReadOnlyOrderedSet, ReadOnlyDict
from .simulation_proxies import *
from .simulation_util import VEC_J, VEC_ZERO
from .snapshot import Snapshot
from itertools import chain
from typing import TYPE_CHECKING
from tqdm import tqdm
import numpy as np
if TYPE_CHECKING:
	from ..util import ReadOnlyDict

# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording snapshots, aggregates (e.g. m, U, n, etc.), etc.
class Simulation:
	NODE_PARAMETERS = ordered_set(NODE_PARAMETERS)
	EDGE_PARAMETERS = ordered_set(EDGE_PARAMETERS)
	STATES = ordered_set(["n", "s", "f", "m", "u", "x", "c"])
	ALL_PROXIES = ordered_set(chain(PARAMETERS, STATES))

	A:   VectorNodeParameterProxy
	B:   VectorNodeParameterProxy
	S:   ScalarNodeParameterProxy
	F:   ScalarNodeParameterProxy
	kT:  ScalarNodeParameterProxy
	Je0: ScalarNodeParameterProxy
	J:   ScalarEdgeParameterProxy
	Je1: ScalarEdgeParameterProxy
	Jee: ScalarEdgeParameterProxy
	b:   ScalarEdgeParameterProxy
	D:   VectorEdgeParameterProxy
	n:   NProxy
	s:   StateProxy
	f:   StateProxy
	m:   MProxy
	u:   UProxy
	x:   ChiProxy
	c:   CProxy

	def __init__(self, rt: Runtime):
		self.rt: Runtime = rt
		self.t: int = 0  # current simulation time since last restart (i.e. reinitialization, or randomization)
		self.history: dict[int, Snapshot] = {}

		config = rt.config
		# Just need to configure nodes, edges, regions, and eregions once and cache
		edges = config.edges    # list[Edge]
		rnodes = config.regions  # Region -> list[Node]
		self._nodes = ReadOnlyOrderedSet(ordered_set(config.nodes))
		self._edges = ReadOnlyOrderedSet(ordered_set(edges))
		self._regions = ReadOnlyDict({
			region: ReadOnlyOrderedSet(ordered_set(nodes))
			for region, nodes in rnodes.items()
		})
		for region in rnodes:  rnodes[region] = set(rnodes[region])  # faster lookup for eregions
		rnodes[None] = set(config.nodes) - set(chain(*rnodes.values()))  # "None" region for eregion lookup
		self._eregions = {
			eregion: [
				edge
				for edge in edges
				if edge[0] in rnodes[eregion[0]] and edge[1] in rnodes[eregion[1]]
			] for eregion in config.regionCombos
		}
		self._eregions = ReadOnlyDict({
			eregion: ReadOnlyOrderedSet(ordered_set(redges))
			for eregion, redges in self._eregions.items()
			if len(redges) != 0
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
				assert p in Simulation.EDGE_PARAMETERS
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

		# pre-allocated buffers for snapshots (i.e. recordings)
		N = len(config.nodes)
		M = len(config.edges)
		L = N + M
		self._buf_mat_node  = np.empty((N, 3), dtype=float)
		self._buf_mat_node2 = np.empty((N, 3), dtype=float)
		self._buf_mat_edge  = np.empty((M, 3), dtype=float)
		self._buf_list_node = np.empty((N,), dtype=float)
		self._buf_list_edge = np.empty((M,), dtype=float)
		self._buf_s_i = np.empty((M, 3), dtype=float)
		self._buf_s_j = np.empty((M, 3), dtype=float)
		self._buf_f_i = np.empty((M, 3), dtype=float)
		self._buf_f_j = np.empty((M, 3), dtype=float)
		self._buf_m_i = np.empty((M, 3), dtype=float)
		self._buf_m_j = np.empty((M, 3), dtype=float)
		self._buf_Je0 = np.empty((N,), dtype=float)
		self._buf_J   = np.empty((M,), dtype=float)
		self._buf_Je1 = np.empty((M,), dtype=float)
		self._buf_Jee = np.empty((M,), dtype=float)
		self._buf_b   = np.empty((M,), dtype=float)

	def seed(self, *seed: int) -> None:
		self.rt.seed(seed)
	
	def reinitialize(self, initSpin: numpy_vec=VEC_J, initFlux: numpy_vec=VEC_ZERO) -> None:
		self.rt.reinitialize(rtvec(initSpin), rtvec(initFlux))
		self.clear_history()
	
	def randomize(self, *seed: int) -> None:
		self.rt.randomize(*seed)
		self.clear_history()

	def metropolis(self, iterations: int, freq: int=0, bookend: bool=True, progress_bar: str=None) -> None:
		"""
		Run the metropolis algorithm on this model for the given number of iterations.
		May specify a recording/sampling period (freq), i.e. record a snapshot every freq iterations.
		If bookend, a final snapshot will be recorded after all iterations are completed. E.g.
			sim.metropolis(100, freq=10, bookend=True) will generate 11 snapshopts at times, t=0, 10, 20, ..., 100.
			sim.metropolis(100, freq=10, bookend=False) will generate 10 snapshots at times, t=0, 10, 20, ..., 90.
				The algorithm will still run for 100 iterations.
			sim.metropolis(100, freq=11, bookend=True) will generate 11 snapshots at times, t=0, 11, 22, ..., 99, 100.
		"""
		if progress_bar is not None:  progress_bar = tqdm(total=iterations, desc=progress_bar)
		if not freq:
			self.rt.metropolis(iterations)
			self.t += iterations
			if progress_bar is not None:  progress_bar.update(iterations)
		else:
			self.record()
			while iterations > freq:
				self.rt.metropolis(freq)
				self.t += freq
				self.record()
				iterations -= freq
				if progress_bar is not None:  progress_bar.update(freq)
			if iterations != 0:
				self.rt.metropolis(iterations)
				self.t += iterations
				if progress_bar is not None:  progress_bar.update(iterations)
			if bookend:
				self.record()
		if progress_bar is not None:  progress_bar.close()
	
	def record(self) -> Snapshot:
		sample = Snapshot(self)
		self.history[sample.t] = sample
		return sample

	def clear_history(self) -> None:
		self.t = 0
		self.history = {}
	
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
