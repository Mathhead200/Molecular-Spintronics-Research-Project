from __future__ import annotations
from .config import Config
from .runtime import Runtime
from .simulation_proxies import *
from .util import ReadOnlyDict
from collections.abc import Mapping, Sequence
from numpy.typing import NDArray
from typing import Annotated, Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec

type numpy_vec = Annotated[NDArray[np.float64], (3,)]
type numpy_list = Annotated[NDArray[np.float64], ("N",)]
type numpy_mat = Annotated[NDArray[np.float64], ("N", 3)]
type Node = Any    # type parameter
type Region = Any  # type parameter
type Edge = tuple[Node, Node]
type ERegion = tuple[Region, Region]


def simvec(v: tuple|None) -> numpy_vec:
	""" Convert a Runtime tuple vec to a Simulation numpy ndarray. """
	if v is None:
		return Simulation.VEC_ZERO
	return np.asarray(v, dtype=float)

def rtvec(v: numpy_vec) -> vec:
	""" Converts a Simulation numpy ndarray to a Runtime tuple vec. """
	return tuple(v.astype(float).tolist())

def simscal(x: float|None) -> float:
	""" Convert a Runtime float|None to a Simulation float. """
	if x is None:
		return 0.0
	return x

def rtscal(x: float) -> float:
	""" Convert a Simulation float to a Runtime float. """
	# Does nothing, but is a placeholder in case we want to change the
	# 	Simulation scalar data type later.
	return x


# -----------------------------------------------------------------------------
# The full state of the Simulation at some simulation time, t.
class Snapshot:
	def __init__(self, sim: Simulation):
		self.t = sim.t
		self.spins = np.array(sim.spins)
		self.fluxes = np.array(sim.fluxes)  # TODO: if fluxes?
		# TODO: copy mutatable parameters
		# TODO: copy other state information: spin/flux, parameters (if they vary), energy, and (maybe) macros like overall magnetization, etc.

	# TODO ...


# -----------------------------------------------------------------------------
# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording samples, aggregates (e.g. m, U, n, etc.), etc.
class Simulation:
	VEC_ZERO = simvec(Runtime.VEC_ZERO)
	VEC_I    = simvec(Runtime.VEC_I)
	VEC_J    = simvec(Runtime.VEC_J)
	VEC_K    = simvec(Runtime.VEC_K)

	NODE_PARAMETERS = Config.ALLOWED_NODE_PARAMETERS
	EDGE_PARAMETERS = Config.ALLOWED_EDGE_PARAMETERS
	PARAMETERS = NODE_PARAMETERS | EDGE_PARAMETERS
	STATES = {"n", "s", "f", "m", "u", "c", "x"}  # TODO: state variable for number of edges?

	def __init__(self, rt: Runtime):
		self.rt: Runtime = rt
		self.t = 0  # current simulation time since last restart (i.e. seed, reinitialization, or randomization)
		self.samples: list[Snapshot] = []
		# self.u_node = {
		# 	node: {
		# 		param: 0.0
		# 		for param in Simulation.NODE_PARAMETERS				
		# 	} for node in self.rt.config.nodes
		# }  # sim.u_node[node][node_param] = local energy for the associated interaction
		# self.u_edge = {
		# 	edge: {
		# 		param: 0.0
		# 		for param in Simulation.EDGE_PARAMETERS
		# 	} for edge in self.rt.config.edges
		# }  # sim.u_edge[edge][edge_param] = bond energy for the associated interaction
		for param in ["A", "B"]:
			setattr(self, f"_{param}_proxy", VectorNodeParameterProxy(self, param))
		for param in ["S", "F", "kT", "Je0"]:
			setattr(self, f"_{param}_proxy", ScalarNodeParameterProxy(self, param))
		for param in ["J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarEdgeParameterProxy(self, param))
		for param in ["D"]:
			setattr(self, f"_{param}_proxy", VectorEdgeParameterProxy(self, param))
		# self._spin_proxy = RuntimeVectorStateProxy(self, "spin", "s")
		# self._flux_proxy = RuntimeVectorStateProxy(self, "flux", "f")
		# self._local_magnetization_proxy = LocalMagnetizationProxy(self, None, "m")

	# TODO: REMOVE?
	def _recalclate_energy(self) -> None:
		# This tracking of interanl energy could be done in ASM,
		#	but since it is not needed during the metropolis algorithm,
		#	for now, we will do a slow (python) compute once after each
		#	metropolis algo. finishes.
		u = 0.0
		
		# node parameters: B, A, Je0
		m_values: numpy_mat = self.m.values()
		u -= np.sum(self.B.values() * m_values)  # U_B: Hadamard product + reduce (sum all elements)
		u -= self.A[i] 

		for edge in self.edges:
			u = 0.0
			# edge parameters: J, Je1, Jee, b, D

	def seed(self, *seed: int) -> None:
		self.rt.seed()
		self.t = 0
		self.samples = []
	
	def reinitialize(self, initSpin: numpy_vec=VEC_J, initFlux: numpy_vec=VEC_ZERO) -> None:
		self.rt.reinitialize(tuple(initSpin.tolist()), tuple(initFlux.tolist()))
		self.t = 0
		self.samples = []
		# TODO: recalculate energy?
	
	def randomize(self, *seed: int) -> None:
		self.rt.randomize(*seed)
		self.t = 0
		self.samples = []
		# TODO: recalculate energy?

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
	def nodes(self) -> Sequence[Node]:
		return ReadOnlyList(self.rt.config.nodes)
	
	@property
	def edges(self) -> Sequence[Edge]:
		return ReadOnlyList(self.rt.config.edges)
	
	@property
	def regions(self) -> Mapping[Region, Sequence[Node]]:
		return ReadOnlyDict(self.rt.config.regions)
	
	@property
	def eregions(self) -> Mapping[ERegion, Sequence[Edge]]:
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
	def s(self) -> Mapping[Any, numpy_vec]:
		""" spin """
		return self._spin_proxy
	
	@property
	def f(self) -> Mapping[Any, numpy_vec]:
		""" flux """
		return self._flux_proxy
	
	@property
	def m(self) -> Mapping[Any, numpy_vec]:
		""" magnetization """
		return self._local_magnetization_proxy
	
	@property
	def parameters(self) -> Mapping[str, Sequence[Node]|Sequence[Edge]]:
		config = self.rt.config
		return {
			param: [
				node
				for node in config.nodes
				if config.hasNodeParameter(node, param)
			]
			for param in config.allKeys
			if param in Simulation.NODE_PARAMETERS
		} | {
			param: [
				edge
				for edge in config.edges
				if config.hasEdgeParameter(edge, param)
			]
			for param in config.allKeys
			if param in Simulation.EDGE_PARAMETERS
		}

	def __getitem__(self, attr: str):
		if attr not in Simulation.PARAMETERS | Simulation.STATES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in Simulation.PARAMETERS:
	setattr(Simulation, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
