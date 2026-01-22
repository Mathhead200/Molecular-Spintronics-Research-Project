from __future__ import annotations
from .constants import __EDGES__, __NODES__
from .simulation_util import ArrangeableDict, ArrangeableMapping, Scalar, Vector, VEC_ZERO, rtscal, rtvec, simscal, simvec
from .util import IInt, Numeric, ReadOnlyCollection, ReadOnlyList, ordered_set
from collections.abc import Mapping
from copy import copy
from itertools import chain
from typing import Any, override, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation import Simulation, Snapshot
	from .simulation_util import Edge, ERegion, Node, Parameter, Region, numpy_vec
	from collections.abc import Callable, Collection, Iterable, Iterator, Sequence


type Filter = Callable[[Simulation, str, Any], Iterable]  # (Proxy.simulation, Proxy.name, key) -> candidate_elements

class Proxy(Mapping[Any, "Proxy"]):
	def __init__(self, simulation: Simulation, sim_name: str, elements: Collection, filter: Filter, subscripts: list=[]):
		self._sim: Simulation = simulation
		self._name = sim_name
		self._elements: Collection = elements
		self._filter = filter
		self._subscripts: list = subscripts
	
	@property
	def history(self) -> dict[int, Any]:
		raise NotImplementedError()  # abstract
	
	def __len__(self) -> int:             len(self._elements)
	def __iter__(self) -> Iterator:       iter(self.elements)  # TODO: do I need ReadOnly wrapper?
	def __contains__(self, key) -> bool:  return key in self._elements

	def __getitem__(self, key: Any) -> Proxy:
		proxy = copy(self)
		candidate_elements = self._filter(self._sim, self._name, key)
		proxy._elements = { x: None for x in candidate_elements if x in self._elements }  # (ordered set) itersection of candidate_elements and self._elements
		if len(proxy._elements) == 0:
			raise ValueError(f"{key} is valid by itself, but disjoint with previous subscripts: {self._subscripts}")
		proxy._subscripts = self._subscripts + [key]
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
	if key in sim.nodes:    return [key]             # key is interpreted a (local) node
	if key in sim.regions:  return sim.regions[key]  # key is interpreted as a (node) region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not a node nor region in Config")

def _edge_filter(sim: Simulation, param: str, key: Edge|ERegion) -> Collection[Edge]:
	if key in sim.edges:     return [key]              # key is interpreted as a (local) edge
	if key in sim.eregions:  return sim.eregions[key]  # key is interpreted as an edge-region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not an edge nor edge-region in Config")

# Allow parameter keys, but only stores Node|Edge keys as elements
def _ntype_filter(sim: Simulation, param: str, key: Node|Edge|Region|ERegion|Parameter) -> Collection[Node|Edge]:
	if key == __NODES__:       return sim.nodes  # interpreted as the set of all nodes
	if key == __EDGES__:       return sim.edges  # interpreted as the set of all edges
	if key in sim.nodes:       return [key]      # interpreted as a (local) node
	if key in sim.edges:       return [key]      # interpreted as a (local) edge
	if key in sim.regions:     return sim.regions[key]     # interpreted as a (node) region
	if key in sim.eregions:    return sim.eregions[key]    # interpreted as an edge-region
	if key in sim.parameters:  return sim.parameters[key]  # interpreted as a (node or edge) parameter
	raise KeyError(f"For parameter {param}, undefined subscript: [{key}]")

# Allows parameter keys, and stores Node|Edge|Parameter keys are elements
def _utype_filter(sim: Simulation, param: str, key: Node|Edge|Region|ERegion|Parameter) -> Collection[Node|Edge|Parameter]:
	if key == __NODES__:       return chain(sim.nodes, sim.parameters)
	if key == __EDGES__:       return chain(sim.edges, sim.parameters)
	if key in sim.nodes:       return chain([key], sim.parameters)
	if key in sim.edges:       return chain([key], sim.parameters)
	if key in sim.regions:     return chain(sim.regions[key], sim.parameters)
	if key in sim.eregions:    return chain(sim.eregions[key], sim.parameters)
	if key in sim.parameters:  return chain(sim.nodes, sim.edges, [key])
	raise KeyError(f"For parameter {param}, Unrecognized subscript: [{key}]")


class NumericProxy(Proxy, Numeric, ArrangeableMapping):
	def __setitem__(self, key, value) -> None:
		self[key].value = value


# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"states", e.g. n, s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy(NumericProxy):
	def __init__(self,
		simulation: Simulation,
		param: str,
		elements: Collection[Node|Edge],
		filter: Filter,
		to_sim: Callable[[vec|float], numpy_vec|float]=None,  # for getting simulation value from runtime
		to_rt: Callable[[numpy_vec|float], vec|float]=None    # for updating runtime value from simualtion
	):
		super().__init__(simulation, param, elements, filter)
		self._runtime_proxy = getattr(simulation.rt, param)  # Returned proxy should be const. Get once and store.
		self._to_sim = to_sim
		self._to_rt = to_rt
	
	@property
	def _key(self) -> Node|Edge|Region|ERegion:
		return self._subscripts[-1]  # only the last subscript is relevant for prarameter proxies
	
	@override
	@property
	def value(self) -> float|numpy_vec:
		if len(self._subscripts) == 0:
			value = self._runtime_proxy.value  # global parameter
		else:
			value = self._runtime_proxy[self._key]
		return self._to_sim(value)

	@value.setter
	def value(self, value: float|numpy_vec) -> None:
		value = self._to_rt(value)
		if len(self._subscripts) == 0:
			self._runtime_proxy.value = value  # global parameter
		else:
			self._runtime_proxy[self._key] = value
	
	def _get_consistant_value(self, snapshot: Snapshot):
		result = None
		for key in self._elements:
			value = getattr(snapshot, self.name)[key]
			if result is None:     # store first value
				result = value
			elif result != value:  # for global or region selections, make sure all values are equal
				raise KeyError(f"Parameter {self._name} is not consistant across selection from subscript [{self._subscripts}]")
		return result

	@override
	@property
	def history(self) -> dict[int, float|numpy_vec]:
		return ArrangeableDict({ snapshot.t: self._get_consistant_value(snapshot) for snapshot in self._sim._history })


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


def _sum(snapshot: Snapshot, hist_name: str):
	# sum of values from dict: _elements (e.g. nodes) -> values
	return np.sum(np.array([ *getattr(snapshot, hist_name).values() ]), axis=0)

# For spin and flux
class StateProxy(NumericProxy, Vector):
	def __init__(self, sim: Simulation, rt_attr: str, sim_name: str):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)
		self._runtime_proxy = getattr(sim.rt, rt_attr)
	
	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.values(), axis=0)  # add all the row (i.e. axis=0) vectors
		else:
			node = [*self._elements][0]
			return simvec(self._runtime_proxy[node])
	
	@value.setter
	def value(self, value: numpy_vec|vec) -> None:
		if len(self._elements) != 1:
			raise ValueError(f"Can't directly assign to {self._name} for a non-local selection. (Subscript(s): {self._subscripts}.) Instead, select nodes individually and set them one at a time.")
		node = [*self._elements][0]
		self._runtime_proxy[node] = rtvec(value)
	
	@override
	@property
	def history(self) -> dict[int, numpy_vec]:
		return ArrangeableDict({
			snapshot.t: _sum(snapshot, self._name)
			for snapshot in self._sim._history
		})

class MProxy(NumericProxy, Vector):
	def __init__(self, sim: Simulation, sim_name: str="m"):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)

	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.values(), axis=0)  # add all the row (i.e. axis=0) vectors
		else:
			runtime = self._sim.rt
			node = [*self._elements][0]
			return simvec(runtime.spin[node]) + simvec(runtime.flux[node])
	
	@override
	@property
	def history(self) -> dict[int, numpy_vec]:
		return ArrangeableDict({ snapshot.t: _sum(snapshot, self._name) for snapshot in self._sim._history })

# Number of nodes in selection
class NProxy(NumericProxy, IInt):
	def __init__(self, sim: Simulation, sim_name: str="n"):
		super().__init__(sim, sim_name, elements=ordered_set(chain(sim.nodes, sim.edges)), filter=_ntype_filter)
	
	@override
	@property
	def value(self) -> int:
		return len(self._elements)
	
	@override
	@property
	def history(self) -> dict[int, int]:
		# n can't change over time
		return ArrangeableDict({ snapshot.t: self.value for snapshot in self._sim._history }, dtype=int)

# Internal Energy in selection
class UProxy(NumericProxy, Scalar):
	def __init__(self, sim: Simulation, sim_name="u"):
		parameters = ordered_set(sim.parameters)
		parameters.pop("S", None)  # these parameters don't contribute to energy
		parameters.pop("F", None)
		super().__init__(sim, sim_name, elements=ordered_set(chain(sim.nodes, sim.edges, parameters)), filter=_utype_filter)
	
	@override
	@property
	def value(self) -> float:
		sim = self._sim
		
		# separate selected nodes, edges, and parameters in elements
		nodes = []
		edges = []
		parameters = set()
		for x in self._elements:
			if x in sim.nodes:
				nodes.append(x)
			elif x in sim.edges:
				edges.append(x)
			else:
				assert x in sim.parameters  # DEBUG
				parameters.add(x)
		
		u = 0.0     # Result: internal energy

		if len(nodes) != 0:
			m = None    # cache (node parameters: B, A)

			if "B" in parameters:
				m = np.array([sim.m[i] for i in nodes], dtype=float)
				B = np.array([sim.B[i] for i in nodes], dtype=float)
				u -= np.sum(B * m)
			
			if "A" in parameters:
				if m is None:  m = np.array([sim.m[i] for i in nodes], dtype=float)
				A = np.array([sim.A[i] for i in nodes], dtype=float)
				u -= np.sum(A * (m * m))
			
			if "Je0" in parameters:
				s = np.array([sim.s[i] for i in nodes], dtype=float)
				f = np.array([sim.f[i] for i in nodes], dtype=float)
				Je0 = np.array([sim.Je0[i] for i in nodes], dtype=float)
				u -= np.sum(Je0 * np.sum(s * f, axis=1), axis=0)  # sum( Je0[i] * (s[i] @ f[i]) )

		if len(edges) != 0:
			s_i = None  # cache (edge parameters: J, Je1)
			s_j = None  # cache (edge parameters: J, Je1)
			f_i = None  # cache (edge parameters: Je1, Jee)
			f_j = None  # cache (edge parameters: Je1, Jee)
			m_i = None  # cache (edge parameters: b, D)
			m_j = None  # cache (edge parameters: b, D)

			if "J" in parameters:
				s_i = np.array([sim.s[i] for i, _ in edges], dtype=float)
				s_j = np.array([sim.s[j] for _, j in edges], dtype=float)
				J = np.array([sim.J[e] for e in edges], dtype=float)
				u -= np.sum(J * np.sum(s_i * s_j, axis=1), axis=0)  # sum( J * (s[i] @ s[j]) )

			if "Je1" in parameters:
				if s_i is None:  s_i = np.array([sim.s[i] for i, _ in edges], dtype=float)
				if s_j is None:  s_j = np.array([sim.s[j] for _, j in edges], dtype=float)
				f_i = np.array([sim.f[i] for i, _ in edges], dtype=float)
				f_j = np.array([sim.f[j] for _, j in edges], dtype=float)
				Je1 = np.array([sim.Je1[e] for e in edges], dtype=float)
				u -= np.sum(Je1 * (np.sum(s_i * f_j, axis=1) + np.sum(s_j * f_i, axis=1)), axis=0)  # sum( Je1 * (s[i] @ f[j] + s[j] @ f[i]) )

			if "Jee" in parameters:
				if f_i is None:  f_i = np.array([sim.f[i] for i, _ in edges], dtype=float)
				if f_j is None:  f_j = np.array([sim.f[j] for _, j in edges], dtype=float)
				Jee = np.array([sim.Jee[e] for e in edges], dtype=float)
				u -= np.sum(Jee * np.sum(f_i * f_j, axis=1), axis=0)  # sum( Jee * (f[i] @ f[j]) )

			if "b" in parameters:
				m_i = np.array([sim.m[i] for i, _ in edges], dtype=float)
				m_j = np.array([sim.m[j] for _, j in edges], dtype=float)
				b = np.array([sim.b[e] for e in edges], dtype=float)
				dotp = np.sum(m_i * m_j, axis=1)
				u -= np.sum(b * (dotp * dotp), axis=0)  # sum( b * (m[i] @ m[j])**2 )

			if "D" in parameters:
				if m_i is None:  m_i = np.array([sim.m[i] for i, _ in edges], dtype=float)
				if m_j is None:  m_j = np.array([sim.m[j] for _, j in edges], dtype=float)
				D = np.array([sim.D[e] for e in edges], dtype=float)
				u -= np.sum(D * np.cross(m_i, m_j))

		return float(u)

	def _keys(self) -> Collection[Node|Edge]:
		sim = self._sim
		nodes = sim.nodes
		edges = sim.edges
		keys = ordered_set(self._elements)
		for k in self._elements:
			if k not in chain(nodes, edges):
				keys.pop(k, None)  # discard parameter elements, leaving only Collection[Node|Edge]
		return keys

	@override
	def __iter__(self) -> Iterator[Node|Edge]:
		return iter(self._keys())
	
	@override
	@property
	def history(self) -> dict[int, float]:
		return ArrangeableDict({ snapshot.t: _sum(snapshot, self._name) for snapshot in self._sim._history })
