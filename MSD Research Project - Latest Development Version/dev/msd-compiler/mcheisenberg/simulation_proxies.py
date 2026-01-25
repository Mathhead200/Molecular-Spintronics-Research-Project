from __future__ import annotations
from .constants import __EDGES__, __NODES__
from .simulation_util import *
from .simulation_util import _cov
from .util import *
from collections.abc import Mapping, MappingView, KeysView, ItemsView, ValuesView
from copy import copy
from itertools import chain
from typing import Any, override, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .config import vec
	from .simulation import Simulation, Snapshot
	from .simulation_util import *
	from collections.abc import Callable, Collection, Iterable, Iterator, Sequence


type Filter = Callable[[Simulation, str, Any], Iterable]  # (Proxy.simulation, Proxy.name, key) -> candidate_elements
class History[H](NumericArrangeable, Mapping[int, H]):  pass

class ProxyItemsView(ItemsView):
	def __repr__(self) -> str:
		return f"{ self.__class__.__name__ }({ list(self) })"

class Proxy[E, K, H]:
	def __init__(self, simulation: Simulation, sim_name: str, elements: Collection[E], filter: Filter, subscripts: list=[]):
		self._sim: Simulation = simulation
		self._name = sim_name
		self._elements: Collection[E] = elements
		self._filter = filter
		self._subscripts: list = subscripts

	@property
	def history(self) -> History[H]:
		raise NotImplementedError()  # abstract
	
	def __len__(self) -> int:             return len(self._elements)
	def __iter__(self) -> Iterator[E]:    return iter(self._elements)
	def __contains__(self, key) -> bool:  return key in self._elements

	def __getitem__(self, key: K) -> Proxy[E, H]:
		proxy = copy(self)
		candidate_elements = self._filter(self._sim, self._name, key)
		proxy._elements = { x: None for x in candidate_elements if x in self._elements }  # (ordered set) itersection of candidate_elements and self._elements
		if len(proxy._elements) == 0:
			raise ValueError(f"{key} is valid by itself, but disjoint with previous subscripts: {self._subscripts}")
		proxy._subscripts = self._subscripts + [key]
		return proxy

	def sub(self, elements: Collection[E]) -> Proxy[E, H]:
		clone = copy(self)
		clone._elements = elements
		return clone

	def get(self, elements: Iterable[E]) -> Proxy[E, H]:
		""" Wrapper extending functionality to general Iterables, e.g. generators. """
		return self.sub([*elements])

	@property
	def name(self) -> str:
		return self._name

	@property
	def elements(self) -> Collection[E]:
		return ReadOnlyCollection(self._elements)

	@property
	def subscripts(self) -> Sequence[K]:
		return ReadOnlyList(self._subscripts)

	def items(self) -> ItemsView:
		return ProxyItemsView(self)

class NumericProxy[E, K, H](Proxy[E, K, H], NumericArrangeable):
	def __setitem__(self, key: K, value: H) -> None:
		self[key].value = value

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


# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"states", e.g. n, s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy[E: Node|Edge, K: Node|Edge|Region|ERegion, H: numpy_vec|float, R: vec|float](NumericProxy[E, K, H]):
	def __init__(self,
		simulation: Simulation,
		param: str,
		elements: Collection[E],
		filter: Filter,
		to_sim: Callable[[R], H],      # for getting simulation value from runtime
		to_rt: Callable[[H], R],       # for updating runtime value from simualtion
		shape: Callable[[int], tuple]  # for values shape (given len(_elements))
	):
		super().__init__(simulation, param, elements, filter)
		self._runtime_proxy = getattr(simulation.rt, param)  # Returned proxy should be const. Get once and store.
		self._to_sim = to_sim
		self._to_rt = to_rt
		self._shape = shape
	
	@property
	def _key(self) -> K:
		return self._subscripts[-1]  # only the last subscript is relevant for prarameter proxies
	
	@override
	@property
	def value(self) -> H:
		if len(self._subscripts) == 0:
			value = self._runtime_proxy.value  # global parameter
		else:
			value = self._runtime_proxy[self._key]
		return self._to_sim(value)

	@value.setter
	def value(self, value: H) -> None:
		value = self._to_rt(value)
		if len(self._subscripts) == 0:
			self._runtime_proxy.value = value  # global parameter
		else:
			self._runtime_proxy[self._key] = value
	
	@override
	def values(self, out: NDArray) -> NDArray:
		keys = self._elements
		if out is None:
			out = np.array(self._shape(len(keys)), dtype=float)
		for idx, key in enumerate(keys):
			out[idx] = self._to_sim(self._runtime_proxy[key])
		return out
	
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
	def history(self) -> ArrangeableDict[int, float|numpy_vec]:
		return ArrangeableDict({ snapshot.t: self._get_consistant_value(snapshot) for snapshot in self._sim._history })


class VectorNodeParameterProxy(ParameterProxy[Node, Node|Region, numpy_vec, vec], Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simvec, to_rt=rtvec, shape=lambda n: (n, 3))

class ScalarNodeParameterProxy(ParameterProxy[Node, Node|Region, float, float], Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simscal, to_rt=rtscal, shape=lambda n: (n,))

class VectorEdgeParameterProxy(ParameterProxy[Edge, Edge|ERegion, numpy_vec, vec], Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simvec, to_rt=rtvec, shape=lambda n: (n, 3))

class ScalarEdgeParameterProxy(ParameterProxy[Edge, Edge|ERegion, float, float], Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simscal, to_rt=rtscal, shape=lambda n: (n,))


def _sum(snapshot: Snapshot, hist_name: str):
	# sum of values from dict: _elements (e.g. nodes) -> values
	return np.sum(np.array([ *getattr(snapshot, hist_name).values() ]), axis=0)

# For spin and flux
class StateProxy(NumericProxy[Node, Node|Region, numpy_vec], Vector):
	def __init__(self, sim: Simulation, rt_attr: str, sim_name: str):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)
		self._runtime_proxy = getattr(sim.rt, rt_attr)
	
	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.array(), axis=0)  # add all the row (i.e. axis=0) vectors
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
	def values(self, out: NDArray=None) -> numpy_mat:
		nodes = self._elements
		if out is None:
			out = np.empty((len(nodes), 3), dtype=float)
		for idx, i in enumerate(nodes):
			out[idx] = simvec(self._runtime_proxy[i])
		return out
	
	@override
	@property
	def history(self) -> ArrangeableDict[int, numpy_vec]:
		return ArrangeableDict({
			snapshot.t: _sum(snapshot, self._name)
			for snapshot in self._sim._history
		})

class MProxy(NumericProxy[Node, Node|Region, numpy_vec], Vector):
	def __init__(self, sim: Simulation, sim_name: str="m"):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)

	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.array(), axis=0)  # add all the row (i.e. axis=0) vectors
		else:
			runtime = self._sim.rt
			node = [*self._elements][0]
			return simvec(runtime.spin[node]) + simvec(runtime.flux[node])
	
	@override
	def values(self, out: NDArray=None) -> numpy_mat:
		nodes = self._elements
		if out is None:
			out = np.empty((len(nodes), 3), dtype=float)
		runtime = self._sim.rt
		for idx, i in enumerate(nodes):
			out[idx] = simvec(runtime.spin[i]) + simvec(runtime.flux[i])
		return out
	
	@override
	@property
	def history(self) -> ArrangeableDict[int, numpy_vec]:
		return ArrangeableDict({ snapshot.t: _sum(snapshot, self._name) for snapshot in self._sim._history })

# Number of nodes in selection
class NProxy(NumericProxy[Node|Edge, Node|Edge, Region|ERegion, int], IInt):
	def __init__(self, sim: Simulation, sim_name: str="n"):
		super().__init__(sim, sim_name, elements=ordered_set(chain(sim.nodes, sim.edges)), filter=_ntype_filter)
	
	@override
	@property
	def value(self) -> int:
		return len(self._elements)

	@override
	def values(self, out: NDArray=None) -> Annotated[NDArray[np.int64], ("N",)]:
		keys = self._elements
		n = len(keys)
		if out is None:
			out = np.ones((n,), dtype=int)
		else:
			out[0:n].fill(1)
		return out
	
	@override
	@property
	def history(self) -> ArrangeableDict[int, int]:
		# n can't change over time
		return ArrangeableDict({ snapshot.t: self.value for snapshot in self._sim._history }, dtype=int)

# Internal Energy in selection
class UProxy(NumericProxy[Node|Edge|Parameter, Node|Edge|Region|ERegion|Parameter, float], Scalar):
	def __init__(self, sim: Simulation, sim_name="u"):
		parameters = sim.parameters - ["S", "F"]  # these parameters don't contribute to energy
		super().__init__(sim, sim_name, elements=ordered_set(chain(sim.nodes, sim.edges, parameters)), filter=_utype_filter)

	@property
	def parameters(self) -> Collection[str]:
		return self._sim.parameters & self._elements
	
	@property
	def nodes(self) -> OrderedSet[Node]:
		return self._elements & self._sim.nodes
	
	@property
	def edges(self) -> OrderedSet[Edge]:
		return self._elements & self._sim.edges
	
	@override
	def __iter__(self) -> Iterator[Node|Edge]:
		return chain(self.nodes, self.edges)
	
	@override
	def __len__(self) -> int:
		return len(self.nodes) + len(self.edges)

	@override
	def values(self, out: NDArray=None) -> numpy_vec:
		sim = self._sim
		parameters = self.parameters
		nodes = self.nodes
		edges = self.edges
		N = len(nodes)
		M = len(edges)
		L = N+M

		# Row vector, u. Each element/column, (scalar) u[key], is the energy for either a node, key=i, or an edges, key=(i,j).
		# Order: first, u[0:N], all nodes in order of sim.nodes. Then, u[N:N+M], all edges in order of sim.edges.
		if out is None:
			out = np.zeros((L,), dtype=float)
		else:
			out[0:L] = 0  # clear (zero) slice of output where results will be calculated and stored

		if N != 0:
			m = None    # cache (used for calculations: B, A); shape=(N, 3)
			buf_mat = np.empty((N, 3), dtype=float)  # temporary buffers
			buf_list = np.empty((N,), dtype=float)

			if "B" in parameters:
				m = sim.m.sub(nodes).array(None)  # (new buffer)
				B = sim.B.sub(nodes).array(buf_mat)
				out[0:N] -= dot(B, m, temp=buf_mat)
			
			if "A" in parameters:
				if m is None:  m = sim.m.sub(nodes).array(None)  # (new buffer)
				A = sim.A.sub(nodes).array(buf_mat)
				np.multiply(m, m, out=m)  # (!) Hadamard product: [[mx * mx, my * my, mz * mz], ...]; don't need m anymore
				out[0:N] -= dot(A, m, temp=buf_mat)
			
			if "Je0" in parameters:
				s = sim.s.sub(nodes).array(m)  # (!) don't need m anymore
				f = sim.f.sub(nodes).array(buf_mat)  # (!) don't need f after this
				Je0 = sim.Je0.sub(nodes).array(buf_list)
				out[0:N] -= np.multiply(Je0, dot(s, f, temp=buf_mat), out=buf_list)
		
		if M != 0:
			s_i = None  # cache (used for calculations: J, Je1); shape=(M, 3)
			s_j = None  # cache (used for calculations: J, Je1); shape=(M, 3)
			f_i = None  # cache (used for calculations: Je1, Jee); shape=(M, 3)
			f_j = None  # cache (used for calculations: Je1, Jee); shape=(M, 3)
			m_i = None  # cache (used for calculations: b, D); shape=(M, 3)
			m_j = None  # cache (used for calculations: b, D); shape=(M, 3)
			buf_mat = np.empty((M, 3), dtype=float)
			buf_list = np.empty((M,), dtype=float)

			if "J" in parameters:
				s_i = sim.s.get(i for i, _ in edges).array(None)  # new buffer
				s_j = sim.s.get(j for _, j in edges).array(None)  # new buffer
				J = sim.J.sub(edges).array(buf_list)
				out[N:L] -= np.multiply(J, dot(s_i, s_j, temp=buf_mat), out=buf_list)

			if "Je1" in parameters:
				if s_i is None:  s_i = sim.s.get(i for i, _ in edges).array(None)  # new buffer
				if s_j is None:  s_j = sim.s.get(j for _, j in edges).array(None)  # new buffer
				f_i = sim.f.get(i for i, _ in edges).array(None)  # new buffer
				f_j = sim.f.get(j for _, j in edges).array(None)  # new buffer
				Je1 = sim.Je1.sub(edges).array(buf_list)
				dotp1 = dot(s_i, f_j, temp=buf_mat)
				dotp2 = dot(f_i, s_j, temp=buf_mat)
				np.add(dotp1, dotp2, out=buf_mat)
				out[N:L] -= np.multiply(Je1, buf_mat, temp=buf_list)

			if "Jee" in parameters:
				if f_i is None:  sim.f.get(i for i, _ in edges).array(None)  # new buffer
				if f_j is None:  sim.f.get(j for _, j in edges).array(None)  # new buffer
				Jee = sim.Jee.sub(edges).array(buf_list)
				out[N:L] -= np.multiply(Jee, dot(f_i, f_j, temp=buf_mat), temp=buf_list)

			if "b" in parameters:
				m_i = sim.m.get(i for i, _ in edges).array(None)  # new buffer
				m_j = sim.m.get(j for _, j in edges).array(None)  # new buffer
				b = sim.b.sub(edges).array(buf_list)
				out[N:L] -= np.multiply(b, dot(m_i, m_j, temp=buf_mat), temp=buf_list)

			if "D" in parameters:
				if m_i is None:  m_i = sim.m.get(i for i, _ in edges).array(None)  # new buffer
				if m_j is None:  m_j = sim.m.get(j for _, j in edges).array(None)  # new buffer
				D = sim.D.sub(edges).array(buf_mat)
				out[N:L] -= dot(D, np.cross(m_i, m_j), temp=buf_mat)

		return out
	
	@override
	@property
	def history(self) -> ArrangeableDict[int, float]:
		return ArrangeableDict({ snapshot.t: float(_sum(snapshot, self._name)) for snapshot in self._sim._history })

class ChiProxy(NumericProxy[Node, Node|Region, numpy_sq], Array):
	def __init__(self, sim: Simulation, sim_name="x"):
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)

	@override
	@property
	def value(self) -> numpy_sq:
		sim = self._sim
		kT = sim.kT.sub(self._elements).value
		m  = sim.m.sub(self._elements).array()  # [[m_x, m_y, m_z], ...]
		x = [[None, None, None], [None, None, None], [None, None, None]]
		for alpha in range(3):     # alpha = 0,1,2; representing the (x,y,z)-components of m, respectively
			for beta in range(3):  # beta = ...(x,y,z)...
				x[alpha][beta] = _cov(m[:, alpha], m[:, beta])
		return (1.0 / kT) * np.array(x)
