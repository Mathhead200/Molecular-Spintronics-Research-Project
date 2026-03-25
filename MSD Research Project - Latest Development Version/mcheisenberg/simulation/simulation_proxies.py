from __future__ import annotations
from ..runtime import scal_in, scal_out, vec_in, vec_out
from ..util import NODES_, EDGES_, ReadOnlyList, ReadOnlyCollection, OrderedSet, ordered_set, IInt
from .simulation_util import NumericArrangeable, Vector, Scalar, simvec, simscal, rtvec, rtscal, dot, _cov_list, _var_list, \
	Node, Edge, Region, ERegion, Parameter, Literal, numpy_vec, numpy_mat, numpy_list, numpy_sq
from collections.abc import KeysView
from copy import copy
from itertools import chain
from typing import override, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	from .simulation import Simulation, Snapshot
	from .data_view_wraper import DataViewWrapper
	from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
	from typing import Any, Annotated
	from numpy.typing import NDArray

type Filter = Callable[[DataViewWrapper, str, Any], Iterable]  # (Proxy._data, Proxy.name, key) -> candidate_elements

class ProxyKeysView(KeysView):
	def __repr__(self) -> str:
		return f"{ self.__class__.__name__ }({ list(self) })"

class _Proxy_get[E, K, V]:
	def __init__(self, proxy: Proxy[E, K, V]):
		self._obj = proxy

	def __call__(self, elements: Iterable[E], key=None, permuted: bool=True) -> Proxy[E, K, V]:
		""" Wrapper extending functionality to general Iterables, e.g. generators. """
		return self._obj.sub(list(elements), key=(elements if key is None else key), permuted=permuted)
	
	def __getitem__(self, elements) -> Proxy[E, K, V]:
		""" Wrapper extending functionality to slices. """
		if isinstance(elements, slice):
			a, b = elements.start, elements.stop
			if any(bound is None for bound in [a, b]):
				raise ValueError(f"Proxy slice must include both a start and stop: get[{a}:{b}]")
			if any(not isinstance(bound, int) for bound in [a, b]):
				raise ValueError(f"Proxy slice currently only supports int type: get[{type(a).__name__} {a}:{type(b).__name__} {b}]")
			if elements.step is not None:
				return self._obj.sub(range(a, b, elements.step), key=elements)
			return self._obj.sub(range(a, b), key=elements)
		if isinstance(elements, tuple):
			return self(elements)   # shorthand syntax: proxy.get([1, 2, 3]) becomes proxy.get[1, 2, 3]
		return self._obj[elements]  # treat as singular element as redirect to core self._obj.__getitem__ behaviour

class Proxy[E, K, V]:
	def __init__(self, data: DataViewWrapper, sim_name: str, elements: Collection[E], filter: Filter, subscripts: list=[], parent: Proxy=None):
		self._data: DataViewWrapper = data
		self._name = sim_name
		self._elements: Collection[E] = elements
		self._filter = filter
		self._subscripts: list = subscripts
		self._get = _Proxy_get(self)
		self._parent = parent  # None for root
		self._root = self  # (const) not modified by subscripting
		self._permuted: bool = False  # is the relative order of self.elements the same as self.root.elements *and* is it a subset?
	
	def __len__(self) -> int:             return len(self._elements)
	def __iter__(self) -> Iterator[E]:    return iter(self._elements)
	def __contains__(self, key) -> bool:  return key in self._elements

	def sub(self, elements: Collection[E], key=None, permuted: bool=True) -> Proxy[E, K, V]:
		clone = copy(self)
		clone._elements = elements
		clone._subscripts = self._subscripts + [elements if key is None else key]
		clone._parent = self
		clone._permuted = permuted
		return clone

	def __getitem__(self, key: K) -> Proxy[E, K, V]:
		candidate_elements = self._filter(self._data, self._name, key)
		elements = { x: None for x in candidate_elements if x in self._elements }  # (ordered set) itersection of candidate_elements and self._elements
		# if len(elements) == 0:
		# 	raise ValueError(f"{key} is valid by itself, but disjoint with previous subscripts: {self._subscripts}")
		return self.sub(elements, key=key, permuted=False)
	
	def assert_unpermuted(self) -> Proxy:
		"""
		Can be used after a call to self.get[], e.g. self.get[1:10:2].assert_unpermuted().values()
		to turn on Proxy._permuted=False optimizations when using DataViewWrapper.ready().
		Since unpermuted elements (relative to the root order), can be much more quickly aggrigated.
		In general, Proxy.sub() Proxy.get(), and Proxy.get[] can not assume an unpermuted element order.
		Proxy[] can, and does by default.
		"""
		self._permuted = False
		return self

	@ property
	def get(self) -> _Proxy_get[E, K, V]:
		"""
			Wrapper extending functionality to general Iterables, e.g. generators; and slices.
			Call like function for Iterables. Index with slice like list for slices. e.g.
				proxy.get(range(1, 11))
				proxy.get[1:11]
			Slices can not omit start or stop becasue the simulation makes no assumptions about (int) nodes being contiguous.
			Slice syntax can also be used to get specific elements, e.g. proxy.get[n0, n1, n2]
		"""
		return self._get

	@property
	def name(self) -> str:
		return self._name

	@property
	def elements(self) -> Collection[E]:
		return ReadOnlyCollection(self._elements)

	@property
	def subscripts(self) -> Sequence[K]:
		return ReadOnlyList(self._subscripts)
	
	@property
	def parent(self) -> Proxy:
		return self._parent
	
	@property
	def root(self) -> Proxy:
		return self._root

	def keys(self) -> KeysView:
		return ProxyKeysView(self)

class NumericProxy[E, K, V](Proxy[E, K, V], NumericArrangeable):
	def __setitem__(self, key: K, value: V) -> None:
		self[key].value = value

def _node_filter(data: DataViewWrapper, param: str, key: Node|Region) -> Collection[Node]:
	if key in data.nodes:    return [key]              # key is interpreted a (local) node
	if key in data.regions:  return data.regions[key]  # key is interpreted as a (node) region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not a node nor region in Config")

def _edge_filter(data: DataViewWrapper, param: str, key: Edge|ERegion) -> Collection[Edge]:
	if key in data.edges:     return [key]               # key is interpreted as a (local) edge
	if key in data.eregions:  return data.eregions[key]  # key is interpreted as an edge-region
	raise KeyError(f"Subscript [{key}] is undefined for parameter {param}: not an edge nor edge-region in Config")

# Allow parameter (and literal) keys, but only stores Node|Edge keys as elements
def _ntype_filter(data: DataViewWrapper, param: str, key: Node|Edge|Region|ERegion|Parameter|Literal) -> Collection[Node|Edge]:
	if key == NODES_:       return data.nodes  # interpreted as the set of all nodes
	if key == EDGES_:       return data.edges  # interpreted as the set of all edges
	if key in data.nodes:       return [key]      # interpreted as a (local) node
	if key in data.edges:       return [key]      # interpreted as a (local) edge
	if key in data.regions:     return data.regions[key]     # interpreted as a (node) region
	if key in data.eregions:    return data.eregions[key]    # interpreted as an edge-region
	if key in data.parameters:  return data.parameters[key]  # interpreted as a (node or edge) parameter
	raise KeyError(f"For parameter {param}, undefined subscript: [{key}]")

# Allows parameter (and literal) keys, and stores Node|Edge|Parameter keys are elements
def _utype_filter(data: DataViewWrapper, param: str, key: Node|Edge|Region|ERegion|Parameter|Literal) -> Iterable[Node|Edge|Parameter]:
	if key == NODES_:       return chain(data.nodes, data.parameters)
	if key == EDGES_:       return chain(data.edges, data.parameters)
	if key in data.nodes:       return chain([key], data.parameters)
	if key in data.edges:       return chain([key], data.parameters)
	if key in data.regions:     return chain(data.regions[key], data.parameters)
	if key in data.eregions:    return chain(data.eregions[key], data.parameters)
	if key in data.parameters:  return chain(data.nodes, data.edges, [key])
	raise KeyError(f"For parameter {param}, Unrecognized subscript: [{key}]")

def _history_filter(sim: Simulation, param: str, key: int|slice|tuple) -> Iterable[int]:
	if isinstance(key, int):    return [key]
	if isinstance(key, slice):  return key.indices(len(sim.history))
	if isinstance(key, tuple):  return key
	raise KeyError(f"For parameter {param}, History subscript must be int, slice, or tuple: {type(key).__name__} [{key}]")


class HistoryProxy[H](NumericProxy[int, int|slice|tuple, H]):
	_data: Simulation

	def __init__(self, data: DataViewWrapper, sim_name: str, evaluate: Callable[[Snapshot], H]):
		if not hasattr(data, "history"):  # if not isinstance(data, Simulation):
			raise ValueError(f"History only exists for Simulations: {type(data)}")
		super().__init__(data, sim_name, elements=ordered_set(data.history), filter=_history_filter)
		self._evaluate = evaluate
	
	@override
	@property
	def value(self) -> H:
		if len(self._elements) != 1:
			raise ValueError(f"History can not be evaluated unless fully specified: len(indices)={len(self._elements)}")
		t = next(iter(self._elements))  # self._elements may not be any Collection type (e.g. ordered_set)
		return self._evaluate(self._data.history[t])

class Historical[H]:
	@property
	def history(self) -> HistoryProxy[H]:
		raise NotImplementedError()  # abstract

class HistoricalNumericProxy[E, K, V](NumericProxy[E, K, V], Historical[V]):  pass


# Proxy for parameters, e.g. J, B, kT, n, etc. Parameters do not aggrigate like
#	"states", e.g. n, s, f, m, u. Instead they access location specific
#	information, e.g. J["FML"] would get the J value used in region "FML", and
#	not the sum of J values used across the region.
class ParameterProxy[E: Node|Edge, K: Node|Edge|Region|ERegion, V: numpy_vec|float, R_in: vec_in|scal_in, R_out: vec_out|scal_out](HistoricalNumericProxy[E, K, V]):
	def __init__(self,
		data: DataViewWrapper,
		param: str,
		elements: Collection[E],
		filter: Filter,
		to_sim: Callable[[R_out], V],     # for getting simulation value from runtime
		to_rt: Callable[[V], R_in],       # for updating runtime value from simualtion
		shape: Callable[[int], tuple]  # for values shape (given len(_elements))
	):
		super().__init__(data, param, elements, filter)
		self._runtime_proxy = getattr(data.view, param)  # Returned proxy should be const(-esk). Get once and store.
		self._to_sim = to_sim
		self._to_rt = to_rt
		self._shape = shape
	
	@property
	def _key(self) -> K:
		return self._subscripts[-1]  # only the last subscript is relevant for prarameter proxies
	
	@override
	@property
	def value(self) -> V:
		if len(self._subscripts) == 0:
			value: R_out = self._runtime_proxy.value  # global parameter
		else:
			value: R_out = self._runtime_proxy[self._key]
		return self._to_sim(value)

	@value.setter
	def value(self, value: V) -> None:
		value: R_in = self._to_rt(value)
		if len(self._subscripts) == 0:
			self._runtime_proxy.value = value  # global parameter
		else:
			self._runtime_proxy[self._key] = value
	
	@override
	def values(self, out: NDArray) -> NDArray:
		values = self._data._ready_cache.get(self._name, None)
		if values is not None and self._parent is None:
			return values  # Note: out is not used/modified in this case
		# Don't bother using the buffer if we are in a sub-proxy since it's more
		#	work to build the dict then to just read these values from memory directly.
		if out is None:
			out = np.empty(self._shape(len(keys)), dtype=float)
		_mat = out.ndim > 1  # bool. e.g. parameter D would be shape=(3, M), _mat=True; J would be (M,), False; and B would be (3, N), True
		keys = self._elements
		for idx, key in enumerate(keys):
			self._to_sim(self._runtime_proxy[key], out=(out[idx] if _mat else out[idx:idx+1]))
		return out
	
	def _get_consistant_value(self, snapshot: Snapshot) -> V:
		result = None
		for key in self._elements:
			value: V = getattr(snapshot, self.name)[key]
			if result is None:     # store first value
				result = value
			elif result != value:  # for global or region selections, make sure all values are equal
				raise KeyError(f"Parameter {self._name} is not consistant across selection from subscript [{self._subscripts}]")
		return result
	
	@override
	@property
	def history(self) -> HistoryProxy[V]:
		return HistoryProxy(self._data, self._name, lambda snapshot: self._get_consistant_value(snapshot))


class VectorNodeParameterProxy(ParameterProxy[Node, Node|Region, numpy_vec, vec_in, vec_out], Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simvec, to_rt=rtvec, shape=lambda n: (n, 3))

class ScalarNodeParameterProxy(ParameterProxy[Node, Node|Region, float, scal_in, scal_out], Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.nodes, filter=_node_filter, to_sim=simscal, to_rt=rtscal, shape=lambda n: (n,))

class VectorEdgeParameterProxy(ParameterProxy[Edge, Edge|ERegion, numpy_vec, vec_in, vec_out], Vector):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simvec, to_rt=rtvec, shape=lambda n: (n, 3))

class ScalarEdgeParameterProxy(ParameterProxy[Edge, Edge|ERegion, float, scal_in, scal_out], Scalar):
	def __init__(self, sim: Simulation, param: str):
		super().__init__(sim, param, elements=sim.edges, filter=_edge_filter, to_sim=simscal, to_rt=rtscal, shape=lambda n: (n,))


def _sum(proxy: Proxy, snapshot: Snapshot):
		# sum of values from dict: _elements (e.g. nodes) -> values
		return np.sum(getattr(snapshot, proxy._name).values(None), axis=0)

class SumProxy[E, K, H](HistoricalNumericProxy[E, K, H]):
	@override
	@property
	def history(self) -> HistoryProxy[numpy_vec]:
		return HistoryProxy(self._data, self._name, lambda snapshot: _sum(self, snapshot))

# For spin and flux
class StateProxy(SumProxy[Node, Node|Region, numpy_vec], Vector):
	def __init__(self, data: DataViewWrapper, rt_attr: str, sim_name: str):
		super().__init__(data, sim_name, elements=data.nodes, filter=_node_filter)
		self._runtime_proxy = getattr(data.view, rt_attr)
	
	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.values(), axis=0)  # add all the row (i.e. axis=0) vectors
		else:
			node = [*self._elements][0]
			return simvec(self._runtime_proxy[node])
	
	@value.setter
	def value(self, value: numpy_vec) -> None:
		if len(self._elements) != 1:
			raise ValueError(f"Can't directly assign to {self._name} for a non-local selection. (Subscript(s): {self._subscripts}.) Instead, select nodes individually and set them one at a time.")
		node = [*self._elements][0]
		self._runtime_proxy[node] = rtvec(value)
	
	@override
	def values(self, out: NDArray=None) -> numpy_mat:
		values = self._data._ready_cache.get(self._name, None)
		if values is not None and self._parent is None:
			return values  # Note: out is not used/modified in this case
		# Don't bother using the buffer if we are in a sub-proxy since it's more
		#	work to build the dict then to just read these values from memory directly.
		nodes = self._elements
		if out is None:
			out = np.empty((len(nodes), 3), dtype=float)
		for idx, i in enumerate(nodes):
			simvec(self._runtime_proxy[i], out=out[idx])  # copy rt values directly into row view of pre-allocated output matrix
		return out

class MProxy(SumProxy[Node, Node|Region, numpy_vec], Vector):
	def __init__(self, data: DataViewWrapper, sim_name: str="m"):
		super().__init__(data, sim_name, elements=data.nodes, filter=_node_filter)

	@override
	@property
	def value(self) -> numpy_vec:
		if len(self._elements) > 1:
			return np.sum(self.values(), axis=0)  # add all the row (i.e. axis=0) vectors
		else:
			view = self._data.view
			node = [*self._elements][0]
			return simvec(view.spin[node]) + simvec(view.flux[node])
	
	@override
	def values(self, out: NDArray=None, s: NDArray=None, f: NDArray=None) -> numpy_mat:
		values = self._data._ready_cache.get("m", None)
		if values is not None and self._parent is None:
			return values  # Note: out is not used/modified in this case
		
		nodes = self._elements
		if out is None:
			out = np.empty((len(nodes), 3), dtype=float)
		if values is not None:
			index_of_node: dict[Node, int] = self._data.view.config.nodeIndex
			for idx, n in enumerate(nodes):
				out[idx] = values[index_of_node[n]]

		data = self._data
		if s is None:  s = data.s.sub(nodes).values(out)
		if f is None:  f = data.f.sub(nodes).values(data._ready_buffers.mat_node[:len(nodes)])
		return np.add(s, f, out=out)

# Number of nodes in selection
class NProxy(HistoricalNumericProxy[Node|Edge, Node|Edge|Region|ERegion, int], IInt):
	def __init__(self, data: DataViewWrapper, sim_name: str="n"):
		super().__init__(data, sim_name, elements=ordered_set(chain(data.nodes, data.edges)), filter=_ntype_filter)
	
	@override
	@property
	def value(self) -> int:
		return len(self._elements)

	@override
	def values(self, out: NDArray=None) -> Annotated[NDArray[np.int64], ("N",)]:
		keys = self._elements
		n = len(keys)
		values = self._data._ready_cache.get("n", None)
		if values is not None:
			return values[:n]  # Note: out is not used/modified in this case
		if out is None:
			out = np.ones((n,), dtype=int)
		else:
			out[:n].fill(1)
		return out
	
	@override
	@property
	def history(self) -> HistoryProxy[int]:
		# n can't change over time
		return HistoryProxy(self._data, self._name, lambda snapshot: self.value)


class UTypeProxy(NumericProxy[Node|Edge|Parameter, Node|Edge|Region|ERegion|Parameter, float], Scalar):
	def __init__(self, data: DataViewWrapper, sim_name: str):
		parameters = data.parameters - ["S", "F"]  # these parameters don't contribute to energy
		super().__init__(data, sim_name, elements=ordered_set(chain(data.nodes, data.edges, parameters)), filter=_utype_filter)
	
	@property
	def parameters(self) -> Collection[str]:
		return self._data.parameters & self._elements
	
	@property
	def nodes(self) -> OrderedSet[Node]:
		return self._elements & self._data.nodes
	
	@property
	def edges(self) -> OrderedSet[Edge]:
		return self._elements & self._data.edges
	
	@override
	def __iter__(self) -> Iterator[Node|Edge]:
		return chain(self.nodes, self.edges)
	
	@override
	def __len__(self) -> int:
		return len(self.nodes) + len(self.edges)

# Internal Energy in selection
class UProxy(UTypeProxy, Historical[float]):
	def __init__(self, data: DataViewWrapper, sim_name: str="u"):
		super().__init__(data, sim_name)
	
	@override
	def __iter__(self) -> Iterator[Node|Edge]:
		return chain(self.nodes, self.edges)

	@override
	@property
	def value(self) -> float:
		return float(np.sum(self.values(), axis=0))

	@override
	def values(self, out: NDArray=None,
		# non-volitile:
		s: numpy_mat=None, f: numpy_mat=None, m: numpy_mat=None,     # pre-calculated s, f, m values
		B: numpy_mat=None, A: numpy_mat=None, Je0: numpy_list=None,  # pre-calculated node parameter
		J: numpy_list=None, Je1: numpy_list=None, Jee: numpy_list=None, b: numpy_list=None, D: numpy_mat=None  # pre-calculated edge parameters
	) -> numpy_list:
		values = self._data._ready_cache.get("u", None)
		if values is not None and self._parent is None:
			return values  # Note: out is not used/modified in this case
		
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

		if values is not None:
			# collect node values
			index_of_node: dict[Node, int] = self._data.view.config.nodeIndex
			for idx, n in enumerate(nodes):
				out[idx] = values[index_of_node[n]]
			# collect edge values
			root = self._root
			zipped_edges: Iterable[tuple[Edge, float]] = zip(root.edges, values[len(root.nodes):])
			if self._permuted:
				# un-optimized: build full edge dict
				value_of_edge: dict[Edge, int] = dict(zipped_edges)
				for idx, e in enumerate(edges, start=N):
					out[N + idx] = value_of_edge[e]
			else:
				# optimized: O(n)
				zip_iter = iter(zipped_edges)
				edge_iter = iter(enumerate(edges, start=N))
				while True:
					try:
						idx, e_target = next(edge_iter)  # edge whos value we need to add, and its index in output array, out
						e_root, u = next(zip_iter)       # root edge to check against, and its cached energy
						while e_target[0] != e_root[0] or e_target[1] != e_root[1]:					
							e_root = next(zip_iter)
						out[idx] = u
					except StopIteration:
						break
			return out

		data = self._data
		parameters = self.parameters
		buf = data._ready_buffers

		if N != 0:
			# m = ...  # is cache (used for calculations: B, A); shape=(N, 3)
			buf_mat = buf.mat_node[:N]
			buf_list = buf.list_node[:N]
			buf_mat2 = buf.mat_node2[:N]

			if "B" in parameters:
				if m is None:  m = data.m.sub(nodes).values(None)  # (new buffer)
				if B is None:  B = data.B.sub(nodes).values(buf_mat)
				out[0:N] -= dot(B, m, temp=buf_mat)
			
			if "A" in parameters:
				if m is None:  m = data.m.sub(nodes).values(None)  # (new buffer)
				if A is None:  A = data.A.sub(nodes).values(buf_mat)
				np.multiply(m, m, out=buf_mat2)  # Hadamard product: [[mx * mx, my * my, mz * mz], ...]
				out[0:N] -= dot(A, m, temp=buf_mat)
			
			if "Je0" in parameters:
				if s is None:  s = data.s.sub(nodes).values(buf_mat)   # (!) don't need m anymore
				if f is None:  f = data.f.sub(nodes).values(buf_mat2)  # (!) don't need f after this
				if Je0 is None:  Je0 = sim.Je0.sub(nodes).values(buf_list)
				out[0:N] -= np.multiply(Je0, dot(s, f, temp=buf_mat), out=buf_list)
		
		if M != 0:
			s_i = None  # cache (used for calculations: J, Je1); shape=(M, 3)
			s_j = None  # cache (used for calculations: J, Je1); shape=(M, 3)
			f_i = None  # cache (used for calculations: Je1, Jee); shape=(M, 3)
			f_j = None  # cache (used for calculations: Je1, Jee); shape=(M, 3)
			m_i = None  # cache (used for calculations: b, D); shape=(M, 3)
			m_j = None  # cache (used for calculations: b, D); shape=(M, 3)
			buf_mat = buf.mat_edge[:M]
			buf_list = buf.list_edge[:M]

			if "J" in parameters:
				s_i = data.s.get(i for i, _ in edges).values(buf.s_i[:M])  # new buffer?
				s_j = data.s.get(j for _, j in edges).values(buf.s_j[:M])  # new buffer?
				if J is None:  J = data.J.sub(edges).values(buf_list)
				out[N:L] -= np.multiply(J, dot(s_i, s_j, temp=buf_mat), out=buf_list)

			if "Je1" in parameters:
				if s_i is None:  s_i = data.s.get(i for i, _ in edges).values(buf.s_i[:M])  # new buffer?
				if s_j is None:  s_j = data.s.get(j for _, j in edges).values(buf.s_j[:M])  # new buffer?
				f_i = data.f.get(i for i, _ in edges).values(buf.f_i[:M])  # new buffer?
				f_j = data.f.get(j for _, j in edges).values(buf.f_j[:M])  # new buffer?
				if Je1 is None:  Je1 = sim.Je1.sub(edges).values(buf_list)
				dotp1 = dot(s_i, f_j, temp=buf_mat)
				dotp2 = dot(f_i, s_j, temp=buf_mat)
				out[N:L] -= np.multiply(Je1, np.add(dotp1, dotp2), out=buf_list)

			if "Jee" in parameters:
				if f_i is None:  f_i = data.f.get(i for i, _ in edges).values(buf.f_i[:M])  # new buffer?
				if f_j is None:  f_j = data.f.get(j for _, j in edges).values(buf.f_j[:M])  # new buffer?
				if Jee is None:  Jee = data.Jee.sub(edges).values(buf_list)
				out[N:L] -= np.multiply(Jee, dot(f_i, f_j, temp=buf_mat), out=buf_list)

			if "b" in parameters:
				m_i = data.m.get(i for i, _ in edges).values(buf.m_i[:M])  # new buffer?
				m_j = data.m.get(j for _, j in edges).values(buf.m_j[:M])  # new buffer?
				if b is None:  b = data.b.sub(edges).values(buf_list)
				out[N:L] -= np.multiply(b, dot(m_i, m_j, temp=buf_mat)**2, out=buf_list)

			if "D" in parameters:
				if m_i is None:  m_i = data.m.get(i for i, _ in edges).values(buf.m_i[:M])  # new buffer?
				if m_j is None:  m_j = data.m.get(j for _, j in edges).values(buf.m_j[:M])  # new buffer?
				if D is None:    D = data.D.sub(edges).values(buf_mat)
				out[N:L] -= dot(D, np.cross(m_i, m_j), temp=buf_mat)

		return out
	
	@override
	@property
	def history(self) -> HistoryProxy[float]:
		return HistoryProxy(self._data, self._name, lambda snapshot: float(_sum(self, snapshot)))  # float(...) converts numpy float to python float

# TODO: revisit these! I don't know if they are correct.
class CProxy(UTypeProxy):
	""" Specific heat """

	_data: Simulation

	def __init__(self, sim: Simulation, sim_name="c"):
		assert hasattr(sim, "history")  # assert isinstance(sim, Simulation)
		super().__init__(sim, sim_name)
	
	@override
	def __iter__(self) -> Iterable[Node]:
		return self.nodes
	
	@override
	@property
	def value(self) -> float:
		sim = self._data
		nodes = self.nodes
		u = sim.u.sub(self._elements).history.values(None)
		kT = sim.kT.sub(nodes).value
		n = len(nodes)
		return _var_list(u, temp=u) / (kT * n)
	
	@override
	@property
	def values(self) -> numpy_list:
		sim = self._data
		parameters = self.parameters
		nodes = self.nodes
		edges = self.edges
		C = np.empty((len(nodes),), dtype=float)
		for idx, node in nodes:
			u = sim.u.sub([node, *parameters]).history.values(None)
			u += 0.5 * sim.u.get(chain((e for e in edges if node in e), parameters)).history.values()
			C[idx] = _var_list(u, temp=u)
		return C

# TODO: revisit these! I don't know if they are correct.
class ChiProxy(NumericProxy[Node, Node|Region, numpy_sq]):
	""" Magnetic susceptibility """

	_data: Simulation

	def __init__(self, sim: Simulation, sim_name="x"):
		assert hasattr(sim, "history")  # assert isinstance(sim, Simulation)
		super().__init__(sim, sim_name, elements=sim.nodes, filter=_node_filter)

	def _calc(self, m_history: numpy_mat, out: NDArray) -> None:
		for alpha in range(3):     # alpha = 0,1,2; representing the (x,y,z)-components of m, respectively
			for beta in range(3):  # beta = ...(x,y,z)...
				out[alpha][beta] = _cov_list(m_history[:, alpha], m_history[:, beta])

	@override
	@property
	def value(self) -> numpy_sq:
		sim = self._data
		nodes = self._elements
		m = sim.m.sub(nodes).history.values(None)
		kT = sim.kT.sub(nodes).value
		n = len(nodes)
		x = np.empty((3, 3), dtype=float)  # tenser (3 by 3 alpha-beta matrix)
		self._calc(m, out=x)
		return np.multiply(1.0/(kT * n), x, out=x)
	
	@override
	def values(self) -> NDArray:  # list of numpy_sq
		sim = self._sim
		nodes = self._elements
		X = np.empty((len(nodes), 3, 3), dtype=float)
		for idx, node in enumerate(nodes):
			node = [node]
			m = sim.m.sub(node).history.array()
			kT = sim.kT.sub(node).value
			x = X[idx, :, :]
			self._calc(m, out=x)
			np.multiply(1.0/kT, x, out=x)
		return X
	