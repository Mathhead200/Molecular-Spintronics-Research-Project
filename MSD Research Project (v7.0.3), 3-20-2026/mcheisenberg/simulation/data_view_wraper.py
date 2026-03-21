from __future__ import annotations
from ..runtime import DataView
from ..util import ordered_set, ReadOnlyOrderedSet, ReadOnlyDict, PARAMETERS
from .simulation_proxies import *
from .from_config import from_config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..driver import Driver
	from ..runtime import MutableStateBuffer
	from .from_config import GrowableBuffer
	from .simulation_util import Parameter
	from collections.abc import Iterable
	from numpy.typing import NDArray

STATES = ordered_set(["n", "s", "f", "m", "u"])
ALL_PROXIES = ordered_set(chain(PARAMETERS, STATES))

class ReadyBuffers:
	s:   NDArray = None
	f:   NDArray = None
	B:   NDArray = None
	A:   NDArray = None
	S:   NDArray = None
	F:   NDArray = None
	kT:  NDArray = None
	Je0: NDArray = None
	J:   NDArray = None
	Je1: NDArray = None
	Jee: NDArray = None
	b:   NDArray = None
	D:   NDArray = None
	n:   NDArray = None
	m:   NDArray = None
	u:   NDArray = None

	mat_node:  GrowableBuffer
	mat_node2: GrowableBuffer
	list_node: GrowableBuffer
	mat_edge:  GrowableBuffer
	list_edge: GrowableBuffer
	s_i: GrowableBuffer
	s_j: GrowableBuffer
	f_i: GrowableBuffer
	f_j: GrowableBuffer
	m_i: GrowableBuffer
	m_j: GrowableBuffer

	def __init__(self, node_count: int, edge_count: int):
		self._node_count = node_count
		self._edge_count = edge_count
	
	def ensure(self, p_set: Collection[Parameter]):
		n = self._node_count
		m = self._edge_count
		for p in ["s", "f", "B", "A", "m"]:
			# e.g. if "s" in p_set and self.s is None:  self.s = np.empty(shape=(n, 3), dtype=float); etc.
			if p in p_set and getattr(self, p) is None:  setattr(self, p, np.empty(shape=(n, 3), dtype=float))
		for p in ["S", "F", "kT", "Je0"]:
			# e.g. if "S" in p_set and self.S is None:  self.S = np.empty(shape=(n,), dtype=float); etc.
			if p in p_set and getattr(self, p) is None:  setattr(self, p, np.empty(shape=(n,), dtype=float))
		for p in ["J", "Je1", "Jee", "b"]:
			# e.g. if "J" in p_set and self.J is None:  self.J = np.empty(shape=(m,), dtype=float)
			if p in p_set and getattr(self, p) is None:  setattr(self, p, np.empty(shape=(m,), dtype=float))
		if "n" in p_set and self.n is None:  self.n = np.ones(shape=(m + n,), dtype=int)
		if "D" in p_set and self.D is None:  self.D = np.empty(shape=(m,3),  dtype=float)
		if "u" in p_set and self.u is None:  self.u = np.empty(shape=(n + m,), dtype=float)
	
	def flag(self, params: Iterable[Parameter], **flags):
		for p in params:
			arr = getattr(self, p)
			if arr is not None:
				for k, v in flags.items():
					setattr(arr.flags, k, v)


class DataViewWrapper[T=Driver|MutableStateBuffer]:
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

	nodes:      ReadOnlyOrderedSet[Node]
	edges:      ReadOnlyOrderedSet[Edge]
	regions:    ReadOnlyDict[Region, ReadOnlyOrderedSet[Node]]
	eregions:   ReadOnlyDict[ERegion, ReadOnlyOrderedSet[Edge]]
	parameters: ReadOnlyDict[Parameter, ReadOnlyOrderedSet[Node]|ReadOnlyOrderedSet[Edge]]

	def __init__(self, view: DataView[T]):
		self.view = view

		config_data = from_config(view.config)  # employs caching
		for attr in config_data.__slots__:
			setattr(self, attr, getattr(config_data, attr))  # e.g. self._nodes = config_data.nodes; etc.

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

		buf = ReadyBuffers(node_count=len(config_data.nodes), edge_count=len(config_data.edges))
		for attr in ["mat_node", "mat_node2", "list_node", "mat_edge", "list_edge", "s_i", "s_j", "f_i", "f_j", "m_i", "m_j"]:
			setattr(buf, attr, getattr(config_data, f"buf_{attr}"))  # e.g. buf.mat_node = config_data.buf_mat_node; etc.
		self._ready_buffers = buf
		self._ready_cache: dict[Parameter, NDArray] = {}

	def ready(self, *params: Parameter) -> None:
		"""
		Pre-calculates full values for given paramenters. e.g. self.ready("s", "f", "m", "u")
		Evaluations will be done efficiently and using preallocated-buffers.
		Caller responsability:
			If changes are made to the underlying state, these buffers must be
			invalidated or udated manually, otherwise future calls to Proxies
			**may** yeild previously readied values, and not the updated values.
			Cache can be manually cleared with self.unready(), or calling
			self.ready(*p) again after changes (with the desired parameters, *p)
			will update the cache as well, but only for *p.
		"""
		self.unready(*params)
		if len(params) == 0:  params = ALL_PROXIES
		p_set = set(params)  # params, but optimized for membership testing
		buf = self._ready_buffers
		buf.flag(params, writeable=True)
		buf.ensure(p_set)
		cache = self._ready_cache
		get = lambda p: cache.get(p, None)
		for p in ["s", "f", "B", "A", "S", "F", "kT", "Je0", "J", "Je1", "Jee", "b", "D"]:  # independent proxies
			if p in p_set:  cache[p] = getattr(self, p).values(getattr(buf, p))  # e.g. if "s" in p_set:  cache["s"] = self.s.values(buf.s); etc.
		# Note: don't need to build n since it's all ones. Already initialized in ReadyBuffer.ensure("n", ...).
		if "m" in p_set:  cache["m"] = self.m.values(buf.m, s=get("s"), f=get("f"))
		if "u" in p_set:  cache["u"] = self.u.values(buf.u,
			s=get("s"), f=get("f"), m=get("m"),
			B=get("B"), A=get("A"), Je0=get("Je0"),
			J=get("J"), Je1=get("Je1"), Jee=get("Jee"), b=get("b"), D=get("D"))
		buf.flag(params, writeable=False)  # So they can be returned from Proxy.values() without worrying about accidental modification
	
	def unready(self, *params: Parameter) -> None:
		if len(params) == 0:
			self._ready_cache = {}
		else:
			cache = self._ready_cache
			for p in params:
				if p in cache:
					del cache[p]

	# @properties added below: Simulation.s, .f, .m, .u, .n, .J, .B, etc.

	def __getitem__(self, attr: str):
		if attr not in ALL_PROXIES:
			raise KeyError(f"{attr} is an unrecognized parameter or state")
		return getattr(self, attr)

for param in ALL_PROXIES:
	setattr(DataViewWrapper, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
