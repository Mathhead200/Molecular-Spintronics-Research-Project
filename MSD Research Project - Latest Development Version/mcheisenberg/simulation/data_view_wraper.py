from __future__ import annotations
from ..runtime import DataView
from ..util import ordered_set, ReadOnlyOrderedSet, ReadOnlyDict, PARAMETERS, NODE_PARAMETERS, EDGE_PARAMETERS
from .simulation_proxies import *
from .from_config import from_config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..driver import Driver
	from ..runtime import MutableStateBuffer

STATES = ordered_set(["n", "s", "f", "m", "u"])
ALL_PROXIES = ordered_set(chain(PARAMETERS, STATES))

class ReadyBuffers:
	s   = None
	f   = None
	B   = None
	A   = None
	Je0 = None
	J   = None
	Je1 = None
	Jee = None
	b   = None
	D   = None
	m   = None

	def __init__(self, node_count: int, edge_count: int):
		self._node_count = node_count
		self._edge_count = edge_count
	
	def ensure(self, p_set: Collection[str]):
		n = self._node_count
		m = self._edge_count
		if "s"   in p_set and self.s   is None:  self.s   = np.empty(shape=(n, 3), dtype=float)
		if "f"   in p_set and self.f   is None:  self.f   = np.empty(shape=(n, 3), dtype=float)
		if "B"   in p_set and self.B   is None:  self.B   = np.empty(shape=(n, 3), dtpye=float)
		if "A"   in p_set and self.A   is None:  self.A   = np.empty(shape=(n, 3), dtpye=float)
		if "Je0" in p_set and self.Je0 is None:  self.Je0 = np.empty(shape=(n,),   dtype=float)
		if "J"   in p_set and self.J   is None:  self.J   = np.empty(shape=(m,),   dtype=float)
		if "Je1" in p_set and self.Je1 is None:  self.Je1 = np.empty(shape=(m,),   dtype=float)
		if "Jee" in p_set and self.Jee is None:  self.Jee = np.empty(shape=(m,),   dtype=float)
		if "b"   in p_set and self.b   is None:  self.b   = np.empty(shape=(m,),   dtype=float)
		if "D"   in p_set and self.D   is None:  self.D   = np.empty(shape=(m,3),  dtype=float)
		if "m"   in p_set:
			if self.m is None:                 self.m        = np.empty(shape=(n, 3), dtype=float)
			if not hasattr(self, "mat_node"):  self.mat_node = np.empty(shape=(n, 3), dtype=float)
		if "u"   in p_set:
			if not hasattr(self, "u"):          self.u = np.empty(shape=(n + m,), dtype=float)
			if not hasattr(self, "mat_node"):   self.mat_node   = np.empty(shape=(n, 3), dtype=float)
			if not hasattr(self, "mat_node2"):  self.mat_node2  = np.empty(shape=(n, 3), dtype=float)
			if not hasattr(self, "list_node"):  self.list_node  = np.empty(shape=(n,), dtype=float)
			if not hasattr(self, "mat_edge"):   self.mat_edge   = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "list_edge"):  self.list_edge  = np.empty(shape=(m,), dtype=float)
			if not hasattr(self, "s_i"):  self.s_i = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "s_j"):  self.s_j = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "f_i"):  self.f_i = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "f_j"):  self.f_j = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "m_i"):  self.m_i = np.empty(shape=(m,3), dtype=float)
			if not hasattr(self, "m_j"):  self.m_j = np.empty(shape=(m,3), dtype=float)

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

		self._ready_buffers = ReadyBuffers(node_count=len(config_data.nodes), edge_count=len(config_data.edges))
		self._ready_cache = {}

	def ready(self, *params: set) -> None:
		buf = self._ready_buffers
		buf.ensure(set(params))
		cache = self._ready_cache
		# TODO: update cache
	
	def unready(self) -> None:
		self._ready_cache = {}

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
