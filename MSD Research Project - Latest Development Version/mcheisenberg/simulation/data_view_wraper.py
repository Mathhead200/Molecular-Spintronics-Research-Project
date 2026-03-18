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
