from __future__ import annotations
from ..util import PARAMETERS
from .data_proxies import *
from .buffers import MutableStateBuffer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..config import Config, vec
	from ..driver import Driver
	from collections.abc import Sequence, Mapping


class Data[T=Driver|MutableStateBuffer]:  # TODO: rename?
	""" A common interface for accessing Runtime-style data from either a Driver or a MutablestateBuffer. """
	def __init__(self, config: Config, source: T):
		self.config: Config = config
		self.source: T      = source
		self._globals_proxy = GlobalsProxy(self)
		self._node_list_proxy    = NodeListProxy(self)
		self._edge_list_proxy    = EdgeListProxy(self)
		self._region_list_proxy  = RegionListProxy(self)
		self._eregion_list_proxy = ERegionListProxy(self)
		self._spin_list_proxy    = StateListProxy(self, "spin")
		self._flux_list_proxy    = StateListProxy(self, "flux")
		for param in ["A", "B", "D"]:
			setattr(self, f"_{param}_proxy", VectorParameterProxy(self, param))  # e.g. self._B_proxy
		for param in ["S", "F", "kT", "Je0", "J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarParameterProxy(self, param))  # e.g. self._J_proxy

	@property
	def node(self) -> Mapping:
		return self._node_list_proxy

	@property
	def spin(self) -> Mapping:
		return self._spin_list_proxy

	@property
	def flux(self) -> Mapping:
		return self._flux_list_proxy

	@property
	def edge(self) -> Mapping:
		return self._edge_list_proxy

	@property
	def region(self) -> Mapping:
		return self._region_list_proxy
	
	@property
	def eregion(self):
		return self._eregion_list_proxy

	def __getitem__(self, region):
		if type(region) is tuple:
			return self.eregion[region]
		else:
			return self.region[region]

for param in PARAMETERS:
	setattr(Data, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(self._globals_proxy, _p, value)
	))
