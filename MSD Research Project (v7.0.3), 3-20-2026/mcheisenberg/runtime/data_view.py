from __future__ import annotations
from ..util import PARAMETERS, ReadOnlyList, ReadOnlyCollection
from .runtime_proxies import *
from .buffers import MutableStateBuffer
from ctypes import Structure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..config import Config
	from ..driver import Driver
	from collections.abc import Mapping, Collection
	from typing import Any

class DataView[T=Driver|MutableStateBuffer]:  # TODO: rename?
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
		
		# Not all of these nodes are defined explicitly in the data (i.e. buffer or DLL), but
		#	if we pass any of the returned keys (i.e. nodes, edges, regions, or eregions)
		#	to a proxy (e.g. DataView.node, .edge, .region, .eregion) we will get the logical value
		#	of that parameter for that node, (resp. edge, region, eregion).
		#	So we treat all logical nodes (resp.) as actual nodes.
		self._nodes_view    = ReadOnlyList(config.nodes)
		self._edges_view    = ReadOnlyList(config.edges)
		self._regions_view  = ReadOnlyCollection(config.regions)
		self._eregions_view = ReadOnlyCollection(config.regionCombos)

		# ... If the user needs the actual nodes (resp. edges, regions, edge_regions) defined in the data,
		#	see: DataView.get_instantiated_nodes() (resp.) for keys and DataView.source.nodes (resp.) for values
		
	def get_instantiated_nodes(self) -> Sequence:
		if isinstance(self.source, Driver):
			return ReadOnlyList(self.config.nodes)
		else:
			return ReadOnlyList(self.config.mutableNodes)
		
	def get_instantiated_edges(self) -> Sequence:
		config = self.config
		if config.EDGE_COUNT != 0 and config.SIZEOF_EDGE != 0:
			return ReadOnlyList(config.edges)
		else:
			return ReadOnlyList([])
	
	def get_instantiated_regions(self) -> Sequence:
		config = self.config
		return ReadOnlyList([r for r in config.regions if r in config.regionNodeParameters])
	
	def get_instantiated_edge_regions(self) -> Sequence:
		config = self.config
		return ReadOnlyList([e for e in config.regionCombos if e in config.regionEdgeParameters])
	
	def build_instantiated_node_map(self) -> dict[Any, Structure]:
		return dict(zip(self.get_instantiated_nodes(), self.source.nodes))
	
	def build_instantiated_edge_map(self) -> dict[Any, Structure]:
		return dict(zip(self.get_instantiated_edges(), self.source.edges))
	
	def build_instantiated_region_map(self) -> dict[Any, Structure]:
		return dict(zip(self.get_instantiated_regions(), self.source.regions))
	
	def build_instantiated_edge_region_map(self) -> dict[Any, Structure]:
		return dict(zip(self.get_instantiated_edge_regions(), self.source.edge_regions))

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
	
	@property
	def nodes(self) -> Sequence:
		return self._nodes_view
	
	@property
	def edges(self) -> Sequence:
		return self._edges_view

	@property
	def regions(self) -> Collection:
		return self._regions_view
	
	@property
	def eregions(self) -> Collection:
		return self._eregions_view

for param in PARAMETERS:
	setattr(DataView, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(self._globals_proxy, _p, value)
	))
