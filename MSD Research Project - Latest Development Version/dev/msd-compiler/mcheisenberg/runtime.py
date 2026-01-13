from __future__ import annotations
from .driver import Driver
from collections.abc import Sequence
import os
from typing import TYPE_CHECKING, KeysView
if TYPE_CHECKING:
	from .config import Config, vec

def div8(n: int) -> int:
	return n // 8 if n is not None else None

class NodeProxy:
	def __init__(self, runtime: Runtime, node):
		self._runtime = runtime
		self._node = node
		self._index = runtime.config.nodeIndex[node]
	
	def _getScalar(self, param: str) -> float:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		return runtime._fromNodes(self._index, offset)
	
	def _fgetScalar(self, param: str) -> vec:
		config = self._runtime.config
		if param in config.localNodeParameters[self._node]:
			return self._getScalar(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return ()  # TODO: (stub) from `region`
		return ()  # TODO: (stub) global
	
	def _getVector(self, param: str) -> vec:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		return tuple(runtime._fromNodes(self._index, offset + i) for i in range(3))
	
	def _fgetVector(self, param: str) -> vec:
		config = self._runtime.config
		if param in config.localNodeParameters[self._node]:
			return self._getVector(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return ()  # TODO: (stub) from `region`
		return ()  # TODO: (stub) global

	def _setScalar(self, param: str, index: int, value: float) -> None:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			raise KeyError(f"{param} is not defined (anywhere) in Config")
		runtime._toNodes(self._index, offset, value)
	
	def _fsetScalar(self, param: str):
		config = self._runtime.config
		if param not in config.localNodeParameters[self._node]:
			raise KeyError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")

	def _setVector(self, param: str, value: vec):
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			raise KeyError(f"{param} is not defined (anywhere) in Config")
		for i in range(3):
			runtime._toNodes(self._index, offset + i, value[i])
	
	def _fsetVector(self, param: str, value: vec):
		config = self._runtime.config
		if param not in config.localNodeParameters[self._node]:
			raise KeyError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")
	
for attr in ["spin", "flux"]:
	setattr(NodeProxy, attr, property(
		fget=lambda self,        _a=attr: self._getVector(_a),
		fset=lambda self, value, _a=attr: self._setVector(_a, value)
	))
for param in ["B", "A"]:
	setattr(NodeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetVector(_p),
		fset=lambda self, value, _p=param: self.fsetVector(_p, value)
	))
for param in ["S", "F", "kT", "Je0"]:
	setattr(NodeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetScalar(_p),
		fset=lambda self, value, _p=param: self._fsetScalar(_p, value)
	))

class NodeListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, node) -> NodeProxy:
		try:
			return self._runtime._node_proxies[node]
		except KeyError as ex:
			raise KeyError(f"Node {node} is not defined in Config") from ex

class EdgeProxy:
	pass  # TODO

class EdgeListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, edge: tuple) -> EdgeProxy:
		try:
			return self._runtime._edge_proxies[edge]
		except KeyError as ex:
			raise KeyError(f"Edge {edge} in not defined in Config") from ex
	
	def __call__(self, node0, node1) -> EdgeProxy:
		return self[(node0, node1)]

class RegionProxy:
	pass  # TODO

class RegionListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, region) -> RegionProxy:
		try:
			return self._runtime._region_proxies[region]
		except KeyError as ex:
			raise KeyError(f"Region {region} is not defined in Config") from ex

class ERegionProxy:
	pass  # TODO

class ERegionListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, eregion: tuple) -> ERegionProxy:
		try:
			return self._runtime._eregion_proxies[eregion]
		except KeyError as ex:
			raise KeyError(f"Edge-region {eregion} in not defined in Config") from ex
	
	def __call__(self, region0, region1) -> ERegionProxy:
		return self[(region0, region1)]

class GlobalsProxy:
	pass  # TODO

class SpinListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, node) -> vec:
		return self._runtime.node[node].spin
	
	def __setitem__(self, node, spin: vec):
		self._runtime.node[node].spin = spin

class FluxListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, node) -> vec:
		return self._runtime.node[node].flux
	
	def __setitem__(self, node, flux: vec):
		self._runtime.node[node].flux = flux

class ReadOnlyList(Sequence):
	def __init__(self, lst):    self._lst = lst

	def __len__(self):          return len(self._lst)
	def __getitem__(self, i):   return self._lst[i]
	def __iter__(self):         return iter(self._lst)
	def __contains__(self, v):  return v in self._lst
	# no __setitem__, append, extend, pop, etc. — attempts to mutate via wrapper fail

class Runtime:
	def __init__(self, config: Config, dll: str, delete: bool=False):
		self.dll = dll
		self.config = config
		self.driver = Driver(config, dll)
		self._delete = delete  # delete on exit?
		self._node_len = config.SIZEOF_NODE // 8  # in doubles (8-byte elements)
		self._offsets = {
			"spin": div8(config.OFFSETOF_SPIN),
			"flux": div8(config.OFFSETOF_FLUX),
			"B": div8(config.OFFSETOF_B),
			"A": div8(config.OFFSETOF_A),
			"S": div8(config.OFFSETOF_S),
			"F": div8(config.OFFSETOF_F),
			"kT": div8(config.OFFSETOF_kT),
			"Je0": div8(config.OFFSETOF_Je0)
		}
		self._node_proxies = { node: NodeProxy(self, node) for node in config.nodes }
		self._edge_proxies = { edge: EdgeProxy(self, edge) for edge in config.edges }
		self._region_proxies = { region: RegionProxy(self, region) for region in config.regions.keys() }
		self._eregion_proxies = { eregion: ERegionProxy(self, eregion) for eregion in config.regionCombos }
		self._node_list_proxy = NodeListProxy(self)
		self._spin_list_proxy = SpinListProxy(self)
		self._flux_list_proxy = FluxListProxy(self)
		self._edge_list_proxy = EdgeListProxy(self)
		self._region_list_proxy = RegionListProxy(self)
		self._eregion_list_proxy = ERegionListProxy(self)
		self._globals_proxy = GlobalsProxy(self)
		self._nodes_view = ReadOnlyList(config.nodes)
		self._edges_view = ReadOnlyList(config.edges)
		self._eregions_view = ReadOnlyList(config.regionCombos)
	
	def _fromNodes(self, index: int, offset: int) -> float:
		return self.driver.nodes[index * self._node_len + offset]
	
	def _toNodes(self, index: int, offset: int, value: float) -> None:
		self.driver.nodes[index * self._node_len + offset] = value

	def shutdown(self):
		self.driver.free()
		if self._delete:
			os.remove(self.dll)

	def __enter__(self):
		return self

	def __exit__(self, ex_type, ex_value, traceback):
		self.shutdown()
		return False

	@property
	def nodes(self) -> Sequence:
		return self._nodes_view
	
	@property
	def node(self):
		return self._node_list_proxy

	@property
	def spin(self):
		return self._spin_list_proxy

	@property
	def flux(self):
		return self._flux_list_proxy
	
	@property
	def edges(self) -> Sequence:
		return self._edges_view

	@property
	def edge(self):
		return self._edge_list_proxy
	
	@property
	def regions(self) -> KeysView:
		return self.config.regions.keys()

	@property
	def region(self):
		return self._region_list_proxy
	
	@property
	def eregions(self) -> Sequence:
		return self._eregions_view
	
	@property
	def eregion(self):
		return self._eregion_list_proxy

	@property
	def globals(self) -> dict:
		return { param: getattr(self._globals_proxy, param) for param in self.config.globalParameters.keys() }

	def __getitem__(self, region):
		if type(region) is tuple:
			return self.eregion[region]
		else:
			return self.region[region]

# TODO: add global properties to runtime
