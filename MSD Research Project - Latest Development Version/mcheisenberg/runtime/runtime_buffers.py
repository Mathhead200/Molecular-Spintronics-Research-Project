from __future__ import annotations
from ctypes import cast, sizeof
from ..driver import Node, Region, GlobalNode, Edge, EdgeRegion, GlobalEdge
from ..util import NODE_PARAMETERS, EDGE_PARAMETERS
from .libc import libc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ctypes import c_void_p
	from collections.abc import Sequence, Mapping
	from .runtime import Runtime

class Buffer:
	def __init__(self, ptr: c_void_p, capacity: int):
		self.ptr = ptr
		self.capacity = capacity
	
	def free(self) -> None:
		libc.free(self.ptr)

	def __len__(self) -> int:
		return self.size

class MutableStateBuffer(Buffer):
	def __init__(self, rt: Runtime, ptr: c_void_p, capacity: int):
		super().__init__(ptr, capacity)
		
		# build scheme from config
		config = rt.config
		struct_node        = Node(config)
		struct_region      = Region(config)
		struct_global_node = GlobalNode(config)
		struct_edge        = Edge(config)
		struct_edge_region = EdgeRegion(config)
		struct_global_edge = GlobalEdge(config)

		addr = ptr.value  # (int) address
		self._nodes = cast(c_void_p(addr), struct_node * config.MUTABLE_NODE_COUNT)
		addr += sizeof(self._nodes)
		self._regions = cast(c_void_p(addr), struct_region * config.REGION_COUNT)
		addr += sizeof(self._regions)
		self._global_node = cast(c_void_p(addr), struct_global_node)
		addr += sizeof(self._global_node)
		self._edges = cast(c_void_p(addr), struct_edge * config.EDGE_COUNT)
		addr += sizeof(self._edges)
		self._eregions = cast(c_void_p(addr), struct_edge_region * config.EDGE_REGION_COUNT)
		addr += sizeof(self._edge_regions)
		self._global_edge = cast(c_void_p(addr), struct_global_edge)
		addr += sizeof(self._global_edge)

		self.size = addr - ptr.value
		if self.size > capacity:
			raise ValueError(f"Buffer wasn't big enough: capacity={capacity} size={self.size}")

		# dict: (E)Region -> struct alias
		self._region_map  = { r: self._regions[i]  for i, r in enumerate(rt.regions)  }
		self._eregion_map = { e: self._eregions[i] for i, e in enumerate(rt.eregions) }
		self._globals = rt.config.globalKeys

	@property
	def nodes(self) -> Sequence:
		return self._nodes
	
	@property
	def regions(self) -> Mapping:
		return self._region_map
	
	@property
	def edges(self) -> Sequence:
		return self._edges
	
	@property
	def eregions(self) -> Mapping:
		return self._eregion_map

	@property
	def globals(self) -> dict:
		return { p: getattr(self, p) for p in self._globals }

for p in NODE_PARAMETERS:
	setattr(MutableStateBuffer, p, property(
		fget=lambda self,        _p=p: getattr(self._global_node, p),
		fset=lambda self, value, _p=p: setattr(self._global_node, p, value)
	))
for p in EDGE_PARAMETERS:
	setattr(MutableStateBuffer, p, property(
		fget=lambda self,        _p=p: getattr(self._global_edge, p),
		fset=lambda self, value, _p=p: setattr(self._global_edge, p, value)
	))
