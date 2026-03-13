from __future__ import annotations
from ctypes import cast, sizeof
from . import Node, Region, GlobalNode, Edge, EdgeRegion, GlobalEdge
from ..util import NODE_PARAMETERS, EDGE_PARAMETERS
from .libc import libc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ctypes import c_void_p, Array, Structure
	from ..config import Config

class Buffer:
	def __init__(self, ptr: c_void_p, capacity: int):
		self.ptr = ptr
		self.capacity = capacity
	
	def free(self) -> None:
		libc.free(self.ptr)

	def __len__(self) -> int:
		return self.size

class MutableStateBuffer(Buffer):
	def __init__(self, config: Config, ptr: c_void_p, capacity: int):
		super().__init__(ptr, capacity)
		
		# build scheme from config
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

		for idx, r in enumerate(r for t in config.regions if r in config.regionNodeParameters):
			rid = config.regionId(r)
			rstruct = self._regions[idx]
			setattr(self, rid, property(
				fget=lambda self, _r=rstruct: _r
			))
		for idx, (r0, r1) in enumerate(e for e in config.regionCombos if e in config.regionEdgeParameters):
			erid = f"{config.regionId(r0)}_{config.regionId(r1)}"
			erstruct = self._eregions[idx]
			setattr(self, erid, property(
				fget=lambda self, _r=erstruct: _r
			))
		for p in config.globalKeys:
			if p in NODE_PARAMETERS:
				setattr(self, p, property(
					fget=lambda self,        _p=p: getattr(self._global_node, p),
					fset=lambda self, value, _p=p: setattr(self._global_node, p, value)
				))
			else:
				assert p in EDGE_PARAMETERS  # DEBUG
				setattr(self, p, property(
					fget=lambda self,        _p=p: getattr(self._global_edge, p),
					fset=lambda self, value, _p=p: setattr(self._global_edge, p, value)
				))

	@property
	def nodes(self) -> Array[Structure]:
		return self._nodes
	
	@property
	def regions(self) -> Array[Structure]:
		return self._region_map
	
	@property
	def global_node(self) -> Structure:
		return self._global_node

	@property
	def edges(self) -> Array[Structure]:
		return self._edges
	
	@property
	def eregions(self) -> Array[Structure]:
		return self._eregion_map
	
	@property
	def global_edge(self) -> Structure:
		return self._global_edge
