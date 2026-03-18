from __future__ import annotations
from ..driver import libc, Node, Region, GlobalNode, Edge, EdgeRegion, GlobalEdge, c_double_3
from ..util import NODE_PARAMETERS, EDGE_PARAMETERS, SCALAR_PARAMETERS, VECTOR_PARAMETERS
from ctypes import sizeof, c_double
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..config import Config
	from ctypes import Array, Structure

NODE_PARAMETER_SET = set(NODE_PARAMETERS)
EDGE_PARAMETER_SET = set(EDGE_PARAMETERS)
SCALAR_PARAMETER_SET = set(SCALAR_PARAMETERS)
VECTOR_PARAMETER_SET = set(VECTOR_PARAMETERS)

class Buffer:
	def __init__(self, ptr: int, capacity: int):
		self.ptr = ptr
		self.capacity = capacity
	
	def free(self) -> None:
		libc.free(self.ptr)

	def __len__(self) -> int:
		return self.size

# Same iterface as Driver: i.e., symbols for each DLL global (e.g. nodes, edges, {rid}, {erid}, and {global_key})
class MutableStateBuffer(Buffer):
	def __init__(self, config: Config, ptr: int, capacity: int):
		super().__init__(ptr, capacity)
		
		# build scheme from config
		struct_node        = Node(config)
		struct_region      = Region(config)
		struct_global_node = GlobalNode(config)
		struct_edge        = Edge(config)
		struct_edge_region = EdgeRegion(config)
		struct_global_edge = GlobalEdge(config)

		addr = ptr  # (int) address
		self._nodes: Array[Structure] = (struct_node * config.MUTABLE_NODE_COUNT).from_address(addr)
		addr += sizeof(self._nodes)
		self._regions: Array[Structure] = (struct_region * config.REGION_COUNT).from_address(addr)
		addr += sizeof(self._regions)
		self._global_node: Structure = struct_global_node.from_address(addr)
		addr += sizeof(self._global_node)
		self._edges: Array[Structure] = (struct_edge * config.EDGE_COUNT).from_address(addr)
		addr += sizeof(self._edges)
		self._edge_regions: Array[Structure] = (struct_edge_region * config.EDGE_REGION_COUNT).from_address(addr)
		addr += sizeof(self._edge_regions)
		self._global_edge: Structure = struct_global_edge.from_address(addr)
		addr += sizeof(self._global_edge)

		self.size = addr - ptr
		if self.size > capacity:
			raise ValueError(f"Buffer wasn't big enough: capacity={capacity} size={self.size}")

		id = config.regionId  # function
		for idx, r in enumerate(r for t in config.regions if r in config.regionNodeParameters):
			rid = id(r)
			setattr(self, rid, self._regions[idx])
		for idx, (r0, r1) in enumerate(e for e in config.regionCombos if e in config.regionEdgeParameters):
			erid = f"{id(r0)}_{id(r1)}"
			setattr(self, erid, self._edge_regions[idx])
		print("globalKeys:", config.globalKeys)  # DEBUG
		for p in config.globalKeys:
			if p in NODE_PARAMETER_SET:
				if p in VECTOR_PARAMETER_SET:
					c_type = c_double_3
				else:
					assert p in SCALAR_PARAMETER_SET  # DEBUG
					c_type = c_double
				setattr(self, p, c_type.from_buffer(self._global_node, getattr(struct_global_node, p).offset))
			else:
				assert p in EDGE_PARAMETER_SET  # DEBUG
				if p in VECTOR_PARAMETER_SET:
					c_type = c_double_3
				else:
					assert p in SCALAR_PARAMETER_SET  # DEBUG
					c_type = c_double
				setattr(self, p, c_type.from_buffer(self._global_edge, getattr(struct_global_edge, p).offset))

	@property
	def nodes(self) -> Array[Structure]:
		return self._nodes
	
	@property
	def regions(self) -> Array[Structure]:
		return self._regions
	
	@property
	def global_node(self) -> Structure:
		return self._global_node

	@property
	def edges(self) -> Array[Structure]:
		return self._edges
	
	@property
	def edge_regions(self) -> Array[Structure]:
		return self._edge_regions
	
	@property
	def global_edge(self) -> Structure:
		return self._global_edge
