from __future__ import annotations
from .structs import Node, Edge, Region, EdgeRegion, c_double_3
from ctypes import WinDLL, c_int64, c_double, c_void_p, wintypes
import ctypes
import gc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..config import Config

def property_wrap(c_mem):
	def fget(self):  return c_mem.value
	def fset(self, value):  c_mem.value = value
	return property(fget, fset)

# Thin wrapper for DLL
class Driver:
	def __init__(self, config: "Config", dll: str):
		self.dll = WinDLL(dll)
		self._procs = []
		self._symbols = []

		self.dll.metropolis.argtypes = [c_int64]
		self.dll.metropolis.restype = None
		self._procs.append("metropolis")

		self.dll.seed.argtypes = ()
		self.dll.seed.restype = None
		self._procs.append("seed")

		self.dll.randomize.argtypes = ()
		self.dll.randomize.restype = None
		self._procs.append("randomize")

		self.dll.mutable_state.argtypes = [c_void_p]
		self.dll.mutable_state.restype = None
		self._procs.append("mutable_state")

		struct_node = Node(config)
		if config.SIZEOF_NODE > 0 and config.NODE_COUNT > 0:
			self.nodes = (struct_node * config.NODE_COUNT).in_dll(self.dll, "nodes")  # Note: all nodes, not just MUTABLE_NODES
			self._symbols.append("nodes")
		
		struct_edge = Edge(config)
		if config.SIZEOF_EDGE > 0 and config.EDGE_COUNT > 0:
			self.edges = (struct_edge * config.EDGE_COUNT).in_dll(self.dll, "edges")
			self._symbols.append("edges")
		
		struct_region = Region(config)
		if config.SIZEOF_REGION != 0:
			for r in config.regions:
				if r in config.regionNodeParameters:
					rid = config.regionId(r)
					setattr(self, rid, (struct_region).in_dll(self.dll, rid))
					self._symbols.append(rid)

		struct_edge_region = EdgeRegion(config)
		if config.SIZEOF_EDGE_REGION:
			for r0, r1 in config.regionCombos:
				if (r0, r1) in config.regionEdgeParameters:
					rid = f"{config.regionId(r0)}_{config.regionId(r1)}"
					setattr(self, rid, (struct_edge_region).in_dll(self.dll, rid))
					self._symbols.append(rid)
		
		if "B" in config.globalKeys:
			self.B = c_double_3.in_dll(self.dll, "B")
			self._symbols.append("B")

		if "A" in config.globalKeys:
			self.A = c_double_3.in_dll(self.dll, "A")
			self._symbols.append("A")

		if any(p in config.globalKeys for p in ["S", "F", "kT", "Je0"]):
			self.S = c_double.in_dll(self.dll, "S")
			self.F = c_double.in_dll(self.dll, "F")
			self.kT = c_double.in_dll(self.dll, "kT")
			self.Je0 = c_double.in_dll(self.dll, "Je0")
			self._symbols.extend(["S", "F", "kT", "Je0"])

		if any(p in config.globalKeys for p in ["J", "Je1", "Jee", "b"]):
			self.J = c_double.in_dll(self.dll, "J")
			self.Je1 = c_double.in_dll(self.dll, "Je1")
			self.Jee = c_double.in_dll(self.dll, "Jee")
			self.b = c_double.in_dll(self.dll, "b")
			self._symbols.extend(["J", "Je1", "Jee", "b"])

		if "D" in config.globalKeys:
			self.D = c_double_3.in_dll(self.dll, "D")
			self._symbols.append("D")

		# TODO: _ref parallel arrays?

		assert config.programParameters["prng"].startswith("xoshiro256")   # TODO: support other PRNGs
		self.prng_state = (c_double * (4 * 4)).in_dll(self.dll, "prng_state")
		self._symbols.append("prng_state")

	def metropolis(self, iterations: c_int64) -> None:
		self.dll.metropolis(iterations)
	
	def seed(self) -> None:
		self.dll.seed()
	
	def randomize(self) -> None:
		self.dll.randomize()
	
	def mutable_state(self, buffer: c_void_p) -> None:
		self.dll.mutable_state(buffer)

	def unload(self):
		_handle = self.dll._handle

		# remove references to (wrapped) dll
		for symbol in self._symbols:
			delattr(self, symbol)
		for proc in self._procs:
			delattr(self.dll, proc)
		self.dll = None
		gc.collect()

		# unload DLLs
		kernel32 = ctypes.windll.kernel32
		kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
		kernel32.FreeLibrary.restype = wintypes.BOOL
		if not kernel32.FreeLibrary(_handle):
			err = ctypes.get_last_error()
			raise OSError(f"FreeLibrary failed: {err}")
		
