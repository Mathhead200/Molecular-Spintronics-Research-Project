from __future__ import annotations
import ctypes
from ctypes import WinDLL, c_int64, c_double, wintypes
import gc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from .config import Config

# Thin wrapper for DLL
class Driver:
	def __init__(self, config: "Config", dll: str):
		self.dll = WinDLL(dll)
		self.symbols = []

		self.dll.metropolis.argtypes = [c_int64]
		self.dll.metropolis.restype = None

		self.dll.seed.argtypes = ()
		self.dll.seed.restype = None

		n = (config.SIZEOF_NODE // 8) * config.NODE_COUNT
		if n > 0:
			self.nodes = (c_double * n).in_dll(self.dll, "nodes")
			self.symbols.append("nodes")
		
		n = (config.SIZEOF_EDGE // 8) * config.EDGE_COUNT
		if n > 0:
			self.edges = (c_double * n).in_dll(self.dll, "edges")
			self.symbols.append("edges")
		
		if config.SIZEOF_REGION != 0:
			for r in config.regions:
				if r in config.regionNodeParameters:
					rid = config.regionId(r)
					n = config.SIZEOF_REGION // 8
					setattr(self, rid, (c_double * n).in_dll(self.dll, rid))
					self.symbols.append(rid)

		if config.SIZEOF_EDGE_REGION:
			for r0, r1 in config.regionCombos:
				if (r0, r1) in config.regionEdgeParameters:
					rid = f"{config.regionId(r0)}_{config.regionId(r1)}"
					n = config.SIZEOF_EDGE_REGION // 8
					setattr(self, rid, (c_double * n).in_dll(self.dll, rid))
					self.symbols.append(rid)
		
		if "B" in config.globalKeys:
			self.B = (c_double * 4).in_dll(self.dll, "B")
			self.symbols.append("B")

		if "A" in config.globalKeys:
			self.A = (c_double * 4).in_dll(self.dll, "A")
			self.symbols.append("A")

		if any(p in config.globalKeys for p in ["S", "F", "kT", "Je0"]):
			self.S = c_double.in_dll(self.dll, "S")
			self.F = c_double.in_dll(self.dll, "F")
			self.kT = c_double.in_dll(self.dll, "kT")
			self.Je0 = c_double.in_dll(self.dll, "Je0")
			self.symbols.extend(["S", "F", "kT", "Je0"])

		if any(p in config.globalKeys for p in ["J", "Je1", "Jee", "b"]):
			self.J = c_double.in_dll(self.dll, "J")
			self.Je1 = c_double.in_dll(self.dll, "Je1")
			self.Jee = c_double.in_dll(self.dll, "Jee")
			self.b = c_double.in_dll(self.dll, "b")
			self.symbols.extend(["J", "Je1", "Jee", "b"])

		if "D" in config.globalKeys:
			self.D = (c_double * 4).in_dll(self.dll, "D")
			self.symbols.append("D")

		# TODO: _ref parallel arrays?

	def metropolis(self, iterations: c_int64) -> None:
		self.dll.metropolis(iterations)
	
	def seed(self) -> None:
		self.dll.seed()

	def free(self):
		_handle = self.dll._handle

		for symbol in self.symbols:
			delattr(self, symbol)
		del self.dll.metropolis
		del self.dll.seed
		self.dll = None
		gc.collect()

		kernel32 = ctypes.windll.kernel32
		kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
		kernel32.FreeLibrary.restype = wintypes.BOOL
		if not kernel32.FreeLibrary(_handle):
			err = ctypes.get_last_error()
			raise OSError(f"FreeLibrary failed: {err}")
