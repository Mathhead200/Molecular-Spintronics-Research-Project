from __future__ import annotations
import ctypes
from ctypes import CDLL, c_void_p, c_size_t, wintypes
import gc
	
class LibcDriver:
	def __init__(self):
		# C Runtime: e.g. heap memory (de)allocation
		self.libc = CDLL("ucrtbase.dll")
		self._libc_procs = []
		
		self.libc.malloc.argtypes = [c_size_t]
		self.libc.malloc.restype = c_void_p
		self._libc_procs.append("malloc")

		self.libc.calloc.argtypes = [c_size_t, c_size_t]
		self.libc.calloc.restype = c_void_p
		self._libc_procs.append("calloc")

		self.libc.realloc.argtypes = [c_void_p, c_size_t]
		self.libc.realloc.restype = c_void_p
		self._libc_procs.append("realloc")

		self.libc.free.argtypes = [c_void_p]
		self.libc.free.restype = None
		self._libc_procs.append("free")
	
	def malloc(self, size: c_size_t) -> c_void_p:
		return self.libc.malloc(size)
	
	def calloc(self, num_of_ele: c_size_t, ele_size: c_size_t) -> c_void_p:
		return self.libc.calloc(num_of_ele, ele_size)
	
	def realloc(self, ptr: c_void_p, new_size: c_size_t) -> c_void_p:
		return self.libc.realloc(ptr, new_size)
	
	def free(self, ptr: c_void_p) -> None:
		self.libc.free(ptr)
	
	def unload(self) -> None:
		_handle_libc = self.libc._handle

		# remove references to libc dll
		for proc in self._libc_procs:
			delattr(self.libc, proc)
		self.libc = None
		gc.collect()

		kernel32 = ctypes.windll.kernel32
		kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
		kernel32.FreeLibrary.restype = wintypes.BOOL
		if not kernel32.FreeLibrary(_handle_libc):
			err = ctypes.get_last_error()
			raise OSError(f"FreeLibrary failed: {err}")
	