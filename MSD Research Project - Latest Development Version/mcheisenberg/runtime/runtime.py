from __future__ import annotations
from ..driver import Driver, GlobalNode, GlobalEdge, libc
from ..prng import SplitMix64
from ..util import ReadOnlyList, ReadOnlyCollection
from .data_view import DataView
from .buffers import MutableStateBuffer
from ctypes import sizeof
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..config import Config, vec
	from ..driver import Driver
	from .runtime_proxies import NodeProxy
	from collections.abc import Sequence, Collection


# Handles communication to and from the DLL, as well as the liftime of the DLL file.
class Runtime(DataView[Driver]):
	VEC_ZERO = (0.0, 0.0, 0.0)  # TODO: different type for runtime data
	VEC_I    = (1.0, 0.0, 0.0)
	VEC_J    = (0.0, 1.0, 0.0)
	VEC_K    = (0.0, 0.0, 1.0)
	
	def __init__(self, config: Config, dll: str, delete: bool=False):
		super().__init__(config, source=Driver(config, dll))
		self.driver: Driver = self.source
		self.dll: str = dll
		self._delete: bool = delete  # delete on exit?

		self.buffers: list[MutableStateBuffer] = []  # track allocated buffers so they can be freed on shutdownvvvvvv
		
		self._buffer_size: int = \
			config.SIZEOF_NODE * config.MUTABLE_NODE_COUNT + \
			config.SIZEOF_REGION * config.REGION_COUNT + \
			sizeof(GlobalNode(config)) + \
			config.SIZEOF_EDGE * config.EDGE_COUNT + \
			config.SIZEOF_EDGE_REGION * config.EDGE_REGION_COUNT + \
			sizeof(GlobalEdge(config))
			# TODO: add extra fields to config for SIZEOF_GLOBAL_NODE/EDGE ?
	
	def _getXoshiro256State(self) -> list[list[int]]:
		prng_state = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
		for i in range(4):
			for j in range(4):
				prng_state[i][j] = self.driver.prng_state[i * 4 + j]
		return prng_state
	
	def _setXoshiro256State(self, prng_state: list[list[int]]) -> None:
		""" Precondition: prng_state has correct dimensions. """
		for i in range(4):
			for j in range(4):
				self.driver.prng_state[i * 4 + j] = prng_state[i][j]
	
	def _validateXoshiro256State(self, prng_state: list[list[int]]) -> None:
		if len(prng_state) != 4:
			raise ValueError("Invalid PRNG state for xoshiro256")
		for i in range(4):
			if len(prng_state[i]) != 4:
				raise ValueError("Invalid PRNG state for xoshiro256")
	
	@staticmethod
	def _generateXoshiro256State(seed: Sequence[int]) -> list[list[int]]:
		"""
		e.g. seed = [A, B, C, D, E, F]
		prng_state = [
			[ (seed)  A,  (seed   B,  (seed)  C,  (seed)  D ],
			[ (seed)  E,  (seed)  F,  (C) -> m0,  (D) -> m0 ],
			[ (E) -> m0,  (F) -> m0,  (C) -> m1,  (D) -> m1 ],
			[ (E) -> m1,  (F) -> m1,  (C) -> m2,  (D) -> m2 ]
		]
		Where (S) -> mi is the i-th SplitMix64 returned from a sequence with seed S.

		e.g. seed = [A]
		prng_state = [
			[ (seed)  A,  (A) -> m11,  (A) -> m7,   (A) -> m3 ]
			[ (A) -> m0,  (A) -> m12,  (A) -> m8,   (A) -> m4 ],
			[ (A) -> m1,  (A) -> m13,  (A) -> m9,   (A) -> m5 ],
			[ (A) -> m2,  (A) -> m14,  (A) -> m10,  (A) -> m6 ]
		]
		"""
		prng_state = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
		for n, s in enumerate(seed):
			prng_state[n // 4][n % 4] = int(s)
		if len(seed) < 16:
			sm64: SplitMix64 = None  # initialized in loop
			for j_offset in range(4):
				n = len(seed) - 1 - j_offset  # final 4 seeds (if they exist)
				j = n % 4
				i = n // 4
				if n >= 0:  # true for at least offset==0 since len(seed) != 0
					sm64 = SplitMix64(int(seed[n]))
				for i_offset in range(i + 1, 4):
					prng_state[i_offset][j] = sm64.next()
		return prng_state

	def shutdown(self):
		# free all allocated buffers (if any)
		for buffer in reversed(self.buffers):
			buffer.free()
		self.buffers = []
		# unload DLL
		self.driver.unload()
		# delete .dll file (unless requested otherwise)
		if self._delete:
			os.remove(self.dll)

	def __enter__(self):
		return self

	def __exit__(self, ex_type, ex_value, traceback):
		self.shutdown()
		return False
	
	def seed(self, *seed: int) -> None:
		"""
		Used to reseed the PRNG-state.
		For deterministic PRNG, give seed.
		If no seed is given, a true-random source of entropy will be used if available.
		"""
		if len(seed) == 0:
			self.driver.seed()
		else:
			assert self.config.programParameters["prng"].startswith("xoshiro256")
			self._setXoshiro256State(Runtime._generateXoshiro256State(seed))

	def reinitialize(self, initSpin: vec=VEC_J, initFlux: vec=VEC_ZERO):
		node_proxy: NodeProxy
		for node_proxy in self.node.values():
			node_proxy.spin = initSpin
			node_proxy.flux = initFlux

	def randomize(self, *seed: int) -> None:
		"""
		Randomize the state of the system using given parameters.
		Optionally, reseed the PRNG as well. (If not seed then PRNG state is
			unchanged before randomization.)
		"""
		if len(seed) != 0:
			self.seed(*seed)
		self.driver.randomize()

	def metropolis(self, iterations: int) -> None:
		""" Run the metropolis algorithm for the given number of iterations. """
		self.driver.metropolis(iterations)
	
	def allocate_buffer(self) -> MutableStateBuffer:
		"""
		Allocates a buffer large enough to store the full mutable state of the model.
		Caller responsibility: Buffer should be manually freed when memory is no longer need (see: Buffer.free())
		All allocated buffers will be freed when this Runntime is shutdown.
		"""
		size = self._buffer_size
		ptr = libc.malloc(size)
		buffer = MutableStateBuffer(self.config, ptr, size)
		self.buffers.append(buffer)
		return buffer
	
	def snapshot(self, buffer: MutableStateBuffer) -> DataView[MutableStateBuffer]:
		"""
		Copies the current mutable state into the given buffer, and
		returns a DataView object which gives access to the data consistant with the Runtime interface.
		"""
		self.driver.mutable_state(buffer.ptr)
		return DataView[MutableStateBuffer](self.config, buffer)

	@property
	def buffer_size(self) -> int:
		return self._buffer_size

	@property
	def prng_state(self) -> list[list[int]]:
		assert self.config.programParameters["prng"].startswith("xoshiro256")
		return self._getXoshiro256State()

	@prng_state.setter
	def prng_state(self, prng_state: list[list[int]]):
		assert self.config.programParameters["prng"].startswith("xoshiro256")
		self._validateXoshiro256State(prng_state)
		self._setXoshiro256State(prng_state)

	@property
	def globals(self) -> dict:
		return { param: getattr(self, param) for param in self.config.globalParameters.keys() }

	@property
	def spins(self) -> Sequence[vec]:
		return ReadOnlyList([ *self.spin.values() ])
	
	@property
	def fluxes(self) -> Sequence[vec]:
		return ReadOnlyList([ *self.flux.values() ])
