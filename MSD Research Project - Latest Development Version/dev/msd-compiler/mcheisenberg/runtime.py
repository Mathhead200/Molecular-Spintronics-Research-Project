from __future__ import annotations
from .driver import Driver
from .prng import SplitMix64
from .runtime_proxies import *
from .util import ReadOnlyList, div8
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from collections.abc import Sequence, Mapping
	from .config import Config, vec

# Handles communication to and from the DLL, as well as the liftime of the DLL file.
class Runtime:
	VEC_ZERO = (0.0, 0.0, 0.0)
	VEC_I    = (1.0, 0.0, 0.0)
	VEC_J    = (0.0, 1.0, 0.0)
	VEC_K    = (0.0, 0.0, 1.0)
	
	def __init__(self, config: Config, dll: str, delete: bool=False):
		self.dll = dll
		self.config = config
		self.driver = Driver(config, dll)
		self._delete = delete  # delete on exit?
		self._node_len = config.SIZEOF_NODE // 8  # in doubles (8-byte elements)
		self._edge_len = config.SIZEOF_EDGE // 8  # ^^
		self._offsets = {
			"spin": div8(config.OFFSETOF_SPIN),  # node offsets
			"flux": div8(config.OFFSETOF_FLUX),
			"B":    div8(config.OFFSETOF_B),
			"A":    div8(config.OFFSETOF_A),
			"S":    div8(config.OFFSETOF_S),
			"F":    div8(config.OFFSETOF_F),
			"kT":   div8(config.OFFSETOF_kT),
			"Je0":  div8(config.OFFSETOF_Je0),
			"J":    div8(config.OFFSETOF_J),     # edge offsets
			"Je1":  div8(config.OFFSETOF_Je1),
			"Jee":  div8(config.OFFSETOF_Jee),
			"b":    div8(config.OFFSETOF_b),
			"D":    div8(config.OFFSETOF_D)
		}
		self._region_offsets ={
			"B":   div8(config.OFFSETOF_REGION_B),    # region node offsets
			"A":   div8(config.OFFSETOF_REGION_A),
			"S":   div8(config.OFFSETOF_REGION_S),
			"F":   div8(config.OFFSETOF_REGION_F),
			"kT":  div8(config.OFFSETOF_REGION_kT),
			"Je0": div8(config.OFFSETOF_REGION_Je0),
			"J":   div8(config.OFFSETOF_REGION_J),    # region edge offsets
			"Je1": div8(config.OFFSETOF_REGION_Je1),
			"Jee": div8(config.OFFSETOF_REGION_Jee),
			"b":   div8(config.OFFSETOF_REGION_b),
			"D":   div8(config.OFFSETOF_REGION_D)
		}
		self._node_proxies    = { node:    NodeProxy(self, node)       for node    in config.nodes          }
		self._edge_proxies    = { edge:    EdgeProxy(self, edge)       for edge    in config.edges          }
		self._region_proxies  = { region:  RegionProxy(self, region)   for region  in config.regions.keys() }
		self._eregion_proxies = { eregion: ERegionProxy(self, eregion) for eregion in config.regionCombos   }
		self._globals_proxy = GlobalsProxy(self)
		self._node_list_proxy    = NodeListProxy(self)
		self._edge_list_proxy    = EdgeListProxy(self)
		self._region_list_proxy  = RegionListProxy(self)
		self._eregion_list_proxy = ERegionListProxy(self)
		self._nodes_view    = ReadOnlyList([ *self._node_proxies.keys()    ])
		self._edges_view    = ReadOnlyList([ *self._edge_proxies.keys()    ])
		self._regions_view  = ReadOnlyList([ *self._region_proxies.keys()  ])
		self._eregions_view = ReadOnlyList([ *self._eregion_proxies.keys() ])
		self._spin_list_proxy    = StateListProxy(self, "spin")
		self._flux_list_proxy    = StateListProxy(self, "flux")
		for param in ["A", "B", "D"]:
			setattr(self, f"_{param}_proxy", VectorParameterProxy(self, param))  # e.g. self._B_proxy
		for param in ["S", "F", "kT", "Je0", "J", "Je1", "Jee", "b"]:
			setattr(self, f"_{param}_proxy", ScalarParameterProxy(self, param))  # e.g. self._J_proxy

	def _fromNodes(self, index: int, offset: int) -> float:
		return self.driver.nodes[index * self._node_len + offset]

	def _fromEdges(self, index: int, offset: int) -> float:
		return self.driver.edges[index * self._edge_len + offset]
	
	def _fromRegions(self, rid: str, offset: int) -> float:
		return getattr(self.driver, rid)[offset]

	def _fromERegions(self, erid: str, offset: int) -> float:
		return getattr(self.driver, erid)[offset]

	def _toNodes(self, index: int, offset: int, value: float) -> None:
		self.driver.nodes[index * self._node_len + offset] = value
	
	def _toEdges(self, index: int, offset: int, value: float) -> None:
		self.driver.edges[index * self._edge_len + offset] = value
	
	def _toRegions(self, rid: str, offset: int, value: float) -> None:
		getattr(self.driver, rid)[offset] = value
	
	def _toERegions(self, erid: str, offset: int, value: float) -> None:
		getattr(self.driver, erid)[offset] = value
	
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
		self.driver.free()
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
		raise NotImplementedError("TODO: implement (in ASM?)")
		# TODO: implement (in ASM?)

	def metropolis(self, iterations: int) -> None:
		""" Run the metropolis algorithm for the given number of iterations. """
		self.driver.metropolis(iterations)

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
	def regions(self) -> Sequence:
		return self._regions_view
	
	@property
	def eregions(self) -> Sequence:
		return self._eregions_view
	
	@property
	def globals(self) -> dict:
		return { param: getattr(self._globals_proxy, param) for param in self.config.globalParameters.keys() }

	@property
	def spins(self) -> Sequence[vec]:
		return ReadOnlyList([ *self.spin.values() ])
	
	@property
	def fluxes(self) -> Sequence[vec]:
		return ReadOnlyList([ *self.flux.values() ])

for param in ["A", "B", "S", "F", "kT", "Je0", "J", "Je1", "Jee", "b", "D"]:
	setattr(Runtime, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(self._globals_proxy, _p, value)
	))
