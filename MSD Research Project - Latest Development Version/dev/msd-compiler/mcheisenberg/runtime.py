from __future__ import annotations
from .driver import Driver
from .prng import SplitMix64
from collections.abc import Sequence, Mapping
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from .config import Config, vec

def div8(n: int) -> int:
	return int(n // 8) if n is not None else None

# Wrapper allowing read-only access to underlying list/Sequence
class ReadOnlyList(Sequence):
	def __init__(self, lst: Sequence):  self._lst = lst

	def __len__(self):          return len(self._lst)	
	def __getitem__(self, i):   return self._lst[i]
	def __iter__(self):         return iter(self._lst)
	def __contains__(self, v):  return v in self._lst

# Parent to be extended/derived from
class AbstractReadOnlyDict(Mapping):
	def keys(self):    return iter(self)
	def values(self):  return (self[k] for k in self)
	def items(self):   return ((k, self[k]) for k in self)
	def get(self, key, default=None):  return self[key] if key in self else default

# Wrapper allowing read-only access to underlying dict
class ReadOnlyDict(AbstractReadOnlyDict):
	def __init__(self, dct: dict):  self._dct = dct

	def __getitem__(self, i):  return self._dct[i]
	def __iter__(self):        return iter(self._dct)
	def __len__(self):         return len(self._dct)

	def get(self, key, default=None):  return self._dct.get(key, default)

# Provides properties (i.e. getters and setters) for local node state and parameters.
# `None` is returned by getters for undefined state/parameters.
# KeyError is raised by setters for undefined state/parameters.
# If node parameters are defined at a higher level (e.g. region, or global),
#	getters will pass through this information, but setters will raise KeyError.
class NodeProxy:
	__slots__ = [ "spin", "flux", "B", "A", "S", "F", "kT", "Je0",
	              "_runtime", "_node", "_index" ]

	def __init__(self, runtime: Runtime, node):
		self._runtime = runtime
		self._node = node
		self._index = runtime.config.nodeIndex[node]
	
	# get scalar (float) only from NODES; or None.
	def _getScalar(self, param: str) -> float:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		return runtime._fromNodes(self._index, offset)
	
	# get vector (tuple) only from NODES; or None.
	def _getVector(self, param: str) -> vec:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		return tuple(runtime._fromNodes(self._index, offset + i) for i in range(3))
	
	# get scalar (float) from either NODES, REGIONS, or GLOBAL_NODE; or None.
	def _fgetScalar(self, param: str) -> vec:
		runtime = self._runtime
		config = runtime.config
		if param in config.localNodeParameters.get(self._node, {}):
			return self._getScalar(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return getattr(runtime.region[region], param)
		return getattr(runtime, param)  # global parameter

	# get vector (tuple) from either NODES, REGION, or GLOBAL_NODE; or None.
	def _fgetVector(self, param: str) -> vec:
		runtime = self._runtime
		config = runtime.config
		if param in config.localNodeParameters.get(self._node, {}):
			return self._getVector(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return getattr(runtime.region[region], param)
		return getattr(runtime, param)  # global parameter
	
	# set scalar (float) in NODES; or KeyError. Bypasses localNodeParameters.
	# For node state, e.g. (no known scalar state -- UNUSED)
	def _setScalar(self, param: str, value: float) -> None:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			raise AttributeError(f"For node {self._node}, {param} is undefined in Config")
		runtime._toNodes(self._index, offset, value)

	# set scalar (float) in NODES; or KeyError.
	def _fsetScalar(self, param: str, value: float) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.localNodeParameters.get(self._node, {}):
			raise AttributeError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")
		offset = runtime._offsets[param];  assert offset is not None  # DEBUG (assert)
		runtime._toNodes(self._index, offset, value)

	# set scalar (float) in NODES; or KeyError. Bypasses localNodeParameters.
	# For node state, e.g. spin and flux.
	def _setVector(self, param: str, value: float) -> None:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			raise AttributeError(f"For node {self._node}, {param} is undefined in Config")
		for i in range(3):
			runtime._toNodes(self._index, offset + i, value[i])
	
	# set vector (tuple) in NODES, or KeyError.
	def _fsetVector(self, param: str, value: vec) -> None:
		runtime = self._runtime
		config = self._runtime.config
		if param not in config.localNodeParameters.get(self._node, {}):
			raise AttributeError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")
		offset = runtime._offsets[param];  assert offset is not None  # DEBUG (assert)
		for i in range(3):
			runtime._toNodes(self._index, offset + i, value[i])

for attr in ["spin", "flux"]:
	setattr(NodeProxy, attr, property(
		fget=lambda self,        _a=attr: self._getVector(_a),
		fset=lambda self, value, _a=attr: self._setVector(_a, value)
	))
for param in ["B", "A"]:
	setattr(NodeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetVector(_p),
		fset=lambda self, value, _p=param: self._fsetVector(_p, value)
	))
for param in ["S", "F", "kT", "Je0"]:
	setattr(NodeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetScalar(_p),
		fset=lambda self, value, _p=param: self._fsetScalar(_p, value)
	))

# Similar to NodeProxy, but for edge parameters.
class EdgeProxy:
	__slots__ = [ "J", "Je1", "Jee", "b", "D",
	              "_runtime", "_edge", "_index" ]

	def __init__(self, runtime: Runtime, edge):
		self._runtime = runtime
		self._edge = edge
		self._index = runtime.config.edgeIndex.get(edge, None)
	
	# get scalar (float) only from EDGES; or None
	def _getScalar(self, param: str) -> float:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		assert self._index is not None
		return runtime._fromEdges(self._index, offset)
	
	# get vector (tuple) only from EDGES; or None
	def _getVector(self, param: str) -> vec:
		runtime = self._runtime
		offset = runtime._offsets[param]
		if offset is None:
			return None
		assert self._index is not None
		return tuple(runtime._fromEdges(self._index, offset + i) for i in range(3))
	
	# get scalar (float) from either EDGES, EDGE_REGIONS, or GLOBAL_EDGE; or None
	def _fgetScalar(self, param: str) -> vec:
		runtime = self._runtime
		config = runtime.config
		if param in config.localEdgeParameters.get(self._edge, {}):
			return self._getScalar(param)
		_, eregion = config.getUnambiguousRegionEdgeParameter(self._edge, param)  # we don't need value, only eregion
		if eregion is not None:
			return getattr(runtime.eregion[eregion], param)
		return getattr(runtime, param)  # global parameter

	# get vector (tuple) from either EDGES, EDGE_REGION, or GLOBAL_EDGE; or None
	def _fgetVector(self, param: str) -> vec:
		runtime = self._runtime
		config = runtime.config
		if param in config.localEdgeParameters.get(self._edge, {}):
			return self._getVector(param)
		_, eregion = config.getUnambiguousRegionEdgeParameter(self._edge, param)  # we don't need value, only eregion
		if eregion is not None:
			return getattr(runtime.eregion[eregion], param)
		return getattr(runtime, param)  # global parameter
	
	# set scalar (float) in EDGES; or KeyError
	def _fsetScalar(self, param: str, value: float) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.localEdgeParameters.get(self._edge, {}):
			raise AttributeError(f"For edge {self._edge}, can't modify {param} which was not defined locally in Config")
		offset = runtime._offsets[param]
		assert offset is not None
		assert self._index is not None
		runtime._toEdges(self._index, offset, value)
	
	# set vector (tuple) in EDGES; or KeyError.
	def _fsetVector(self, param: str, value: vec) -> None:
		runtime = self._runtime
		config = self._runtime.config
		if param not in config.localEdgeParameters.get(self._edge, {}):
			raise AttributeError(f"For edge {self._edge}, can't modify {param} which was not defined locally in Config")
		offset = runtime._offsets[param]
		assert offset is not None
		assert self._index is not None
		for i in range(3):
			runtime._toEdges(self._index, offset + i, value[i])

for param in ["J", "Je1", "Jee", "b"]:
	setattr(EdgeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetScalar(_p),
		fset=lambda self, value, _p=param: self._fsetScalar(_p, value)
	))
for param in ["D"]:
	setattr(EdgeProxy, param, property(
		fget=lambda self,        _p=param: self._fgetVector(_p),
		fset=lambda self, value, _p=param: self._fsetVector(_p, value)
	))

# Similar to NodeProxy/EdgeProxy but for region (node) parameters
class RegionProxy:
	__slots__ = [ "B", "A", "S", "F", "kT", "Je0",
	              "_runtime", "_region", "_rid" ]

	def __init__(self, runtime: Runtime, region):
		self._runtime = runtime
		self._region = region
		self._rid = runtime.config.regionId(region)
	
	# get scalar (float) only from REGIONS; or None
	def _getScalar(self, param: str) -> float:
		runtime = self._runtime
		offset = runtime._region_offsets[param]
		if offset is None:
			return None
		return runtime._fromRegions(self._rid, offset)
	
	# get vector (tuple) only from REGIONS; or None
	def _getVector(self, param: str) -> vec:
		runtime = self._runtime
		offset = runtime._region_offsets[param]
		if offset is None:
			return None
		return tuple(runtime._fromRegions(self._rid, offset + i) for i in range(3))

	# get scalar (float) from either REGION or GLOBAL_NODE; or None
	def _fgetScalar(self, param: str) -> float:
		runtime = self._runtime
		config = runtime.config
		if param in config.regionNodeParameters.get(self._region, {}):
			return self._getScalar(param)
		return getattr(runtime, param)  # global parameter
	
	# get vector (tuple) from either REGION or GLOBAL_NODE; or None
	def _fgetVector(self, param: str) -> vec:
		runtime= self._runtime
		config = runtime.config
		if param in config.regionNodeParameters.get(self._region, {}):
			return self._getVector(param)
		return getattr(runtime, param)  # global parameter

	# set scalar (float) in REGION; or KeyError
	def _fsetScalar(self, param: str, value: float) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.regionNodeParameters.get(self._region, {}):
			raise AttributeError(f"For region {self._region}, can't modify {param} which was not defined in Config")
		offset = runtime._region_offsets[param];  assert offset is not None  # DEBUG (assert)
		runtime._toRegions(self._rid, offset, value)
	
	# set vector (tuple) in REGION; or KeyError
	def _fsetVector(self, param: str, value: vec) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.regionNodeParameters.get(self._region, {}):
			raise AttributeError(f"For region {self._region}, can't modify {param} which was not defined in Config")
		offset = runtime._region_offsets[param];  assert offset is not None  # DEBUG (assert)
		for i in range(3):
			runtime._toRegions(self._rid, offset + i, value)

for param in ["B", "A"]:
	setattr(RegionProxy, param, property(
		fget=lambda self,        _p=param: self._fgetVector(_p),
		fset=lambda self, value, _p=param: self._fsetVector(_p, value)
	))
for param in ["S", "F", "kT", "Je0"]:
	setattr(RegionProxy, param, property(
		fget=lambda self,        _p=param: self._fgetScalar(_p),
		fset=lambda self, value, _p=param: self._fsetScalar(_p, value)
	))

# Similar to RegionProxy but for region edge parameters
class ERegionProxy:
	__slots__ = [ "J", "Je1", "Jee", "b", "D",
	              "_runtime", "_eregion", "_erid" ]

	def __init__(self, runtime: Runtime, eregion: tuple):
		self._runtime = runtime
		self._eregion = eregion
		regionId = runtime.config.regionId  # function
		self._erid = f"{regionId(eregion[0])}_{regionId(eregion[1])}"
	
	# get scalar (float) only from EDGE_REGIONS; or None
	def _getScalar(self, param: str) -> float:
		runtime = self._runtime
		offset = runtime._region_offsets[param]
		if offset is None:
			return None
		return runtime._fromERegions(self._erid, offset)
	
	# get vector (tuple) only from EDGE_REGIONS; or None
	def _getVector(self, param: str) -> vec:
		runtime = self._runtime
		offset = runtime._region_offsets[param]
		if offset is None:
			return None
		return tuple(runtime._fromERegions(self._erid, offset + i) for i in range(3))

	# get scalar (float) from either EDGE_REGION or GLOBAL_EDGE; or None
	def _fgetScalar(self, param: str) -> float:
		runtime = self._runtime
		config = runtime.config
		if param in config.regionEdgeParameters.get(self._eregion, {}):
			return self._getScalar(param)
		return getattr(runtime, param)  # global parameter
	
	# get vector (tuple) from either EDGE_REGION or GLOBAL_EDGE; or None
	def _fgetVector(self, param: str) -> vec:
		runtime = self._runtime
		config = runtime.config
		if param in config.regionEdgeParameters.get(self._eregion, {}):
			return self._getVector(param)
		return getattr(runtime, param)  # global parameter

	# set scalar (float) in EDGE_REGION; or KeyError
	def _fsetScalar(self, param: str, value: float) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.regionEdgeParameters.get(self._eregion, {}):
			raise AttributeError(f"For edge-region {self._eregion}, can't modify {param} which was not defined in Config")
		offset = runtime._region_offsets[param];  assert offset is not None  # DEBUG (assert)
		runtime._toERegions(self._erid, offset, value)
	
	# set vector (tuple) in EDGE_REGION; or KeyError
	def _fsetVector(self, param: str, value: vec) -> None:
		runtime = self._runtime
		config = runtime.config
		if param not in config.regionEdgeParameters.get(self._eregion, {}):
			raise AttributeError(f"For edge region {self._eregion}, can't modify {param} which was not defined in Config")
		offset = runtime._region_offsets[param];  assert offset is not None  # DEBUG (assert)
		for i in range(3):
			runtime._toERegions(self._erid, offset + i, value)

for param in ["J", "Je1", "Jee", "b"]:
	setattr(ERegionProxy, param, property(
		fget=lambda self,        _p=param: self._fgetScalar(_p),
		fset=lambda self, value, _p=param: self._fsetScalar(_p, value)
	))
for param in ["D"]:
	setattr(ERegionProxy, param, property(
		fget=lambda self,        _p=param: self._fgetVector(_p),
		fset=lambda self, value, _p=param: self._fsetVector(_p, value)
	))

class GlobalsProxy:
	__slots__ = [ "B", "A", "S", "F", "kT", "Je0",
	              "J", "Je1", "Jee", "b", "D",
	              "_runtime" ]

	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def _getScalar(self, param: str) -> float:
		return getattr(self._runtime.driver, param, None).value
	
	def _getVector(self,  param: str) -> vec:
		try:
			v = getattr(self._runtime.driver, param)
			return tuple(v[i] for i in range(3))
		except AttributeError:
			return None
	
	def _setScalar(self, param: str, value: float) -> None:
		try:
			getattr(self._runtime.driver, param).value = value
		except AttributeError as ex:
			raise AttributeError(f"Global parameter {param} is undefined in Config") from ex

	def _setVector(self, param: str, value: vec) -> None:
		try:
			v = getattr(self._runtime.driver, param)
			v[0], v[1], v[2] = value[0], value[1], value[2]
		except AttributeError as ex:
			raise AttributeError(f"Global parameter {param} is undefined in Config") from ex

for param in ["A", "B", "D"]:
	setattr(GlobalsProxy, param, property(
		fget=lambda self,        _p=param: self._getVector(_p),
		fset=lambda self, value, _p=param: self._setVector(_p, value)
	))
for param in ["S", "F", "kT", "Je0", "J", "Je1", "Jee", "b"]:
	setattr(GlobalsProxy, param, property(
		fget=lambda self,        _p=param: self._getScalar(_p),
		fset=lambda self, value, _p=param: self._setScalar(_p, value)
	))

class NodeListProxy(ReadOnlyDict):
	def __init__(self, runtime: Runtime):
		super().__init__(runtime._node_proxies)
		self._runtime = runtime
	
	def __getitem__(self, node) -> NodeProxy:
		try:
			return super().__getitem__(node)
		except KeyError as ex:
			raise KeyError(f"Node {node} is not defined in Config") from ex

class EdgeListProxy(ReadOnlyDict):
	def __init__(self, runtime: Runtime):
		super().__init__(runtime._edge_proxies)
		self._runtime = runtime
	
	def __getitem__(self, edge: tuple) -> EdgeProxy:
		try:
			return super().__getitem__(edge)
		except KeyError as ex:
			raise KeyError(f"Edge {edge} in not defined in Config") from ex
	
	def __call__(self, node0, node1) -> EdgeProxy:
		return self[(node0, node1)]

class RegionListProxy:
	def __init__(self, runtime: Runtime):
		self._runtime = runtime
	
	def __getitem__(self, region) -> RegionProxy:
		try:
			return self._runtime._region_proxies[region]
		except KeyError as ex:
			raise KeyError(f"Region {region} is not defined in Config") from ex

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

class StateListProxy(AbstractReadOnlyDict):
	def __init__(self, runtime: Runtime, state: str):
		self._proxy_list = runtime.node
		self._state = state  # e.g. "spin", or "flux"

	def __getitem__(self, node):
		return getattr(self._proxy_list[node], self._state)

	def __setitem__(self, node, value) -> None:
		return setattr(self._proxy_list[node], self._state, value)

	def __len__(self):
		return len(self._proxy_list)
	
	def __iter__(self):
		return iter(self._proxy_list)
	
	def get(self, node, default=None):
		proxy = self._proxy_list.get(node, None)
		if proxy is None:
			return None
		return getattr(proxy, self._state)

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
		self._spin_list_proxy    = StateListProxy(self, "spin")
		self._flux_list_proxy    = StateListProxy(self, "flux")
		self._edge_list_proxy    = EdgeListProxy(self)
		self._region_list_proxy  = RegionListProxy(self)
		self._eregion_list_proxy = ERegionListProxy(self)
		self._nodes_view    = ReadOnlyList([ *self._node_proxies.keys()    ])
		self._edges_view    = ReadOnlyList([ *self._edge_proxies.keys()    ])
		self._regions_view  = ReadOnlyList([ *self._region_proxies.keys()  ])
		self._eregions_view = ReadOnlyList([ *self._eregion_proxies.keys() ])
	
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
		fget=lambda self,        _p=param: getattr(self._globals_proxy, _p),
		fset=lambda self, value, _p=param: setattr(self._globals_proxy, _p, value)
	))
