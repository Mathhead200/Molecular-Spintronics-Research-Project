from __future__ import annotations
from .util import AbstractReadableDict, ReadOnlyDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from .config import vec
	from .runtime import Runtime

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
		runtime = self._runtime
		if param not in runtime.config.globalParameters:
			return None
		return getattr(runtime.driver, param).value
	
	def _getVector(self,  param: str) -> vec:
		runtime = self._runtime
		if param not in runtime.config.globalParameters:
			return None
		v = getattr(runtime.driver, param)
		return tuple(v[i] for i in range(3))
	
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
	
	def __getitem__(self, node) -> NodeProxy:
		try:
			return super().__getitem__(node)
		except KeyError as ex:
			raise KeyError(f"Node {node} is not defined in Config") from ex

class EdgeListProxy(ReadOnlyDict):
	def __init__(self, runtime: Runtime):
		super().__init__(runtime._edge_proxies)
	
	def __getitem__(self, edge: tuple) -> EdgeProxy:
		try:
			return super().__getitem__(edge)
		except KeyError as ex:
			raise KeyError(f"Edge {edge} in not defined in Config") from ex
	
	def __call__(self, node0, node1) -> EdgeProxy:
		return self[(node0, node1)]

class RegionListProxy(ReadOnlyDict):
	def __init__(self, runtime: Runtime):
		super().__init__(runtime._region_proxies)
	
	def __getitem__(self, region) -> RegionProxy:
		try:
			return super().__getitem__(region)
		except KeyError as ex:
			raise KeyError(f"Region {region} is not defined in Config") from ex

class ERegionListProxy(ReadOnlyDict):
	def __init__(self, runtime: Runtime):
		super().__init__(runtime._eregion_proxies)
	
	def __getitem__(self, eregion: tuple) -> ERegionProxy:
		try:
			return super().__getitem__(eregion)
		except KeyError as ex:
			raise KeyError(f"Edge-region {eregion} in not defined in Config") from ex
	
	def __call__(self, region0, region1) -> ERegionProxy:
		return self[(region0, region1)]

class StateListProxy(AbstractReadableDict):
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

# Acts like the global parameter when used as a value, but allows the lookup
#	for local/region parameters via __getitem__/__setitem__.
class ParameterProxy:
	def __init__(self, runtime: Runtime, param: str):
		self._runtime = runtime
		self._param = param
	
	@property
	def name(self):
		return self._param

	@property
	def value(self):
		return getattr(self._runtime._globals_proxy, self.name)
	
	@value.setter
	def value(self, value) -> None:
		setattr(self._runtime._globals_proxy, self.name, value)

	def __getitem__(self, key):
		param = self.name
		runtime = self._runtime
		config = runtime.config
		if param in config.localNodeParameters.get(key, {}):
			return getattr(runtime.node[key], param)
		if param in config.localEdgeParameters.get(key, {}):
			return getattr(runtime.edge[key], param)
		if param in config.regionNodeParameters.get(key, {}):
			return getattr(runtime.region[key], param)
		if param in config.regionEdgeParameters.get(key, {}):
			return getattr(runtime.eregion[key], param)
		return self.value  # global parameter

	def __setitem__(self, key, value) -> None:
		param = self.name
		runtime = self._runtime
		config = runtime.config
		if param in config.localNodeParameters.get(key, {}):
			setattr(runtime.node[key], param, value)
		elif param in config.localEdgeParameters.get(key, {}):
			setattr(runtime.edge[key], param, value)
		elif param in config.regionNodeParameters.get(key, {}):
			setattr(runtime.region[key], param, value)
		elif param in config.regionEdgeParameters.get(key, {}):
			setattr(runtime.eregion[key], param, value)
		else:
			raise KeyError(f"Parameter {param} is not defined for (node, edge, region, or edge-region) {key}")
	
	def __str__(self) -> str:  return str(self.value)
	def __repr__(self) -> str: return repr(self.value)

# Acts like a float
class ScalarParameterProxy(ParameterProxy):
	def __float__(self) -> float:  return self.value

# Acts like a tuple
class VectorParameterProxy(ParameterProxy):
	def __iter__(self) -> vec:  return self.value
	def __len__(self) -> int:  return len(self.value)
