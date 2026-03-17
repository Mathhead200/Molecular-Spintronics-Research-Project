from __future__ import annotations
from ..util import AbstractReadableDict, ReadOnlyDict, Numeric
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from .data_view import DataView
	from collections.abc import Sequence
	from ctypes import Array, c_double
	from typing import Annotated

type vec_out = Annotated[Array[c_double], 3]
type vec_in = Annotated[Sequence[float], 3]
type scal_out = c_double
type scal_in = float|c_double  # something assignable to a c_double.value

# Provides properties (i.e. getters and setters) for local node state and parameters.
# `None` is returned by getters for undefined state/parameters.
# KeyError is raised by setters for undefined state/parameters.
# If node parameters are defined at a higher level (e.g. region, or global),
#	getters will pass through this information, but setters will raise KeyError.
class NodeProxy:
	__slots__ = [ "spin", "flux", "B", "A", "S", "F", "kT", "Je0",
	              "_runtime", "_node", "_index" ]

	def __init__(self, data: DataView, node):
		self._data = data
		self._node = node
	
	# get scalar only from NODES; or None.
	def _getScalar(self, param: str) -> scal_out:
		data = self._data
		index = data.config.nodeIndex[self._node]
		return getattr(data.source.nodes[index], param)
	
	# get vector only from NODES; or None.
	def _getVector(self, param: str) -> vec_out:
		data = self._data
		index = data.config.nodeIndex[self._node]
		return getattr(data.source.nodes[index], param)
	
	# get scalar from either NODES, REGIONS, or GLOBAL_NODE; or None.
	def _fgetScalar(self, param: str) -> scal_out:
		data = self._data
		config = data.config
		if param in config.localNodeParameters.get(self._node, {}):
			return self._getScalar(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return getattr(data.region[region], param)
		return getattr(data.source, param)  # global parameter

	# get vector from either NODES, REGION, or GLOBAL_NODE; or None.
	def _fgetVector(self, param: str) -> vec_out:
		data = self._data
		config = data.config
		if param in config.localNodeParameters.get(self._node, {}):
			return self._getVector(param)
		_, region = config.getUnambiguousRegionNodeParameter(self._node, param)  # we don't need value, only region
		if region is not None:
			return getattr(data.region[region], param)
		return getattr(data.source, param)  # global parameter
	
	# set scalar in NODES; or KeyError. Bypasses localNodeParameters.
	# For node state, e.g. (no known scalar state -- UNUSED)
	def _setScalar(self, param: str, value: scal_in) -> None:
		data = self._data
		index = data.config.nodeIndex[self._node]
		try:
			setattr(data.source.nodes[index], param, value)
		except IndexError|KeyError as ex:
			ex.add_note(f"Node {self._node}, index={index}, not found in the data. Immutable?")
			raise
		except AttributeError as ex:
			ex.add_note(f"For node {self._node}, {param} is undefined in Config")
			raise
	
	# set vector (c_double_3) in NODES; or KeyError. Bypasses localNodeParameters.
	# For node state, e.g. spin and flux.
	def _setVector(self, param: str, value: vec_in) -> None:
		data = self._data
		index = data.config.nodeIndex[self._node]
		try:
			getattr(data.source.nodes[index], param)[:3] = value
		except IndexError|KeyError as ex:
			ex.add_note(f"Node {self._node}, index={index}, not found in the data. Immutable?")
			raise
		except AttributeError as ex:
			ex.add_note(f"For node {self._node}, {param} is undefined in Config")
			raise

	# set scalar (float) in NODES; or KeyError.
	def _fsetScalar(self, param: str, value: float) -> None:
		data = self._data
		if param not in data.config.localNodeParameters.get(self._node, {}):
			raise AttributeError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")
		index = data.config.nodeIndex[self._node]
		setattr(data.source.nodes[index], param, value)
	
	# set vector (c_double_3) in NODES, or KeyError.
	def _fsetVector(self, param: str, value: vec_in) -> None:
		data = self._data
		if param not in data.config.localNodeParameters.get(self._node, {}):
			raise AttributeError(f"For node {self._node}, can't modify {param} which was not defined locally in Config")
		index = data.config.nodeIndex[self._node]
		getattr(data.source.nodes[index], param)[:3] = value

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

	def __init__(self, data: DataView, edge):
		self._data = data
		self._edge = edge
	
	# get scalar only from EDGES; or None
	def _getScalar(self, param: str) -> scal_out:
		data = self._data
		index = data.config.edgeIndex[self._edge]
		return getattr(data.source.edges[index], param)
	
	# get vector only from EDGES; or None
	def _getVector(self, param: str) -> vec_out:
		data = self._data
		index = data.config.edgeIndex[self._edge]
		return getattr(data.source.edges[index], param)
	
	# get scalar from either EDGES, EDGE_REGIONS, or GLOBAL_EDGE; or None
	def _fgetScalar(self, param: str) -> scal_out:
		data = self._data
		config = data.config
		if param in config.localEdgeParameters.get(self._edge, {}):
			return self._getScalar(param)
		_, eregion = config.getUnambiguousRegionEdgeParameter(self._edge, param)  # we don't need value, only eregion
		if eregion is not None:
			return getattr(data.eregion[eregion], param)
		return getattr(data.source, param)  # global parameter

	# get vector from either EDGES, EDGE_REGION, or GLOBAL_EDGE; or None
	def _fgetVector(self, param: str) -> vec_out:
		data = self._data
		config = data.config
		if param in config.localEdgeParameters.get(self._edge, {}):
			return self._getVector(param)
		_, eregion = config.getUnambiguousRegionEdgeParameter(self._edge, param)  # we don't need value, only eregion
		if eregion is not None:
			return getattr(data.eregion[eregion], param)
		return getattr(data.source, param)  # global parameter
	
	# set scalar in EDGES; or KeyError
	def _fsetScalar(self, param: str, value: scal_in) -> None:
		data = self._data
		config = data.config
		if param not in config.localEdgeParameters.get(self._edge, {}):
			raise AttributeError(f"For edge {self._edge}, can't modify {param} which was not defined locally in Config")
		index = data.config.edgeIndex[self._edge]
		setattr(data.source.edges[index], param, value)
	
	# set vector in EDGES; or KeyError.
	def _fsetVector(self, param: str, value: vec_in) -> None:
		data = self._data
		config = data.config
		if param not in config.localEdgeParameters.get(self._edge, {}):
			raise AttributeError(f"For edge {self._edge}, can't modify {param} which was not defined locally in Config")
		index = data.config.edgeIndex[self._edge]
		getattr(data.source.edges[index], param)[:3] = value

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

	def __init__(self, data: DataView, region):
		self._data = data
		self._region = region
	
	# get scalar only from REGIONS; or None
	def _getScalar(self, param: str) -> scal_out:
		data = self._data
		rid = data.config.regionId(self._region)
		return getattr(getattr(data.source, rid), param)
	
	# get vector only from REGIONS; or None
	def _getVector(self, param: str) -> vec_out:
		data = self._data
		rid = data.config.regionId(self._region)
		return getattr(getattr(data.source, rid), param)

	# get scalar from either REGION or GLOBAL_NODE; or None
	def _fgetScalar(self, param: str) -> scal_out:
		data = self._data
		if param in data.config.regionNodeParameters.get(self._region, {}):
			return self._getScalar(param)
		return getattr(data.source, param)  # global parameter
	
	# get vector from either REGION or GLOBAL_NODE; or None
	def _fgetVector(self, param: str) -> vec_out:
		data = self._data
		if param in data.config.regionNodeParameters.get(self._region, {}):
			return self._getVector(param)
		return getattr(data.source, param)  # global parameter

	# set scalar in REGION; or KeyError
	def _fsetScalar(self, param: str, value: scal_in) -> None:
		data = self._runtime
		if param not in data.config.regionNodeParameters.get(self._region, {}):
			raise AttributeError(f"For region {self._region}, can't modify {param} which was not defined in Config")
		rid = data.config.regionId(self._region)
		setattr(getattr(data.source, rid), param, value)
	
	# set vector in REGION; or KeyError
	def _fsetVector(self, param: str, value: vec_in) -> None:
		data = self._data
		if param not in data.config.regionNodeParameters.get(self._region, {}):
			raise AttributeError(f"For region {self._region}, can't modify {param} which was not defined in Config")
		rid = data.config.regionId(self._region)
		getattr(getattr(data.source, rid), param)[:3] = value

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

	def __init__(self, data: DataView, eregion: tuple):
		self._data = data
		self._eregion = eregion
	
	# get scalar only from EDGE_REGIONS; or None
	def _getScalar(self, param: str) -> scal_out:
		data = self._data
		e = self._eregion
		id = data.config.regionId  # function
		erid = f"{id(e[0])}_{id(e[1])}"
		return getattr(getattr(data.source, erid), param)
	
	# get vector only from EDGE_REGIONS; or None
	def _getVector(self, param: str) -> vec_out:
		data = self._data
		e = self._eregion
		id = data.config.regionId  # function
		erid = f"{id(e[0])}_{id(e[1])}"
		return getattr(getattr(data.source, erid), param)

	# get scalar from either EDGE_REGION or GLOBAL_EDGE; or None
	def _fgetScalar(self, param: str) -> scal_out:
		data = self._data
		if param in data.config.regionEdgeParameters.get(self._eregion, {}):
			return self._getScalar(param)
		return getattr(data.source, param)  # global parameter
	
	# get vector from either EDGE_REGION or GLOBAL_EDGE; or None
	def _fgetVector(self, param: str) -> vec_out:
		data = self._data
		if param in data.config.regionEdgeParameters.get(self._eregion, {}):
			return self._getVector(param)
		return getattr(data.source, param)  # global parameter

	# set scalar in EDGE_REGION; or KeyError
	def _fsetScalar(self, param: str, value: scal_in) -> None:
		data = self._data
		if param not in data.config.regionEdgeParameters.get(self._eregion, {}):
			raise AttributeError(f"For edge-region {self._eregion}, can't modify {param} which was not defined in Config")
		e = self._eregion
		id = data.config.regionId  # function
		erid = f"{id(e[0])}_{id(e[1])}"
		setattr(getattr(data.source, erid), param, value)
	
	# set vector in EDGE_REGION; or KeyError
	def _fsetVector(self, param: str, value: vec_in) -> None:
		data = self._data
		if param not in data.regionEdgeParameters.get(self._eregion, {}):
			raise AttributeError(f"For edge region {self._eregion}, can't modify {param} which was not defined in Config")
		e = self._eregion
		id = data.config.regionId  # function
		erid = f"{id(e[0])}_{id(e[1])}"
		getattr(getattr(data.source, erid), param)[:3] = value

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
	              "_data" ]

	def __init__(self, data: DataView):
		self._data = data
	
	def _getScalar(self, param: str) -> scal_out:
		data = self._data
		return getattr(data.source, param) if param in data.config.globalParameters else None
	
	def _getVector(self,  param: str) -> vec_out:
		data = self._data
		return getattr(data.source, param) if param in data.config.globalParameters else None
	
	def _setScalar(self, param: str, value: scal_in) -> None:
		try:
			getattr(self._data.source, param).value = value
		except AttributeError as ex:
			raise AttributeError(f"Global parameter {param} is undefined in Config") from ex

	def _setVector(self, param: str, value: vec_in) -> None:
		try:
			getattr(self._runtime.driver, param)[:3] = value
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

class NodeListProxy(Mapping):
	def __init__(self, data: DataView):
		self._data = data

	def __iter__(self):  return iter(self._data._nodes_view)
	def __len__(self):   return len(self._data._nodes_view)

	def __getitem__(self, node) -> NodeProxy:
		data = self._data
		if node not in data.config.nodes:
			raise KeyError(f"Node {node} is not defined in Config")
		return NodeProxy(data, node)

class EdgeListProxy(Mapping):
	def __init__(self, data: DataView):
		self._data = data
	
	def __iter__(self):  return iter(self._data._edges_view)
	def __len__(self):   return len(self._data._edges_view)
	
	def __getitem__(self, edge: tuple) -> EdgeProxy:
		data = self._data
		if edge not in data.config.edges:
			raise KeyError(f"Edge {edge} in not defined in Config")
		return EdgeProxy(data, edge)

	def __call__(self, node0, node1) -> EdgeProxy:
		return self[(node0, node1)]

class RegionListProxy(Mapping):
	def __init__(self, data: DataView):
		self._data = data
	
	def __iter__(self):  return iter(self._data._regions_view)
	def __len__(self):   return len(self._data._regions_view)
	
	def __getitem__(self, region) -> RegionProxy:
		data = self._data
		if region not in data.config.regions:
			raise KeyError(f"Region {region} is not defined in Config")
		return RegionProxy(data, region)

class ERegionListProxy(ReadOnlyDict):
	def __init__(self, data: DataView):
		self._data = data
	
	def __iter__(self):  return iter(self._data._eregions_view)
	def __len__(self):   return len(self._data._eregions_view)
	
	def __getitem__(self, eregion: tuple) -> ERegionProxy:
		data = self._data
		if eregion not in data.config.regionCombos:
			raise KeyError(f"Edge-region {eregion} in not defined in Config")
		return ERegionProxy(data, eregion)
	
	def __call__(self, region0, region1) -> ERegionProxy:
		return self[(region0, region1)]

class StateListProxy(AbstractReadableDict):
	def __init__(self, data: DataView, state: str):
		self._proxy_list = data.node
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
	def __init__(self, data: DataView, param: str):
		self._data = data
		self._param = param
	
	@property
	def name(self):
		return self._param

	@property
	def value(self):
		return getattr(self._data._globals_proxy, self.name)
	
	@value.setter
	def value(self, value) -> None:
		setattr(self._data._globals_proxy, self.name, value)

	def __getitem__(self, key):
		param = self.name
		data = self._data
		config = data.config
		if param in config.localNodeParameters.get(key, {}):
			return getattr(data.node[key], param)
		if param in config.localEdgeParameters.get(key, {}):
			return getattr(data.edge[key], param)
		if param in config.regionNodeParameters.get(key, {}):
			return getattr(data.region[key], param)
		if param in config.regionEdgeParameters.get(key, {}):
			return getattr(data.eregion[key], param)
		return self.value  # global parameter

	def __setitem__(self, key, value) -> None:
		param = self.name
		data = self._data
		config = data.config
		if param in config.localNodeParameters.get(key, {}):
			setattr(data.node[key], param, value)
		elif param in config.localEdgeParameters.get(key, {}):
			setattr(data.edge[key], param, value)
		elif param in config.regionNodeParameters.get(key, {}):
			setattr(data.region[key], param, value)
		elif param in config.regionEdgeParameters.get(key, {}):
			setattr(data.eregion[key], param, value)
		else:
			raise KeyError(f"Parameter {param} is not defined for (node, edge, region, or edge-region) {key}")
	
	def __str__(self) -> str:  return str(self.value)
	def __repr__(self) -> str: return repr(self.value)

# Acts like a float
class ScalarParameterProxy(ParameterProxy, Numeric):
	def __float__(self) -> float:  return self.value.value  # second .value converts (via copy) c_double -> float

# Acts like a tuple
class VectorParameterProxy(ParameterProxy, Numeric):
	def __iter__(self) -> vec_out:  return self.value
	def __len__(self) -> int:  return len(self.value)
