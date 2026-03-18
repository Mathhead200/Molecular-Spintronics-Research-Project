from __future__ import annotations
from ..config import Config
from ..util import ordered_set, ReadOnlyOrderedSet, ReadOnlyDict, NODE_PARAMETERS, EDGE_PARAMETERS
from itertools import chain

NODE_PARAMETER_SET = set(NODE_PARAMETERS)
EDGE_PARAMETER_SET = set(EDGE_PARAMETERS)

class ConfigData:
	__slots__ = ("nodes", "edges", "regions", "eregions", "parameters")

	def __init__(self, config: Config):
		# Just need to configure nodes, edges, regions, and eregions once and cache
		edges = config.edges    # list[Edge]
		rnodes = config.regions  # Region -> list[Node]
		self.nodes = ReadOnlyOrderedSet(ordered_set(config.nodes))
		self.edges = ReadOnlyOrderedSet(ordered_set(edges))
		self.regions = ReadOnlyDict({
			region: ReadOnlyOrderedSet(ordered_set(nodes))
			for region, nodes in rnodes.items()
		})
		for region in rnodes:  rnodes[region] = set(rnodes[region])  # faster lookup for eregions
		rnodes[None] = set(config.nodes) - set(chain(*rnodes.values()))  # "None" region for eregion lookup
		self.eregions = {
			eregion: [
				edge
				for edge in edges
				if edge[0] in rnodes[eregion[0]] and edge[1] in rnodes[eregion[1]]
			] for eregion in config.regionCombos
		}
		self.eregions = ReadOnlyDict({
			eregion: ReadOnlyOrderedSet(ordered_set(redges))
			for eregion, redges in self._eregions.items()
			if len(redges) != 0
		})
		# set of defined parameters, preserving order defined in Config:
		node_p =    [ p for params in config.localNodeParameters.values()  for p in params ]
		edge_p =    [ p for params in config.localEdgeParameters.values()  for p in params ]
		region_p =  [ p for params in config.regionNodeParameters.values() for p in params ]
		eregion_p = [ p for params in config.regionEdgeParameters.values() for p in params ]
		global_p =  [ p for p in config.globalParameters.keys() ]
		parameters = {}
		for p in chain(global_p, region_p, eregion_p, node_p, edge_p):
			if p in NODE_PARAMETER_SET:
				parameters[p] = ReadOnlyOrderedSet(ordered_set(node for node in config.nodes if config.hasNodeParameter(node, p)))
			else:
				assert p in EDGE_PARAMETER_SET
				parameters[p] = ReadOnlyOrderedSet(ordered_set(edge for edge in config.edges if config.hasEdgeParameter(edge, p)))
		self.parameters = ReadOnlyDict(parameters)


config_data_cache: dict[Config, ConfigData] = {}

def from_config(config: Config) -> ConfigData:
	# Assumes config is immutable after first call
	try:
		return config_data_cache[config]
	except KeyError:
		data = ConfigData(config)
		config_data_cache[config] = data
		return data
