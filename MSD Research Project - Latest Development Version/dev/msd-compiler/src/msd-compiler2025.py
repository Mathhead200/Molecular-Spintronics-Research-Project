from typing import Callable, Generic, Iterable, Optional, Self, TypedDict, TypeVar
from datetime import datetime
from collections import defaultdict
import os
from math import log2, ceil

type vec = tuple[float, float, float]

class NodeParameters(TypedDict, total=False):
	spin: vec   # inital spin (None = random)
	flux: vec   # inital flux
	S: float    # spin magnatude
	F: float    # max flux magnitude
	kT: float   # temperature
	B: vec      # external magnetic field
	A: vec      # anisotropy
	Je0: float  # local (r=0) spin-flux exchange coupling constant

class EdgeParameters(TypedDict, total=False):
	J: float    # Heisenberg exchange coupling constant
	Je1: float  # neighboring (r=1) spin-flux exchange coupling constant
	Jee: float  # neighboring (r=1) flux-flux exchange coupling constant
	b: float    # biquadratic coupling
	D: vec      # Dzyaloshinsky–Moriya interaction

class NodeAndEdgeParameters(NodeParameters, EdgeParameters):
	pass

class ProgramParameters(TypedDict):
	t_eq: Optional[int]
	simCount: int
	freq: Optional[int]
	seed: Optional[list[int]]
	prng: str  # which algorithm?

Index = TypeVar("Index")    # type of node indicies; e.g., int, or tuple[int]
Region = TypeVar("Region")  # type of region names; e.g. str

class _Model(Generic[Index, Region], TypedDict, total=False):
	type Node = Index
	type Edge = tuple[Node, Node]

	type Nodes = Iterable[Node]
	type Edges = Iterable[Edge]

	nodes: Nodes
	edges: Edges  # tuple elemenets must be elements of (in) self.nodes
	mutableNodes: Nodes  # must be a subset of self.nodes
	
	globalParameters: NodeAndEdgeParameters

	regions: dict[Region, Nodes]
	regionNodeParameters: dict[Region, NodeParameters]
	regionEdgeParameters: dict[tuple[Region, Region], EdgeParameters]
	
	localNodeParameters: dict[Node, NodeParameters]
	localEdgeParameters: dict[Edge, EdgeParameters]

	programParameters: ProgramParameters

	# TODO: specify which parameters may change durring execution.
	#	Use this information to allow for optimization on constant parameters.
	#	e.g. 0: remove;  1: skip load and mul;  -1: _vneg
	variableParamters: NodeAndEdgeParameters  # set

	nodeId: Callable[[Node], str]      # returned str must conform to C identifier spec.
	regionId: Callable[[Region], str]  # returned str must conform to C identifier spec. Must be able to handle None as input.

def floats(values: Iterable) -> tuple[float]:
	return (float(x) for x in values)

class Model:
	def __init__(self, **kw):
		self.__dict__ = _Model(*kw)
		if "nodeId" not in self.__dict__:
			self.nodeId = lambda node: str(node)
		if "regionId" not in self.__dict__:
			self.regionId = lambda region: str(region)
		# fields set durring compilation 
		self.localNodeKeys: set[str] = None    # set[str]: which localNodeParameters are being used?
		self.localEdgeKeys: set[str] = None    # set[str]: which localEdgeParameters are being used?
		self.regionNodeKeys: set[str] = None   # set[str]: which regionNodeParameters are being used?
		self.regionEdgeKeys: set[str] = None   # set[str]: which regionEdgeParameters are being used?
		self.globalKeys: set[str] = None       # set[str]: which globalParameters are being used?
		self.allKeys: set[str] = None          # set[str]: which parameter keys are being used anywhere in the model? (union of all of keys sets.)
		self.immutableNodes: list = None  # list[Node]: disjoint of self.nodes - self.mutableNodes
		self.nodeIndex: dict = None       # map: (Node) -> (int) index in ASM nodes array
		self.edgeIndex: dict = None       # map: (Edge) -> (int) index in ASM edges array
		self.regionCombos: list[tuple] = None  # list[Edge] containing all region combos of 1 or 2 regions. Counter-part to self.regions, but for edges.
		self.constParameters: set[str] = None  # set[str in NodeAndEdgeParameters.keys()] which will not change durring runtime leading to optimizations

	def __repr__(self) -> str:
		return str(self.__dict__)
	
	@staticmethod
	def innerKeys(parameters: dict) -> set[str]:
		"""
		Computes a set of all the local/region parameter keys used at least
		once across all nodes/edges/regions in the system. e.g.
		<pre>
			msd = Model()
			# ...
			msd.localNodeParameters = {
				node1 : {kT: 0.1},
				node2 : {B: (0, 0.1, 0), F: 1}
			}
			msd.regionNodeParameters = {
				regionA : {kT: 0.2},
				regionB : {A: (0, 0, 0.1), S: 0.5}
			}
			msd.regionEdgeParameters = {
				edge1 : {J: -0.5},
				edge2 : {b: 0.25}
			}
			print(Model.innerKeys(self.localNodeParameters))   # {kT, B, F}
			print(Model.innerKeys(self.regionNodeParameters))  # {kT, A, S}
			print(Model.innerKeys(sele.regionEdgeParameters))  # {J, b}
		</pre>
		"""
		keys = set()
		for innerDict in parameters.values():
			keys |= innerDict.keys()
		return keys
	
	def calcImmutableNodes(self) -> Iterable:
		"""
		Computes all immutable nodes by calculating the disjoint of
		<code>self.nodes - self.mutableNodes</code>.
		The returned nodes will be in the same order they appear in
		<code>self.nodes</code>.
		"""
		M = set(self.mutableNodes)
		return [n for n in self.nodes if n not in M]

	def calcAllRegionCombos(self) -> list[tuple]:
		"""
		Computes all possible 1 or 2 region combinations. e.g.
		<pre>
		msd = Model()
		# ...
		msd.regions = {
			A:  # [... nodes ...],
			B:  # [... nodes ...],
			C:  # [... nodes ...],
		}
		print(msd.calcRegionCombos())
		# Output: [
		#	(A, None), (A, A), (A, B), (A, C), (None, A),
		#	(B, None), (B, A), (B, B), (B, C), (None, B),
		#	(C, None), (C, A), (C, B), (C, C), (None, C),
		# ]
		</pre>
		Notes:
		* Some of these combinatorically possible regions may contain zero edges.
		* (None, None) edges can be convered by self.globalParameters
		"""
		combos = []
		for src in self.regions:
			combos.append((src, None))
			for dest in self.regions:
				combos.append((src, dest))
			combos.append((None, src))
		return combos
	
	def calcFluxMode(self) -> bool:
		"""
		Returns True if any parameters are set which would require flux
		vectors for each node's state. Otherwise, False.
		Preconditon:
			* Must be called after self.__keys sets are initialized.
		"""
		return "F" in self.globalKeys + self.regionNodeKeys + \
			self.regionEdgeKeys + self.localNodeKeys + self.localEdgeKeys

	def getRegions(self, node) -> list:
		""" Get the list of all regions containing this node. May be empty list []. """
		return [label for label, subnodes in self.regions.items() if node in subnodes]
	
	def getRegionCombos(self, edge) -> list:
		"""
		Get the list of all (region, region) tuples possible for this edge.
		May be empty list [].
		Tuple elements may contain a None (e.g. (None, regionA)) exactly when one node is in no regions.
		"""
		R0 = self.getRegions(edge[0])
		R1 = self.getRegions(edge[1])
		
		if len(R0) == 0 and len(R1) == 0:
			return []
		if len(R0) == 0:
			return [(None, r1) for r1 in R1]
		if len(R1) == 0:
			return [(r0, None) for r0 in R0]
		# else:
		return [(r0, r1) for r0 in R0 for r1 in R1]
	
	def getAllRegionNodeParameterValues(self, node, param) -> list[tuple]:
		"""
		Find the value(s) defined at the region level of this parameter and node.
		Value is returned as a list of tuples: [(value, region), ...].
		The list may be empty if this parameter is not defined at the region
			level of any regions this node is in.
		Or it may have multiple values if multiple regions contain this node,
			define this same parameter. This should be okay as long as all the
			regions agree on the same value. Otherwise, th parameter value is
			ambiguous for this node.
		"""
		values = []
		for region in self.getRegions(node):
			regionParams = self.regionNodeParameters.get(region, {})
			if param in regionParams:
				values.append((regionParams[param], region))
		return values
	
	def getAllRegionEdgeParameterValues(self, edge, param) -> list[tuple]:
		""" 
		Similar to getAllRegionNodeParameterValues, but returned tuples contain
		[(value, combo), ...] where combo = tuple(region0, region1).
		"""
		values = []
		for combo in self.getRegionCombos(edge):
			regionParams = self.regionEdgeParameters.get(combo, {})
			if param in regionParams:
				values.append((regionParams[param], combo))
		return values

	def getUnambiguousRegionNodeParameter(self, node, param):
		"""
		Similar to getAllRegionNodeParameterValues, except
		(1) only returns one tuple, (value, region), not a list, and instead
		(2) raises an exception if the value is ambiguous.
		Returns (None, None) if the list of region values is empty.
		"""
		values = self.getAllRegionNodeParameterValues(node, param)
		if len(values) == 0:
			return None, None
		# all the values should be equal to the first value (transative property of equality)
		value, region = values[0]
		collisions = []
		for v, r in values:
			if v != value:
				collisions.append((v, r))
		if len(collisions) != 0:
			raise ValueError(
				f"For node {self.nodeId(node)}, {param} is ambiguously defined in multiple regions: {[r for _, r in values]}." + \
				f"{param} is defined as {value} in {region}, but also defined as {[f"{v} in {r}" for v, r in collisions]}." )
		return value, region
	
	def getUnambiguousRegionEdgeParameter(self, edge, param):
		""" 
		Similar to getUnambiguousRegionNodeParameter, but returned tuple contains
		(value, combo) where combo = tuple(region0, region1), or (None, None).
		"""
		values = self.getAllRegionEdgeParameterValues(edge, param)
		if len(values) == 0:
			return (None, None)
		# all the values should be equal to the first value (transative property of equality)
		value, combo = values[0]
		collisions = []
		for v, c in values:
			if v != value:
				collisions.append((v, c))
		if len(collisions) != 0:
			raise ValueError(
				f"For edge {self.nodeId(edge[0])} -> {self.nodeId(edge[1])}, {param} is ambiguously defined in multiple edge-regions: {[c for _, c in values]}." + \
				f"{param} is defined as {value} in {combo}, but also dedfined as {[f"{v} in {c}" for v, c in collisions]}." )
		return value, combo
	
	def hasNodeParameter(self, node, param) -> bool:
		"""
		Return True if this node has a definiton for the parameter at some level
		(local, region, or global). Otherwise, False.
		"""
		return param in self.globalKeys or \
			param in self.localNodeParameters.get(node, {}) or \
			len(self.getAllRegionNodeParameterValues(node, param)) != 0
	
	def hasEdgeParameter(self, node, param) -> bool:
		"""
		Return True if an edge connected to this node has a definiton for the
		parameter at some level (local, region, or global). Otherwise, False.
		"""
		edges = [edge for edge in self.edges if node in edge]
		combos = [combo for edge in edges for combo in self.getRegionCombos(edge)]
		return param in self.globalKeys \
			or any(param in self.localEdgeParameters.get(edge, {}) for edge in edges) \
			or any(param in self.regionEdgeParameters.get(combo, {}) for combo in combos)
	
	def hasFlux(self, node):  return self.hasNodeParameter(node, "F")
	def hasJe0(self, node):   return self.hasNodeParameter(node, "Je0")
	def hasJe1(self, node):   return self.hasEdgeParameter(node, "Je1")
	def hasJee(self, node):   return self.hasEdgeParameter(node, "Jee")

	def connections(self, node) -> list:
		""" List of edges containing (i.e. connected to) a node. """
		return [edge for edge in self.edges if node in edge]
	
	@staticmethod
	def neighbor(node, edge) -> tuple:
		"""
		Return (neighbor, direction):
			direction = 1: forward edge (node -> neighbor)
			direction = -1: backward edge (neightbor -> node)
			direction = 0: self-loop (node -> node)
		or (None, None) if edge isn't connected to node.
		"""
		if node == edge[0]:
			return (edge[1], 1 if node != edge[1] else 0)
		if node == edge[1]:
			return (edge[0], -1)
		return (None, None)
	
	def compile(self, out_path: str):
		# Check for and fix missing required attributes;
		if "nodes"                not in self.__dict__:  self.nodes = {}
		if "mutableNodes"         not in self.__dict__:  self.mutableNodes = self.nodes
		if "edges"                not in self.__dict__:  self.edges = {}
		if "regions"              not in self.__dict__:  self.regions = {}
		if "localNodeParameters"  not in self.__dict__:  self.localNodeParameters = {}
		if "localEdgeParameters"  not in self.__dict__:  self.localEdgeParameters = {}
		if "regionNodeParameters" not in self.__dict__:  self.regionNodeParameters = {}
		if "regionEdgeParameters" not in self.__dict__:  self.regionEdgeParameters = {}
		if "globalParameters"     not in self.__dict__:  self.globalParameters = {}
		if "variableParameters"   not in self.__dict__:  self.variableParameters = set()
		if "programParameters"    not in self.__dict__:  self.programParameters = {}

		if "prng" not in self.programParameters:
			self.programParameters["prng"] = "xoshiro256**"

		# aliases
		prng = self.programParameters["prng"]
		seed = self.programParameters.get("seed", None)

		# Do all nodes referenced in self.edges actually exist in self.nodes?
		for edge in self.edges:
			if len(edge) != 2:
				raise ValueError(f"All edges must have exactly 2 nodes: len={len(edge)} for edge {edge}")
			for node in edge:
				if node not in self.nodes:
					raise KeyError(f"All nodes referenced in model.edges must exist in model.nodes: Couldn't find node {node} referenced in edge {edge}")
				
		# Is PRNG algorithm supported?
		if prng not in ["xoshiro256**", "xoshiro256++", "xoshiro256+"]:
			raise ValueError(f"Unsupported psuedo-random number generator algorithm: {prng}. Suggested algorithm model.programParameters[\"prng\"] = \"xoshiro256**\"")
		if seed is not None:
			for s in seed:
				if s < 0 or s >= 2**64:
					raise ValueError(f"Seed value {s} is out-of-range [0, 2**64)")

		# other (dependant) variables
		self.localNodeKeys = Model.innerKeys(self.localNodeParameters)    # set[str]
		self.localEdgeKeys = Model.innerKeys(self.localEdgeParameters)    # set[str]
		self.regionNodeKeys = Model.innerKeys(self.regionNodeParameters)  # set[str]
		self.regionEdgeKeys = Model.innerKeys(self.regionEdgeParameters)  # set[str]
		self.globalKeys = self.globalParameters.keys()                    # set[str]
		self.allKeys = self.localNodeKeys | self.localEdgeKeys | self.regionNodeKeys | self.regionEdgeKeys | self.globalKeys
		self.constParameters = self.allKeys - self.variableParameters  # set[str]
		self.immutableNodes = self.calcImmutableNodes()  # list[Node]
		self.regionCombos = self.calcAllRegionCombos()  # list[Edge]
		self.nodeIndex = {}  # dict[Node, int]
		self.edgeIndex = {}  # dict[Edge, int]

		src = f"; Generated by {__name__} (python) at {datetime.now()}\n"
		# options
		src += "OPTION CASEMAP:NONE\n\n"
		# includes
		src += "include vec.inc  ; _vdotp, etc.\n"
		src += "include prng.inc  ; splitmix64, xoshiro256ss, etc. \n"
		src += "include dumpreg.inc  ; DEBUG\n\n"  # TODO: (DEBUG)
		
		# ---------------------------------------------------------------------
		src += ".data\n"
		# nodes:
		# allocate memory for "nodes" as an array of structures
		# generate a comment explaining nodes' memory structure
		src += "; struct node {\n"
		src += ";\tdouble spin[4];  // 32 bytes. last element is ignored.\n"
		if "F" in self.allKeys:
			src += ";\tdouble flux[4];  // 32 bytes. last element is ignored.\n"
		if "B" in self.localNodeKeys:
			src += ";\tdouble B[4];     // 32 bytes. last element is ignored.\n"
		if "A" in self.localNodeKeys:
			src += ";\tdouble A[4];     // 32 bytes. last element is ignored.\n"
		if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
			src += ";\tdouble S, F, kT, Je0;  // 32 bytes. unused parameters are ignored.\n"
		src += "; }\n"
		# symbolic constants related to nodes' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		src += f"OFFSETOF_SPIN EQU 32*({offset32})\n";  offset32 += 1
		if "F" in self.allKeys:
			src += f"OFFSETOF_FLUX EQU 32*({offset32})\n";  offset32 += 1
		if "B" in self.localNodeKeys:
			src += f"OFFSETOF_B    EQU 32*({offset32})\n";  offset32 += 1
		if "A" in self.localNodeKeys:
			src += f"OFFSETOF_A    EQU 32*({offset32})\n";  offset32 += 1
		if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
			if "S" in self.localNodeKeys:    src += f"OFFSETOF_S    EQU 32*({offset32}) + 8*(0)\n"
			if "F" in self.localNodeKeys:    src += f"OFFSETOF_F    EQU 32*({offset32}) + 8*(1)\n"
			if "kT" in self.localNodeKeys:   src += f"OFFSETOF_kT   EQU 32*({offset32}) + 8*(2)\n"
			if "Je0" in self.localNodeKeys:  src += f"OFFSETOF_Je0  EQU 32*({offset32}) + 8*(3)\n"
			offset32 += 1
		src += f"SIZEOF_NODE   EQU 32*({offset32})\n"
		src += f"MUTABLE_NODE_COUNT   EQU {len(self.mutableNodes)}\n"
		src += f"IMMUTABLE_NODE_COUNT EQU {len(self.immutableNodes)}\n"
		src += f"NODE_COUNT           EQU {len(self.nodes)}\n\n"
		# define memeory for nodes
		src += "NODES SEGMENT ALIGN(32)  ; AVX-256 (i.e. 32-byte) alignment\n"
		src += "nodes\t"
		for i, node in enumerate([*self.mutableNodes, *self.immutableNodes]):
			self.nodeIndex[node] = i  # store map: node -> array index
			if i != 0:  src += "\t\t"  # ASM formating: first struct must be on same line as symbol (i.e. "nodes")
			params = self.localNodeParameters.get(node, {})  # dict
			src += "dq 0.0, 1.0, 0.0, 0.0"  # init spin <0, 1, 0>
			if "F" in self.allKeys:
				src += ", \t0.0, 0.0, 0.0, 0.0"  # init flux <0, 0, 0>
			if "B" in self.localNodeKeys:
				Bx, By, Bz = floats(params.get("B", (0.0, 0.0, 0.0)))  # unpack generator
				src += f", \t{Bx}, {By}, {Bz}, 0.0"
			if "A" in self.localNodeKeys:
				Ax, Ay, Az = floats(params.get("A", (0.0, 0.0, 0.0)))  # unpack generator
				src += f", \t{Ax}, {Ay}, {Az}, 0.0"
			if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
				S = float(params.get("S", 0.0))
				F = float(params.get("F", 0.0))
				kT = float(params.get("kT", 0.0))
				Je0 = float(params.get("Je0", 0.0))
				src += f", \t{S}, {F}, {kT}, {Je0}"
			const = "const " if i >= len(self.mutableNodes) else ""
			regions = self.getRegions(node)
			regions = f" in {regions}" if len(regions) != 0 else ""
			src += f"  ; {const}nodes[{i}]: id=\"{self.nodeId(node)}\"{regions}\n"
		src += "NODES ENDS\n\n"

		# regions:
		# generate a comment explaining (node) regions' memory structure
		src += "; struct region {\n"
		if "B" in self.regionNodeKeys:
			src += ";\tdouble B[4];     // 32 bytes. last element is ignored.\n"
		if "A" in self.regionNodeKeys:
			src += ";\tdouble A[4];     // 32 bytes. last element is ignored.\n"
		if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
			src += ";\tdouble S, F, kT, Je0;  // 32 bytes. unused parameters are ignored.\n"
		src += "; }\n"
		# symbolic constants related to regions' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		if "B" in self.regionNodeKeys:
			src += f"OFFSETOF_REGION_B   EQU 32*({offset32})\n";  offset32 += 1
		if "A" in self.regionNodeKeys:
			src += f"OFFSETOF_REGION_A   EQU 32*({offset32})\n";  offset32 += 1
		if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
			if "S"   in self.regionNodeKeys:  src += f"OFFSETOF_REGION_S   EQU 32*({offset32}) + 8*(0)\n"
			if "F"   in self.regionNodeKeys:  src += f"OFFSETOF_REGION_F   EQU 32*({offset32}) + 8*(1)\n"
			if "kT"  in self.regionNodeKeys:  src += f"OFFSETOF_REGION_kT  EQU 32*({offset32}) + 8*(2)\n"
			if "Je0" in self.regionNodeKeys:  src += f"OFFSETOF_REGION_Je0 EQU 32*({offset32}) + 8*(3)\n"
			offset32 += 1
		src += f"SIZEOF_REGION EQU 32*({offset32})\n"
		src += f"REGION_COUNT  EQU {len(self.regionNodeParameters.keys())}\n\n"
		# define memeory for regions
		src += "REGIONS SEGMENT ALIGN(32)  ; AVX-256 (i.e. 32-byte) alignment\n"
		for region in self.regions:
			if region not in self.regionNodeParameters:
				src += f"; {self.regionId(region)}\n"
				continue  # skip defining this region. it has no special parameters.
			params = self.regionNodeParameters[region]  # dict
			src += f"  {self.regionId(region)}\tdq "
			if "B" in self.regionNodeKeys:
				Bx, By, Bz = floats(params.get("B", (0.0, 0.0, 0.0)))  # unpack generator
				src += f"{Bx}, {By}, {Bz}, 0.0,\t"
			if "A" in self.regionNodeKeys:
				Ax, Ay, Az = floats(params.get("A", (0.0, 0.0, 0.0)))  # unpack generator
				src += f"{Ax}, {Ay}, {Az}, 0.0,\t"
			if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
				S   = float(params.get("S",   0.0))
				F   = float(params.get("F",   0.0))
				kT  = float(params.get("kT",  0.0))
				Je0 = float(params.get("Je0", 0.0))
				src += f"{S}, {F}, {kT}, {Je0},\t"
			src = src[0:-2]  # remove last 2-char ",\t" delimeter
			src += "\n"
		src += "REGIONS ENDS\n\n"
		
		# global parameters (node only):
		src += "GLOBAL_NODE SEGMENT ALIGN(32)\n"
		params = self.globalParameters
		if "B" in self.globalKeys:
			Bx, By, Bz = floats(params["B"])  # unpack generator
			src += f"B   dq {Bx}, {By}, {Bz}, 0.0\n"
		if "A" in self.globalKeys:
			Ax, Ay, Az = floats(params["A"])  # unpack generator
			src += f"A   dq {Ax}, {Ay}, {Az}, 0.0\n"
		if any(p in self.globalKeys for p in ["S", "F", "kT", "Je0"]):
			S   = float(params.get("S",   0.0))
			F   = float(params.get("F",   0.0))
			kT  = float(params.get("kT",  0.0))
			Je0 = float(params.get("Je0", 0.0))
			src += f"S   dq {S}\n"
			src += f"F   dq {F}\n"
			src += f"kT  dq {kT}\n"
			src += f"Je0 dq {Je0}\n"
		src += "GLOBAL_NODE ENDS\n\n"

		# edges:
		# generate a comment explaining edges' memory structure
		src += "; struct edge {\n"
		if any(p in self.localEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
			src += ";\tdouble J, Je1, Jee, b;  // 32 bytes. unused fields ignored.\n"
		if "D" in self.localEdgeKeys:
			src += ";\tdouble D[4];            // 32 bytes. last element ignored.\n"
		src += "; }\n"
		# symbolic constants related to regions' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		if any(p in self.localEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
			if "J"   in self.localEdgeKeys:  src += f"OFFSETOF_J   EQU 32*({offset32}) + 8*(0)\n"
			if "Je1" in self.localEdgeKeys:  src += f"OFFSETOF_Je1 EQU 32*({offset32}) + 8*(1)\n"
			if "Jee" in self.localEdgeKeys:  src += f"OFFSETOF_Jee EQU 32*({offset32}) + 8*(2)\n"
			if "b"   in self.localEdgeKeys:  src += f"OFFSETOF_Je1 EQU 32*({offset32}) + 8*(3)\n"
			offset32 += 1
		if "D" in self.localEdgeKeys:
			src += f"OFFSETOF_D  EQU 32*({offset32})";  offset32 += 1
		src += f"SIZEOF_EDGE  EQU 32*({offset32})\n"
		src += f"EDGE_COUNT EQU {len(self.edges)}\n\n"
		# define memeory for edges
		src += "EDGES SEGMENT ALIGN(32)\n"
		if len(self.localEdgeKeys) == 0:
			src += "; edges  ; (0 bytes) array of empty edge structs\n"
		else:
			src += "edges\t"
			for i, edge in enumerate(self.edges):
				self.edgeIndex[edge] = i
				if i != 0:  src += "\t\t"  # ASM formating: first struct must be on same line as symbol (i.e. "edges")
				src += "dq "
				params = self.localEdgeParameters.get(edge, {})
				if any(p in self.localEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
					J   = float(params.get("J",   0.0))
					Je1 = float(params.get("Je1", 0.0))
					Jee = float(params.get("Jee", 0.0))
					b   = float(params.get("b",   0.0))
					src += f"{J}, {Je1}, {Jee}, {b},\t"
				if "D" in self.localEdgeKeys:
					Dx, Dy, Dz = floats(params.get("D", (0.0, 0.0, 0.0)))
					src += f"{Dx}, {Dy}, {Dz}, 0.0,\t"
				src = src[0:-2]  # remove last 2-char ",\t" delimeter
				regions = self.getRegionCombos(edge)
				regions = f" in {regions}" if len(regions) != 0 else ""
				src += f"  ; edges[{i}]: {self.nodeId(edge[0])} -> {self.nodeId(edge[1])}{regions}\n"
		src += "EDGES ENDS\n\n"

		# edge_regions:
		# generate a comment explaining edge_regions' memory structure
		src += "; struct edge_region {\n"
		if any(p in self.regionEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
			src += ";\tdouble J, Je1, Jee, b;  // 32 bytes. unused fields are ignored.\n"
		if "D" in self.regionEdgeKeys:
			src += ";\tdouble D[4];            // 32 bytes. last element ignored.\n"
		src += "; }\n"
		# symbolic constants related to edge_regions' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		if any(p in self.regionEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
			if "J"   in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_J   EQU 32*({offset32}) + 8*(0)\n"
			if "Je1" in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_Je1 EQU 32*({offset32}) + 8*(1)\n"
			if "Jee" in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_Jee EQU 32*({offset32}) + 8*(2)\n"
			if "b"   in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_b   EQU 32*({offset32}) + 8*(3)\n"
			offset32 += 1
		if "D" in self.regionEdgeKeys:
			src += f"OFFSETOF_REGION_D   EQU 32*({offset32})\n";  offset32 += 1
		src += "SIZEOF_EDGE_REGION  EQU 32*({offset32})"
		src += f"EDGE_REGION_COUNT EQU {len(self.regionCombos)}\n\n"
		# define memeory for edge_regions
		src += "EDGE_REGIONS SEGMENT ALIGN(32)\n"
		for r0, r1 in self.regionCombos:
			rid = f"{self.regionId(r0)}_{self.regionId(r1)}"
			if (r0, r1) not in self.regionEdgeParameters:
				src += f"; {rid}\n"
				continue  # skip defining this edge_region. it has no special parameters.
			params = self.regionEdgeParameters[(r0, r1)]
			src += f"  {rid}\tdq "
			if any(p in self.regionEdgeKeys for p in ["J", "Je1", "Jee", "b"]):
				J   = float(params.get("J",   0.0))
				Je1 = float(params.get("Je1", 0.0))
				Jee = float(params.get("Jee", 0.0))
				b   = float(params.get("b",   0.0))
				src += f"{J}, {Je1}, {Jee}, {b},\t"
			if "D" in self.regionEdgeKeys:
				Dx, Dy, Dz = floats(params.get("D", (0.0, 0.0, 0.0)))
				src += f"{Dx}, {Dy}, {Dz}, 0.0,\t"
			src = src[0:-2]  # remove last 2-char ",\t" delimeter
			src += "\n"
		src += "EDGE_REGIONS ENDS\n\n"

		# global parameters (edge only):
		src += "GLOBAL_EDGE SEGMENT ALIGN(32)\n"
		params = self.globalParameters
		if any(p in self.globalKeys for p in ["J", "Je1", "Jee", "b"]):
			J   = float(params.get("J",   0.0))
			Je1 = float(params.get("Je1", 0.0))
			Jee = float(params.get("Jee", 0.0))
			b   = float(params.get("b",   0.0))
			src += f"J   dq {J}\n"
			src += f"Je1 dq {Je1}\n"
			src += f"Jee dq {Jee}\n"
			src += f"b   dq {b}\n"
		if "D" in self.globalKeys:
			Dx, Dy, Dz = floats(params["D"])
			src += f"D   dq {Dx}, {Dy}, {Dz}, 0.0\n"
		src += "GLOBAL_EDGE ENDS\n\n"

		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			src += f"seed\tdq 0, 0, 0, 0  ; for {prng}\n"
		src += "\n"

		# deltaU array (function pointers):
		src += "; array of function pointers paralell to (mutable) nodes\n"
		src += "deltaU\t"
		if len(self.mutableNodes) == 0:
			src += "; 0 byte empty array. no multable nodes.\n"
		else:
			for i, node in enumerate(self.mutableNodes):
				if i != 0:  # loop index, not node index.
					src += "\t\t"  # ASM formating: first struct must be on same line as symbol (i.e. "dU")
				src += f"dq deltaU_{self.nodeId(node)}\n"
		src += "\n"

		# TODO: Place more paralell array of function pointers here as needed.
		
		# ---------------------------------------------------------------------
		src += ".code\n"
		# deltaU PROCs:
		# documentation for deltaU_{nodeId} PROCs
		src += "; @param ymm0:  \\Delta spin\n"
		flux: str = " (unused)" if "F" not in self.allKeys else ""
		src += f"; @param ymm1:  \\Delta flux{flux}\n"
		src += "; @retun xmm15: -\\Delta U (Negated for Boltzmann distribution.)\n\n"
		# code for deltaU_{nodeId} PROCs
		if len(self.mutableNodes) == 0:
			src += "; deltaU_{nodeId} PROCs missing. No mutable nodes!\n\n"
		else:
			for node in self.mutableNodes:
				nid = self.nodeId(node)
				proc_id = f"deltaU_{nid}"
				index = self.nodeIndex[node]
				hasFlux: bool = self.hasFlux(node)
				flux_mode: bool = hasFlux and (self.hasJe0(node) or self.hasJe1(node) or self.hasJee(node))

				src += f"; node[{index}]\n"
				src += f"{proc_id} PROC\n"
				# No preable: no local vars. so no stack space needed.
				# Set up some initial vars for calculations:
				#	(Inputs) ymm0-2: s', f', m';  ymm3-5: s, f, m;  ymm6-8: Δs, Δf, Δm;
				#	(Output) xmm15: -ΔU = -ΔU_B - ΔU_A - ΔU_Je0 - ΔU_J - ΔU_Je1 - ΔU_Jee - ΔU_b - ΔU_D
				reg_m1 = "ymm2"  # or ymm0 if m' == s'
				reg_m  = "ymm5"  # or ymm3 if m  == s
				reg_dm = "ymm8"  # or ymm6 if Δm == Δs
				out_init: bool = False  # track if output register has been initialized yet
				src += "\t; (param) ymm0: s' (new)\n"
				if flux_mode:
					src += "\t; (param) ymm1: f' (new)\n"
					src += "\tvaddpd ymm2, ymm0, ymm1  ; m' (new)\n"
				else:                # Not in flux mode:
					reg_m1 = "ymm0"  # m' == s'; just use the same register for both symbols
					reg_m  = "ymm3"  # m  == s ; just use the same register for both symbols
					reg_dm = "ymm6"  # Δm == Δs; just use the same register for both symbols
				src += f"\tvmovapd ymm3, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_SPIN]  ; s (current)\n"
				if flux_mode:
					src += f"\tvmovapd ymm4, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_FLUX]  ; f (current)\n"
					src += "\tvaddpd ymm5, ymm3, ymm4  ; m (current)\n"
				src += "\tvsubpd ymm6, ymm0, ymm3  ; \\Delta s\n"
				if flux_mode:
					src += "\tvsubpd ymm7, ymm1, ymm4  ; \\Delta f\n"
					src += "\tvsubpd ymm8, ymm2, ymm5  ; \\Delta m\n"
				src += "\n"
				# comments about what is being skipped
				s = ""
				if "B"   not in self.allKeys:  s += "\t; skipping -deltaU_B\n"
				if "A"   not in self.allKeys:  s += "\t; skipping -deltaU_A\n"
				if "Je0" not in self.allKeys:  s += "\t; skipping -deltaU_Je0\n"
				if "J"   not in self.allKeys:  s += "\t; skipping -deltaU_J\n"
				if "Je1" not in self.allKeys:  s += "\t; skipping -deltaU_Je1\n"
				if "Jee" not in self.allKeys:  s += "\t; skipping -deltaU_Jee\n"
				if "b"   not in self.allKeys:  s += "\t; skipping -deltaU_b\n"
				if "D"   not in self.allKeys:  s += "\t; skipping -deltaU_D\n"
				src += s
				if len(s) != 0:  src += "\n"
				# compute -ΔU_B = B ⋅ Δm_i:
				if "B" in self.allKeys:
					src += "\t; compute -delta_U_B\n"
					# where is B defined?
					load_insn = None  # How should this node be loaded?
					if "B" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovapd ymm9, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_B]  ; load B_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "B")  # don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovapd ymm9, ymmword ptr [{rid} + OFFSETOF_REGION_B]  ; load B_{rid} (region)\n"
						elif "B" in self.globalKeys:
							load_insn = "\tvmovapd ymm9, ymmword ptr B  ; load B (global)\n"
					if load_insn is None:
						src += f"\t; skip. For this node, B is not defined at any level (local, region, nor global).\n\n"
					else:
						src += load_insn  # load B into ymm9
						if not out_init:
							src += f"\t_vdotp xmm15, ymm9, {reg_dm}, xmm9, ymm9  ; (ymm9, {reg_dm})\n\n"
							out_init = True
						else:
							src += f"\t_vdotadd xmm15, ymm15, ymm9, {reg_dm}, xmm9  ; (ymm9, {reg_dm})\n\n"
				# compute -ΔU_A: A ⋅ (m'_i^{⊙2} - m_i^{⊙2})
				if "A" in self.allKeys:
					src += "\t; -deltaU_A calculation\n"
					# where is A defined?
					load_insn = None
					if "A" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovapd ymm9, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_A]  ; load A_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "A")  # don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovapd ymm9, ymmword ptr [{rid} + OFFSETOF_REGION_A]  ; load A_{rid} (region)\n"
						elif "A" in self.globalKeys:
							load_insn = "\tvmovapd ymm9, ymmword ptr A  ; load A (global)\n"
					if load_insn is None:
						src += f"\t; skip. For this node, A is not defined at any level (local, region, nor global).\n\n"
					else:
						src += load_insn  # load A into ymm9
						src += f"\tvmulpd ymm10, {reg_m1}, {reg_m1}  ; m' Hadamard squared: m'2\n"
						src += f"\tvfnmadd213pd ymm10, {reg_m}, {reg_m}  ; ymm10 -= {reg_m} * {reg_m} -> m'2 - m2\n"
						if not out_init:
							src += f"\t_vdotp xmm15, ymm9, ymm10, xmm9, ymm9  ; (ymm9, ymm10)\n\n"
							out_init = True
						else:
							src += f"\t_vdotadd xmm15, ymm15, ymm9, ymm10, xmm9  ; (ymm9, ymm10)\n\n"
				# compute -ΔU_Je0 = Je0 (s'⋅f' - s⋅f):
				if "Je0" in self.allKeys:
					src += "\t; -deltaU_Je0 calculation\n"
					# where is Jeo defined?
					load_insn = None
					if "Je0" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovsd xmm9, qword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_Je0]  ; load Je0_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "Je0")  # we don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovsd xmm9, qword ptr [{rid} + OFFSETOF_REGION_Je0]  ; load Je0_{rid} (region)\n"
						elif "Je0" in self.globalKeys:
							load_insn = f"\tvmovsd xmm9, qword ptr Je0  ; load Je0 (global)\n"
					# TODO: add optimization_remove_scalar
					# TODO: add optimization_neg_scalar
					if load_insn is None:
						src += "\t; skip. For this node, Je0 is not defined at any level (local, region, nor global).\n\n"
					elif not hasFlux:
						raise KeyError(f"For node {nid}, Je0 is defined, but F is not. Potential fix: model.globalParameters[F] = 0.0")
					else:
						# compute s_i'·f_i' - s_i·f_i (difference between new and current dot products for local spin·flux)
						src += "\t_vdotp xmm10, ymm0, ymm1, xmm9, ymm9       ; (ymm0, ymm1) = (s'f)\n"
						src += "\t_vndotadd xmm10, ymm10, ymm3, ymm4, xmm9   ; (ymm3, ymm4) = (s'f' - sf)\n"
						src += load_insn  # load Je0 into xmm9
						if not out_init:
							src += f"\tvmulsd xmm15, xmm9, xmm10\n\n"
							out_init = True
						else:
							src += f"\tvfmadd213sd xmm15, xmm9, xmm10  ; xmm15 += xmm9 * xmm10\n\n"
				# compute -ΔU_J = Σ_j{J Δs_i·s_j}:
				if "J" in self.allKeys:
					src += "\t; -deltaU_J calculation\n"
					# figure out where all neighboring edges load J from, and group the common load instructions
					load_groups: dict[str, list] = defaultdict(list)  # dict: str load_insn -> list[Edge]
					for edge in self.connections(node):
						if "J" in self.localEdgeParameters.get(edge, {}):
							eindex = self.edgeIndex[edge]
							nid0 = self.nodeId(edge[0])
							nid1 = self.nodeId(edge[1])
							load_insn = f"\tvmovsd xmm9, qword ptr [edges + ({eindex})*SIZEOF_EDGE + OFFSETOF_J]  ; load J_{nid0}_{nid1}\n"
						else:
							_, combo = self.getUnambiguousRegionEdgeParameter(edge, "J")  # don't need value, only region
							if combo is not None:
								rid0, rid1 = self.regionId(combo[0]), self.regionId(combo[1])
								load_insn = f"\tvmovsd xmm9, qword ptr [{rid0}_{rid1} + OFFSETOF_REGION_J]  ; load J_{rid0}_{rid1} (region)\n"
							elif "J" in self.globalKeys:
								load_insn = "\tvmovsd xmm9, qword ptr J  ; load J (global)\n"
							else:
								load_insn = None
						load_groups[load_insn].append(edge)
					for i, load_group in enumerate(load_groups.items(), 1):
						load_insn, edges = load_group
						optimization_remove_scalar = False  # TODO: implement optimization. if scalar is const and exactly 1.0
						optimization_neg_scalar = False     # TODO: implement optimization. if scalar is const and exactly -1.0
						src += f"\t; [load group {i}]\n"
						if optimization_remove_scalar or optimization_neg_scalar:
							src += "\t; optimization (J*=1,-1): skip load\n"
						else:
							src += load_insn  # load J into xmm9
						for edge in edges:
							# Note: edge in self.edgeIndex may be false if ASM edge array is empty or missing local variables for this edge. This is fine.
							edgelbl = f"edges[{self.edgeIndex[edge]}]" if edge in self.edgeIndex else "edge"
							nid0 = self.nodeId(edge[0])
							nid1 = self.nodeId(edge[1])
							neighbor, _ = Model.neighbor(node, edge)  # just need neighbor. communative operation: doesn't care about direction.
							nindex = self.nodeIndex[neighbor]
							nnid = self.nodeId(neighbor)  # neighbor's node id
							src += f"\t; {edgelbl}: {nid0} -> {nid1}\n"
							src += f"\tvmovapd ymm10, ymmword ptr [nodes + ({nindex})*SIZEOF_NODE + OFFSETOF_SPIN]  ; load s_{nnid} (neighbor)\n"
							if optimization_remove_scalar:
								if not out_init:
									src += "\t_vdotp xmm15, ymm6, ymm10, xmm9, ymm9  ; optimization (J*=1), (ymm6, ymm10)\n"
									out_init = True
								else:
									src += "\t_vdotadd xmm15, ymm15, ymm6, ymm10, xmm9  ; optimization (J*=1), (ymm6, ymm10)\n"
							elif optimization_neg_scalar:
								if not out_init:
									src += "\tvxorsd xmm15, xmm15, xmm15   ; init. xmm15 = 0\n"
									out_init = True
								src += "\t_vndotadd xmm15, ymm15, ymm6, ymm10, xmm9  ; optimization (J*=-1), (ymm6, ymm10)\n"
							else:
								src += f"\t_vdotp xmm10, ymm6, ymm10, xmm11, ymm11  ; (ymm6, ymm10)\n"
								if not out_init:
									src += f"\tvmulsd xmm15, xmm9, xmm10\n"
									out_init = True
								else:
									src += f"\tvfmadd213sd xmm15, xmm9, xmm10  ; xmm15 += xmm9 * xmm10\n"
					src += "\n"

				# compute -ΔU_Je1:
				if "Je1" in self.allKeys:
					src += "\t; -deltaU_Je1 calculation\n"
					src += "\t; TODO ...\n\n" # TODO (stub)
				# compute -ΔU_Jee:
				if "Jee" in self.allKeys:
					src += "\t; -deltaU_Jee calculation\n"
					src += "\t; TODO ...\n\n" # TODO (stub)
				# compute -ΔU_b:
				if "b" in self.allKeys:
					src += "\t; -deltaU_b calculation\n"
					src += "\t; TODO ...\n\n" # TODO (stub)
				# compute -ΔU_D:
				if "D" in self.allKeys:
					src += "\t; -deltaU_D calculation\n"
					src += "\t; TODO ...\n\n" # TODO (stub)
				# return:
				if not out_init:
					src += "\tvxorpd xmm15, xmm15, xmm15  ; return 0.0\n"  # fall back in case there are no parameters set for this node
				src += "\tret\n"
				src += f"{proc_id} ENDP\n\n"
		
		# metropolis PROC
		src += "; Runs the standard metropolis algorithm\n"
		src += "; @param RCX (uint64) - number of iterations\n"
		src += "; @return (void)\n"
		src += "PUBLIC metropolis\n"
		src += "metropolis PROC\n"
		# src += "\t; preamble. needed if using local variables (i.e. stack meory)\n"
		# src += "\tpush rbp\n"
		# src += "\tmov rbp, rsp\n\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			src += f"\t; load PRNG ({prng}) state\n"
			src += "\tmov r8, qword ptr [seed + (0)*8]\n"
			src += "\tmov r9, qword ptr [seed + (1)*8]\n"
			src += "\tmov r10, qword ptr [seed + (2)*8]\n"
			src += "\tmov r11, qword ptr [seed + (3)*8]\n"
		src += "\t; load mutable nodes array size\n"
		src += "\tmov rdi, NODE_COUNT\n\n"
		src += "\tmov rbx, rcx  ; non-volitile loop counter\n"  # TODO: do we actaully call any functions in metropolis that need RCX (1st arg)?
		src += "\tcmp rbx, 0\n"
		src += "\tLOOP_START:\n"
		src += "\t\tjz LOOP_END\n\n"
		# select random node
		src += "\t\t; select random index, rdx, of a (mutable) node\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			macro = prng.replace("*", "s").replace("+", "p")
			src += f"\t\t_{macro} rax, r8, r9, r10, r11, rdx\n"
			src += "\t\tmul rdi  ; rdx:rax = rax * rdi = random * NODE_COUNT\n\n"
		# TODO: (stub) pick uniformally random new state for the node
		src += "\t\t; pick uniformally random new state for the node\n"
		src += "\t\t_vputj ymm0  ; TODO: (stub) new spin, s'=-J\n"
		src += "\t\t_vneg ymm0, ymm0, ymm1\n"
		src += "\t\t_vput0 ymm1  ; TODO: (stub) new flux, f'=0\n\n"
		# compute -deltaU for the  state change
		src += "\t\t; compute -deltaU for the proposed state change\n"
		src += "\t\tlea rax, deltaU         ; pointer to array of function pointers\n"
		src += "\t\tmov rax, [rax + rsi*8]  ; deltaU[rsi], dereferenced to get the actual functino pointer\n"
		src += "\t\tcall rax  ; args: (ymm0, ymm1?) -> return xmm15\n\n"
		# TODO: (stub) compute probability, p = e^{-deltaU/kT}
		src += "\t\t; compute probability, p = e^(-deltaU/kT)\n"
		src += "\t\t; TODO: (stub)\n\n"
		# TODO: (stub) (maybe) change the node's state
		src += "\t\t; (maybe) change the node's state\n"
		src += "\t\t; TODO: (stub)\n\n"
		# TODO?: parameter modification(s); e.g. global.B -= dB
		# ...
		src += "\t\tdec rbx\n"
		src += "\t\tjmp LOOP_START\n"
		src += "\tLOOP_END:\n\n"
		# src += "pop rbp\n"
		src += "\tret\n"
		src += "metropolis ENDP\n\n"

		# TODO: (DEBUG) test main
		src += "PUBLIC main\n"
		src += "main PROC\n"
		src += f"\t;seed PRNG ({prng})\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			if seed is None or len(seed) == 0:
				# use hardware entropy for initial seed
				cores = os.cpu_count()
				rot = ceil(log2(cores))
				src += "\trdseed rcx   ; try hardware entropy\n"
				src += "\tjc TSC_END   ; skip timestamp counter (TSC) method on success\n"
				src += "\trdtscp       ; sets edx:eax (TSC) and ecx (processor core). high bits 0ed.\n"
				src += "\tshl rdx, 32  ; reconstruct 64-bit TSC in rax\n"
				src += "\tor rax, rdx\n"
				if rot % 64 == 0:
					src += ";"  # skip rotate right if it would have no effect
				src += f"\tror rcx, {rot}    ; rotate small number to highest order bits (CPU cores = {cores})\n"
				src += "\txor rcx, rax\n"
				src += "\tTSC_END:       ; initial seed is now in rcx. Now SplitMix64.\n"
				for i in range(4):
					src += "\t_splitmix64 rax, rcx, rdx\n"
					src += f"\tmov qword ptr [seed + ({i})*8], rax\n"
			else:
				# use given seed, plus SplitMix64 if len < 4
				for i, s in enumerate(seed):
					src += f"\tmov qword [seed + ({i})*8], {int(s)}\n"
				if len(seed) < 4:
					src += f"\tmov rcx, {int(seed[-1])}\n"
					for i in range(len(seed), 4):
						src +="\t_splitmix64 rax, rcx, rdx\n"
						src += f"\tmov qword ptr [sedd + ({i})*8], rax\n"
			src += "\n"
		src += "\tmov rcx, 1  ; number of iterations\n"
		src += "\t _dumpreg\n"
		src += "\tcall metropolis\n"
		src += "\t _dumpreg\n\n"
		src += "\txor rax, rax  ; return 0\n"
		src += "\tret\n"
		src += "main ENDP\n\n"

		# ---------------------------------------------------------------------
		src += "END"  # absolute end of ASM file

		with open(out_path, "w", encoding="utf-8") as file:
			file.write(src)
		
		# TODO: compile/assemble
		# TODO: dynamically link to python??


# Examples:
def example_3d():
	width = 11
	height = 10
	depth = 10
	molPosL = 5
	molPosR = 5
	topL = 3
	bottomL = 6
	frontR = 3
	backR = 6

	msd = Model()
	msd.edges = []
	
	fml = []
	for x in range(0, molPosL):
		for y in range(topL, bottomL + 1):
			for z in range(0, depth):
				fml.append((x, y, z))
				# internal FML_FML edges:
				if x + 1 < molPosL:
					msd.edges.append(((x, y, z), (x + 1, y, z)))
				if y + 1 <= bottomL:
					msd.edges.append(((x, y, z), (x, y + 1, z)))
				if z + 1 < depth:
					msd.edges.append(((x, y, z), (x, y, z + 1)))
	
	mol = []
	for x in range(molPosL, molPosR + 1):
		for y in range(topL, bottomL + 1):
			for z in range(frontR, backR + 1):
				if y == topL or y == bottomL or z == frontR or z == backR:
					mol.append((x, y, z))
					# internal mol_mol edges:
					if x + 1 <= molPosR:
						msd.edges.append(((x, y, z), (x + 1, y, z)))
					# FML_mol edges:
					if molPosL - 1 >= 0:
						msd.edges.append(((molPosL - 1, y, z), (molPosL, y, z)))
					# mol_FMR edges:
					if molPosR + 1 < width:
						msd.edges.append(((molPosR, y, z), (molPosR + 1, y, z)))
	
	fmr = []
	for x in range(molPosR + 1, width):
		for y in range(0, height):
			for z in range(frontR, backR + 1):
				fmr.append((x, y, z))
				# internal FMR_FMR edges
				if x + 1 < width:
					msd.edges.append(((x, y, z), (x + 1, y, z)))
				if y + 1 < height:
					msd.edges.append(((x, y, z), (x, y + 1, z)))
				if z + 1 <= backR:
					msd.edges.append(((x, y, z), (x, y, z + 1)))
	
	# LR direct coupling
	for y in range(topL, bottomL + 1):
		for z in range(frontR, backR + 1):
			if y == topL or y == bottomL or z == frontR or z == backR:
				# FML_FMR edges:
				if molPosL - 1 >= 0 and molPosR + 1 < width:
					msd.edges.append(((molPosL - 1, y, z), (molPosR + 1, y, z)))

	msd.regions = { "FML": fml, "mol": mol, "FMR": fmr }
	msd.nodes = fml + mol + fmr
	msd.nodeId = lambda node: f"{node[0]}_{node[1]}_{node[2]}"
	# msd.mutableNodes = msd.nodes  # now automatic

	msd.globalParameters = {
		"kT": 0.25,
		"S": 1,
		"J": 1
	}
	msd.regionNodeParameters = {
		"mol": { "S": 10 }
	}
	msd.regionEdgeParameters = {
		("mol", "FMR"): { "J": -1 }
	}
	msd.programParameters = {
		"simCount": 1000000,
		"freq": 50000
	}

	# testing localNodeParameters
	def is_dictesk(obj):
		return hasattr(obj, "keys") and callable(obj.keys) and hasattr(obj, "__getitem__")
	def deep_union_2(a, b):
		u = {**a, **b}
		for k in a.keys() & b.keys():
			v1, v2 = a[k], b[k]
			if is_dictesk(v1) and is_dictesk(v2):
				u[k] = deep_union_2(v1, v2)
		return u
	def deep_union(*dicts):
		from functools import reduce
		return reduce(deep_union_2, dicts, {})
	
	# testing immutableNodes()
	msd.mutableNodes = [*msd.nodes]
	for y in range(topL, bottomL + 1):
		for z in range(0, depth):
			msd.mutableNodes.remove((0, y, z))
	for y in range(0, height):
		for z in range(frontR, backR + 1):
			msd.mutableNodes.remove((width - 1, y, z, ))

	msd.localNodeParameters = deep_union({
		(0, topL, z): { "S": 2 } for z in range(depth)
	}, {
		(0, y, 0): { "F": 1 } for y in range(topL, bottomL + 1)
	})

	return msd

	# TODO: (though) Currently, msd.nodes mst be defined, and msd.mutableNodes
	#	is optional, then self.immutableNodes gets computed. Is this the best
	#	pattern for the user? SHould we allow alternate patterns like defining
	#	self.nodes and self.immutableNodes; or
	#	self.mutabelNodes and self.immutableNodes: forcing them to specify?
	#	(idk)

def example_1d():
	# 1D model with the classic 3 sections.
	#
	#   |  FML  |mol|  FMR  |
	# 0*--1---2---3---4---5---6*
	# ^        \     /        ^ {0,6} are leads and immutable
	#           \---/ Direct coupling (2,4) (e.g. JLR)

	msd = Model()
	msd.nodes = [0, 1, 2, 3, 4, 5, 6]
	msd.mutableNodes = [1, 2, 3, 4, 5]  # excluding 0, 6
	msd.edges = [(i, i+1) for i in range(len(msd.nodes) - 1)]
	msd.edges += [(2, 4)]
	msd.globalParameters = {
		"S": 1.0,
		"F": 0.5,
		"kT": 0.1,
		"B": (0.1, 0, 0),
		"J": 1.0
	}
	msd.regions = {
		"FML": [1, 2],
		"mol": [3],
		"FMR": [4, 5]
	}
	msd.regionNodeParameters = {
		"FML": { "A": (0.0, 0.0, 0.2) },
		"mol": { "Je0": 3.0 },
		"FMR": { "A": (0.0, 0.0, -0.2) }
	}
	msd.regionEdgeParameters = {
		("mol", "FMR"): { "J": -1.0 },
		("FML", "FMR"): { "J": 0.1 }
	}
	msd.localNodeParameters = {
		0: { "S": 10.0 },
		6: { "S": 10.0 }
	}
	return msd

# tests
if __name__ == "__main__":
	msd = example_1d()
	msd.compile("out/msd-example_1d.asm")

	msd = example_3d()
	msd.compile("out/msd-example_3d.asm")
