from typing import Callable, Generic, Iterable, Optional, Self, TypedDict, TypeVar
from datetime import datetime

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
	seed: Optional[int]

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
				f"For node {self.nodeId(node)}, {param} is ambiguously defined in multiple regions: {[r for _, r in values]}.\n" + \
				f"{param} is defined as {value} in {region}, but also defined as {[f"{v} in {r}" for v, r in collisions]}.\n" )
		return value, region
	
	def hasFlux(self, node) -> bool:
		return "F" in self.globalKeys or \
			"F" in self.localNodeParameters.get(node, {}) or \
			len(self.getAllRegionNodeParameterValues(node, "F")) != 0
	
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

		# other (dependant) variables
		self.localNodeKeys = Model.innerKeys(self.localNodeParameters)    # set[str]
		self.localEdgeKeys = Model.innerKeys(self.localEdgeParameters)    # set[str]
		self.regionNodeKeys = Model.innerKeys(self.regionNodeParameters)  # set[str]
		self.regionEdgeKeys = Model.innerKeys(self.regionEdgeParameters)  # set[str]
		self.globalKeys = self.globalParameters.keys()                    # set[str]
		self.allKeys = self.localNodeKeys | self.localEdgeKeys | self.regionNodeKeys | self.regionEdgeKeys | self.globalKeys
		self.immutableNodes = self.calcImmutableNodes()  # list[Node]
		self.regionCombos = self.calcAllRegionCombos()  # list[Edge]
		self.nodeIndex = {}  # dict[Node, int]
		self.edgeIndex = {}  # dict[Edge, int]

		src = f"; Generated by {__name__} (python) at {datetime.now()}\n"
		# options
		src += "OPTION CASEMAP:NONE\n\n"
		# includes
		src += "include vec.inc  ; _vdotp, etc.\n\n"
		
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
				flux_mode: bool = self.hasFlux(node)  # TODO: (OPTIMIZATION) Also check if has Je0, Je1, or Jee!

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
				# compute -ΔU_B = B \cdot Δm_i:
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
						else:
							# raise KeyError(
							# 	f"For node {self.nodeId(node)}, B is not defined at any level (local, region, nor global).\n" + \
							# 	"Potential fix: model.globalParameters[B] = (0.0, 0.0, 0.0)\n" )
							pass
					if load_insn is None:
						src += f"\t; skip. For this node, B is not defined at any level (local, region, nor global).\n\n"
					else:
						# load B
						src += load_insn
						# do math (B)
						if not out_init:
							src += f"\t_vdotp xmm15, ymm9, {reg_dm}, xmm9, ymm9  ; clobbers ymm9!\n\n"
							out_init = True
						else:
							src += f"\t_vdotp xmm10, ymm9, {reg_dm}, xmm9, ymm9  ; cobbers ymm9!\n"
							src += "\tvaddsd xmm15, xmm15, xmm10\n\n"
				# compute -ΔU_A:
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
						else:
							# raise KeyError(
							# 	f"For node {self.nodeId(node)}, A is not defined at any level (local, region, nor global).\n" + \
							# 	"Potential fix: model.globalParameters[A] = (0.0, 0.0, 0.0)\n" )
							pass
					if load_insn is None:
						src += f"\t; skip. For this node, A is not defined at any level (local, region, nor global).\n\n"
					else:
						# Hadamard squared diff first so we can reuse a register
						src += f"\tvmulpd ymm9, {reg_m}, {reg_m}  ; m Hadamard squared: m2\n"
						src += f"\tvmulpd ymm10, {reg_m1}, {reg_m1}  ; m' Hadamard squared: m'2\n"
						src += "\tvsubpd ymm10, ymm10, ymm9  ; m'2 - m2\n"
						# load A
						src += load_insn
						# do math (A)
						if not out_init:
							src += f"\t_vdotp xmm15, ymm10, ymm9, xmm9, ymm9  ; clobbers ymm9!\n\n"
							out_init = True
						else:
							src += f"\t_vdotp xmm10, ymm10, ymm9, xmm9, ymm9  ; clobbers ymm9!\n"
							src += "\tvaddsd xmm15, xmm15, xmm10\n\n"
				# compute -ΔU_Je0:
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
					if load_insn is None:
						src += "\t; skip. For this node, Je0 is not defined at any level (local, region, nor global).\n\n"
					elif not flux_mode:
						raise KeyError(f"For node {nid}, Je0 is defined, but F is not.\nPotential fix: model.globalParameters[F] = 0.0\n")
					else:
						# compute s'·f' - s·f (difference between new and current dot products for local spin·flux)
						src += "\t_vdotp xmm10, ymm0, ymm1, xmm11, ymm11  ; s'f'\n"
						src += "\t_vdotp xmm9, ymm3, ymm4, xmm11, ymm11   ; sf\n"
						src += "\tvsubsd xmm10, xmm10, xmm9               ; s'f - sf\n"
						src += load_insn  # load Je0
						if not out_init:
							src += f"\tvmulsd xmm15, xmm10, xmm9  ; (s'f' - sf)(Je0)\n\n"
							out_init = True
						else:
							src += f"\tvmulsd xmm10, xmm10, xmm9  ; (s'f' - sf)(Je0)\n"
							src += f"\tvaddsd xmm15, xmm15, xmm10\n\n"
				# compute -ΔU_J:
				if "J" in self.allKeys:
					src += "\t; -deltaU_J calculation\n"
					src += "\t; TODO ...\n\n" # TODO (stub)
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
				src += "\tret\n"
				src += f"{proc_id} ENDP\n\n"

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
				if x + 1 < molPosL:
					msd.edges.append(((x, y, z), (x + 1, y, z)))
				if y + 1 <= bottomL:
					msd.edges.append(((x, y, z), (x, y + 1, z)))
				if z + 1 < depth:
					msd.edges.append(((x, y, z), (z, y, z + 1)))
	
	mol = []
	for x in range(molPosL, molPosR + 1):
		for y in range(topL, bottomL + 1):
			for z in range(frontR, backR + 1):
				if y == topL or y == bottomL or z == frontR or z == backR:
					mol.append((x, y, z))
					if x + 1 <= molPosR:
						msd.edges.append(((x, y, z), (x + 1, y, z)))
	
	fmr = []
	for x in range(molPosR + 1, width):
		for y in range(0, height):
			for z in range(frontR, backR + 1):
				fmr.append((x, y, z))
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
	msd.compile("msd-example_1d.asm")

	msd = example_3d()
	msd.compile("msd-example_3d.asm")
