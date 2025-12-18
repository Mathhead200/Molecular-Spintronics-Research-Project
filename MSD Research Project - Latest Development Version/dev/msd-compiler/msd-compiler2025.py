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
	regionId: Callable[[Region], str]  # returned str must conform to C identifier spec.

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
		self.immutableNodes: list = None  # list[Node]: disjoint of self.nodes - self.mutableNodes
		self.nodeIndex: dict = None       # map: (Node) -> (int) index in nodes array

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

	def calcRegionNodeKeys(self) -> set[str]:
		"""
		Compute a set of all the region paramaters used at least once across
		all regions in the system. e.g.
		<pre>
			msd = Model()
			# ...
			
			print(msd.calcRegionNodeKeys())  # {kT, B, F}
		</pre>
		"""
		regionNodeKeys = set()
		for regionParameters in self.regionNodeParameters.values():
			regionNodeKeys |= regionParameters

	def calcImmutableNodes(self) -> Iterable:
		"""
		Computes all immutable nodes by calculating the disjoint of
		<code>self.nodes - self.mutableNodes</code>.
		The returned nodes will be in the same order they appear in
		<code>self.nodes</code>.
		"""
		M = set(self.mutableNodes)
		return [n for n in self.nodes if n not in M]

	def getRegions(self, node) -> list:
		""" Get the list of all regions containing this node. May be empty list []. """
		return [label for label, subnodes in self.regions.items() if node in subnodes]

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
		# TODO: add more checks?

		# other (dependant) variables
		self.localNodeKeys = Model.innerKeys(self.localNodeParameters)    # set[str]
		self.localEdgeKeys = Model.innerKeys(self.localEdgeParameters)    # set[str]
		self.regionNodeKeys = Model.innerKeys(self.regionNodeParameters)  # set[str]
		self.regionEdgeKeys = Model.innerKeys(self.regionEdgeParameters)  # set[str]
		self.globalKeys = self.globalParameters.keys()                    # set[str]
		self.immutableNodes = self.calcImmutableNodes()  # list[Node]
		self.nodeIndex = {}                              # dict[Node, int]

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
		src += ";\tdouble spin[4];  // 32-bytes. last element is ignored.\n"
		src += ";\tdouble flux[4];  // 32-bytes. last element is ignored.\n"
		if "B" in self.localNodeKeys:
			src += ";\tdouble B[4];     // 32-bytes. last element is ignored.\n"
		if "A" in self.localNodeKeys:
			src += ";\tdouble A[4];     // 32-bytes. last element is ignored.\n"
		if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
			src += ";\tdouble S, F, kT, Je0;  // 32-bytes. unused parameters are ignored.\n"
		src += "; }\n"
		# symbolic constants related to nodes' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		src += f"OFFSETOF_SPIN EQU 32*({offset32})\n";  offset32 += 1
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
		# generate a comment explaining nodes' memory structure
		src += "; struct region {\n"
		if "B" in self.regionNodeKeys:
			src += ";\tdouble B[4];     // 32-bytes. last element is ignored.\n"
		if "A" in self.regionNodeKeys:
			src += ";\tdouble A[4];     // 32-bytes. last element is ignored.\n"
		if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
			src += ";\tdouble S, F, kT, Je0;  // 32-bytes. unused parameters are ignored.\n"
		src += "; }\n"
		# symbolic constants related to regions' memory structure
		offset32 = 0  # tracks current position as we iterate through struct in 32-byte chunks
		if "B" in self.regionNodeKeys:
			src += f"OFFSETOF_REGION_B   EQU 32*({offset32})\n";  offset32 += 1
		if "A" in self.localNodeKeys:
			src += f"OFFSETOF_REGION_A   EQU 32*({offset32})\n";  offset32 += 1
		if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
			if "S" in self.localNodeKeys:    src += f"OFFSETOF_REGION_S   EQU 32*({offset32}) + 8*(0)\n"
			if "F" in self.localNodeKeys:    src += f"OFFSETOF_REGION_F   EQU 32*({offset32}) + 8*(1)\n"
			if "kT" in self.localNodeKeys:   src += f"OFFSETOF_REGION_kT  EQU 32*({offset32}) + 8*(2)\n"
			if "Je0" in self.localNodeKeys:  src += f"OFFSETOF_REGION_Je0 EQU 32*({offset32}) + 8*(3)\n"
			offset32 += 1
		src += f"SIZEOF_REGION EQU 32*({offset32})\n"
		src += f"REGION_COUNT  EQU {len(self.regionNodeParameters.keys())}\n\n"
		# define memeory for regions
		src += "REGIONS SEGMENT ALIGN(32)  ; AVX-256 (i.e. 32-byte) alignment\n"
		for region in self.regions:
			if region not in self.regionNodeParameters:
				continue  # skip defining this region. it has no special parameters.
			src += f"{self.regionId(region)}\tdq "
			params = self.regionNodeParameters[region]  # dict
			if "B" in self.regionNodeKeys:
				Bx, By, Bz = floats(params.get("B", (0.0, 0.0, 0.0)))  # unpack generator
				src += f"{Bx}, {By}, {Bz}, 0.0,\t"
			if "A" in self.regionNodeKeys:
				Ax, Ay, Az = floats(params.get("A", (0.0, 0.0, 0.0)))  # unpack generator
				src += f"{Ax}, {Ay}, {Az}, 0.0,\t"
			if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
				S = float(params.get("S", 0.0))
				F = float(params.get("F", 0.0))
				kT = float(params.get("kT", 0.0))
				Je0 = float(params.get("Je0", 0.0))
				src += f"{S}, {F}, {kT}, {Je0},\t"
			src = src[0:-2]  # remove last ,\t delimeter
			src += "\n"
		src += "REGIONS ENDS\n\n"
		
		# global parameters (node only):
		src += "GLOBAL_NODE SEGMENT ALIGN(32)\n"
		params = self.globalParameters
		if "B" in self.globalKeys:
			Bx, By, Bz = floats(params.get("B", (0.0, 0.0, 0.0)))  # unpack generator
			src += f"B   dq {Bx}, {By}, {Bz}, 0.0\n"
		if "A" in self.globalKeys:
			Ax, Ay, Az = floats(params.get("A", (0.0, 0.0, 0.0)))  # unpack generator
			src += f"A   dq {Ax}, {Ay}, {Az}, 0.0\n"
		if any(p in self.globalKeys for p in ["S", "F", "kT", "Je0"]):
			S = float(params.get("S", 0.0))
			F = float(params.get("F", 0.0))
			kT = float(params.get("kT", 0.0))
			Je0 = float(params.get("Je0", 0.0))
			src += f"S   dq {S}\n"
			src += f"F   dq {F}\n"
			src += f"kT  dq {kT}\n"
			src += f"Je0 dq {Je0}\n"
		src += "GLOBAL_NODE ENDS\n\n"

		# edges:
		# TODO ...
		src += f"EDGE_COUNT EQU {len(self.edges)}\n\n"
		# TODO ...

		# edge_regions:
		# TODO ...

		# global parameters (edge only):
		# TODO ...

		# dU array (function pointers):
		# TODO ...

		# ---------------------------------------------------------------------
		src += ".code\n"
		# dU PROCs:
		# TODO ...


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
		"kT": 0.1,
		"B": (0.1, 0, 0),
		"J": 1.0
	}
	msd.regions = {
		"FML": [1, 2],
		"mol": [3],
		"FMR": [4, 5]
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
