from __future__ import annotations
from .build import VisualStudio
from .prng import SplitMix64
from .runtime import Runtime
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from importlib import resources
from math import log2, ceil
from subprocess import CalledProcessError
from tempfile import mkstemp
from typing import Callable, Generic, Iterable, Optional, Self, TypedDict, TypeVar
import os

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

class _Config(Generic[Index, Region], TypedDict, total=False):
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

def is_pow2(n: int) -> bool:
	return n > 0 and (n & (n - 1)) == 0

class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)

class Config:
	def __init__(self, **kw):
		self.__dict__ = _Config(*kw)
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
		# ASM "EQU" defines
		self.OFFSETOF_SPIN:        int = None  # node
		self.OFFSETOF_FLUX:        int = None
		self.OFFSETOF_B:           int = None
		self.OFFSETOF_A:           int = None
		self.OFFSETOF_S:           int = None
		self.OFFSETOF_F:           int = None
		self.OFFSETOF_kT:          int = None
		self.OFFSETOF_Je0:         int = None
		self.SIZEOF_NODE:          int = None
		self.MUTABLE_NODE_COUNT:   int = None
		self.IMMUTABLE_NODE_COUNT: int = None
		self.NODE_COUNT:           int = None
		self.OFFSETOF_REGION_B:    int = None  # region
		self.OFFSETOF_REGION_A:    int = None
		self.OFFSETOF_REGION_S:    int = None
		self.OFFSETOF_REGION_F:    int = None
		self.OFFSETOF_REGION_kT:   int = None
		self.OFFSETOF_REGION_Je0:  int = None
		self.SIZEOF_REGION:        int = None
		self.REGION_COUNT:         int = None
		self.OFFSETOF_J:           int = None  # edge
		self.OFFSETOF_Je1:         int = None
		self.OFFSETOF_Jee:         int = None
		self.OFFSETOF_b:           int = None
		self.OFFSETOF_D:           int = None
		self.SIZEOF_EDGE:          int = None
		self.EDGE_COUNT:           int = None
		self.OFFSETOF_REGION_J:    int = None  # edge region
		self.OFFSETOF_REGION_Je1:  int = None
		self.OFFSETOF_REGION_Jee:  int = None
		self.OFFSETOF_REGION_b:    int = None
		self.OFFSETOF_REGION_D:    int = None
		self.SIZEOF_EDGE_REGION:   int = None
		self.EDGE_REGION_COUNT:    int = None

	def __repr__(self) -> str:
		return str(self.__dict__)
	
	@staticmethod
	def innerKeys(parameters: dict) -> set[str]:
		"""
		Computes a set of all the local/region parameter keys used at least
		once across all nodes/edges/regions in the system. e.g.
		<pre>
			msd = Config()
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
			print(Config.innerKeys(self.localNodeParameters))   # {kT, B, F}
			print(Config.innerKeys(self.regionNodeParameters))  # {kT, A, S}
			print(Config.innerKeys(sele.regionEdgeParameters))  # {J, b}
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
		msd = Config()
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

	def getNodeConst(self, node, k: str) -> float:
		"""	If parameter k is a const, return its value for the given node.
			Returns None if k isn't defined or if k isn't constant.
		"""
		if k not in self.constParameters:
			return None
		v = self.localNodeParameters[node].get(k, None)
		if v is not None:
			return v
		v, _ = self.getUnambiguousRegionNodeParameter(node, k)
		if v is not None:
			return v
		return self.globalParameters.get(k, None)

	def getEdgeConst(self, edge, k: str) -> float:
		"""	If parameter k is a const, return its value for the given edge.
			Returns None if k isn't defined or if k isn't constant.
		"""
		if k not in self.constParameters:
			return None
		v = self.localEdgeParameters[edge].get(k, None)
		if v is not None:
			return v
		v, _ = self.getUnambiguousRegionEdgeParameter(edge, k)
		if v is not None:
			return v
		return self.globalParameters.get(k, None)
	
	def compile(self, tool=VisualStudio(), asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True) -> Runtime:
		# Check for and fix missing required attributes;
		if "nodes"                not in self.__dict__:  self.nodes = []
		if "mutableNodes"         not in self.__dict__:  self.mutableNodes = self.nodes
		if "edges"                not in self.__dict__:  self.edges = []
		if "regions"              not in self.__dict__:  self.regions = {}
		if "localNodeParameters"  not in self.__dict__:  self.localNodeParameters = {}
		if "localEdgeParameters"  not in self.__dict__:  self.localEdgeParameters = {}
		if "regionNodeParameters" not in self.__dict__:  self.regionNodeParameters = {}
		if "regionEdgeParameters" not in self.__dict__:  self.regionEdgeParameters = {}
		if "globalParameters"     not in self.__dict__:  self.globalParameters = {}
		if "variableParameters"   not in self.__dict__:  self.variableParameters = set()
		if "programParameters"    not in self.__dict__:  self.programParameters = {}
		if "constParameters"      not in self.__dict__:  self.constParameters = set()

		if "prng" not in self.programParameters:
			self.programParameters["prng"] = "xoshiro256**"

		# aliases
		prng = self.programParameters["prng"]
		seed = self.programParameters.get("seed", None)

		# CHECK: Do all nodes referenced in self.edges actually exist in self.nodes?
		for edge in self.edges:
			if len(edge) != 2:
				raise ValueError(f"All edges must have exactly 2 nodes: len={len(edge)} for edge {edge}")
			for node in edge:
				if node not in self.nodes:
					raise KeyError(f"All nodes referenced in model.edges must exist in model.nodes: Couldn't find node {node} referenced in edge {edge}")
		
		# CHECK: Are all parameters (i.e. dict keys) supported by the program and
		# 	in the correct dicts (i.e. edge parameter in node dict)?
		ALLOWED_NODE_PARAMETERS = ["B", "A", "S", "F", "kT", "Je0"]
		ALLOWED_EDGE_PARAMETERS = ["J", "Je1", "Jee", "b", "D"]
		ALLOWED_PRGM_PARAMETERS = ["prng", "seed"]
		for n, p in self.localNodeParameters.items():
			for k, v in p.items():
				if k not in ALLOWED_NODE_PARAMETERS:
					if k in ALLOWED_EDGE_PARAMETERS:
						raise KeyError(f'{k} is an edge parameters and unsupported in localNodeParameters[{n}]["{k}"]. Fix: config.localEdgeParameters[{n}, {n}]["{k}"] = {v}')
					elif k in ALLOWED_PRGM_PARAMETERS:
						raise KeyError(f'{k} is a program parameter and unsupported in localNodeParameters[{n}]["{k}"]. Fix: config.programParameters[{n}, {n}]["{k}"] = {v}')
					else:
						raise KeyError(f'{k} is an unrecognized parameter and unsupported in localNodeParameters[{n}]["{k}"]. Typo? (Allowable node parameters: {ALLOWED_NODE_PARAMETERS})')
		for e, p in self.localEdgeParameters.items():
			n0, n1 = e
			for k, v in p.items():
				if k not in ALLOWED_EDGE_PARAMETERS:
					if k in ALLOWED_NODE_PARAMETERS:
						raise KeyError(f'{k} is a node parameters and unsupported in localEdgeParameters[{n0}, {n1}]["{k}"]. Fix: config.localNodeParameters[{n0}]["{k}"] = {v}')
					elif k in ALLOWED_PRGM_PARAMETERS:
						raise KeyError(f'{k} is a program parameter and unsupported in localEdeParameters[{n0}, {n1}]["{k}"]. Fix: config.programParameters["{k}"] = {v}')
					else:
						raise KeyError(f'{k} is an unrecognized parameter and unsupported in localEdgeParameters["{k}"]. Typo? (Allowable edge parameters: {ALLOWED_EDGE_PARAMETERS})')
		for r, p in self.regionNodeParameters.items():
			for k, v in p.items():
				if k not in ALLOWED_NODE_PARAMETERS:
					if k in ALLOWED_EDGE_PARAMETERS:
						raise KeyError(f'{k} is an edge parameters and unsupported in regionNodeParameters[{r}]["{k}"]. Fix: config.regionEdgeParameters[{r}]["{k}"] = {v}')
					elif k in ALLOWED_PRGM_PARAMETERS:
						raise KeyError(f'{k} is a program parameter and unsupported in regionNodeParameters[{r}]["{k}"]. Fix: config.programParameters[{r}]["{k}"] = {v}')
					else:
						raise KeyError(f'{k} is an unrecognized parameter and unsupported in regionNodeParameters[{r}]["{k}"]. Typo? (Allowable node parameters: {ALLOWED_NODE_PARAMETERS})')
		for r, p in self.regionNodeParameters.items():
			for k, v in p.items():
				if k not in ALLOWED_EDGE_PARAMETERS:
					if k in ALLOWED_NODE_PARAMETERS:
						raise KeyError(f'{k} is a node parameters and unsupported in regionEdgeParameters[{r}]["{k}"]. Fix: config.regionNodeParameters[{r}]["{k}"] = {v}')
					elif k in ALLOWED_PRGM_PARAMETERS:
						raise KeyError(f'{k} is a program parameter and unsupported in regionEdgeParameters[{r}]["{k}"]. Fix: config.programParameters[{r}]["{k}"] = {v}')
					else:
						raise KeyError(f'{k} is an unrecognized parameter and unsupported in regionNodeParameters[{r}]["{k}"]. Typo? (Allowable edge parameters: {ALLOWED_EDGE_PARAMETERS})')
		for k, v in self.globalParameters.items():
			if k not in ALLOWED_NODE_PARAMETERS + ALLOWED_EDGE_PARAMETERS:
				if k in ALLOWED_PRGM_PARAMETERS:
					raise KeyError(f'{k} is a program parameter and unsupported in globalParameters["{k}"]. Fix: config.prograParameters["{k}"] = {v}')
				else:
					raise KeyError(f'{k} is an unrecognized parameter and unsupported in globalParameters["{k}"]. Typo? (Allowable node/edge parameters: {ALLOWED_NODE_PARAMETERS + ALLOWED_EDGE_PARAMETERS})')
		for k, v in self.programParameters.items():
			if k not in ALLOWED_PRGM_PARAMETERS:
				if k in ALLOWED_NODE_PARAMETERS:
					raise KeyError(f'{k} is a node parameter and unsupported in programParameters["{k}"]. Fix: config.globalParameters["{k}"] = {v}')
				elif k in ALLOWED_EDGE_PARAMETERS:
					raise KeyError(f'{k} is an edge parameter and unsupported in programParameters["{k}"]. Fix: config.globalParameters["{k}"] = {v}')
				else:
					raise KeyError(f'{k} is an unrecognized parameter and unsupported in programParameters["{k}"]. Typo? (Allowable program parameters: {ALLOWED_PRGM_PARAMETERS})')
		
		# CHECK: Is PRNG algorithm supported?
		if prng not in ["xoshiro256**", "xoshiro256++", "xoshiro256+"]:
			raise ValueError(f"Unsupported psuedo-random number generator algorithm: {prng}. Suggested algorithm model.programParameters[\"prng\"] = \"xoshiro256**\"")
		if seed is not None:
			for s in seed:
				if s < 0 or s >= 2**64:
					raise ValueError(f"Seed value {s} is out-of-range [0, 2**64)")

		# other (dependant) variables
		self.localNodeKeys = Config.innerKeys(self.localNodeParameters)    # set[str]
		self.localEdgeKeys = Config.innerKeys(self.localEdgeParameters)    # set[str]
		self.regionNodeKeys = Config.innerKeys(self.regionNodeParameters)  # set[str]
		self.regionEdgeKeys = Config.innerKeys(self.regionEdgeParameters)  # set[str]
		self.globalKeys = { *self.globalParameters.keys() }                # set[str]
		self.allKeys = self.localNodeKeys | self.localEdgeKeys | self.regionNodeKeys | self.regionEdgeKeys | self.globalKeys
		self.constParameters = self.allKeys - self.variableParameters  # set[str]
		self.immutableNodes = self.calcImmutableNodes()  # list[Node]
		self.regionCombos = self.calcAllRegionCombos()  # list[Edge]
		self.nodeIndex = {}  # dict[Node, int]
		self.edgeIndex = {}  # dict[Edge, int]

		exports = []  # [(symbol: str, data: bool), ...]

		src = StrJoiner()
		src += f"; Generated by {__name__} (python) at {datetime.now()}\n"
		# options
		src += "OPTION CASEMAP:NONE\n\n"
		# includes
		src += "include vec.inc  ; _vdotp, etc.\n"
		src += "include prng.inc  ; splitmix64, xoshiro256ss, etc. \n"
		src += "include ln.inc  ; _vln\n\n"
		
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
		self.OFFSETOF_SPIN = 32 * offset32
		src += f"OFFSETOF_SPIN EQU 32*({offset32})\n";  offset32 += 1
		if "F" in self.allKeys:
			self.OFFSETOF_FLUX = 32 * offset32
			src += f"OFFSETOF_FLUX EQU 32*({offset32})\n";  offset32 += 1
		if "B" in self.localNodeKeys:
			self.OFFSETOF_B = 32 * offset32
			src += f"OFFSETOF_B    EQU 32*({offset32})\n";  offset32 += 1
		if "A" in self.localNodeKeys:
			self.OFFSETOF_A = 32 * offset32
			src += f"OFFSETOF_A    EQU 32*({offset32})\n";  offset32 += 1
		if any(p in self.localNodeKeys for p in {"S", "F", "kT", "Je0"}):
			if "S" in self.localNodeKeys:    src += f"OFFSETOF_S    EQU 32*({offset32}) + 8*(0)\n";  self.OFFSETOF_S   = 32 * offset32 + 8*0
			if "F" in self.localNodeKeys:    src += f"OFFSETOF_F    EQU 32*({offset32}) + 8*(1)\n";  self.OFFSETOF_F   = 32 * offset32 + 8*1
			if "kT" in self.localNodeKeys:   src += f"OFFSETOF_kT   EQU 32*({offset32}) + 8*(2)\n";  self.OFFSETOF_kT  = 32 * offset32 + 8*2
			if "Je0" in self.localNodeKeys:  src += f"OFFSETOF_Je0  EQU 32*({offset32}) + 8*(3)\n";  self.OFFSETOF_Je0 = 32 * offset32 + 8*3
			offset32 += 1
		self.SIZEOF_NODE = 32 * offset32
		src += f"SIZEOF_NODE   EQU 32*({offset32})\n"
		self.MUTABLE_NODE_COUNT = len(self.mutableNodes)
		src += f"MUTABLE_NODE_COUNT   EQU {len(self.mutableNodes)}\n"
		self.IMMUTABLE_NODE_COUNT = len(self.immutableNodes)
		src += f"IMMUTABLE_NODE_COUNT EQU {len(self.immutableNodes)}\n"
		self.NODE_COUNT = len(self.nodes)
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
		exports.append(("nodes", True))

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
			self.OFFSETOF_REGION_B = 32 * offset32
			src += f"OFFSETOF_REGION_B   EQU 32*({offset32})\n";  offset32 += 1
		if "A" in self.regionNodeKeys:
			self.OFFSETOF_REGION_A = 32 * offset32
			src += f"OFFSETOF_REGION_A   EQU 32*({offset32})\n";  offset32 += 1
		if any(p in self.regionNodeKeys for p in {"S", "F", "kT", "Je0"}):
			if "S"   in self.regionNodeKeys:  src += f"OFFSETOF_REGION_S   EQU 32*({offset32}) + 8*(0)\n";  self.OFFSETOF_REGION_S   = 32 * offset32 + 8*0
			if "F"   in self.regionNodeKeys:  src += f"OFFSETOF_REGION_F   EQU 32*({offset32}) + 8*(1)\n";  self.OFFSETOF_REGION_F   = 32 * offset32 + 8*1
			if "kT"  in self.regionNodeKeys:  src += f"OFFSETOF_REGION_kT  EQU 32*({offset32}) + 8*(2)\n";  self.OFFSETOF_REGION_kT  = 32 * offset32 + 8*2
			if "Je0" in self.regionNodeKeys:  src += f"OFFSETOF_REGION_Je0 EQU 32*({offset32}) + 8*(3)\n";  self.OFFSETOF_REGION_Je0 = 32 * offset32 + 8*3
			offset32 += 1
		self.SIZEOF_REGION = 32 * offset32
		src += f"SIZEOF_REGION EQU 32*({offset32})\n"
		self.REGION_COUNT = len(self.regionNodeParameters.keys())
		src += f"REGION_COUNT  EQU {len(self.regionNodeParameters.keys())}\n\n"
		# define memeory for regions
		src += "REGIONS SEGMENT ALIGN(32)  ; AVX-256 (i.e. 32-byte) alignment\n"
		for region in self.regions:
			rid = self.regionId(region)
			if region not in self.regionNodeParameters:
				src += f"; {rid}\n"
				continue  # skip defining this region. it has no special parameters.
			params = self.regionNodeParameters[region]  # dict
			src += f"  {rid}\tdq "
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
			src.pieces[-1] = src.pieces[-1][0:-2]  # remove last 2-char ",\t" delimeter
			src += "\n"
			exports.append((rid, True))
		src += "REGIONS ENDS\n\n"
		
		# global parameters (node only):
		src += "GLOBAL_NODE SEGMENT ALIGN(32)\n"
		params = self.globalParameters
		if "B" in self.globalKeys:
			Bx, By, Bz = floats(params["B"])  # unpack generator
			src += f"B   dq {Bx}, {By}, {Bz}, 0.0\n"
			exports.append(("B", True))
		if "A" in self.globalKeys:
			Ax, Ay, Az = floats(params["A"])  # unpack generator
			src += f"A   dq {Ax}, {Ay}, {Az}, 0.0\n"
			exports.append(("A", True))
		if any(p in self.globalKeys for p in ["S", "F", "kT", "Je0"]):
			S   = float(params.get("S",   0.0))
			F   = float(params.get("F",   0.0))
			kT  = float(params.get("kT",  0.0))
			Je0 = float(params.get("Je0", 0.0))
			src += f"S   dq {S}\n"
			src += f"F   dq {F}\n"
			src += f"kT  dq {kT}\n"
			src += f"Je0 dq {Je0}\n"
			exports.append(("S", True))
			exports.append(("F", True))
			exports.append(("kT", True))
			exports.append(("Je0", True))
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
			if "J"   in self.localEdgeKeys:  src += f"OFFSETOF_J   EQU 32*({offset32}) + 8*(0)\n";  self.OFFSETOF_J   = 32 * offset32 + 8*0
			if "Je1" in self.localEdgeKeys:  src += f"OFFSETOF_Je1 EQU 32*({offset32}) + 8*(1)\n";  self.OFFSETOF_Je1 = 32 * offset32 + 8*1
			if "Jee" in self.localEdgeKeys:  src += f"OFFSETOF_Jee EQU 32*({offset32}) + 8*(2)\n";  self.OFFSETOF_Jee = 32 * offset32 + 8*2
			if "b"   in self.localEdgeKeys:  src += f"OFFSETOF_Je1 EQU 32*({offset32}) + 8*(3)\n";  self.OFFSETOF_b   = 32 * offset32 + 8*3
			offset32 += 1
		if "D" in self.localEdgeKeys:
			self.OFFSETOF_D = 32 * offset32
			src += f"OFFSETOF_D  EQU 32*({offset32})";  offset32 += 1
		self.SIZEOF_EDGE = 32 * offset32
		src += f"SIZEOF_EDGE  EQU 32*({offset32})\n"
		self.EDGE_COUNT = len(self.edges)
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
				src.pieces[-1] = src.pieces[-1][0:-2]  # remove last 2-char ",\t" delimeter
				regions = self.getRegionCombos(edge)
				regions = f" in {regions}" if len(regions) != 0 else ""
				src += f"  ; edges[{i}]: {self.nodeId(edge[0])} -> {self.nodeId(edge[1])}{regions}\n"
			exports.append(("edges", True))
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
			if "J"   in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_J   EQU 32*({offset32}) + 8*(0)\n";  self.OFFSETOF_REGION_J   = 32 * offset32 + 8*0
			if "Je1" in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_Je1 EQU 32*({offset32}) + 8*(1)\n";  self.OFFSETOF_REGION_Je1 = 32 * offset32 + 8*1
			if "Jee" in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_Jee EQU 32*({offset32}) + 8*(2)\n";  self.OFFSETOF_REGION_Jee = 32 * offset32 + 8*2
			if "b"   in self.regionEdgeKeys:  src += f"OFFSETOF_REGION_b   EQU 32*({offset32}) + 8*(3)\n";  self.OFFSETOF_REGIONN_b  = 32 * offset32 + 8*3
			offset32 += 1
		if "D" in self.regionEdgeKeys:
			self.OFFSETOF_REGION_D = 32 * offset32
			src += f"OFFSETOF_REGION_D   EQU 32*({offset32})\n";  offset32 += 1
		self.SIZEOF_EDGE_REGION = 32 * offset32
		src += f"SIZEOF_EDGE_REGION  EQU 32*({offset32})\n"
		self.EGDE_REGION_COUNT = len(self.regionCombos)
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
			src.pieces[-1] = src.pieces[-1][0:-2]  # remove last 2-char ",\t" delimeter
			src += "\n"
			exports.append((rid, True))
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
			exports.append(("J", True))
			exports.append(("Je1", True))
			exports.append(("Jee", True))
			exports.append(("b", True))
		if "D" in self.globalKeys:
			Dx, Dy, Dz = floats(params["D"])
			src += f"D   dq {Dx}, {Dy}, {Dz}, 0.0\n"
			exports.append(("D", True))
		src += "GLOBAL_EDGE ENDS\n\n"

		# parallel arrays:
		def node_ref_array(src, exports, ks: list, vs=0.0, name=None, align: int=None) -> None:
			""" Generate .data array of pointers to some node field, k, for each mutable node. """
			if name is None:
				name = f"{''.join(ks)}ref"
			src += f"; array of {ks} pointers (double *) parallel to (mutable) nodes\n"
			if align is not None:
				src += f"{name.upper()} SEGMENT ALIGN({align})\n"
			src += f"{name}\t"
			if len(self.mutableNodes) == 0:
				src += "; 0 byte array. no mutable nodes.\n"
			first_line = True
			for node in self.mutableNodes:
				idx = self.nodeIndex[node]
				for k, v in zip(ks, vs):
					if first_line:
						first_line = False
					else:
						src += "\t\t"  # ASM formating: first struct must be on same line as symbol (i.e. "dU")
					if k in self.localNodeParameters.get(node, {}):
						src += f"dq nodes + SIZEOF_NODE*{idx} + OFFSETOF_{k}  ; nodes[{idx}] -> &nodes[{idx}].{k} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, k)  # we don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							src += f"dq {rid} + OFFSETOF_REGION_{k}  ; nodes[{idx}] -> &{rid}.{k} (region)\n"
						elif k in self.globalKeys:
							src += f"dq {k}  ; nodes[{idx}] -> &{k} (global)\n"
						else:
							raise ValueError(f"For mutable node {self.nodeId(node)}, {k} is not defined at any level. Potential fix: model.globalParameters[\"{k}\"] = {v}")
			if align is not None:
				src += f"{name.upper()} ENDS\n"
			src += "\n"
			exports.append((name, True))
			return str(src)

		# S,F parallel array
		if "F" in self.allKeys:
			node_ref_array(src, exports, ["S", "F"], [1.0, 0.0], align=16)
		else:
			node_ref_array(src, exports, ["S"], [1.0])

		# deltaU array (function pointers):
		src += "; array of function pointers parallel to (mutable) nodes\n"
		src += "deltaU\t"
		if len(self.mutableNodes) == 0:
			src += "; 0 byte empty array. no multable nodes.\n"
		for i, node in enumerate(self.mutableNodes):
			if i != 0:  # loop index, not node index.
				src += "\t\t"  # ASM formating: first struct must be on same line as symbol (i.e. "dU")
			src += f"dq deltaU_{self.nodeId(node)}  ; nodes[{self.nodeIndex[node]}]\n"
		src += "\n"
		exports.append(("deltaU", True))

		# kT parallel array
		node_ref_array(src, exports, ["kT"], [0.1])

		# misc. constants
		# src += "; misc. constants\n"
		# src += "ONE dq 1.0\n\n"

		# PRNG state
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			if seed is None or len(seed) == 0:
				src += ".data?\n"
				src += "PRNG SEGMENT ALIGN(32)\n"
				src += f"; Vectorized {prng}\n"
				src += "prng_state dq 4 dup (?, ?, ?, ?)  ; initialized at runtime\n"
				src += "PRNG ENDS\n\n"
			else:
				src += ".data\n"
				src += "PRNG SEGMENT ALIGN(32)\n"
				# use given seed, plus SplitMix64 if len < 16
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
				src += "prng_state dq\t"
				for i, row in enumerate(prng_state):
					if i != 0:
						src += "\t\t\t\t"
					for j, value in enumerate(row):
						src += f"0{value:016x}h"
						if   j != 3:  src += ",\t"
						elif i != 3:  src += ","
					src += "\n"
				src += "PRNG ENDS\n\n"
			exports.append(("prng_state", True))

		# ---------------------------------------------------------------------
		src += ".code\n"
		# deltaU PROCs:
		# documentation for deltaU_{nodeId} PROCs
		src +=  "; @return ymm0: -\\Delta U (Negated for Boltzmann distribution.)\n"
		src +=  "; @param  ymm1: s'_i\n"
		flux_note: str = " (unused)" if "F" not in self.allKeys else ""
		src += f"; @param  ymm2: f'_i{flux_note}\n"
		src +=  ";         ymm3: Parameter (Je0; A, J, Je1, Jee, b; B, then D)\n"
		src +=  ";         ymm4: s_i, m_i, then (unused)\n"
		src +=  ";         ymm5: f_i, m'_i, then (unused)\n"
		src +=  ";         ymm6: (unused), \\Delta s_i, then \\Delta m_i\n"
		src +=  ";         ymm7: (unused), \\Delta f_i, then (unused)\n"
		src +=  ";         ymm8: s_j, f_j, or m_j\n"
		src +=  ";         ymm9: temp/scratch reg.\n"
		src +=  ";         ymm10: TODO: (remove) temp/scratch reg.\n"  # TODO: remove
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

				# registers:
				resx, resy = "xmm0", "ymm0"   # result (return: -ΔU = -ΔU_Je0 - ΔU_A - ΔU_J - ΔU_Je1 - ΔU_Jee - ΔU_b - ΔU_B - ΔU_D)
				s1i = "ymm1"                  # s'_i (param: new spin)
				f1i = "ymm2"                  # f'_i (param: new flux)
				prmx, prmy = "xmm3", "ymm3"  # scalar parameter (Je0, J, Je1, Jee, b), or vector parameter (A, B, D)
				smi = "ymm4"                  # s_i, m_i, then (unused)
				fm1 = "ymm5"                  # f_i, m'_i, then (unused)
				dsm = "ymm6"                  # (unused), Δs_i, then Δm_i
				dfi = "ymm7"                  # (unused), Δf_i, then (unused)
				sfmj = "ymm8"                 # s_j, f_j, or m_j
				tmpx, tmpy = "xmm9", "ymm9"   # temp/scratch reg. (scalar/vector)
				tmp2x, tmp2y = "xmm10", "ymm10"  # TODO: (stub) REMOVE! Used by ΔU_J. We can (likely) reorder computations in phase 2 to use smi, fm1, or dfi as temp2

				# state
				out_init: bool = False  # track if output register has been initialized yet
				
				src += f"{proc_id} PROC  ; node[{index}]\n"
				# No preable: no local vars. so no stack space needed.
				# comments about what is being skipped:
				s = ""
				if "Je0" not in self.allKeys:  s += "\t; skipping -deltaU_Je0\n"
				if "A"   not in self.allKeys:  s += "\t; skipping -deltaU_A\n"
				if "J"   not in self.allKeys:  s += "\t; skipping -deltaU_J\n"
				if "Je1" not in self.allKeys:  s += "\t; skipping -deltaU_Je1\n"
				if "Jee" not in self.allKeys:  s += "\t; skipping -deltaU_Jee\n"
				if "b"   not in self.allKeys:  s += "\t; skipping -deltaU_b\n"
				if "B"   not in self.allKeys:  s += "\t; skipping -deltaU_B\n"
				if "D"   not in self.allKeys:  s += "\t; skipping -deltaU_D\n"
				src += s
				if len(s) != 0:  src += "\n"
				# Phase 1: load (s_i, f_i) -> -ΔU_(Je0):
				phase1: bool = False  # does phase 1 contain any code?
				src1 = StrJoiner()    # buffer phase 1 src to emit after load
				# try to compute -ΔU_Je0 = Je0 (s'⋅f' - s⋅f):
				if "Je0" in self.allKeys:
					src1 += "\t; -deltaU_Je0 calculation\n"
					# where is Jeo defined?
					load_insn = None
					if "Je0" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovsd {prmx}, qword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_Je0]  ; load Je0_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "Je0")  # we don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovsd {prmx}, qword ptr [{rid} + OFFSETOF_REGION_Je0]  ; load Je0_{rid} (region)\n"
						elif "Je0" in self.globalKeys:
							load_insn = f"\tvmovsd {prmx}, qword ptr Je0  ; load Je0 (global)\n"
					# TODO: add optimization_remove_scalar
					# TODO: add optimization_neg_scalar
					if load_insn is None:
						src1 += "\t; skip. For this node, Je0 is not defined at any level (local, region, nor global).\n\n"
					elif not hasFlux:
						raise KeyError(f"For node {nid}, Je0 is defined, but F is not. Potential fix: model.globalParameters[F] = 0.0")
					else:
						phase1 = True
						# compute s_i'·f_i' - s_i·f_i (difference between new and current dot products for local spin·flux)
						src1 += f"\t_vdotp {tmpx}, {tmpy}, {s1i}, {f1i}, {prmx}      ; ({s1i}, {f1i}) = (s'f')\n"
						src1 += f"\t_vndotadd {tmpx}, {tmpy}, {smi}, {fm1}, {prmx}   ; ({smi}, {fm1}) = (s'f' - sf)\n"
						src1 += load_insn  # load Je0 into prmx
						if not out_init:
							src1 += f"\tvmulsd {resx}, {prmx}, {tmpx}\n\n"
							out_init = True
						else:
							src1 += f"\tvfmadd231sd {resx}, {prmx}, {tmpx}  ; {resx} += {prmx} * {tmpx}\n\n"
				# commit phase 1
				if phase1:
					src += "\t; Phase 1:\n"
					src += f"\tvmovapd {smi}, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_SPIN]  ; load s_i\n"
					if flux_mode:
						src += f"\tvmovapd {fm1}, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_FLUX]  ; load f_i\n"
				else:
					src += "\t; Phase 1: (skip)\n"
				src += "\n" + str(src1)  # actual ASM (or comments)
				# Phase 2: compute (m_i, m'_i, Δs_i, Δf_i) -> -ΔU_(A, J, Je1, Jee, b):
				phase2: bool = False  # does phase 2 contain any code?
				src2 = StrJoiner()    # buffer phase 2 src to emit after load
				m1i = fm1 if flux_mode else s1i
				# compute -ΔU_A: A ⋅ (m'_i^{⊙2} - m_i^{⊙2})
				if "A" in self.allKeys:
					src2 += "\t; -deltaU_A calculation\n"
					# where is A defined?
					load_insn = None
					if "A" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovapd {prmy}, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_A]  ; load A_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "A")  # don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovapd {prmy}, ymmword ptr [{rid} + OFFSETOF_REGION_A]  ; load A_{rid} (region)\n"
						elif "A" in self.globalKeys:
							load_insn = f"\tvmovapd {prmy}, ymmword ptr A  ; load A (global)\n"
					if load_insn is None:
						src2 += f"\t; skip. For this node, A is not defined at any level (local, region, nor global).\n\n"
					else:
						phase2 = True
						src2 += load_insn  # load A into prmy
						src2 += f"\tvmulpd {tmpy}, {m1i}, {m1i}  ; m' Hadamard squared: m'2\n"
						src2 += f"\tvfnmadd231pd {tmpy}, {smi}, {smi}  ; {tmpy} -= {smi} * {smi} -> m'2 - m2\n"
						if not out_init:
							src2 += f"\t_vdotp {resx}, {resy}, {prmy}, {tmpy}, {prmx}  ; ({prmy}, {tmpy})\n\n"
							out_init = True
						else:
							src2 += f"\t_vdotadd {resx}, {resy}, {prmy}, {tmpy}, {prmy}  ; ({prmy}, {tmpy})\n\n"
				# compute -ΔU_J = Σ_j{J Δs_i·s_j}:
				if "J" in self.allKeys:
					src2 += "\t; -deltaU_J calculation\n"
					# figure out where all neighboring edges load J from, and group the common load instructions
					load_groups: dict[str, list] = defaultdict(list)  # dict: str load_insn -> list[Edge]
					for edge in self.connections(node):
						if "J" in self.localEdgeParameters.get(edge, {}):
							eindex = self.edgeIndex[edge]
							nid0 = self.nodeId(edge[0])
							nid1 = self.nodeId(edge[1])
							load_insn = f"\tvmovsd {prmx}, qword ptr [edges + ({eindex})*SIZEOF_EDGE + OFFSETOF_J]  ; load J_{nid0}_{nid1}\n"
						else:
							_, combo = self.getUnambiguousRegionEdgeParameter(edge, "J")  # don't need value, only region
							if combo is not None:
								rid0, rid1 = self.regionId(combo[0]), self.regionId(combo[1])
								load_insn = f"\tvmovsd {prmx}, qword ptr [{rid0}_{rid1} + OFFSETOF_REGION_J]  ; load J_{rid0}_{rid1} (region)\n"
							elif "J" in self.globalKeys:
								load_insn = f"\tvmovsd {prmx}, qword ptr J  ; load J (global)\n"
							else:
								load_insn = None
						load_groups[load_insn].append(edge)
					for i, load_group in enumerate(load_groups.items(), 1):
						load_insn, edges = load_group
						optimization_remove_scalar = False  # TODO: implement optimization. if scalar is const and exactly 1.0
						optimization_neg_scalar = False     # TODO: implement optimization. if scalar is const and exactly -1.0
						src2 += f"\t; [load group {i}]\n"
						if optimization_remove_scalar or optimization_neg_scalar:
							src2 += "\t; optimization (J*=1,-1): skip load\n"
						else:
							src2 += load_insn  # load J into prmx
						for edge in edges:
							# Note: edge in self.edgeIndex may be false if ASM edge array is empty or missing local variables for this edge. This is fine.
							edgelbl = f"edges[{self.edgeIndex[edge]}]" if edge in self.edgeIndex else "edge"
							if load_insn is None:
								src += "; skip. For the edge, J is not defined at any level (local, region, nor global).\n"
							else:
								phase2 = True
								nid0 = self.nodeId(edge[0])
								nid1 = self.nodeId(edge[1])
								neighbor, _ = Config.neighbor(node, edge)  # just need neighbor. communative operation: doesn't care about direction.
								nindex = self.nodeIndex[neighbor]
								nnid = self.nodeId(neighbor)  # neighbor's node id
								src2 += f"\t; {edgelbl}: {nid0} -> {nid1}\n"
								src2 += f"\tvmovapd {sfmj}, ymmword ptr [nodes + ({nindex})*SIZEOF_NODE + OFFSETOF_SPIN]  ; load s_{nnid} (neighbor)\n"
								if optimization_remove_scalar:
									if not out_init:
										src2 += f"\t_vdotp {resx}, {resy}, {dsm}, {sfmj}, {prmx}  ; optimization (J*=1), ({dsm}, {sfmj})\n"
										out_init = True
									else:
										src2 += f"\t_vdotadd {resx}, {resy}, {dsm}, {sfmj}, {prmx}  ; optimization (J*=1), ({dsm}, {sfmj})\n"
								elif optimization_neg_scalar:
									if not out_init:
										src2 += f"\t_vput0 {resx}\n"
										out_init = True
									src2 += f"\t_vndotadds {resx}, {resy}, {dsm}, {sfmj}, {prmx}  ; optimization (J*=-1), ({dsm}, {sfmj})\n"
								else:
									src2 += f"\t_vdotp {tmpx}, {tmpy}, {dsm}, {sfmj}, {tmp2x}  ; ({dsm}, {sfmj})\n"
									if not out_init:
										src2 += f"\tvmulsd {resx}, {prmx}, {tmpx}\n"
										out_init = True
									else:
										src2 += f"\tvfmadd231sd {resx}, {prmx}, {tmpx}  ; {resx} += {prmx} * {tmpx}\n"
					src2 += "\n"
				# compute -ΔU_Je1:
				if "Je1" in self.allKeys:
					src2 += "\t; -deltaU_Je1 calculation\n"
					src2 += "\t; TODO ...\n\n" # TODO (stub)
				# compute -ΔU_Jee:
				if "Jee" in self.allKeys:
					src2 += "\t; -deltaU_Jee calculation\n"
					src2 += "\t; TODO ...\n\n" # TODO (stub)
				# compute -ΔU_b:
				if "b" in self.allKeys:
					src2 += "\t; -deltaU_b calculation\n"
					src2 += "\t; TODO ...\n\n" # TODO (stub)
				# commit phase 2
				if phase2:
					src += "\t; Phase 2:\n"
					src += f"\tvsubpd {dsm}, {s1i}, {smi}  ; \\Delta s_i = s'_i - s_i\n"
					if flux_mode:
						src += f"\tvsubpd {dfi}, {f1i}, {fm1}  ; \\Detla f = f'_i - f_i\n"
						src += f"\tvaddpd {smi}, {smi}, {fm1}  ; m_i = s_i + f_i\n"
						src += f"\tvaddpd {fm1}, {s1i}, {f1i}  ; m'_i = s'_i + f'_i\n"
					else:
						src += f"\t; ({smi}) m_i = s_i since f_i = 0\n"
						src += f"\t; ({m1i}) m'_i = s'_i since f'_i = 0\n"
				else:
					src += "\t; Phase 2: (skip)\n"
				src += "\n" + str(src2)
				# Phase 3 compute (Δm_i) -> -ΔU_(B, D):
				phase3: bool = False  # does phase 3 contain any code?
				src3 = StrJoiner()    # buffer phase 3 src to emit after load
				# compute -ΔU_B = B ⋅ Δm_i:
				if "B" in self.allKeys:
					src3 += "\t; compute -delta_U_B\n"
					# where is B defined?
					load_insn = None  # How should this node be loaded?
					if "B" in self.localNodeParameters.get(node, {}):
						load_insn = f"\tvmovapd {prmy}, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_B]  ; load B_{nid} (local)\n"
					else:
						_, region = self.getUnambiguousRegionNodeParameter(node, "B")  # don't need value, only region
						if region is not None:
							rid = self.regionId(region)
							load_insn = f"\tvmovapd {prmy}, ymmword ptr [{rid} + OFFSETOF_REGION_B]  ; load B_{rid} (region)\n"
						elif "B" in self.globalKeys:
							load_insn = f"\tvmovapd {prmy}, ymmword ptr B  ; load B (global)\n"
					if load_insn is None:
						src3 += f"\t; skip. For this node, B is not defined at any level (local, region, nor global).\n\n"
					else:
						phase3 = True
						src3 += load_insn  # load B into ymm9
						if not out_init:
							src3 += f"\t_vdotp {resx}, {resy}, {prmy}, {dsm}, {prmx}  ; ({prmy}, {dsm})\n\n"
							out_init = True
						else:
							src3 += f"\t_vdotadd {resx}, {resy}, {prmy}, {dsm}, {prmx}  ; ({prmy}, {dsm})\n\n"
				# compute -ΔU_D:
				if "D" in self.allKeys:
					src3 += "\t; -deltaU_D calculation\n"
					src3 += "\t; TODO ...\n\n" # TODO (stub)
				if phase3:
					src += "\t; Phase 3:\n"
					if flux_mode:
						src += f"\tvsubpd {dsm}, {m1i}, {smi}  ; \\Delta m_i = m'_i - m_i\n\n"
					else:
						src += f"\t; ({dsm}) \\Delta m_i = \\Delta s_i since \\Delta f_i = 0\n\n"
				src += str(src3)
				# return:
				if not out_init:
					src += f"\tvxorpd {resx}, {resx}, {resx}  ; return 0.0\n"  # fall back in case there are no parameters set for this node
				src += "\tret\n"
				src += f"{proc_id} ENDP\n\n"
				exports.append((proc_id, False))
		
		# metropolis PROC
		src += "; Helpers for calculating random unit vectors (Marsaglia's method)\n"
		src += "_gen_ws MACRO w, s, one, temp\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			macro = prng.replace("*", "s").replace("+", "p")
			src += f"\t_v{macro} w, ymm12, ymm13, ymm14, ymm15, temp\n"
			src += "\t; assert one == [1.0, 1.0, 1.0, 1.0]\n"
			src += "\t_vomega w, w, one   ; [0, 1)\n"
		src += "\tvaddpd w, w, w      ; mul by 2 -> [0, 2)\n"
		src += "\tvsubpd w, w, one    ; sub by 1 -> [-1, 1); w = [u_1, v_1, u_2, v_2]\n"
		src += "\tvmulpd s, w, w   ; Hadamard product: [u_1^2, v_1^2, ...]\n"
		src += "\tvhaddpd s, s, s  ; Radius squared: [s_1, s_1, s_2, s_2] where s_i = u_i^2 + v_i^2\n"
		src += "ENDM\n\n"
		src += "_calc_xyz MACRO w, z, s, one\n"
		src += "\tvaddpd w, w, w           ; 2w\n"
		src += "\tvsubpd z, one, s         ; 1 - s\n"
		src += "\tvsqrtpd z, z             ; sqrt(1 - s)\n"
		src += "\tvmulpd w, w, z           ; 2w * sqrt(1 - s) = [x_1, y_1, x_2, y_2] -> x_i = 2u_i * sqrt(1-s_i) and y_i = 2v_i * sqrt(1-s)\n"
		src += "\tvaddpd s, s, s           ; 2s\n"
		src += "\tvsubpd z, one, s         ; 1 - 2s = [z_1, z_1, z_2, z_2] -> z_i = 1 - 2s_i\n"
		src += "\tvpxor s, s, s            ; [0, 0, 0, 0]\n"
		src += "\tvblendpd z, z, s, 1010b  ; [z_1, 0, z_2, 0]\n"
		src += "ENDM\n\n"
		src += "; Runs the standard metropolis algorithm\n"
		src += "; @param RCX (uint64) - number of iterations\n"
		src += "; @return (void)\n"
		src += "PUBLIC metropolis\n"
		src += "metropolis PROC\n"
		src += "\t; *rax: scratch reg.\n"
		src += "\t;  rcx: loop counter\n"
		src += "\t;  rdx: index[3] (or index[2] in TAIL)\n"
		src += "\t;  rbx: \n"
		src += "\t;  rsp: (stack pointer)\n"
		src += "\t;  rbp: (base pointer)\n"
		src += "\t;  *rsi: scratch reg.\n"
		src += "\t;  rdi: \n"
		src += "\t;  r8:  index[0]\n"
		src += "\t;  r9:  index[1]\n"
		src += "\t;  r10: index[2]\n"
		src += "\t;  r11: MUTABLE_NODE_COUNT\n"
		src += "\t;  ... \n"
		src += "\t;  ymm0: -\\DeltaU\n"
		src += "\t;  ymm1: new spin, s'\n"
		src += "\t;  ymm2: new flux, f'\n"
		src += "\t;  ...\n"
		src += "\t;  ymm11: vectorized ln(p)\n"
		src += "\t;  ymm12: vectorized PNRG state[0]\n"
		src += "\t;  ymm13: vectorized PRNG state[1]\n"
		src += "\t;  ymm14: vectorized PRNG state[2]\n"
		src += "\t;  ymm15: vectorized PRNG state[3]\n\n"
		# preamble
		src += "\t; preamble: save non-volitile registers\n"
		src += "\tpush rbp\n"
		src += "\tmov rbp, rsp\n"
		src += "\tsub rsp, (10)*16 + 8  ; space for the 10 non-volitile XMM6-15 reg. and RSI\n"
		for i in range(10):  # XMM6-15
			src += f"\tvmovdqu xmmword ptr [rbp - ({i+1})*16], xmm{6+i}\n"
		src += f"\tmov qword ptr [rbp - (10)*16 - 8], rsi\n"
		src += "\n"
		# load PRNG state, and other constants
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			src += f"\t; load vectorized PRNG ({prng}) state into YMM12, YMM13, YMM14, YMM15; and load other constants\n"
			src += "\tvmovdqa ymm12, ymmword ptr [prng_state + (0)*32]\n"
			src += "\tvmovdqa ymm13, ymmword ptr [prng_state + (1)*32]\n"
			src += "\tvmovdqa ymm14, ymmword ptr [prng_state + (2)*32]\n"
			src += "\tvmovdqa ymm15, ymmword ptr [prng_state + (3)*32]\n"
		src += "\tmov r11, MUTABLE_NODE_COUNT ; load mutable nodes array size\n"
		# loop metropolis algo. in batchs of 4 (`True`) until TAIL (`False`)
		for batch4 in [True, False]:
			if batch4:
				regis = ["r8", "r9", "r10", "rdx"]
				src += "\tLOOP_START:  ; process nodes in batchs of 4\n"
				src += "\t\tcmp rcx, 4\n"
				src += "\t\tjb LOOP_END  ; while ((unsigned) rcx >= 4)\n\n"
			else:
				regis = ["r8", "r9", "rdx"]
				src += "\tTAIL_START:  ; Last 0-3 nodes\n"
				src += "\t\tcmp rcx, 0\n"
				src += "\t\tje TAIL_END\n\n"
			# select (3 or) 4 random nodes
			src += f"\t\t; select {len(regis)} random indices for mutable nodes -> {regis}\n"
			if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
				macro = prng.replace("*", "s").replace("+", "p")
				src += f"\t\t_v{macro} ymm0, ymm12, ymm13, ymm14, ymm15, ymm10  ; [a, b, c, d]\n"
				src += "\t\tvmovq rax, xmm0     ; extract a\n"
				src += "\t\tmul r11             ; rdx:rax = rax * r11 = random * NODE_COUNT\n"
				src += "\t\tmov r8, rdx         ; save 1st future random index (r8)\n"
				src += "\t\t_vpermj ymm0, ymm0  ; [b, a, c, d]\n"
				src += "\t\tvmovq rax, xmm0     ; extract b\n"
				src += "\t\tmul r11\n"
				src += "\t\tmov r9, rdx         ; save 2nd future random index (r9)\n"
				src += "\t\t_vpermk ymm0, ymm0  ; [c, d, b, a]\n"
				src += "\t\tvmovq rax, xmm0     ; extract c\n"
				src += "\t\tmul r11"
				if batch4:
					src += "\n"
					src += "\t\tmov r10, rdx        ; save 3rd future random index (r10)\n"
					src += "\t\t_vpermj ymm0, ymm0  ; [d, c, b, a]\n"
					src += "\t\tvmovq rax, xmm0     ; extract d\n"
					src += "\t\tmul r11             ; save 4th future random index (rdx)\n\n"
				else:
					src += "             ; save 3rd future random index (rdx)\n\n"
			# generate 4 ω ∈ [0, 1) for probabilistic branching (ymm11)
			one = "ymm6"
			src += "\t\t; generate 4 \\omega \\in [0, 1) for probabilistic branching (ymm11)\n"
			src += "\t\t_vxoshiro256ss ymm11, ymm12, ymm13, ymm14, ymm15, ymm10       ; (vectorized) uint64\n"
			src += f"\t\t_vones {one}                                                   ; [1.0, 1.0, 1.0, 1.0]\n"
			src += f"\t\t_vomega ymm11, ymm11, {one}                                    ; (vectorized) \\omega \\in [0, 1)\n"
			src += f"\t\t_vln ymm11, ymm11, {one}, ymm7, ymm8, ymm9, ymm10, xmm10, rax  ; (vectorized) ln(\\omega) \\in [-inf, 0)\n\n"
			# (repeat 3 or 4 times)
			for regi in regis:
				pad = " " * (3 - len(regi))
				# pick uniformally random new state for the node
				wxys = "ymm1"  # [u_1, v_1, u_2, v_2] -> [x_1, y_1, x_2, y_2] -> s'
				flux = "ymm2"  # f'
				zmsk = "ymm3"  # comparision bit-mask -> [z_1, 0, z_2, 0]
				s = "ymm4"  # [s_1, s_1, s_2, s_2] where s_i = (u_i)^2 + (v_i)^2
				lbl_suffix = f"{regi.upper()}_{["TAIL", "BATCH4"][int(batch4)]}"
				padl = " " * (len(regi) + 1 + 6 - len(lbl_suffix))
				MARSAGLIA_START_2 = f"MARSAGLIA_START_2_{lbl_suffix}"
				MARSAGLIA_START_1 = f"MARSAGLIA_START_1_{lbl_suffix}"
				MARSAGLIA_END = f"MARSAGLIA_END_{lbl_suffix}"
				num = regis.index(regi)
				F = ["G", "F"][int(batch4)]
				S = ["T", "S"][int(batch4)]
				F1 = f"{F}{2*num+1}"
				F2 = f"{F}{2*num+2}"
				S1 = f"{S}{2*num+1}"
				S2 = f"{S}{2*num+2}"
				src += f"\t\t; pick uniformally random new state for node [{regi}] (Marsaglia's method)\n"
				if regis.index(regi) == 0:
					src += "\t\t                                          ; assert ymm6 == [1.0, 1.0, 1.0, 1.0]\n"
				else:
					src += "\t\t_vones ymm6                               ; [1.0, 1.0, 1.0, 1.0]\n"
				if "F" in self.allKeys:
					src += f"\t\t{MARSAGLIA_START_2}:             {pad}{padl}; 2 vectors, f' -> {flux}, s' -> {wxys}\n"
					src += f"\t\t\t_gen_ws {wxys}, {s}, {one}, ymm10\n"
					src += f"\t\t\tvcmppd {zmsk}, {s}, {one}, 11h          ; {s} < {one}\n"
					src += f"\t\t\tvmovmskpd rax, {zmsk}                   ; rax = 0...bbaa where a = (s_1 < 1.0) and b = (s_2 < 1.0)\n"
					src += f"\t\t\t{F1}:\ttest rax, 0100b                   ; test bit (b)\n"
					src += f"\t\t\t\tjz {F2}                             ; !b -> s_2 >= 1.0\n"
					src += f"\t\t\t\t_calc_xyz {wxys}, {zmsk}, {s}, ymm6\n"
					src += f"\t\t\t\tvperm2f128 {flux}, {wxys}, {zmsk}, 31h  ; [x_2, y_2, z_2, 0] = f'\n"
					src += "\t\t\t\ttest rax, 0001b                   ; test bit (a)\n"
					src += f"\t\t\t\tjz {MARSAGLIA_START_1}   {pad}{padl}; !a -> s_1 >= 1.0\n"
					src += f"\t\t\t\tvperm2f128 {wxys}, {wxys}, {zmsk}, 20h  ; [x_1, y_1, z_1, 0] = s'\n"
					src += f"\t\t\t\tjmp {MARSAGLIA_END}\n"
					src += f"\t\t\t{F2}:\ttest rax, 0001b                   ; test bit (a)\n"
					src += f"\t\t\t\tjz {MARSAGLIA_START_2}   {pad}{padl}; !a -> s_1 >= 1.0\n"
					src += f"\t\t\t\t_calc_xyz {wxys}, {zmsk}, {s}, ymm6\n"
					src += f"\t\t\t\tvperm2f128 {flux}, {wxys}, {zmsk}, 20h  ; [x_1, y_1, z_1, 0] -> f'\n"
					src += f"\t\t\t\t; jmp {MARSAGLIA_START_1}\n"
				# either no flux. or already haveflux. only need s'
				src += f"\t\t{MARSAGLIA_START_1}:             {pad}{padl}; only 1 vector, s' -> {wxys}\n"
				src += f"\t\t\t_gen_ws {wxys}, {s}, ymm6, ymm10\n"
				src += f"\t\t\tvcmppd {zmsk}, {s}, ymm6, 11h          ; {s} < ymm6\n"
				src += f"\t\t\tvmovmskpd rax, {zmsk}                   ; rax = 0...bbaa where a = (s_1 < 1.0) and b = (s_2 < 1.0)\n"
				src += f"\t\t\t{S1}:\ttest rax, 0001b                   ; test bit (a)\n"
				src += f"\t\t\t\tjz {S2}                             ; !a -> s_1 >= 1.0\n"
				src += f"\t\t\t\t_calc_xyz {wxys}, {zmsk}, {s}, ymm6\n"
				src += f"\t\t\t\tvperm2f128 {wxys}, {wxys}, {zmsk}, 20h  ; [x_1, y_1, z_1, 0] = s'\n"
				src += f"\t\t\t\tjmp {MARSAGLIA_END}\n"
				src += f"\t\t\t{S2}:\ttest rax, 0100b                   ; test bit (b)\n"
				src += f"\t\t\t\tjz {MARSAGLIA_START_1}   {pad}{padl}; !b -> s_2 >= 1.0\n"
				src += f"\t\t\t\t_calc_xyz {wxys}, {zmsk}, {s}, ymm6\n"
				src += f"\t\t\t\tvperm2f128 {wxys}, {wxys}, {zmsk}, 31h  ; [x_2, y_2, z_2, 0] = s'\n"
				src += f"\t\t{MARSAGLIA_END}:\n"
				# scale by S and F
				spin = wxys
				flux = flux
				coef = "ymm3"  # scalar S or F broadcast, e.g. [S, S, S, S]
				msk = "xmm3"   # Note: vgather mask used before coef. overlap okay
				addr = "xmm4"  # Addresses [&S, &F] IMPORTANT: Can *not* not overlap with dest `sf` nor mask `msk`. Will cause runtime #UD fault!
				sf = "xmm5"    # scalars [S, F] packed
				# TODO: (optimization) skip for S=1.0 and/or F=1.0 (Note: S, F >= 0)
				if "F" in self.allKeys:
					src += f"\t\t; scale s' ({spin}) and f' ({flux})\n"
					src += f"\t\tlea rax, SFref\n"
					src += f"\t\tmov rsi, {regi}                           {pad}; calculate array offset\n"
					src += f"\t\tshl rsi, 4                             ; mul 16\n"
					src += f"\t\tvmovapd {addr}, xmmword ptr [rax + rsi]  ; pointers (double*[]) to [S, F]\n"
					src += f"\t\txor rax, rax                           ; 0 base for absolute vgather\n"
					src += f"\t\tvpcmpeqq {msk}, {msk}, {msk}              ; load mask (all 1's)\n"
					src += f"\t\tvgatherqpd {sf}, [rax + {addr}], {msk}\n"
					src += f"\t\tvbroadcastsd {coef}, {sf}                ; {coef} = [S, S, S, S]\n"
					src += f"\t\tvmulpd {spin}, {spin}, {coef}                ; scale {spin} by S\n"
					src += f"\t\t_vpermj {sf}, {sf}                     ; [F, S]\n"
					src += f"\t\tvbroadcastsd {coef}, {sf}                ; {coef} = [F, F, F, F]\n"
					src += f"\t\tvmulpd {flux}, {flux}, {coef}                ; scale {flux} by F\n"
				else:
					src += f"\t\t; scale s' ({spin})\n"
					src += f"\t\tlea rax, Sref\n"
					src += f"\t\tmov rax, qword ptr [rax + {regi}*8]   {pad}; pointer (double*) to S\n"
					src += f"\t\tvbroadcastsd {coef}, qword ptr [rax]  ; [S, S, S, S]\n"
					src += f"\t\tvmulpd {spin}, {spin}, {coef}  ; scale {spin} by S\n"
				src += "\n"
				# compute -ΔU for the propsed state change
				src += "\t\t; compute -deltaU for the proposed state change\n"
				src += "\t\tlea rax, deltaU         ; pointer to array of function pointers\n"
				src += f"\t\tmov rax, [rax + {regi}*8]  {pad}; deltaU[{regi}], dereferenced to get the actual function pointer\n"
				src += f"\t\tcall rax                ; args: ({wxys}, {flux}?) -> return xmm0\n\n"
				# compute probability factor, ω < p = e^{-ΔU/kT} -> kT*ln(ω) < -ΔU
				ln_omegaX, ln_omegaY = "xmm11", "ymm11"
				kT = s.replace("y", "x")
				fact = coef.replace("y", "x")  # stores probability factor, kT*ln(ω)
				src += f"\t\t; compute probability factor ({fact}), \\omega < p = e^(-deltaU/kT) -> kT*ln(\\omega) < -deltaU\n"
				src += "\t\tlea rax, kTref\n"
				src += f"\t\tmov rax, qword ptr [rax + {regi}*8]  {pad}; double* -> rax\n"
				src += f"\t\tvmovq {kT}, qword ptr [rax]       ; {kT} = [kT, ?, 0, 0]\n"
				if regis.index(regi) == 0:
					src += f"\t\tvmulsd {fact}, {ln_omegaX}, {kT}          ; kT * (scalar) [(a), b, c, d] = kT * a\n\n"
				else:
					if regis.index(regi) == 1:
						src += f"\t\t_vpermj {ln_omegaY}, {ln_omegaY}              ; [b, a, c, d]\n"
						src += f"\t\tvmulsd {fact}, {ln_omegaX}, {kT}          ; kT * b\n\n"
					elif regis.index(regi) == 2:
						src += f"\t\t_vpermk {ln_omegaY}, {ln_omegaY}              ; [c, d, b, a]\n"
						src += f"\t\tvmulsd {fact}, {ln_omegaX}, {kT}          ; kT * c\n\n"
					elif regis.index(regi) == 3:
						src += f"\t\t_vpermj {ln_omegaY}, {ln_omegaY}              ; [d, c, b, a]\n"
						src += f"\t\tvmulsd {fact}, {ln_omegaX}, {kT}          ; kT * d\n\n"
				# (maybe) change the node's state
				spin = spin
				flux = flux
				SKIP = f"SKIP_{regi.upper()}_{["TAIL", "BATCH4"][int(batch4)]}"
				src += "\t\t; (maybe) change the node's state\n"
				src += f"\t\tvucomisd {fact}, xmm0  ; compare {fact} - xmm0\n"
				src += f"\t\tjae {SKIP}  {pad}; kT * ln(\\omega) >= -deltaU\n"
				src += "\t\tlea rax, nodes\n"
				if is_pow2(self.SIZEOF_NODE):
					src += f"\t\tshl {regi}, {self.SIZEOF_NODE.bit_length() - 1}  ; mul SIZEOF_NODE ({self.SIZEOF_NODE})\n"
				else:
					src += f"\t\timul {regi}, SIZEOF_NODE\n"
				src += f"\t\tvmovapd [rax + {regi} + OFFSETOF_SPIN], {spin}  {pad}; update s' -> spin\n"
				if "F" in self.allKeys:
					src += f"\t\tvmovapd [rax + {regi} + OFFSETOF_FLUX], {flux}  {pad}; update f' -> flux\n"
				src += f"\t\t{SKIP}:\n\n"
				# TODO?: parameter modification(s); e.g. global.B -= dB
				# ...
				if not batch4 and regi != "rdx":
					src += "\t\t; done?\n"
					src += "\t\tdec rcx\n"
					src += "\t\tjz TAIL_END\n\n"
			if batch4:
				src += "\t\tsub rcx, 4\n"
				src += "\t\tjmp LOOP_START\n"
				src += "\tLOOP_END:\n\n"
			else:
				src += "\tTAIL_END:\n\n"
		# save PRNG state
		src += "\t; save PRNG state\n"
		src += "\tvmovdqa ymmword ptr [prng_state + (0)*32], ymm12\n"
		src += "\tvmovdqa ymmword ptr [prng_state + (1)*32], ymm13\n"
		src += "\tvmovdqa ymmword ptr [prng_state + (2)*32], ymm14\n"
		src += "\tvmovdqa ymmword ptr [prng_state + (3)*32], ymm15\n\n"
		# epilogue
		src += "\t; epilogue: restore non-volitile registers\n"
		for i in range(10):  # XMM6-15
			src += f"\tvmovdqu xmm{6+i}, xmmword ptr [rbp - ({i+1})*16]\n"
		src += f"\tmov rsi, qword ptr [rbp - (10)*16 - 8]\n"
		src += "\tadd rsp, (10)*16 + 8  ; (aligned) space for the 10 non-volitile XMM6-15 reg. and RSI\n"
		src += "\tpop rbp\n"
		src += "\tret\n"
		src += "metropolis ENDP\n\n"
		exports.append(("metropolis", False))

		# seed PROC
		src += "; Seeds (or reseeds) the PRNG using rdseed if availbe, otherwise rdtscp."
		src += "; @param (void)\n"
		src += "; @return (void)\n"
		src += "PUBLIC seed\n"
		src += "seed PROC\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
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
			src += f"\tror rcx, {rot}   ; rotate small number to highest order bits (CPU cores = {cores})\n"
			src += "\txor rcx, rax\n"
			src += "\tTSC_END:     ; initial seed is now in rcx. Now SplitMix64.\n"
			for j in range(4):  # column-major b.c. each column represents an independent PRNG channel
				for i in range(4):
					src += "\t_splitmix64 rax, rcx, rdx\n"
					src += f"\tmov qword ptr [prng_state + ({i})*32 + ({j})*8], rax\n"
		src += "\tret\n"
		src += "seed ENDP\n\n"
		exports.append(("seed", False))

		# DLL main (for (un)initialization) -- bypass CRT
		src += "; Windows DLL main (for (un)initialization)\n"
		src += "; @param (rcx) hinstDLL\n"
		src += "; @param (rdx) fwdReason: enum 1, 2, 3, or 4\n"
		src += "; @param (r8) lpvReserved\n"
		src += "; @return (rax) bool: success or failure\n"
		src += "PUBLIC DllMain\n"
		src += "DllMain PROC\n"
		src += "\tcmp rdx, 1\n"
		src += "\tje DLL_PROCESS_ATTACH\n"
		src += "\tcmp rdx, 2\n"
		src += "\tje DLL_THREAD_ATTACH\n"
		src += "\tcmp rdx, 3\n"
		src += "\tje DLL_THREAD_DETACH\n"
		src += "\tcmp rdx, 4\n"
		src += "\tje DLL_PROCESS_DETACH\n"
		src += "\txor rax, rax  ; return false (This should never happen!)\n"
		src += "\tret\n\n"
		src += "\tDLL_PROCESS_ATTACH:\n"
		src += f"\t\t; seed PRNG ({prng})\n"
		if prng.startswith("xoshiro256"):  # TODO: support other PRNGs
			if seed is None or len(seed) == 0:
				src += "\t\tcall seed\n\n"
			else:
				src += "\t\t; (skip) seed was given and was expanded at compile time.\n\n"
		# src += "\tmov rcx, 7   ; number of iterations\n"
		# src += "\tsub rsp, 40  ; reserve shadow space\n"
		# src += "\tcall metropolis\n"
		# src += "\tadd rsp, 40  ; free shadow space\n\n"
		# src += "\txor rax, rax  ; return 0\n"
		src += "\tDLL_THREAD_ATTACH:   ; do nothing\n"
		src += "\tDLL_THREAD_DETACH:   ; do nothing\n"
		src += "\tDLL_PROCESS_DETACH:  ; do nothing\n\n"
		src += "\tmov rax, 1  ; return true\n"
		src += "\tret\n"
		src += "DllMain ENDP\n\n"

		# ---------------------------------------------------------------------
		for symbol, _ in exports:
			src += f"PUBLIC {symbol}\n"

		src += "END"  # absolute end of ASM file


		def reserve_tempfile(suffix):
			""" Create an empty temp file for later use. """
			fd, path = mkstemp(suffix=suffix, dir=dir)
			os.close(fd)
			return path

		asm_temp, obj_temp, _def_temp, dll_temp = False, False, False, False  # bool flags for cleanup
		if  asm is None:  asm = reserve_tempfile(".asm");  asm_temp = True
		if  obj is None:  obj = reserve_tempfile(".obj");  obj_temp = True
		if _def is None: _def = reserve_tempfile(".def"); _def_temp = True
		if  dll is None:  dll = reserve_tempfile(".dll");  dll_temp = True
		
		# DEBUG
		print("ASM:", asm)
		print("OBJ:", obj)
		print("DEF:", _def)
		print("DLL:", dll)
		
		with open(asm, "w", encoding="utf-8") as file:
			file.write(str(src))
		
		with open(_def, "w", encoding="utf-8") as file:
			file.write("EXPORTS\n")
			for symbol, data in exports:
				file.write(symbol)
				if data:
					file.write("\tDATA")
				file.write("\n")
		
		# compile/assemble
		with resources.as_file(resources.files(__package__)) as package_path:
			try:
				tool.assemble(asm, out=obj, include=[str(package_path)])
				tool.dlink(obj, out=dll, entry="DllMain", exports=_def)
			except CalledProcessError as ex:
				raise  # TODO: (stub) Re-raise exception
		
		# clean up tempp files
		if  asm_temp:  os.remove( asm)
		if  obj_temp:  os.remove( obj)
		if _def_temp:  os.remove(_def)

		# dynamically link to python
		return Runtime(config=deepcopy(self), dll=dll, delete=dll_temp)
