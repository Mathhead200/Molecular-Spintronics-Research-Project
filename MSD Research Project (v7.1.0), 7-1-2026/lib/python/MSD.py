# The Python binding of MSD.
# Uses ctypes to call underlying C code.
# See: MSD-export.cpp
#
# Author: Christopher D'Angelo
# Last Updated: November 30, 2022
# Version: 1.2

from ctypes import *
from typing import *
import os


# Load DLL
# TODO: how should the py lib. be configured, e.g. path?
MSD_EXPORT_DLL_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "MSD-export.dll")
msd_clib = CDLL(MSD_EXPORT_DLL_PATH)


# Helpers (classes, functions, etc.)
class _StructWithDict(Structure):
	''' Version of ctypes.Structure that defines __dict__ and overrides __repr__ '''
	def __init__(self, *args, **kw): super().__init__(*args, **kw)
	__dict__ = property(fget = lambda self: {k: getattr(self, k) for k, _ in self._fields_})
	__iter__ = lambda self: iter([getattr(self, k) for k,_ in self._fields_])
	__repr__ = lambda self: str(vars(self))
	
	# https://python-list.python.narkive.com/JjDfqgnK/iterable-ctypes-struct
	def __getitem__(self, idx):
		if isinstance(idx, str):
			return getattr(self, idx)
		else:
			return getattr(self, self._fields_[idx][0])
	
	def __setitem__(self, idx, val):
		if isinstance(idx, str):
			setattr(self, idx, val)
		else:
			setattr(self, self._fields_[idx][0], val)

class _MMTStruct(_StructWithDict):
	''' Adds the MSD-Molecule-Text, mmt() method for subclasses '''
	def __init__(self, *args, **kw): super().__init__(*args, **kw)
	def mmt(self) -> str:
		return "; ".join([f"{k}={v}" for k, v in vars(self).items()])

def _tupler(f, obj, c_types):
	values = [c_type() for c_type in c_types]  # instantiate each c_type in c_types list
	f(obj, *[pointer(v) for v in values])  # pass-by-pointer
	return tuple(values)

def _sig(restype, f, argtypes):
	'''Specify the signature of the given C fucntion'''
	f.restype = restype
	f.argtypes = argtypes

def _boundsCheck(index, max, min = 0):
	if index < min or index > max:
		raise IndexError(f"Index ({index}) out of bounds. Must be between {min} (inclusive) and {max} (inclusive).")


# public Types (and Globals) exposed by MSD.py library
PI = c_double.in_dll(msd_clib, "PI").value
E = c_double.in_dll(msd_clib, "E").value


class Vector(_StructWithDict):
	'''Python version of udc::Vector class in Vector.h'''
	_fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]

	def __init__(self, *args, **kw): super().__init__(*args, **kw)

	def __len__(self): return 3
	def __repr__(self): return f"({self.x}, {self.y}, {self.z})"

	@classmethod
	def cylindricalForm(cls, r, theta, z): return msd_clib.cylindricalForm(r, theta, z)

	@classmethod
	def polarForm(cls, r, theta): return msd_clib.polarForm(r, theta)

	@classmethod
	def sphericalForm(cls, rho, theta, phi): return msd_clib.sphericalForm(rho, theta, phi)

	normSq = property(fget = lambda self: msd_clib.normSq(byref(self)))
	norm = property(fget = lambda self: msd_clib.norm(byref(self)))
	theta = property(fget = lambda self: msd_clib.theta(byref(self)))
	phi = property(fget = lambda self: msd_clib.phi(byref(self)))

	def __eq__(self, other): return msd_clib.eq_v(byref(self), byref(other))
	def __ne__(self, other): return msd_clib.ne_v(byref(self), byref(other))

	def __add__(self, other): return msd_clib.add_v(byref(self), byref(other))
	def __neg__(self): return msd_clib.neg_v(byref(self))
	def __sub__(self, other): return msd_clib.sub_v(byref(self), byref(other))
	
	def __mul__(self, other):
		return self.dotProduct(other) if isinstance(other, Vector) else msd_clib.mul_v(byref(self), other)
	
	def distanceSq(self, other): return msd_clib.distanceSq(byref(self), byref(other))
	def distance(self, other): return msd_clib.distance(byref(self), byref(other))
	def dotProduct(self, other): return msd_clib.dotProduct(byref(self), byref(other))
	def angleBetween(self, other): return msd_clib.angleBetween(byref(self), byref(other))
	def crossProduct(self, other): return msd_clib.crossProduct(byref(self), byref(other))

	def __iadd__(self, other): msd_clib.iadd_v(pointer(self), byref(other))
	def __isub__(self, other): msd_clib.isub_v(pointer(self), byref(other))
	def __imul__(self, other): msd_clib.imul_v(pointer(self), byref(other))

	def negate(self): msd_clib.negate(pointer(self))
	
	def rotate(self, theta, phi = None):
		if theta is None:
			msd_clib.rotate_2d(pointer(self), theta)
		else:
			msd_clib.rotate_3d(pointer(self), theta, phi)
	
	def normalize(self): msd_clib.normalize(pointer(self))

Vector.I = Vector.in_dll(msd_clib, "I")
Vector.J = Vector.in_dll(msd_clib, "J")
Vector.K = Vector.in_dll(msd_clib, "K")
Vector.ZERO = Vector.in_dll(msd_clib, "ZERO")
# TODO: ^^ These are not actually const.
# 	Will this cause problems? Testing needed.


class Molecule:

	# inner classes
	class EdgeParameters(_MMTStruct):
		_fields_ = [
			("Jm", c_double),
			("Je1m", c_double),
			("Jeem", c_double),
			("bm", c_double),
			("Dm", Vector)
			]

		def __init__(self, *args, **kw):
			self.Jm = 0.0
			self.Je1m = 0.0
			self.Jeem = 0.0
			self.bm = 0.0
			self.Dm = Vector.ZERO
			super().__init__(*args, **kw)

	class NodeParameters(_MMTStruct):
		_fields_ = [
			("Sm", c_double),
			("Fm", c_double),
			("Je0m", c_double),
			("Am", Vector)
			]
		
		def __init__(self, *args, **kw):
			self.Sm = 1.0
			self.Fm = 0.0
			self.Je0m = 0.0
			self.Am = Vector.ZERO
			super().__init__(*args, **kw)
	

	class _NodeIterable:
		'''
		Represents an iterable collection of molecule nodes.
		Do not construct directly!
		Should be constructed using Molecule methods/properties, namely:
		Molecule.nodes
		'''
		def __init__(self, nodes):
			self._nodes = nodes
			self._begin = msd_clib.createBeginNodeIter(nodes)
			self._end = msd_clib.createEndNodeIter(nodes)

		def __del__(self):
			msd_clib.destroyNodeIter(self._end)
			msd_clib.destroyNodeIter(self._begin)
			msd_clib.destroyNodes(self._nodes)

		begin = property(fget = lambda self: Molecule._Node(iterable = self))
		end = property(fget = lambda self: Molecule._Node(iterable = self, start = (self._end, len(self))))
		def __iter__(self): return self.begin
		def __reversed__(self): return Molecule._Node(iterable = self, start = (self._end, len(self)), direction = Molecule._Node.BACKWARD)
		def __len__(self): return msd_clib.size_n(self._nodes)
		
		def __getitem__(self, index):
			index = index % len(self)
			if index >= 0:
				return self.begin + index
			else:
				return self.end - -index

	class _Node:
		'''
		Represents a specific molecule node.
		Also acts as a node iterator.
		Should not be construct directly!
		Should be constructed using Molecule._NodeIterable methods/properties.
		'''
		# these must be defined before __init__ since they are used as default arguments
		def next(self):
			if msd_clib.eq_n(self._node, self._iterable._end):
				raise StopIteration
			
			curr = self.clone()
			msd_clib.next_n(self._node)
			self._i += 1
			return curr
		
		def prev(self):
			if msd_clib.eq_n(self._node, self._iterable._begin):
				raise StopIteration
			
			msd_clib.prev_n(self._node)
			self._i -= 1
			return self.clone()
		
		# direction = tuple(__next__)
		FORWARD = (next,)
		BACKWARD = (prev,)

		def __init__(self, iterable, start = (None, 0), direction = FORWARD):
			self._iterable = iterable
			self._node = msd_clib.copyNodeIter(iterable._begin if start[0] is None else start[0])
			self._i = start[1]
			
			self._next = direction[0]
		
		def __del__(self): msd_clib.destroyNodeIter(self._node)
		
		def clone(self): return Molecule._Node(self._iterable, start = (self._node, self._i), direction = (self._next,))

		metaIndex = property(fget = lambda self: self._i)

		index = property(fget = lambda self: msd_clib.nodeIndex_i(self._node))
		def getParameters(self): return msd_clib.getNodeParameters_i(self._node)
		neighbors = property(fget = lambda self: Molecule._EdgeIterable(msd_clib.getNeighbors(self._node)))

		def __eq__(self, other): return msd_clib.eq_n(self._node, other._node)
		def __ne__(self, other): return msd_clib.ne_n(self._node, other._node)
		def __lt__(self, other): return msd_clib.lt_n(self._node, other._node)
		def __gt__(self, other): return msd_clib.gt_n(self._node, other._node)
		def __le__(self, other): return msd_clib.le_n(self._node, other._node)
		def __ge__(self, other): return msd_clib.ge_n(self._node, other._node)
		
		def __next__(self): return self._next(self)

		def __iadd__(self, offset):
			newPos = self._i + offset
			_boundsCheck(newPos, len(self._iterable))
			msd_clib.add_n(self._node, offset)
			self._i = newPos
		
		def __isub__(self, offset):
			newPos = self._i - offset
			_boundsCheck(newPos, len(self._iterable))
			msd_clib.sub_n(self._node, offset)
			self._i = newPos
		
		def __add__(self, offset):
			copy = self.clone()
			copy += offset
			return copy
		
		def __sub__(self, offset):
			copy = self.clone()
			copy -= offset
			return copy

	class _EdgeIterable:
		'''
		Represents an iterable collection of molecule edges.
		Do not construct directly!
		Should be constructed using Molecule methods/properties, namely:
		Molecule.edges and Molecule.getAdjacencyList()
		'''
		def __init__(self, edges):
			self._edges = edges
			self._begin = msd_clib.createBeginEdgeIter(edges)
			self._end = msd_clib.createEndEdgeIter(edges)
		
		def __del__(self):
			msd_clib.destroyEdgeIter(self._end)
			msd_clib.destroyEdgeIter(self._begin)
			msd_clib.destroyEdges(self._edges)
		
		begin = property(fget = lambda self: Molecule._Edge(iterable = self))
		end = property(fget = lambda self: Molecule._Edge(iterable = self, start = (self._end, len(self))))
		def __iter__(self): return self.begin
		def __reversed__(self): return Molecule._Edge(iterable = self, start = (self._end, len(self)), direction = Molecule._Edge.BACKWARD)
		def __len__(self): return msd_clib.size_e(self._edges)
		
		def __getitem__(self, index):
			index = index % len(self)
			if index >= 0:
				return self.begin + index
			else:
				return self.end - -index

	class _Edge:
		'''
		Represents a specific molecule edge.
		Also acts as an edge iterator.
		Should not be construct directly!
		Should be constructed using Molecule._EdgeIterable methods/properties.
		'''
		def next(self):
			if msd_clib.eq_e(self._edge, self._iterable._end):
				raise StopIteration
			
			curr = self.clone()
			msd_clib.next_e(self._edge)
			self._i += 1
			return curr
		
		def prev(self):
			if msd_clib.eq_e(self._edge, self._iterable._begin):
				raise StopIteration
			
			msd_clib.prev_e(self._edge)
			self._i -= 1
			return self.clone()
		
		# direction = (__next__)
		FORWARD = (next,)
		BACKWARD = (prev,)

		def __init__(self, iterable, start = (None, 0), direction = FORWARD):
			self._iterable = iterable
			self._edge = msd_clib.copyEdgeIter(iterable._begin if start[0] is None else start[0])
			self._i = start[1]
			
			self._next = direction[0]
		
		def __del__(self): msd_clib.destroyEdgeIter(self._edge)

		def clone(self): return Molecule._Edge(self._iterable, start = (self._edge, self._i), direction = (self._next,))
		
		metaIndex = property(fget = lambda self: self._i)

		index = property(fget = lambda self: msd_clib.edgeIndex_i(self._edge))
		def getParameters(self): return msd_clib.getEdgeParameters_i(self._edge)
		src = property(fget = lambda self: msd_clib.src_e(self._edge))
		dest = property(fget = lambda self: msd_clib.dest_e(self._edge))
		direction = property(fget = lambda self: msd_clib.getDirection(self._edge))
		
		def __eq__(self, other): return msd_clib.eq_e(self._edge, other._edge)
		def __ne__(self, other): return msd_clib.ne_e(self._edge, other._edge)
		def __lt__(self, other): return msd_clib.lt_e(self._edge, other._edge)
		def __gt__(self, other): return msd_clib.gt_e(self._edge, other._edge)
		def __le__(self, other): return msd_clib.le_e(self._edge, other._edge)
		def __ge__(self, other): return msd_clib.ge_e(self._edge, other._edge)
		
		def __next__(self): return self._next(self)
		
		def __iadd__(self, offset):
			newPos = self._i + offset
			_boundsCheck(newPos, len(self._iterable))
			msd_clib.add_e(self._edge, offset)
			self._i = newPos
		
		def __isub__(self, offset):
			newPos = self._i - offset
			_boundsCheck(newPos, len(self._iterable))
			msd_clib.sub_e(self._edge, offset)
			self._i = newPos
		
		def __add__(self, offset):
			copy = self.clone()
			copy += offset
			return copy
		
		def __sub__(self, offset):
			copy = self.clone()
			copy -= offset
			return copy

	# (export "C") Globals as static constants
	HEADER = c_char_p.in_dll(msd_clib, "HEADER")
	HEADER_SIZE = c_size_t.in_dll(msd_clib, "HEADER_SIZE")
	NOT_FOUND = c_uint.in_dll(msd_clib, "NOT_FOUND")

	# Molecule methods and properties
	def __init__(self, nodeCount = None, nodeParams = None, _proto: c_void_p = None):
		'''
		Construct a new Molecule prototype.
		An Optional nodeCount, and initial NodeParameters can be given.
		If nodeCount is None, an empty prototype is created with
		no nodes, and no edges.

		Both the left and right leads are initialized to 0.

		Note: _proto parameter is reserved for internal use only!
		'''
		self._proto: c_void_p = _proto

		if _proto is None:
			if nodeCount is None:
				self._proto = msd_clib.createMolProto_e()
			elif nodeParams is None:
				self._proto = msd_clib.createMolProto_n(nodeCount)
			else:
				self._proto = msd_clib.createMolProto_p(nodeCount, byref(nodeParams))

	def __del__(self): msd_clib.destroyMolProto(self._proto)

	def __repr__(self): return self.mmt()

	def serialize(self, buffer: bytearray): msd_clib.serialize(self._proto, buffer)
	def deserialize(self, buffer: bytearray): msd_clib.deserialize(self._proto, buffer)  # TODO, known bug: edges aren't preserved
	serializationSize = property(fget = lambda self: msd_clib.serializationSize(self._proto))
	
	# TODO: read, write, and load are not yet defined
	def read(self, istream): raise NotImplementedError()  # TODO stub
	def write(self, ostream): raise NotImplementedError()  # TODO stub
	def load(istream): raise NotImplementedError()  # static, TODO stub
	
	def createNode(self, nodeParams = None):
		'''
		Creates a new node in the molecule prototype, and returns its index.
		Optional NodeParameters can be given. Returns the new node's index.
		
		If nodeParam is None, this method uses C++ defaults for nodeParams. This should
		have the same effect as using a default-constructed Python NodeParameter object.
		'''
		if nodeParams is None:
			return msd_clib.createNode_d(self._proto)
		return msd_clib.createNode_p(self._proto, byref(nodeParams))

	nodeCount = property(fget = lambda self: msd_clib.nodeCount(self._proto))
	
	def connectNodes(self, nodeA, nodeB, edgeParams = None):
		'''
		Connects the two given nodes (by index), and
		returns an edge index of the newly constructed edge.
		Optional EdgeParameters can be given.

		Note: this method does not check if the two nodes are already connected.
	 	It is possible to creates more then one edge connecting the same two nodes.

		If edgeParam is None, this method the uses C++ defaults for nodeParams. This should
		have the same effect as using a default-constructed Python NodeParameter object.
		'''
		if edgeParams is None:
			return msd_clib.connectNodes_d(self._proto, nodeA, nodeB)
		else:
			return msd_clib.connectNodes_p(self._proto, nodeA, nodeB, byref(edgeParams))

	def edgeIndex(self, nodeA, nodeB):
		'''
		Search for and return the index of the edge connecting the given nodes if one exists,
		or Molecule.NOT_FOUND if one doesn't.
	 	'''
		return msd_clib.edgeIndex(self._proto, nodeA, nodeB)
	
	def getEdgeParameters(self, index): return msd_clib.getEdgeParameters(self._proto, index)
	def setEdgeParamters(self, index, edgeParams): msd_clib.setEdgeParameters(self._proto, index, byref(edgeParams))
	def getNodeParameters(self, index): return msd_clib.getNodeParameters(self._proto, index)
	def setNodeParamters(self, index, nodeParams): msd_clib.setNodeParameters(self._proto, index, byref(nodeParams))
	def setAllParameters(self, nodeParams, edgeParams): msd_clib.setAllParameters(self._proto, byref(nodeParams), byref(edgeParams))

	leftLead = property(
		fget = lambda self: msd_clib.getLeftLead(self._proto),
		fset = lambda self, node: msd_clib.setLeftLead(self._proto, node)
		)
	
	rightLead = property(
		fget = lambda self: msd_clib.getRightLead(self._proto),
		fset = lambda self, node: msd_clib.setRightLead(self._proto, node)
		)

	leads = property(
		fget = lambda self: _tupler(msd_clib.getLeads, self._proto, 2 * [c_uint]),
		fset = lambda self, leads: msd_clib.setLeads(self._proto, *leads)
		)
	
	nodes = property(fget = lambda self: Molecule._NodeIterable(msd_clib.createNodes(self._proto)))
	edges = property(fget = lambda self: Molecule._EdgeIterable(msd_clib.createEdges(self._proto)))
	def getAdjacencyList(self, nodeIndex): return Molecule._EdgeIterable(msd_clib.getAdjacencyList(self._proto, nodeIndex))
	
	@property
	def edgesUnique(self):
		'''
		Generator of edges which contain uniqie edgeIndexes.
		Useful since edges are directional and edges between two different nodes will be duplicated.

		The elements of the list will be Molecule._Edge objects, which are themselves iterable.
		But iterating on these objects will return "duplicated" edges (edges with the same edgeIndex)
		from the underlying Molecule._EdgeIterable.
		'''
		visited: Set[int] = set()
		for edge in self.edges:
			edgeIndex = edge.index
			if edgeIndex not in visited:
				visited.add(edgeIndex)
				yield edge

	def mmt(self):
		''' MMT formated string '''
		# Nodes
		nodes = self.nodes
		mmt = str(len(nodes)) + "\n"
		for node in nodes:
			mmt += node.getParameters().mmt() + "\n"
		mmt += "\n"

		# Edges
		edges = self.edgesUnique
		mmt += str(len(edges)) + "\n"
		for edge in edges:
			mmt += edge.getParameters().mmt()
			mmt += f"; srcNode={edge.src}; destNode={edge.dest}\n"
		mmt += "\n"

		# Leads
		mmt += f"{self.leftLead}\n{self.rightLead}\n"

		return mmt

class MSD:

	# static typedef
	MolProto = Molecule

	# (export "C") Globals as static constants
	LINEAR_MOL = c_void_p.in_dll(msd_clib, "LINEAR_MOL")
	CIRCULAR_MOL = c_void_p.in_dll(msd_clib, "CIRCULAR_MOL")

	UP_DOWN_MODEL = c_void_p.in_dll(msd_clib, "UP_DOWN_MODEL")
	CONTINUOUS_SPIN_MODEL = c_void_p.in_dll(msd_clib, "CONTINUOUS_SPIN_MODEL")


	# inner classes
	class Parameters(_StructWithDict):
		_fields_ = [
			("kT", c_double),
			("B", Vector),
			("SL", c_double), ("SR", c_double),
			("FL", c_double), ("FR", c_double),
			("JL", c_double), ("JR", c_double), ("JmL", c_double), ("JmR", c_double), ("JLR", c_double),
			("Je0L", c_double), ("Je0R", c_double),
			("Je1L", c_double), ("Je1R", c_double), ("Je1mL", c_double), ("Je1mR", c_double), ("Je1LR", c_double),
			("JeeL", c_double), ("JeeR", c_double), ("JeemL", c_double), ("JeemR", c_double), ("JeeLR", c_double),
			("bL", c_double), ("bR", c_double), ("bmL", c_double), ("bmR", c_double), ("bLR", c_double),
			("AL", Vector), ("AR", Vector),
			("DL", Vector), ("DR", Vector), ("DmL", Vector), ("DmR", Vector), ("DLR", Vector)				
			]
		
		def __init__(self, *args, **kw):
			self.kT = 0.25
			self.B = Vector.ZERO
			self.SL, self.SR = 1.0, 1.0
			self.FL, self.FR = 0.0, 0.0
			self.JL, self.JR, self.JmL, self.JmR, self.JLR = 1.0, 1.0, 1.0, -1.0, 0.0
			self.Je0L, self.Je0R = 0.0, 0.0
			self.Je1L, self.Je1R, self.Je1mL, self.Je1mR, self.Je1LR = 0.0, 0.0, 0.0, 0.0, 0.0
			self.JeeL, self.JeeR, self.JeemL, self.JeemR, self.JeeLR = 0.0, 0.0, 0.0, 0.0, 0.0
			self.bL, self.bR, self.bmL, self.bmR, self.bLR = 0.0, 0.0, 0.0, 0.0, 0.0
			self.AL, self.AR = Vector.ZERO, Vector.ZERO
			self.DL, self.DR, self.DmL, self.DmR, self.DLR = Vector.ZERO, Vector.ZERO, Vector.ZERO, Vector.ZERO, Vector.ZERO
			super().__init__(*args, **kw)

	class Results(_StructWithDict):
		_fields_ = [
			("t", c_ulonglong),
			("M", Vector), ("ML", Vector), ("MR", Vector), ("Mm", Vector),
			("MS", Vector), ("MSL", Vector), ("MSR", Vector), ("MSm", Vector),
			("MF", Vector), ("MFL", Vector), ("MFR", Vector), ("MFm", Vector),
			("U", c_double), ("UL", c_double), ("UR", c_double), ("Um", c_double), ("UmL", c_double), ("UmR", c_double), ("ULR", c_double)
			]

		def __init__(self, *args, **kw):
			self.t = 0
			self.M, self.ML, self.MR, self.Mm = Vector.ZERO, Vector.ZERO, Vector.ZERO, Vector.ZERO
			self.MS, self.MSL, self.MSR, self.MSm = Vector.ZERO, Vector.ZERO, Vector.ZERO, Vector.ZERO
			self.MF, self.MFL, self.MFR, self.MFm = Vector.ZERO, Vector.ZERO, Vector.ZERO, Vector.ZERO
			self.U, self.UL, self.UR, self.Um, self.UmL, self.UmR, self.ULR = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
			super().__init__(*args, **kw)


	class _Iterator:
		def next(self):
			if msd_clib.eq_a(self._iter, self._msd._end):
				raise StopIteration
			
			curr = self.clone()
			msd_clib.next_a(self._iter)
			self._i += 1
			return curr
		
		def prev(self):
			if msd_clib.eq_a(self._iter, self._msd._begin):
				raise StopIteration
			
			msd_clib.prev_a(self._iter)
			self._i -= 1
			return self.clone()

		FORWARD = (next,)
		BACKWARD = (prev,)

		def __init__(self, msd, start = (None, 0), direction = FORWARD):
			self._msd = msd
			self._iter = msd_clib.copyMSDIter(msd._begin if start[0] is None else start[0])
			self._i = start[1]

			self._next = direction[0]
		
		def __del__(self): msd_clib.destroyMSDIter(self._iter)
		
		def clone(self): return MSD._Iterator(self._msd, start = (self._iter, self._i), direction = (self._next,))

		metaIndex = property(fget = lambda self: self._i)

		index = property(fget = lambda self: msd_clib.msdIndex(self._iter))
		x = property(fget = lambda self: msd_clib.getX(self._iter))
		y = property(fget = lambda self: msd_clib.getY(self._iter))
		z = property(fget = lambda self: msd_clib.getZ(self._iter))
		pos = property(fget = lambda self: (self.x, self.y, self.z))
		spin = property(fget = lambda self: msd_clib.getSpin_a(self._iter))
		flux = property(fget = lambda self: msd_clib.getFlux_a(self._iter))
		localM = property(fget = lambda self: msd_clib.getLocalM_a(self._iter))

		def __eq__(self, other): return msd_clib.eq_a(self._iter, other._iter)
		def __ne__(self, other): return msd_clib.ne_a(self._iter, other._iter)
		def __lt__(self, other): return msd_clib.lt_a(self._iter, other._iter)
		def __gt__(self, other): return msd_clib.gt_a(self._iter, other._iter)
		def __le__(self, other): return msd_clib.le_a(self._iter, other._iter)
		def __ge__(self, other): return msd_clib.ge_a(self._iter, other._iter)
		
		def __next__(self): return self._next(self)

		def __iadd__(self, offset):
			newPos = self._i + offset
			_boundsCheck(newPos, len(self._msd))
			msd_clib.add_a(self._iter, offset)
			self._i = newPos
		
		def __isub__(self, offset):
			newPos = self._i - offset
			_boundsCheck(newPos, len(self._msd))
			msd_clib.sub_a(self._iter, offset)
			self._i = newPos
		
		def __add__(self, offset):
			copy = self.clone()
			copy += offset
			return copy
		
		def __sub__(self, offset):
			copy = self.clone()
			copy -= offset
			return copy
		


	# MSD Methods and Properties
	def __init__(self, width, height, depth, \
			molProto: Optional[MolProto] = None, molType: Optional[c_void_p] = None, \
			molPosL = None, molPosR = None, \
			topL = None, bottomL = None, frontR = None, backR = None, \
			molPos = None, molLen = None, heightL = None, depthR = None
	):
		self._msd: c_void_p = None
		
		if molProto is not None:
			nodeCount = molProto.nodeCount
			if molLen is None:
				molLen = nodeCount
			elif nodeCount != molLen:
				raise ArgumentError(f"molProto.nodeCount ({nodeCount}) != molLen ({molLen})")				

		if molPos is not None:
			if molPosL is None:
				molPosL = molPos
			elif molPosL != molPos:
				raise ArgumentError(f"molPos ({molPos}) != molPosL ({molPosL})")
			
			if molPosR is None:
				molPosR = molPos
			elif molPos != molPos:
				raise ArgumentError(f"molPos ({molPos}) != molPosR ({molPosR})")

		if molLen is not None:
			L = molPosL is not None
			R = molPosR is not None
			if L and R:
				if molLen != molPosR - molPosL + 1:
					raise ArgumentError(f"molLen ({molLen}) != molPosR ({molPosR}) - molPosL ({molPosL}) + 1")
			elif L:
				molPosR = molPosL + (molLen - 1)
			elif R:
				molPosL = molPosR - (molLen - 1)
			else:
				molPosL = int((width - molLen) / 2)
				molPosR = molPosL + (molLen - 1)

		if heightL is not None:
			T = topL is not None
			B = bottomL is not None
			if T and B:
				if heightL != bottomL - topL + 1:
					raise ArgumentError(f"heightL ({heightL}) != bottomL ({bottomL}) - topL ({topL}) + 1")
			elif T:
				bottomL = topL + (heightL - 1)
			elif B:
				topL = bottomL - (heightL - 1)
			else:
				topL = int((height - heightL) / 2)
				bottomL = topL + (heightL - 1)
		
		if depthR is not None:
			F = frontR is not None
			B = backR is not None
			if F and B:
				if depthR != backR - frontR + 1:
					raise ArgumentError(f"depthR ({depthR}) != backR ({backR}) - frontR ({frontR}) + 1")
			elif F:
				backR = frontR + (depthR - 1)
			elif B:
				frontR = backR - (depthR - 1)
			else:
				frontR = int((depth - depthR) / 2)
				backR = frontR + (depthR - 1)
		
		if molPosL is None != molPosR is None:
			raise ArgumentError("molPosL and molPosR must both be specified, implied, or omitted together")
		
		if molPosL is None:  # and molPosR is None
			molPosL = int((width - 1) / 2)
			molPosR = molPosL
		
		if topL is None:
			topL = 0
		if bottomL is None:
			bottomL = height - 1
		if frontR is None:
			frontR = 0
		if backR is None:
			backR = depth - 1

		if molProto is not None and molType is not None:
			raise ArgumentError("Can not specify both molProto and molType")

		if molProto is not None:
			self._msd = msd_clib.createMSD_p(width, height, depth, molProto._proto, molPosL, topL, bottomL, frontR, backR)
		elif molType is not None:
			self._msd = msd_clib.createMSD_f(width, height, depth, molType, molPosL, molPosR, topL, bottomL, frontR, backR)
		else:
			self._msd = msd_clib.createMSD_i(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR)
		
		# get iterators at construction because msd dimensions are immutable
		self._begin = msd_clib.createBeginMSDIter(self._msd)
		self._end = msd_clib.createEndMSDIter(self._msd)

	def __del__(self):
		msd_clib.destroyMSDIter(self._end)
		msd_clib.destroyMSDIter(self._begin)
		msd_clib.destroyMSD(self._msd)

	@property
	def record(self):
		n = msd_clib.getRecordSize(self._msd)
		cArr = msd_clib.getRecord(self._msd)
		return [cArr[i] for i in range(n)]
	
	@record.setter
	def record(self, record):
		n = len(record)
		msd_clib.setRecord(self._msd, (MSD.Results * n)(*record), n)
	
	flippingAlgorithm = property(fset = lambda self, algo: msd_clib.setFlippingAlgorithm(self._msd, algo))

	def getParameters(self): return msd_clib.getParameters(self._msd)
	def setParameters(self, parameters): msd_clib.setParameters(self._msd, parameters)
	# No parameters property fget because I don't want to accidentally expose the anti-pattern:
	# msd.parameters.kT = 0.2 wouldn't doesn't change the underlying parameters!!
	parameters = property(fset = setParameters)
	def getResults(self): return msd_clib.getResults(self._msd)
	results = property(fget = getResults)

	def set_kT(self, kT): msd_clib.set_kT(self._msd, kT)
	kT = property(fset = set_kT)
	def setB(self, B): msd_clib.setB(self._msd, byref(B))
	B = property(fset = setB)

	def getMolProto(self): return Molecule(_proto = msd_clib.getMolProto(self._msd))
	def setMolProto(self, mol: Molecule): msd_clib.setMolProto(self._msd, mol._proto)
	molProto = property(fset = setMolProto)
	
	def setMolParameters(self, nodeParams = Molecule.NodeParameters(), edgeParams = Molecule.EdgeParameters()):
		msd_clib.setMolParameters(self._msd, nodeParams, edgeParams)
	
	# params = tuple(nodeParams, edgeParams)
	molParameters = property(fset = lambda self, params: self.setMolParameters(*params))

	def getSpin(self, x, y = None, z = None):
		if y is None:
			return msd_clib.getSpin_i(self._msd, x)
		return msd_clib.getSpin_v(self._msd, x, y, z)
	
	def getFlux(self, x, y = None, z = None):
		if y is None:
			return msd_clib.getFlux_i(self._msd, x)
		return msd_clib.getFlux_v(self._msd, x, y, z)
	
	def getLocalM(self, x, y = None, z = None):
		if y is None:
			return msd_clib.getLocalM_i(self._msd, x)
		return msd_clib.getLocalM_v(self._msd, x, y, z)
	
	def setSpin(self, x, y = None, z = None, spin = None):
		'''
		Version 1: setSpin(a, spin)
		Version 2: setSpin(x, y, z, spin)
		
		Note: spin can be passed as a positional argument or a named argument
		'''
		if spin is None:
			if y is None:
				raise ArgumentError("spin must be passed as second positional argument, or as named parameter")
			spin = y
		
		spin = byref(spin)
		if z is None:
			msd_clib.setSpin_i(self._msd, x, spin)
		else:
			msd_clib.setSpin_v(self._msd, x, y, z, spin)
	
	def setFlux(self, x, y = None, z = None, flux = None):
		'''
		Version 1: setFlux(a, flux)
		Version 2: setFlux(x, y, z, flux)

		Note: flux can be passed as a positional argument or a named argument
		'''
		if flux is None:
			if y is None:
				raise ArgumentError("flux must be passed as second positional argument, or as named parameter")
			flux = y
		
		flux = byref(flux)
		if z is None:
			msd_clib.setFlux_i(self._msd, x, flux)
		else:
			msd_clib.setFlux_v(self._msd, x, y, z, flux)
	
	def setLocalM(self, x, y = None, z = None, spin = None, flux = None):
		'''
		Version 1: setLocalM(a, spin, flux)
		Version 2: setLocalM(x, y, z, spin, flux)

		Note: spin and flux can be passed as positional arguments or named arguments
		'''
		mode_i = y is None or spin is None

		if spin is None:
			if y is None:
				raise ArgumentError("spin must be passed as second positional argument, or as named parameter")
			spin = y
		
		if flux is None:
			if z is None:
				raise ArgumentError("flux must be passed as third positional argument, or as named parameter")
			flux = z

		spin = byref(spin)
		flux = byref(flux)
		if mode_i:
			msd_clib.setLocalM_i(self._msd, x, spin, flux)
		else:
			msd_clib.setLocalM_v(self._msd, x, y, z, spin, flux)
	
	def __getitem__(self, idx):
		if isinstance(idx, Iterable):
			return (self.getSpin(*idx), self.getFlux(*idx))
		return (self.getSpin(idx), self.getFlux(idx))
	
	def __setitem__(self, idx, localM):
		if isinstance(idx, Iterable):
			self.setLocalM(*idx, *localM)
		else:
			self.setLocalM(idx, spin = localM[0], flux = localM[1])
	
	n = property(fget = lambda self: msd_clib.getN(self._msd))
	nL = property(fget = lambda self: msd_clib.getNL(self._msd))
	nR = property(fget = lambda self: msd_clib.getNR(self._msd))
	n_m = property(fget = lambda self: msd_clib.getNm(self._msd))
	n_mL = property(fget = lambda self: msd_clib.getNmL(self._msd))
	n_mR = property(fget = lambda self: msd_clib.getNmR(self._msd))
	nLR = property(fget = lambda self: msd_clib.getNLR(self._msd))
	width = property(fget = lambda self: msd_clib.getWidth(self._msd))
	height = property(fget = lambda self: msd_clib.getHeight(self._msd))
	depth = property(fget = lambda self: msd_clib.getDepth(self._msd))
	dimensions = property(fget = lambda self : _tupler(msd_clib.getDimensions, self._msd, 3 * [c_uint]))
	molPosL = property(fget = lambda self: msd_clib.getMolPosL(self._msd))
	molPosR = property(fget = lambda self: msd_clib.getMolPosR(self._msd))
	molPos = property(fget = lambda self: _tupler(msd_clib.getMolPos, self._msd, 2 * [c_uint]))
	topL = property(fget = lambda self: msd_clib.getTopL(self._msd))
	bottomL = property(fget = lambda self: msd_clib.getBottomL(self._msd))
	frontR = property(fget = lambda self: msd_clib.getFrontR(self._msd))
	backR = property(fget = lambda self: msd_clib.getBackR(self._msd))
	innerBounds = property(fget = lambda self: _tupler(msd_clib.getInnerBounds, self._msd, 4 * [c_uint]))
	FM_L_exists = property(fget = lambda self: msd_clib.getFM_L_exists(self._msd))
	FM_R_exists = property(fget = lambda self: msd_clib.getFM_R_exists(self._msd))
	mol_exists = property(fget = lambda self: msd_clib.getMol_exists(self._msd))
	regions = property(fget = lambda self: _tupler(msd_clib.getRegions, self._msd, 3 * [c_bool]))

	seed = property(
		fget = lambda self : msd_clib.getSeed(self._msd),
		fset = lambda self, seed: msd_clib.setSeed(self._msd, seed)
		)

	def reinitialize(self, reseed = True): msd_clib.reinitialize(self._msd, reseed)
	def randomize(self, reseed = True): msd_clib.randomize(self._msd, reseed)

	def metropolis(self, N, freq = None):
		if freq is None:
			msd_clib.metropolis_o(self._msd, N)
		else:
			msd_clib.metropolis_r(self._msd, N, freq)
	
	specificHeat = property(fget = lambda self : msd_clib.specificHeat(self._msd))
	specificHeat_L = property(fget = lambda self : msd_clib.specificHeat_L(self._msd))
	specificHeat_R = property(fget = lambda self : msd_clib.specificHeat_R(self._msd))
	specificHeat_m = property(fget = lambda self : msd_clib.specificHeat_m(self._msd))
	specificHeat_mL = property(fget = lambda self : msd_clib.specificHeat_mL(self._msd))
	specificHeat_mR = property(fget = lambda self : msd_clib.specificHeat_mR(self._msd))
	specificHeat_LR = property(fget = lambda self : msd_clib.specificHeat_LR(self._msd))
	magneticSusceptibility = property(fget = lambda self: msd_clib.magneticSusceptibility(self._msd))
	magneticSusceptibility_L = property(fget = lambda self: msd_clib.magneticSusceptibility_L(self._msd))
	magneticSusceptibility_R = property(fget = lambda self: msd_clib.magneticSusceptibility_R(self._msd))
	magneticSusceptibility_m = property(fget = lambda self: msd_clib.magneticSusceptibility_m(self._msd))

	meanM = property(fget = lambda self : msd_clib.meanM(self._msd))
	meanML = property(fget = lambda self : msd_clib.meanML(self._msd))
	meanMR = property(fget = lambda self : msd_clib.meanMR(self._msd))
	meanMm = property(fget = lambda self : msd_clib.meanMm(self._msd))
	meanMS = property(fget = lambda self : msd_clib.meanMS(self._msd))
	meanMSL = property(fget = lambda self : msd_clib.meanMSL(self._msd))
	meanMSR = property(fget = lambda self : msd_clib.meanMSR(self._msd))
	meanMSm = property(fget = lambda self : msd_clib.meanMSm(self._msd))
	meanMF = property(fget = lambda self : msd_clib.meanMF(self._msd))
	meanMFL = property(fget = lambda self : msd_clib.meanMFL(self._msd))
	meanMFR = property(fget = lambda self : msd_clib.meanMFR(self._msd))
	meanMFm = property(fget = lambda self : msd_clib.meanMFm(self._msd))
	meanU = property(fget = lambda self : msd_clib.meanU(self._msd))
	meanUL = property(fget = lambda self : msd_clib.meanUL(self._msd))
	meanUR = property(fget = lambda self : msd_clib.meanUR(self._msd))
	meanUm = property(fget = lambda self : msd_clib.meanUm(self._msd))
	meanUmL = property(fget = lambda self : msd_clib.meanUmL(self._msd))
	meanUmR = property(fget = lambda self : msd_clib.meanUmR(self._msd))
	meanULR = property(fget = lambda self : msd_clib.meanULR(self._msd))

	begin = property(fget = lambda self: MSD._Iterator(msd = self))
	end = property(fget = lambda self: MSD._Iterator(msd = self, start = (self._end, len(self))))
	def __iter__(self): return self.begin
	def __reversed__(self): return MSD._Iterator(msd = self, start = (self._end, len(self)), direction = MSD._Iterator.BACKWARD)
	def __len__(self): return self.n


# (export "C") Function Signatures/Declarations
_sig(c_void_p, msd_clib.createMSD_p, 3 * [c_uint] + [c_void_p] + 5 * [c_uint])
_sig(c_void_p, msd_clib.createMSD_f, 3 * [c_uint] + [c_void_p] + 6 * [c_uint])
_sig(c_void_p, msd_clib.createMSD_i, 9 * [c_uint])
_sig(c_void_p, msd_clib.createMSD_c, 5 * [c_uint])
_sig(c_void_p, msd_clib.createMSD_d, 3 * [c_uint])
_sig(None, msd_clib.destroyMSD, [c_void_p])

_sig(POINTER(MSD.Results), msd_clib.getRecord, [c_void_p])  # returns c-array
_sig(c_size_t, msd_clib.getRecordSize, [c_void_p])
_sig(None, msd_clib.setRecord, [c_void_p, POINTER(MSD.Results), c_size_t])  # takes c-array
_sig(None, msd_clib.setFlippingAlgorithm, 2 * [c_void_p])

_sig(MSD.Parameters, msd_clib.getParameters, [c_void_p])
_sig(None, msd_clib.setParameters, [c_void_p, MSD.Parameters])
_sig(MSD.Results, msd_clib.getResults, [c_void_p])

_sig(None, msd_clib.set_kT, [c_void_p, c_double])
_sig(None, msd_clib.setB, [c_void_p, POINTER(Vector)])

_sig(c_void_p, msd_clib.getMolProto, [c_void_p])
_sig(None, msd_clib.setMolProto, 2 * [c_void_p])
_sig(None, msd_clib.setMolParameters, [c_void_p, Molecule.NodeParameters, Molecule.EdgeParameters])

_sig(Vector, msd_clib.getSpin_i, [c_void_p, c_uint])
_sig(Vector, msd_clib.getSpin_v, [c_void_p] + 3 * [c_uint])
_sig(Vector, msd_clib.getFlux_i, [c_void_p, c_uint])
_sig(Vector, msd_clib.getFlux_v, [c_void_p] + 3 * [c_uint])
_sig(Vector, msd_clib.getLocalM_i, [c_void_p, c_uint])
_sig(Vector, msd_clib.getLocalM_v, [c_void_p] + 3 * [c_uint])
_sig(None, msd_clib.setSpin_i, [c_void_p, c_uint, POINTER(Vector)])
_sig(None, msd_clib.setSpin_v, [c_void_p] + 3 * [c_uint] + [POINTER(Vector)])
_sig(None, msd_clib.setFlux_i, [c_void_p, c_uint, POINTER(Vector)])
_sig(None, msd_clib.setFlux_v, [c_void_p] + 3 * [c_uint] + [POINTER(Vector)])
_sig(None, msd_clib.setLocalM_i, [c_void_p, c_uint] + 2 * [POINTER(Vector)])
_sig(None, msd_clib.setLocalM_v, [c_void_p] + 3 * [c_uint] + 2 * [POINTER(Vector)])

_sig(c_uint, msd_clib.getN, [c_void_p])
_sig(c_uint, msd_clib.getNL, [c_void_p])
_sig(c_uint, msd_clib.getNR, [c_void_p])
_sig(c_uint, msd_clib.getNm, [c_void_p])
_sig(c_uint, msd_clib.getNmL, [c_void_p])
_sig(c_uint, msd_clib.getNmR, [c_void_p])
_sig(c_uint, msd_clib.getNLR, [c_void_p])
_sig(c_uint, msd_clib.getWidth, [c_void_p])
_sig(c_uint, msd_clib.getHeight, [c_void_p])
_sig(c_uint, msd_clib.getDepth, [c_void_p])
_sig(None, msd_clib.getDimensions, [c_void_p]  + 3 * [POINTER(c_uint)])
_sig(c_uint, msd_clib.getMolPosL, [c_void_p])
_sig(c_uint, msd_clib.getMolPosR, [c_void_p])
_sig(None, msd_clib.getMolPos, [c_void_p] + 2 * [POINTER(c_uint)])
_sig(c_uint, msd_clib.getTopL, [c_void_p])
_sig(c_uint, msd_clib.getBottomL, [c_void_p])
_sig(c_uint, msd_clib.getFrontR, [c_void_p])
_sig(c_uint, msd_clib.getBackR, [c_void_p])
_sig(None, msd_clib.getInnerBounds, [c_void_p] + 4 * [POINTER(c_uint)])
_sig(c_bool, msd_clib.getFM_L_exists, [c_void_p])
_sig(c_bool, msd_clib.getFM_R_exists, [c_void_p])
_sig(c_bool, msd_clib.getMol_exists, [c_void_p])
_sig(None, msd_clib.getRegions, [c_void_p] + 3 * [POINTER(c_bool)])

_sig(None, msd_clib.setSeed, [c_void_p, c_ulong])
_sig(c_ulong, msd_clib.getSeed, [c_void_p])

_sig(None, msd_clib.reinitialize, [c_void_p, c_bool])
_sig(None, msd_clib.randomize, [c_void_p, c_bool])
_sig(None, msd_clib.metropolis_o, [c_void_p, c_ulonglong])
_sig(None, msd_clib.metropolis_r, [c_void_p] + 2 * [c_ulonglong])

_sig(c_double, msd_clib.specificHeat, [c_void_p])
_sig(c_double, msd_clib.specificHeat_L, [c_void_p])
_sig(c_double, msd_clib.specificHeat_R, [c_void_p])
_sig(c_double, msd_clib.specificHeat_m, [c_void_p])
_sig(c_double, msd_clib.specificHeat_mL, [c_void_p])
_sig(c_double, msd_clib.specificHeat_mR, [c_void_p])
_sig(c_double, msd_clib.specificHeat_LR, [c_void_p])
_sig(c_double, msd_clib.magneticSusceptibility, [c_void_p])
_sig(c_double, msd_clib.magneticSusceptibility_L, [c_void_p])
_sig(c_double, msd_clib.magneticSusceptibility_R, [c_void_p])
_sig(c_double, msd_clib.magneticSusceptibility_m, [c_void_p])

_sig(Vector, msd_clib.meanM, [c_void_p])
_sig(Vector, msd_clib.meanML, [c_void_p])
_sig(Vector, msd_clib.meanMR, [c_void_p])
_sig(Vector, msd_clib.meanMm, [c_void_p])
_sig(Vector, msd_clib.meanMS, [c_void_p])
_sig(Vector, msd_clib.meanMSL, [c_void_p])
_sig(Vector, msd_clib.meanMSR, [c_void_p])
_sig(Vector, msd_clib.meanMSm, [c_void_p])
_sig(Vector, msd_clib.meanMF, [c_void_p])
_sig(Vector, msd_clib.meanMFL, [c_void_p])
_sig(Vector, msd_clib.meanMFR, [c_void_p])
_sig(Vector, msd_clib.meanMFm, [c_void_p])
_sig(c_double, msd_clib.meanU, [c_void_p])
_sig(c_double, msd_clib.meanUL, [c_void_p])
_sig(c_double, msd_clib.meanUR, [c_void_p])
_sig(c_double, msd_clib.meanUm, [c_void_p])
_sig(c_double, msd_clib.meanUmL, [c_void_p])
_sig(c_double, msd_clib.meanUmR, [c_void_p])
_sig(c_double, msd_clib.meanULR, [c_void_p])

_sig(c_void_p, msd_clib.createMolProto_e, [])
_sig(c_void_p, msd_clib.createMolProto_n, [c_size_t])
_sig(c_void_p, msd_clib.createMolProto_p, [c_size_t, POINTER(Molecule.NodeParameters)])
_sig(c_void_p, msd_clib.destroyMolProto, [c_void_p])

_sig(None, msd_clib.serialize, [c_void_p, c_char_p])
_sig(None, msd_clib.deserialize, [c_void_p, c_char_p])
_sig(c_size_t, msd_clib.serializationSize, [c_void_p])

_sig(c_uint, msd_clib.createNode_d, [c_void_p])
_sig(c_uint, msd_clib.createNode_p, [c_void_p, POINTER(Molecule.NodeParameters)])
_sig(c_uint, msd_clib.nodeCount, [c_void_p])
_sig(c_uint, msd_clib.connectNodes_d, [c_void_p] + 2 * [c_uint])
_sig(c_uint, msd_clib.connectNodes_p, [c_void_p] + 2 * [c_uint] + [POINTER(Molecule.EdgeParameters)])
_sig(c_uint, msd_clib.edgeIndex, [c_void_p] + 2 * [c_uint])

_sig(Molecule.EdgeParameters, msd_clib.getEdgeParameters, [c_void_p, c_uint])
_sig(None, msd_clib.setEdgeParameters, [c_void_p, c_uint, POINTER(Molecule.EdgeParameters)])
_sig(Molecule.NodeParameters, msd_clib.getNodeParameters, [c_void_p, c_uint])
_sig(None, msd_clib.setNodeParameters, [c_void_p, c_uint, POINTER(Molecule.NodeParameters)])
_sig(None, msd_clib.setAllParameters, [c_void_p, POINTER(Molecule.NodeParameters), POINTER(Molecule.EdgeParameters)])

_sig(None, msd_clib.setLeftLead, [c_void_p, c_uint])
_sig(None, msd_clib.setRightLead, [c_void_p, c_uint])
_sig(None, msd_clib.setLeads, [c_void_p] + 2 * [c_uint])
_sig(c_uint, msd_clib.getLeftLead, [c_void_p])
_sig(c_uint, msd_clib.getRightLead, [c_void_p])
_sig(None, msd_clib.getLeads, [c_void_p] + 2 * [POINTER(c_uint)])

_sig(c_void_p, msd_clib.createBeginMSDIter, [c_void_p])
_sig(c_void_p, msd_clib.createEndMSDIter, [c_void_p])
_sig(c_void_p, msd_clib.copyMSDIter, [c_void_p])
_sig(None, msd_clib.destroyMSDIter, [c_void_p])
_sig(c_uint, msd_clib.getX, [c_void_p])
_sig(c_uint, msd_clib.getY, [c_void_p])
_sig(c_uint, msd_clib.getZ, [c_void_p])
_sig(c_uint, msd_clib.msdIndex, [c_void_p])
_sig(Vector, msd_clib.getSpin_a, [c_void_p])
_sig(Vector, msd_clib.getFlux_a, [c_void_p])
_sig(Vector, msd_clib.getLocalM_a, [c_void_p])
_sig(c_bool, msd_clib.eq_a, 2 * [c_void_p])
_sig(c_bool, msd_clib.ne_a, 2 * [c_void_p])
_sig(c_bool, msd_clib.lt_a, 2 * [c_void_p])
_sig(c_bool, msd_clib.gt_a, 2 * [c_void_p])
_sig(c_bool, msd_clib.le_a, 2 * [c_void_p])
_sig(c_bool, msd_clib.ge_a, 2 * [c_void_p])
_sig(None, msd_clib.next_a, [c_void_p])
_sig(None, msd_clib.prev_a, [c_void_p])
_sig(None, msd_clib.add_a, [c_void_p, c_int])
_sig(None, msd_clib.sub_a, [c_void_p, c_int])

_sig(c_void_p, msd_clib.createNodes, [c_void_p])
_sig(None, msd_clib.destroyNodes, [c_void_p])
_sig(c_void_p, msd_clib.createEdges, [c_void_p])
_sig(c_void_p, msd_clib.createAdjacencyList, [c_void_p, c_uint])
_sig(None, msd_clib.destroyEdges, [c_void_p])

_sig(c_uint, msd_clib.size_n, [c_void_p])
_sig(c_uint, msd_clib.size_e, [c_void_p])

_sig(c_void_p, msd_clib.createBeginNodeIter, [c_void_p])
_sig(c_void_p, msd_clib.createEndNodeIter, [c_void_p])
_sig(c_void_p, msd_clib.copyNodeIter, [c_void_p])
_sig(None, msd_clib.destroyNodeIter, [c_void_p])
_sig(c_uint, msd_clib.nodeIndex_i, [c_void_p])
_sig(Molecule.NodeParameters, msd_clib.getNodeParameters_i, [c_void_p])
_sig(c_void_p, msd_clib.getNeighbors, [c_void_p])
_sig(c_bool, msd_clib.eq_n, 2 * [c_void_p])
_sig(c_bool, msd_clib.ne_n, 2 * [c_void_p])
_sig(c_bool, msd_clib.lt_n, 2 * [c_void_p])
_sig(c_bool, msd_clib.gt_n, 2 * [c_void_p])
_sig(c_bool, msd_clib.le_n, 2 * [c_void_p])
_sig(c_bool, msd_clib.ge_n, 2 * [c_void_p])
_sig(None, msd_clib.next_n, [c_void_p])
_sig(None, msd_clib.prev_n, [c_void_p])
_sig(None, msd_clib.add_n, [c_void_p, c_int])
_sig(None, msd_clib.sub_n, [c_void_p, c_int])

_sig(c_void_p, msd_clib.createBeginEdgeIter, [c_void_p])
_sig(c_void_p, msd_clib.createEndEdgeIter, [c_void_p])
_sig(c_void_p, msd_clib.copyEdgeIter, [c_void_p])
_sig(None, msd_clib.destroyEdgeIter, [c_void_p])
_sig(c_uint, msd_clib.edgeIndex_i, [c_void_p])
_sig(Molecule.EdgeParameters, msd_clib.getEdgeParameters_i, [c_void_p])
_sig(c_uint, msd_clib.src_e, [c_void_p])
_sig(c_uint, msd_clib.dest_e, [c_void_p])
_sig(c_double, msd_clib.getDirection, [c_void_p])
_sig(c_bool, msd_clib.eq_e, 2 * [c_void_p])
_sig(c_bool, msd_clib.ne_e, 2 * [c_void_p])
_sig(c_bool, msd_clib.lt_e, 2 * [c_void_p])
_sig(c_bool, msd_clib.gt_e, 2 * [c_void_p])
_sig(c_bool, msd_clib.le_e, 2 * [c_void_p])
_sig(c_bool, msd_clib.ge_e, 2 * [c_void_p])
_sig(None, msd_clib.next_e, [c_void_p])
_sig(None, msd_clib.prev_e, [c_void_p])
_sig(None, msd_clib.add_e, [c_void_p, c_int])
_sig(None, msd_clib.sub_e, [c_void_p, c_int])

_sig(Vector, msd_clib.createVector_3, 3 * [c_double])
_sig(Vector, msd_clib.createVector_2, 2 * [c_double])
_sig(Vector, msd_clib.createVector_0, [])
_sig(Vector, msd_clib.cylindricalForm, 3 * [c_double])
_sig(Vector, msd_clib.polarForm, 2 * [c_double])
_sig(Vector, msd_clib.sphericalForm, 3 * [c_double])

_sig(c_double, msd_clib.normSq, [POINTER(Vector)])
_sig(c_double, msd_clib.norm, [POINTER(Vector)])
_sig(c_double, msd_clib.theta, [POINTER(Vector)])
_sig(c_double, msd_clib.phi, [POINTER(Vector)])

_sig(c_bool, msd_clib.eq_v, 2 * [POINTER(Vector)])
_sig(c_bool, msd_clib.ne_v, 2 * [POINTER(Vector)])

_sig(Vector, msd_clib.add_v, 2 * [POINTER(Vector)])
_sig(Vector, msd_clib.neg_v, [POINTER(Vector)])
_sig(Vector, msd_clib.sub_v, 2 * [POINTER(Vector)])
_sig(Vector, msd_clib.mul_v, [POINTER(Vector), c_double])

_sig(c_double, msd_clib.distanceSq, 2 * [POINTER(Vector)])
_sig(c_double, msd_clib.distance, 2 * [POINTER(Vector)])
_sig(c_double, msd_clib.dotProduct, 2 * [POINTER(Vector)])
_sig(c_double, msd_clib.angleBetween, 2 * [POINTER(Vector)])
_sig(Vector, msd_clib.crossProduct, 2 * [POINTER(Vector)])

_sig(None, msd_clib.iadd_v, 2 * [POINTER(Vector)])
_sig(None, msd_clib.isub_v, 2 * [POINTER(Vector)])
_sig(None, msd_clib.imul_v, [POINTER(Vector), c_double])

_sig(None, msd_clib.negate, [POINTER(Vector)])
_sig(None, msd_clib.rotate_2d, [POINTER(Vector), c_double])
_sig(None, msd_clib.rotate_3d, [POINTER(Vector), c_double, c_double])
_sig(None, msd_clib.normalize, [POINTER(Vector)])




# Note: Tests moved to ~/src/tests
