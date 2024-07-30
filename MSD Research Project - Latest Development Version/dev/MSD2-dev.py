from typing import Hashable

class vec:
	i = lambda: (1, 0, 0)
	j = lambda: (0, 1, 0)
	k = lambda: (0, 0, 1)
	w = lambda: (None, None, None)  # TODO: (stub) return random unit vector

# 1. Optimize meta-indicies so edges are as close as possible (RAM spatial locality).
# 2. Cluster like parameters so that if those parameters are updated, the
#	system only needs to recalculate those parameters times the cluster's macro-properties.
#   Do we want to link this optimization to regions to avoid monitoring too many clusters?
# 3. Macro-properties like M only need to be calculated at the end of a sequence of metropolis updates
#	and not intermitenly. (Assuming delta-t >> n). Lazy evaluation of macro-properties??

class _Region:
	spins: dict[Hashable, vec] = {}

	S: dict[Hashable, float] = {}
	F: dict[Hashable, float] = {}
	B: dict[Hashable, vec] = {}
	Je0: dict[Hashable, float] = {}
	A: dict[Hashable, vec] = {}

	J: dict[tuple[Hashable, Hashable], float] = {}
	Je1: dict[tuple[Hashable, Hashable], float] = {}
	Jee: dict[tuple[Hashable, Hashable], float] = {}
	b: dict[tuple[Hashable, Hashable], float] = {}
	D: dict[tuple[Hashable, Hashable], vec] ={}

class SpinDevice:
	_regions: dict[str, _Region] = {}
	_current: str = ""
	_g = property(fget=lambda self: self._regions[self._current])

	def __init__(self) -> None:
		self._regions[self._current] = _Region()  # create default region

	def region(self, label: str = "") -> bool:
		""" Select a new or existing sub-graph. """
		added = label not in self._regions
		if added:
			self._regions[label] = _Region()  # create new region
		self._current = label
		return added
	
	def spin(self, index: Hashable, value: vec = None) -> int:
		""" Create a new node in the selected region with the given index. """
		s = self._g.spins
		added = int(index not in s)
		s[index] = value
		return added

	def spins(self, spins: list[tuple[Hashable, vec]]) -> int:
		return sum([self.spin(*s) for s in spins])

	def J(self, i: Hashable, j: Hashable, J: float) -> None:
		""" Set J edge connecting i and j """
		self._g.J[(i, j)] = J
	
	def Js(self, Js: list[tuple[Hashable, Hashable, float]]) -> None:
		""" Set J edges connecting the given i's and j's respectively """
		for i, j, J in Js:
			self.J(i, j, J)

model = SpinDevice()
model.region("FML")  # model.selectRegion(label: str)
w, h, d = 5, 4, 10
idx = [(x, y, z) for x in range(w) for y in range(h) for z in range(d)]
edge = lambda i, j: abs(i[0] - j[0]) + abs(i[1] - j[1]) + abs(i[3] - j[3]) == 1
model.spins(idx)
model.Js([(i, j, 1.0) for i in idx for j in idx if edge(i, j)])

model.region("mol")
w, h, d = 1, 4, 4
idx = [ (x, y, z)
	for x in range(w) for y in range(h) for z in range(d)
	if y == 0 or y == h-1 or z == 0 or z == d-1 ]
edge = lambda i, j: abs(i[0] - j[0]) == 1 and i[1] == j[1] and i[3] == j[3]
model.spins(idx)
model.Js([(i, j, 1.0) for i in idx for j in idx if edge(i, j)])

model.region("FMR")
w, h, d = 5, 10, 4
idx = [(x, y, z) for x in range(w) for y in range(h) for z in range(d)]
edge = lambda i, j: abs(i[0] - j[0]) + abs(i[1] - j[1]) + abs(i[3] - j[3]) == 1
model.spins(idx)
model.Js([(i, j, 1.0) for i in idx for j in idx if edge(i, j)])


