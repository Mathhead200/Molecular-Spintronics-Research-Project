from typing import Iterable, Hashable, Callable, Mapping, TypedDict
from functools import reduce

class NodeParameters(TypedDict):
    s: tuple[float] = None  # spin
    f: tuple[float] = None  # flux
    S: float = None         # spin norm
    F: float = None         # max flux norm
    B: tuple[float] = None  # external magnetic field
    Je0: float = None       # s_i \cdot f_i
    A: tuple[float] = None  # anisotropy

class EdgeParameters(TypedDict):
    J: float = None         # Heisenburg exchange coupling
    Je1: float = None       # s_i \cdot f_j + s_j \cdot f_i
    Jee: float = None       # s_i \cdot f_j
    b: float = None         # biquadratic coupling
    D: tuple[float] = None  # Dzyaloshinsky–Moriya interaction (i.e., skyrmions)

# Returns the union of the sets of parameters given.
_union = lambda parameters: reduce( lambda total, p: total.union(p.keys()),
    [p for p in parameters if p is not None], set() )


class SpinDevice:
    
    # All of the indices in this SpinDevice
    indices: Iterable[Hashable] = []

    # Returns a collection of region labels associated with the given index.
    # Should return an empty collection, [], for indices not in a regions.
    # All indices are automatically assumed to be in the global region, "";
    # this region does not need to be included in the returned collection.
    regions: Callable[[Hashable], Iterable[Hashable]] = lambda i: []

    # Returns a set of NodeParameters for this specific location, i.
    # Return None if this location has no parameters and is inactive.
    # Return an empty NodeParameters, {}, if this node has no specialized
    # parameters, and all parameters should be inherited from ```node_regions```.
    nodes: Callable[[Hashable], NodeParameters] = lambda i: NodeParameters()

    # Return a set of EdgeParameters for this specific edge/bond, (i, j).
    # Return None if there is no edge or bond, (i, j).
    edges: Callable[[Hashable, Hashable], EdgeParameters] = lambda i, j: None

    # Return a set of NodeParameters to be applied for this entire region, R;
    # or None if not applicable.
    node_regions: Callable[[Hashable], NodeParameters] = lambda R: None
    
    # Return a set of EdgeParameters to be applied for interactions anywhere
    # bewteen the given regions, Q and R; or return None if not applicable.
    edge_regions: Callable[[Hashable, Hashable], EdgeParameters] = lambda Q, R: None

    # Global node parameters, if any.
    node_globals: NodeParameters = {}

    # Global edge parameters, if any.
    edge_globals: EdgeParameters = {}

    def _reindex(self) -> dict[Hashable, int]:
        """ Return meta-index table which maps ints in range(0, n) to user provided indices """
        # filter inactive nodes
        NZ = lambda node: node is not None and ("s" in node or "f" in node)
        idx = [i for i in self.indices if NZ(self.nodes(i))]
        
        # TODO: optimize graph structure to minimize destance between neighboring indices
        return { k: v for k, v in zip(range(len(idx)), idx) }

    def compile(self, filename: str):
        meta = self._reindex()
        n = len(meta)

        def VEC(vec: tuple[float]) -> str:
            return "{" + f"{vec[0]}, {vec[1]}, {vec[2]}" + "}"

        with open(filename, "w") as file:
            # header stuff
            file.write("typedef struct { double x, y, z } Vector;\n\n")

            # struct Node
            nodeKeys = _union([self.nodes(i) for i in self.indices])
            file.write("struct Node {\n")
            if   "s" in nodeKeys:  file.write("\tVector s;\n")
            if   "f" in nodeKeys:  file.write("\tVector f;\n")
            if   "B" in nodeKeys:  file.write("\tVector B;\n")
            if   "A" in nodeKeys:  file.write("\tVector A;\n")
            if   "S" in nodeKeys:  file.write("\tfloat S;\n")
            if   "F" in nodeKeys:  file.write("\tfloat F;\n")
            if "Je0" in nodeKeys:  file.write("\tfloat Je0;\n")
            file.write("} nodes[] = {\n")
            for i in range(n):
                file.write("\t{ ")
                if   "s" in nodeKeys:  file.write(f".s = {VEC(self.nodes(meta[i])["s"])}, ")
                if   "f" in nodeKeys:  file.write(f".f = {VEC(self.nodes(meta[i])["f"])}, ")
                if   "B" in nodeKeys:  file.write(f".B = {VEC(self.nodes(meta[i])["B"])}, ")
                if   "A" in nodeKeys:  file.write(f".A = {VEC(self.nodes(meta[i])["A"])}, ")
                if   "S" in nodeKeys:  file.write(f".S = {self.nodes(meta[i])["S"]}, ")
                if   "F" in nodeKeys:  file.write(f".F = {self.nodes(meta[i])["F"]}, ")
                if "Je0" in nodeKeys:  file.write(f".Je0 = {self.nodes(meta[i])["Je0"]}, ")
                file.write("},\n")
            file.write("};\n\n")

            # struct Edge
            edgeKeys = _union([self.edges(i, j) for i in self.indices for j in self.indices])
            file.write("struct Edge {\n")
            if   "D" in edgeKeys:  file.write("\tVector D;\n")
            if   "J" in edgeKeys:  file.write("\tdouble J;\n")
            if "Je1" in edgeKeys:  file.write("\tdouble Je1;\n")
            if "Jee" in edgeKeys:  file.write("\tdouble Jee;\n")
            if   "b" in edgeKeys:  file.write("\tdouble b;\n")
            file.write("} edges[" + str(n) + "][" + str(n) + "];\n\n")

if __name__ == "__main__":
    model = SpinDevice()

    width, height, depth = 11, 10, 10
    molPosL, molPosR = 5, 5
    topL, bottomL, frontR, backR = 3, 6, 3, 6

    SL, SR, Sm = 1, 1, 2
    JL, JR, Jm, JmL, JmR, JLR = 1, 1, 1, 1, -1, 0.1
    B = (0.1, 0, 0)

    model.indices = [ (x, y, z)
        for x in range(width) for y in range(height) for z in range(depth)
        if (x < molPosL and topL <= y <= bottomL)
        or (x > molPosR and frontR <= z <= backR)
        or (molPosL <= x <= molPosR and (y == topL or y == bottomL or z == frontR or z == backR)) ]
    
    model.regions = lambda i: ["FML" if i[0] < molPosL else "FMR" if i[0] > molPosR else "mol"]

    model.nodes = lambda i: { "s": (0, 1, 0) }

    model.edges = lambda i, j: {} \
        if abs(i[0] - j[0]) + abs(i[1] - j[1]) + abs(i[2] - j[2]) == 1 \
        or (i[0] == molPosL - 1 and j[0] == molPosR + 1 and topL <= i[1] <= bottomL and frontR <= j[2] <= backR) \
        else None

    model.node_regions = lambda R: \
        { "S": SL } if R == "FML" else \
        { "S": SR } if R == "FMR" else \
        { "S": Sm } if R == "mol" else \
        None

    EQ = lambda p, q: p == q or p == (q[1], q[0])
    model.edge_regions = lambda Q, R: \
        { "J": JL } if Q == R == "FML" else \
        { "J": JR } if Q == R == "FMR" else \
        { "J": Jm } if Q == R == "mol" else \
        { "J": JmL } if EQ((Q, R), ("FML", "mol")) else \
        { "J": JmR } if EQ((Q, R), ("mol", "FMR")) else \
        { "J": JLR } if EQ((Q, R), ("FML", "FMR")) else \
        None

    model.node_globals = { "B": B }

    model.compile(filename="model.c")


