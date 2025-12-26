
def compile(
    indicies, edges, mutableIndicies = None,
    initialSpins = None, initialFluxes = None,
    # TODO ...
):
    """
    indicies: list of some type <T> (e.g., int) which defines the nodes
    edges: dict mapping <T> to tuples of (<T>, <T>) which defines the connections
    mutableIndicies: (optional) subset of indicies which are mutable
    initialSpins: (optional) dict mapping <T> to vector3
    initialFluxes: (optional) dict mapping <T> to vector3
    """
    if mutableIndicies is None:
        mutableIndicies = indicies
    
    pass  # TODO: stub

I = range(11)  # [0, 1, 2, ..., 9, 10]
J = range(1, 10)  # [1, 2, ..., 9]
E = {i: (i, i+1) for i in range(10)}  # [0: (0, 1), 1: (1, 2), ..., 9: (9, 10)]
