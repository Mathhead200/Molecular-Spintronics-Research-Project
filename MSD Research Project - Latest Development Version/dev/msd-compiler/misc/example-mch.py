from mcheisenberg import Config

graph = Config()

# define the nodes and edges
graph.nodes = [ 0, 1, 2, 3, 4, 5 ]
graph.edges = [ (0, 1), (1, 2), (2, 3), (3, 4), (4, 5) ]

# define the regions
graph.regions = {
	"regionA": [ 0, 1, 2 ],
	"regionB": [ 3, 4, 5 ]
}

# apply these parameters globally
graph.globalParameters = {
	"kT": 0.1,         # temperature
	"B": (1.0, 0, 0),  # applied magnetic field
	"J": 1.0,          # Heisenberg exchange coupling
	"S": 1.0           # spin magnitude
}

# apply these parameters to a specific region
graph.regionNodeParameters = {
	"regionA": { "A": (0, 0, 1.0) }  # anisotropy
}
graph.regionEdgeParameters = {
	("regionA", "regionB"): { "J": -1.0 }
}

# apply these parameters to a specific location
graph.localNodeParameters = {
	0: { "kT": 10.0 },  # node 0
	5: { "kT": 0.001 }  # node 5
}

# ---------------------------------------------------------

from mcheisenberg import Simulation

runtime = graph.compile()
simulation = Simulation(runtime)

simulation.metropolis(iterations=10_000_000)  # progress_bar="Reaching equalibrium"
simulation.metropolis(iterations=100_000_000, freq=2_000_000)  # progress_bar="Gathering data"

from mcheisenberg.io import csv
csv(simulation, dir="out", prefix="example-mch", progress_bar=True)

runtime.shutdown()  # clean up temp. files

# ---------------------------------------------------------
