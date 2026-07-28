import mcheisenberg as mc
from mcheisenberg.model import MSD

msd = MSD(11, 10, 10, 5, 5, 3, 6, 3, 6)
msd.globalParameters["B"] = (0.0, 0.0, 0.0)
for r in msd.regions:
	msd.regionNodeParameters[r] = {
		"A": (0.0, 0.0, 0.0)
	}

msd.regionEdgeParameters[("L", "R")] = { "J": 0.0}
for e in [("L", "L"), ("L", "m"), ("m", "m"), ("m", "R"), ("R", "R"), ("L", "R")]:
	msd.regionEdgeParameters[e]["b"] = 0.0
	msd.regionEdgeParameters[e]["D"] = (0.0, 0.0, 0.0)
