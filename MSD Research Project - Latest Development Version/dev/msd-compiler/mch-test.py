import mcheisenberg as mch
import numpy as np

if __name__ == "__main__":
	n = 2
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = { "S": 1.0, "kT": 1.0, "J": 1.0, "B": (1.0, 0.0, 0.0) }

	with model.compile(dir=".", asm="mch-test.asm") as rt:
		sim = mch.Simulation(rt)

		sim.metropolis(7)
		print(sim.n[__NODES__])
