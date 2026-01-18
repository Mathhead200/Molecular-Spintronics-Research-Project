import mcheisenberg as mch
from mcheisenberg import __NODES__, __EDGES__
import numpy as np

if __name__ == "__main__":
	n = 4
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = {
		"S": 1.0,
		"kT": 0.0001,
		"J": 1.0,
	}

	with model.compile(dir=".", asm="mch-test.asm") as rt:
		sim = mch.Simulation(rt)
		s = sim.s
		u = sim.u

		s[0] = (0.0,  1.0,  0.0)
		s[1] = (0.0, -1.0,  0.0)
		s[2] = (0.0,  1.0,  0.0)
		s[3] = (1.0,  0.0,  0.0)
		print("u:", u)
		for i in range(10):
			sim.metropolis(10000000)
			print("s:", s.values())
			print("u:", u[__EDGES__].items())
			print("u:", u)
