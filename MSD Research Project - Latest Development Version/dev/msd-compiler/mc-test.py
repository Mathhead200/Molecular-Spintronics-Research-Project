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
		"F": 1.0,
		"kT": 0.01,
		"Je0": 1.0
	}

	with model.compile(dir=".", asm="mc-test.asm") as rt:
		sim = mch.Simulation(rt)
		s = sim.s
		f = sim.f
		u = sim.u

		for i in sim.nodes:
			s[i] = (0.0, (-1.0)**i, 0.0)
			f[i] = np.array([0.0, (-1.0)**(i+1), 0.0])
		print(f"u[t={sim.t}]:", u, u.items())
		print(f"s|f[t={sim.t}]:", s.values(), f.values())
		
		for i in range(5):
			sim.metropolis(1)
			print(f"u[t={sim.t}]:", u, u.items())
			print(f"s|f[t={sim.t}]:", s.values(), f.values())
