import mcheisenberg as mc
import numpy as np

if __name__ == "__main__":
	n = 4
	
	model = mc.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = {
		"S": 1.0,
		"F": 1.0,
		"kT": 0.01,
		"J": 1.0,
		"Je0": 1.0,
		"B": (1.0, 0, 0)
	}

	with model.compile(dir=".", asm="mc-test.asm") as rt:
		sim = mc.Simulation(rt)
		s = sim.s
		f = sim.f
		u = sim.u

		for i in sim.nodes:
			s[i] = (0.0, (-1.0)**i, 0.0)
			f[i] = np.array([0.0, (-1.0)**(i+1), 0.0])
		u_array = u.array()
		print(f"u[t={sim.t}]:", u, { k: float(v) for k, v in zip(u.keys(), u_array) })
		print(f"s|f[t={sim.t}]:", s.values(), f.values())
		
		for i in range(5):
			sim.metropolis(1)
			u_brray = u.array()
			print(f"u[t={sim.t}]:", u, { k: (float(v), float(d)) for k, v, d in zip(u.keys(), u_brray, u_brray - u_array) })
			print(f"s|f[t={sim.t}]:", s.values(), f.values())
			u_array = u_brray
