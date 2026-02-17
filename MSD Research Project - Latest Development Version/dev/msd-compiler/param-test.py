import mcheisenberg as mc
from mcheisenberg import VEC_I

N = 4

if __name__ == "__main__":
	config = mc.Config()
	config.nodes = list(range(N))
	config.edges = [(i-1, i) for i in range(1, N)]
	config.globalParameters = {
		"S": 1.0,
		"F": 1.0,
		"kT": 0.1,
		"Je0": 1.0,            # Je0 working
		"A": (1.0, 0.0, 0.0),  # A working
		"J": 1.0,              # J working
		"Je1": 1.0,
		"Jee": 1.0,            # Jee working
		# "b": 1.0,
		# D
		"B": (1.0, 0.0, 0.0)  # B working
	}
	with config.compile(dir=".", asm="param-test.asm") as rt:
		sim = mc.Simulation(rt)
		for i in sim.nodes:
			sim.s[i] = (-1)**i * VEC_I
			sim.f[i] = (-1)**(i+1) * VEC_I
		
		def print_u():
			print("u:", sim.u)
			for key in sim.u.keys():
				print(f"u[{key}]", sim.u[key])
		
		print_u()
		for i in range(8):
			sim.metropolis(1)
			print_u() 
