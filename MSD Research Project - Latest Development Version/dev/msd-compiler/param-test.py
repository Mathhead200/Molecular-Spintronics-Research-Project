import mcheisenberg as mc
from mcheisenberg import VEC_I, VEC_J, VEC_K, VEC_ZERO

N = 4

if __name__ == "__main__":
	config = mc.Config()
	config.nodes = list(range(N))
	config.edges = [(i-1, i) for i in range(1, N)]
	config.globalParameters = {
		"S": 1.0,
		"F": 1.0,
		"kT": 10.0,
		"Je0": 1.0,          # Je0 working
		"A": (1.0, 0.0, 0.0),  # A working
		"J": 1.0,              # J working
		"Je1": 1.0,            # Je1 working
		"Jee": 1.0,            # Jee working
		"b": 1.0,            # b working??
		# D
		"B": (1.0, 0.0, 0.0)  # B working
	}
	config.debug = { "deltaU_ret_dump" }
	with config.compile(dir=".", asm="param-test.asm") as rt:
		sim = mc.Simulation(rt)
		for i in sim.nodes:
			sim.s[i] = (VEC_I, VEC_J, VEC_K)[i % 3]
			sim.f[i] = VEC_ZERO
		
		def print_state():
			print("_" * 140)
			u, m, s, f = sim.u, sim.m, sim.s, sim.f
			print(f"{'node/edge':>10} {'u':>10} {'m':>40} {'s':>40} {'f':>40}")
			print("_" * 140)
			print(f"{'':>10} {u:>10.10} {m:>40} {s:>40} {f:>40}")
			for i in sim.nodes:
				print(f"{i:>10} {u[i]:>10.10} {m[i]:>40} {s[i]:>40} {f[i]:>40}")
			for e in sim.edges:
				print(f"{str(e):>10} {u[e]:>10.10}")
			print("_" * 140)
		
		print_state()
		for i in range(8):
			sim.metropolis(1)
			print_state() 
