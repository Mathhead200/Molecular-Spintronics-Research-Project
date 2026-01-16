import mcheisenberg as mch

def print_spins(sim: mch.Simulation):
	for s in sim.s.values():
		print(f"{s} ", end="")
	print()

def print_fluxess(sim: mch.Simulation):
	for f in sim.f.values():
		print(f"{f} ", end="")
	print()

def print_magnetizations(sim: mch.Simulation):
	for m in sim.m.values():
		print(f"{m} ", end="")
	print()

if __name__ == "__main__":
	n = 2
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = { "S": 1.0, "kT": 1.0, "J": 1.0, "B": (1.0, 0.0, 0.0) }

	with model.compile(dir=".", asm="mch-test.asm") as rt:
		sim = mch.Simulation(rt)

		sim.rt.kT = 0.1
		sim.metropolis(7)
		print_magnetizations(sim)
		print(sim.rt.kT)
