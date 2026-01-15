import mcheisenberg as mch

def print_spins(rt: mch.Runtime):
	for s in rt.spins:
		print(f"{s} ", end="")
	print()

if __name__ == "__main__":
	n = 2
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = { "S": 1.0, "kT": 1.0, "J": 1.0 }

	with model.compile(dir=".", asm="mch-test.asm") as rt:
		sim = mch.Simulation(rt)

		print_spins(rt)
		rt.driver.metropolis(7)
		print_spins(rt)
