import mcheisenberg as mch

def print_spins(rt: mch.Runtime) -> None:
	for n in rt.nodes:
		print(f"spin[{n}]: {rt.spin[n]}")

def print_Js(rt: mch.Runtime) -> None:
	for e in rt.edges:
		print(f"J{e}: {rt.edge[e].J}")

def print_kTs(rt: mch.Runtime) -> None:
	for n in rt.nodes:
		print(f"kT[{n}]: {rt.node[n].kT}")

def print_bs(rt: mch.Runtime) -> None:
	for e in rt.edges:
		print(f"b{e}: {rt.edge[e].b}")

if __name__ == "__main__":
	n = 10
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.localEdgeParameters = {
		(0, 1): { "J": -1.0 },
		(8, 9): { "J": -1.0 }
	}
	model.regions = {
		"L": [0, 1, 2, 3, 4],
		"R": [5, 6, 7, 8, 9]
	}
	model.regionNodeParameters = {
		"L": {
			"kT": 0.2,
			"A": (0.1, 0, 0)
		},
	}
	model.regionEdgeParameters = {
		"R": {
			"B": -0.1
		}
	}
	model.globalParameters = {
		"S": 1.0,
		"J": 1.0,
		"kT": 0.1,
		"B": (2.0, 0.0, 0.0),
	}

	with model.compile(dir=".", asm="mch-test.asm") as rt:
		print_spins(rt)
		print_Js(rt)
		print_kTs(rt)
		print_bs(rt)
		rt.driver.metropolis(20000000)
		rt.spin[9] = (0.0, 0.0, 0.0)
		rt.edge[0, 1].J = 2.0
		rt.region["L"].kT = 1.0
		rt.region["R"].b = -0.1
		print_spins(rt)
		rt.driver.metropolis(20000000)
		print_spins(rt)
		print_Js(rt)
		print_kTs(rt)
		print_bs(rt)
		print(rt.region["R"].b)
