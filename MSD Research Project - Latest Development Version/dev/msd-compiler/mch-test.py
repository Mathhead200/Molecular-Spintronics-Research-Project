import mcheisenberg as mch

def print_nodes(model: mch.Config, rt: mch.Runtime) -> None:
	for n in model.nodes:
		print(f"{n}: {rt.spin[n]}")

if __name__ == "__main__":
	n = 10
	
	model = mch.Config()
	model.nodes = [*range(n)]
	model.edges = [(i, i+1) for i in range(n - 1)]
	model.globalParameters = {
		"S": 1.0,
		"J": 1.0,
		"kT": 0.1,
		"B": (2.0, 0.0, 0.0),
	}

	with model.compile(dir=".") as rt:
		print_nodes(model, rt)
		rt.driver.metropolis(20000000)
		rt.spin[9] = (0.0, 0.0, 0.0)
		print_nodes(model, rt)
