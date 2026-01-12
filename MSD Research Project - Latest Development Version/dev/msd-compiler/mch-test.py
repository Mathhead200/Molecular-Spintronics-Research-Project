import mcheisenberg as mch

def print_nodes(model: mch.Config, rt: mch.Runtime) -> None:
	SPAN = model.SIZEOF_NODE // 8
	for j in range(model.NODE_COUNT):
		print(j, ": ", sep="", end="")
		for i in range(SPAN):
			print(rt.driver.nodes[j * SPAN + i], ", ", sep="", end="")
		print()

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
		rt.driver.metropolis(100000000)
		print_nodes(model, rt)
