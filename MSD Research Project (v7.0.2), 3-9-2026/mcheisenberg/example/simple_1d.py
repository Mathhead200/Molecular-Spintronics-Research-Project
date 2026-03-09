from mcheisenberg import Config

def simple_1d():
	n = 10

	msd = Config()
	msd.nodes = list(range(n))
	msd.edges = [(i, i+1) for i in range(n-1)]
	msd.globalParameters = {
		"kT": 0.01,
		"S": 1.0,
		"J": 1.0,
		"B": (1.0, 0, 0)
	}
	return msd

if __name__ == "__main__":
	msd = simple_1d()
	msd.compile(asm="simple_1d.asm", dir=".")  # arguments optional
