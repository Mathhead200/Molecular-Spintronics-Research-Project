from ..config import Config
from ..simulation import Simulation, Snapshot
import numpy as np

test_sizes = [1, 2, 3, 10, 100, 1000, 10000, 20000]  # TODO: 1000000. Takes too long to compile! Why?
LEFT = "LEFT"
RIGHT = "RIGHT"
epsilon = 1e-14

for n in test_sizes:
	config = Config()
	config.nodes = list(range(n))
	config.edges = [(i - 1, i) for i in range(1, n)]
	half = n // 2
	config.regions = {
		LEFT: list(range(0, half)),
		RIGHT: list(range(half, n))
	}
	config.globalParameters = {
		"kT": 0.1,
		"S": 1.0,
		"J": 1.0,
		"A": [0.1, 0, 0],
		"b": 0.1
	}
	config.regionEdgeParameters = {
		(LEFT, RIGHT): { "J": -1.0 }
	}
	config.localNodeParameters = {
		0:     { "kT": 100.0 },
		n - 1: { "kT": 100.0 }
	}

	with config.compile(progress_bars=True) as rt:
		sim = Simulation(rt)
		assert len(sim.nodes) == n
		assert len(sim.edges) == n - 1
		assert len(sim.regions) == 2
		if n <= 1:    assert len(sim.eregions) == 0
		elif n == 2:  assert len(sim.eregions) == 1  # LEFT-RIGHT
		elif n == 3:  assert len(sim.eregions) == 2  # LEFT-RIGHT + either LEFT-LEFT or RIGHT-RIGHT, but not both
		else:         assert len(sim.eregions) == 3  # LEFT-LEFT, LEFT-RIGHT, and RIGHT_RIGHT
		for i, j in zip(sim.nodes, range(0, n)):  assert i == j
		for e, f in zip(sim.edges, ((i - 1, i) for i in range(1, n))):  assert e[0] == f[0] and e[1] == f[1]
		try:
			L = sim.regions[LEFT]
			R = sim.regions[RIGHT]
			len(L) == half
			len(R) == n - half
			for i, j in zip(L, range(0, half)):  assert i == j
			for i, j in zip(R, range(half, n)):  assert i == j
		except KeyError:
			assert False
		assert abs(sim.S - 1.0) < epsilon
		for A, A_ in zip(sim.A.value, [0.1, 0.0, 0.0]):  assert abs(A - A_) < epsilon
		assert abs(sim.b - 0.1) < epsilon

		def handle_snapshot(ss):
			assert type(ss) is Snapshot
			assert ss.t % n == 0
			assert isinstance(ss.s.value, np.ndarray)
			assert isinstance(ss.m.value, np.ndarray)
			assert isinstance(ss.u.value, float)
			
			assert len(ss.nodes) == len(sim.nodes)
			assert len(ss.edges) == len(sim.edges)
			assert len(ss.regions) == len(sim.regions)
			assert len(ss.eregions) == len(sim.eregions)
			for i, j in zip(ss.nodes, sim.nodes):  assert i == j
			for e, f in zip(ss.edges, sim.edges):  assert e[0] == f[0] and e[1] == f[1]
			for r, s in zip(ss.regions, sim.regions):
				for i, j in zip(ss.regions[r], sim.regions[s]):  assert i == j
			for r, s in zip(ss.eregions, sim.eregions):
				for e, f in zip(ss.eregions[r], sim.eregions[s]):  assert e[0] == f[0] and e[1] == f[1]
			
			for x in range(3):
				assert abs(ss.s.value[x] - sim.s.value[x]) < epsilon
				for i in ss.nodes:
					assert abs(ss.s[i].value[x] - sim.s[i].value[x]) < epsilon
			for x in range(3):
				assert abs(ss.m.value[x] - sim.m.value[x]) < epsilon
				for i in ss.nodes:
					assert abs(ss.m[i].value[x] - sim.m[i].value[x]) < epsilon
			for x in range(3):
				assert abs(sim.s.value[x] - sim.m.value[x]) < epsilon
				for i in sim.nodes:
					assert abs(sim.s[i].value[x] - sim.m[i].value[x]) < epsilon  # b.c. no flux
			for x in range(3):
				assert abs(ss.s.value[x] - ss.m.value[x]) < epsilon
				for i in ss.nodes:
					assert abs(ss.s[i].value[x] - ss.m[i].value[x]) < epsilon    # b.c. no flux
			sim.record(ss)

		sim.randomize(clear_history=False)
		sim.metropolis(100 * n, freq=n, callback=handle_snapshot, progress_bar=True)
		assert len(sim.history) == 101
