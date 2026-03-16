from .. import Config
from ..util import S_, F_, kT_, B_, A_, J_, Je0_, Je1_, Jee_, b_, D_, PARAMETERS, NODE_PARAMETERS, EDGE_PARAMETERS

test_sizes = [0, 1, 2, 3, 10, 100, 1000, 10000, 1000000]

test_globalParameters = {
	"No spin-flux": {
		S_:  1.0,
		kT_: 0.2,
		B_:  (0.1, 0, 0),
		A_:  (0.1, 0, 0),
		J_:  1.0,
		b_:  0.1,
		D_:  (0.01, 0.0, 0.0)
	},
	"Spin-flux": {
		S_:   1.0,
		F_:   0.5,
		kT_:  0.2,
		B_:   (0.1, 0, 0),
		A_:   (0.1, 0, 0),
		J_:   1.0,
		Je0_: 0.1,
		Je1_: 0.1,
		Jee_: 0.1,
		b_:   0.1,
		D_:   (0.01, 0.0, 0.0)
	}
}

for g_lbl, g in test_globalParameters.items():
	print("g:", g_lbl)
	for n in test_sizes:
		print("n:", n)
		config = Config()
		config.nodes = list(range(n))
		config.edges = [(i - 1, i) for i in range(1, n)]
		config.globalParameters = g
		if n == 0:
			try:  config.compile()
			except ValueError:  pass  # expected
			except:             assert False
			continue  # skip the rest of the check since compilation is expected to fail in this case
		with config.compile() as rt:
			assert config.NODE_COUNT == n
			assert config.MUTABLE_NODE_COUNT == n
			assert config.IMMUTABLE_NODE_COUNT == 0
			assert config.EDGE_COUNT == n - 1
			assert config.REGION_COUNT == 0
			assert config.EDGE_REGION_COUNT == 0
			epsilon = 1e-14
			for p in PARAMETERS:
				if p in g:  assert abs(getattr(rt, p) - g[p]) < epsilon
			
			simCount = 10 * n
			print(" -- metropolis:", simCount)
			rt.randomize()
			rt.metropolis(simCount)
			buf = rt.allocate_buffer()
			data = rt.record(buf)

		assert len(data.source.nodes) == n
		assert len(data.source.edges) == n - 1
		assert len(data.source.regions) == 0
		assert len(data.source.eregions) == 0
		epsilon = 1e-14
		for p in PARAMETERS:
			if p in g:  assert abs(getattr(data, p) - g[p]) < epsilon
		buf.free()
