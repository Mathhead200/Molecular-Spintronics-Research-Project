from __future__ import annotations
from .. import Config
from ..util import S_, F_, kT_, B_, A_, J_, Je0_, Je1_, Jee_, b_, D_, SCALAR_PARAMETERS, VECTOR_PARAMETERS
from ..driver import c_double_3
from ctypes import c_double
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..runtime import scal_out, vec_out

test_sizes = [0, 1, 2, 3, 10, 100, 1000, 10000, 20000]  # TODO: 1000000. Takes too long to compile! Why?

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
			try:  rt = config.compile()
			except ValueError:  pass  # expected
			except:             assert False
			continue  # skip the rest of the check since compilation is expected to fail in this case
		print("Compiling...")
		with config.compile(progress_bars=True) as rt:
			print("Testing...")
			assert config.NODE_COUNT == n
			assert config.MUTABLE_NODE_COUNT == n
			assert config.IMMUTABLE_NODE_COUNT == 0
			assert config.EDGE_COUNT == n - 1
			assert config.REGION_COUNT == 0
			assert config.EDGE_REGION_COUNT == 0
			epsilon = 1e-14
			for p in SCALAR_PARAMETERS:
				if p not in g:  continue
				v: scal_out = getattr(rt, p).value
				assert type(v) == c_double
				assert abs(v.value - g[p]) < epsilon
			for p in VECTOR_PARAMETERS:
				if p not in g:  continue
				v: vec_out = getattr(rt, p).value
				assert type(v) is c_double_3
				for i in  range(3):
					assert abs(v[i] - g[p][i]) < epsilon
			
			simCount = 10 * n
			print(" -- metropolis:", simCount)
			rt.randomize()
			rt.metropolis(simCount)
			buf = rt.allocate_buffer()
			data = rt.snapshot(buf)

		assert len(data.source.nodes) == n
		assert len(data.source.edges) == n - 1
		assert len(data.source.regions) == 0
		assert len(data.source.edge_regions) == 0
		epsilon = 1e-14
		for p in SCALAR_PARAMETERS:
			if p not in g:  continue
			v: scal_out = getattr(data.source, p)
			assert type(v) is c_double
			assert abs(v.value - g[p]) < epsilon
			v: scal_out = getattr(data, p).value
			assert type(v) is c_double
			assert abs(v.value - g[p]) < epsilon
		for p in VECTOR_PARAMETERS:
			if p not in g:  continue
			v: vec_out = getattr(data.source, p)
			assert type(v) is c_double_3
			for i in range(3):
				assert abs(v[i] - g[p][i]) < epsilon
			v: vec_out = getattr(data, p).value
			assert type(v) is c_double_3
			for i in range(3):
				assert abs(v[i] - g[p][i]) < epsilon
		buf.free()
