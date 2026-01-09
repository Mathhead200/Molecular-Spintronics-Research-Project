from mcheisenberg import Config

# Examples:
def example_3d():
	width = 11
	height = 10
	depth = 10
	molPosL = 5
	molPosR = 5
	topL = 3
	bottomL = 6
	frontR = 3
	backR = 6

	msd = Config()
	msd.edges = []
	
	fml = []
	for x in range(0, molPosL):
		for y in range(topL, bottomL + 1):
			for z in range(0, depth):
				fml.append((x, y, z))
				# internal FML_FML edges:
				if x + 1 < molPosL:
					msd.edges.append(((x, y, z), (x + 1, y, z)))
				if y + 1 <= bottomL:
					msd.edges.append(((x, y, z), (x, y + 1, z)))
				if z + 1 < depth:
					msd.edges.append(((x, y, z), (x, y, z + 1)))
	
	mol = []
	for x in range(molPosL, molPosR + 1):
		for y in range(topL, bottomL + 1):
			for z in range(frontR, backR + 1):
				if y == topL or y == bottomL or z == frontR or z == backR:
					mol.append((x, y, z))
					# internal mol_mol edges:
					if x + 1 <= molPosR:
						msd.edges.append(((x, y, z), (x + 1, y, z)))
					# FML_mol edges:
					if molPosL - 1 >= 0:
						msd.edges.append(((molPosL - 1, y, z), (molPosL, y, z)))
					# mol_FMR edges:
					if molPosR + 1 < width:
						msd.edges.append(((molPosR, y, z), (molPosR + 1, y, z)))
	
	fmr = []
	for x in range(molPosR + 1, width):
		for y in range(0, height):
			for z in range(frontR, backR + 1):
				fmr.append((x, y, z))
				# internal FMR_FMR edges
				if x + 1 < width:
					msd.edges.append(((x, y, z), (x + 1, y, z)))
				if y + 1 < height:
					msd.edges.append(((x, y, z), (x, y + 1, z)))
				if z + 1 <= backR:
					msd.edges.append(((x, y, z), (x, y, z + 1)))
	
	# LR direct coupling
	for y in range(topL, bottomL + 1):
		for z in range(frontR, backR + 1):
			if y == topL or y == bottomL or z == frontR or z == backR:
				# FML_FMR edges:
				if molPosL - 1 >= 0 and molPosR + 1 < width:
					msd.edges.append(((molPosL - 1, y, z), (molPosR + 1, y, z)))

	msd.regions = { "FML": fml, "mol": mol, "FMR": fmr }
	msd.nodes = fml + mol + fmr
	msd.nodeId = lambda node: f"{node[0]}_{node[1]}_{node[2]}"
	# msd.mutableNodes = msd.nodes  # now automatic

	msd.globalParameters = {
		"kT": 0.25,
		"S": 1,
		"F": 0,
		"J": 1
	}
	msd.regionNodeParameters = {
		"mol": { "S": 10 }
	}
	msd.regionEdgeParameters = {
		("mol", "FMR"): { "J": -1 }
	}
	msd.programParameters = {
		"simCount": 1000000,
		"freq": 50000
	}

	# testing localNodeParameters
	def is_dictesk(obj):
		return hasattr(obj, "keys") and callable(obj.keys) and hasattr(obj, "__getitem__")
	def deep_union_2(a, b):
		u = {**a, **b}
		for k in a.keys() & b.keys():
			v1, v2 = a[k], b[k]
			if is_dictesk(v1) and is_dictesk(v2):
				u[k] = deep_union_2(v1, v2)
		return u
	def deep_union(*dicts):
		from functools import reduce
		return reduce(deep_union_2, dicts, {})
	
	# testing immutableNodes()
	msd.mutableNodes = [*msd.nodes]
	for y in range(topL, bottomL + 1):
		for z in range(0, depth):
			msd.mutableNodes.remove((0, y, z))
	for y in range(0, height):
		for z in range(frontR, backR + 1):
			msd.mutableNodes.remove((width - 1, y, z, ))

	msd.localNodeParameters = deep_union({
		(0, topL, z): { "S": 2 } for z in range(depth)
	}, {
		(0, y, 0): { "F": 1 } for y in range(topL, bottomL + 1)
	})

	return msd

	# TODO: (though) Currently, msd.nodes mst be defined, and msd.mutableNodes
	#	is optional, then self.immutableNodes gets computed. Is this the best
	#	pattern for the user? SHould we allow alternate patterns like defining
	#	self.nodes and self.immutableNodes; or
	#	self.mutabelNodes and self.immutableNodes: forcing them to specify?
	#	(idk)

def example_1d():
	# 1D model with the classic 3 sections.
	#
	#   |  FML  |mol|  FMR  |
	# 0*--1---2---3---4---5---6*
	# ^        \     /        ^ {0,6} are leads and immutable
	#           \---/ Direct coupling (2,4) (e.g. JLR)

	msd = Config()
	msd.nodes = [0, 1, 2, 3, 4, 5, 6]
	msd.mutableNodes = [1, 2, 3, 4, 5]  # excluding 0, 6
	msd.edges = [(i, i+1) for i in range(len(msd.nodes) - 1)]
	msd.edges += [(2, 4)]
	msd.globalParameters = {
		"S": 1.0,
		# "F": 0.5,
		"kT": 0.1,
		"B": (0.1, 0, 0),
		"J": 1.0
	}
	msd.regions = {
		"FML": [1, 2],
		"mol": [3],
		"FMR": [4, 5]
	}
	msd.regionNodeParameters = {
		"FML": { "A": (0.0, 0.0, 0.2) },
		# "mol": { "Je0": 3.0 },
		"FMR": { "A": (0.0, 0.0, -0.2) }
	}
	msd.regionEdgeParameters = {
		("mol", "FMR"): { "J": -1.0 },
		("FML", "FMR"): { "J": 0.1 }
	}
	msd.localNodeParameters = {
		0: { "S": 10.0 },
		6: { "S": 10.0 }
	}
	msd.programParameters = {
		"seed": [0, 42, 1234567, 200]
	}
	return msd

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

# tests
if __name__ == "__main__":
	msd = example_1d()
	msd.compile("out/msd-example_1d.asm")

	msd = example_3d()
	msd.compile("out/msd-example_3d.asm")

	msd = simple_1d()
	msd.compile("out/simple_1d.asm")
