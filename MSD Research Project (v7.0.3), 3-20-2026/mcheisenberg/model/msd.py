from ..config import Config
from ..util import clamp

# region names
__FML__ = "L"
__FMR__ = "R"
__mol__ = "m"

def MSD(width: int, height: int, depth: int,
		molPosL: int=None, molPosR: int=None,
		topL: int=None, bottomL: int=None, frontR: int=None, backR: int=None
):
	# validate arguments
	width   = clamp(width,  0, None)
	height  = clamp(height, 0, None)
	depth   = clamp(depth,  0, None)
	molPosL = clamp(molPosL, 0,           width)
	molPosR = clamp(molPosR, molPosL - 1, width)
	topL    = clamp(topL,    0,        height - 1)
	bottomL = clamp(bottomL, topL - 1, height - 1)
	frontR  = clamp(frontR,  0,          depth - 1)
	backR   = clamp(backR,   frontR - 1, depth - 1)

	msd = Config()
	msd.nodes = {}  # ordered set (as dict)
	msd.edges = []
	msd.regions = { __FML__: [], __FMR__: [], __mol__: [] }

	# TODO: build nodes chucks to optimize for L1 cache
	# add all nodes
	for z in range(depth):
		for y in range(topL, bottomL + 1):
			for x in range(0, molPosL):
				i = (x, y, z)
				msd.nodes[i] = None  # add to ordered set
				msd.regions[__FML__].append(i)
	
	for z in range(frontR, backR + 1):
		for y in range(topL, bottomL + 1):
			if y == topL or y == bottomL or z == frontR or z == backR:
				for x in range(molPosL, molPosR + 1):
					i = (x, y, z)
					msd.nodes[i] = None
					msd.regions[__mol__].append(i)
	
	for z in range(frontR, backR + 1):
		for y in range(0, height):
			for x in range(molPosR + 1, width):
				i = (x, y, z)
				msd.nodes[i] = None
				msd.regions[__FMR__].append(i)
	
	# connect nodes
	for i in msd.nodes:
		x, y, z = i
		x_neighbor = (x + 1, y, z)  # right neighbor
		y_neighbor = (x, y + 1, z)  # neighbor below
		z_neighbor = (x, y, z + 1)  # neighbor behind
		
		if x < molPosL:  # FML
			if x_neighbor in msd.nodes:  msd.edges.append((i, x_neighbor))
			if y_neighbor in msd.nodes:  msd.edges.append((i, y_neighbor))
			if z_neighbor in msd.nodes:  msd.edges.append((i, z_neighbor))
		
		elif x <= molPosR:  # mol
			if x_neighbor in msd.nodes:  msd.edges.append((i, x_neighbor))
		
		else:  # FMR
			if x_neighbor in msd.nodes:  msd.edges.append((i, x_neighbor))
			if y_neighbor in msd.nodes:  msd.edges.append((i, y_neighbor))
			if z_neighbor in msd.nodes:  msd.edges.append((i, z_neighbor))
		
		if x == molPosL - 1 and molPosL <= molPosR:  # FML-FMR direct coupling
			dc_neighbor = (molPosR + 1, y, z)
			if dc_neighbor in msd.nodes:  msd.edges.append((i, dc_neighbor))

	# default parameters
	msd.globalParameters = {
		"kT": 0.1,
		"S": 1.0
	}
	msd.regionNodeParameters = {}
	msd.regionEdgeParameters = {
		(__FML__, __FML__): { "J": 1.0 },
		(__FML__, __mol__): { "J": 1.0 },
		(__mol__, __mol__): { "J": 1.0 },
		(__mol__, __FMR__): { "J": -1.0 },
		(__FMR__, __FMR__): { "J": 1.0 }
		# no default JLR direct coupling
	}

	msd.nodes = list(msd.nodes)
	msd.nodeId = lambda node: f"{node[0]}_{node[1]}_{node[2]}"  # converts nodes (tuples) to IDs (str)
	return msd
