from ..models import MSD, FML_, FMR_, mol_

OuterLayer_ = "Outer"
InnerLayer_ = "Inner"

def AsymmetricTwoLayerMSD(width: int, height: int, depth: int,
		molPosL: int=None, molPosR: int=None,
		topL: int=None, bottomL: int=None, frontR: int=None, backR: int=None,
		x0: int=None):
	msd = MSD(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR)
	if x0 is None:  x0 = molPosL // 2
	msd.regions[OuterLayer_] = [(x, y, z) for x, y, z in msd.regions[FML_] if x <= x0]
	msd.regions[InnerLayer_] = [(x, y, z) for x, y, z in msd.regions[FML_] if x > x0]
	# msd.regionNodeParameters[OuterLayer_] = { "A": (0.0, 0.0, 0.0) }
	JL = msd.regionEdgeParameters[(FML_, FML_)]["J"]
	del msd.regionEdgeParameters[(FML_, FML_)]["J"]
	msd.regionEdgeParameters[(OuterLayer_, OuterLayer_)] = { "J": JL }
	msd.regionEdgeParameters[(OuterLayer_, InnerLayer_)] = { "J": JL }
	msd.regionEdgeParameters[(InnerLayer_, InnerLayer_)] = { "J": JL }
	return msd
