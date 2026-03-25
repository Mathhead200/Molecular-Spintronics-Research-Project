from .. import Simulation, NODES_, EDGES_, VisualStudio
from ..model import MSD, FML_, FMR_, mol_

TOOL = VisualStudio(year=2022)

dims = {
	# sim_name: (width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR)
	"a": (11, 10, 10, 5, 5, 3, 6, 3, 6),
	"b": (11, 10, 10, 5, 5, 0, 9, 0, 9),
	"c": (10, 10, 10, 10, 9, 0, 9, 0, 9),
	"d": (25, 25, 25, 25, 24, 0, 24, 0, 24),
	"e": (21, 105, 105, 10, 10, 51, 53, 51, 53),
	"f": (70, 10, 10, 10, 59, 0, 9, 0, 9)
}

for sim_name, dim in dims.items():
	print("--", sim_name)
	width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR = dim
	msd = MSD(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR)
	with msd.compile(tool=TOOL, progress_bars=True) as rt:
		sim = Simulation(rt)
		n = sim.n[NODES_]
		assert n[FML_] == molPosL * (bottomL - topL + 1) * depth
		assert n[FMR_] == (width - 1 - molPosR) * height * (backR - frontR + 1)
		assert n[FML_] + n[FMR_] + n[mol_] == n
		print("Randomize...")
		sim.randomize()
		print("Metropolis...")
		sim.metropolis(100 * n, freq=n, callback=lambda snapshot: None, reuse_buffer=True, progress_bar=True)
