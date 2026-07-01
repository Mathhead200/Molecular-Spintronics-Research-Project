import mcheisenberg as mc
from mcheisenberg.build import VisualStudio
from mcheisenberg.model import MSD, FML_, FMR_
import numpy as np

msd = MSD(11, 10, 10, 5, 5, 3, 6, 3, 6)
msd.globalParameters["B"] = (0.0, 0.0, 0.0)
msd.globalParameters["dB"] = (1e-9, 0.0, 0.0) 

with msd.compile(tool=VisualStudio(year=2022), asm="./example_dB.asm") as rt:
	sim = mc.Simulation(rt)

	def print_stats():
		print("t: ", sim.t)
		print("B: ", sim.B)
		print("dB:", sim.dB)
		print("<M>: ", np.mean(sim.m.values(), axis=0))
		print("<ML>:", np.mean(sim.m[FML_].values(), axis=0))
		print("<MR>:", np.mean(sim.m[FMR_].values(), axis=0))
		print("-" * 80)
	
	sim.randomize()
	print_stats()

	sim.metropolis(int(1e8))
	print_stats()

	sim.dB = (-1e-9, 0.0, 0.0)
	print_stats()

	sim.metropolis(int(2e8))
	print_stats()
