import mcheisenberg as mc
from mcheisenberg.models import MSD
import numpy as np

if __name__ == "__main__":
	msd = MSD(11, 10, 10, 5, 5, 3, 6, 3, 6)
	with msd.compile(dir=".") as rt:
		sim = mc.Simulation(rt)
		sim.metropolis(5_000_000)                  # to reach equilibrium
		sim.metropolis(2_000_000, freq=200_000)  # to gather uncorrelated samples
		print("kT:", sim.kT, sim.rt.kT, sim.rt.kT[(0, 3, 0)])
		print("kT.history:", sim.kT.history)
		print("m.history:", sim.m.history)
		print(mc.mean(sim.kT))
		print(mc.norm(mc.mean(sim.m)))
		print(mc.mean(mc.norm(sim.m)))
