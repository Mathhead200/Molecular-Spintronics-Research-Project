import mcheisenberg as mc
from mcheisenberg.models import MSD
import numpy as np

if __name__ == "__main__":
	msd = MSD(11, 10, 10, 5, 5, 3, 6, 3, 6)
	with msd.compile(dir=".", asm="history-test.asm") as rt:
		sim = mc.Simulation(rt)
		sim.metropolis(5_000_000)                  # to reach equilibrium
		sim.metropolis(100_000_000, freq=200_000)  # to gather uncorrelated samples
		print(np.mean(sim.m.history.values()))
