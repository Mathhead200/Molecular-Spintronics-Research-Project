import mcheisenberg as mc
from mcheisenberg.model import MSD
import numpy as np
import time

if __name__ == "__main__":
	msd = MSD(11, 10, 10, 5, 5, 3, 6, 3, 6)
	with msd.compile(dir=".") as rt:
		sim = mc.Simulation(rt)

		clock = time.perf_counter()
		sim.metropolis(100_000_000)  # to reach equilibrium
		elapsed1 = time.perf_counter() - clock
		print(f"1. Elapsed time: {elapsed1:.3f} s")

		clock = time.perf_counter()
		sim.metropolis(100_000_000, freq=10_000_000, bookend=False)  # to gather uncorrelated samples
		elapsed2 = time.perf_counter() - clock
		print(f"2. Elapsed time: {elapsed2:.3f} s")
		print(f"          Delta: {elapsed2 - elapsed1:.3f} s")
		print(f"      Snapshots: {len(sim.history)}")
		print()
		
		print("Avg. U:", mc.mean(sim.u))
		print("Avg. M:", mc.mean(sim.m))
		print("Avg. |M|:", mc.mean(mc.norm(sim.m)))
		print("|Avg. M|:", mc.norm(mc.mean(sim.m)))
