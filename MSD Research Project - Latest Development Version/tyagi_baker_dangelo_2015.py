# Reference: https://iopscience.iop.org/article/10.1088/0957-4484/26/30/305602/meta
# Rerunning the same experiment to see if we get the same results

from mcheisenberg import BalancedRuntimePoolExecutor, Simulation, VisualStudio, J_, kT_
from mcheisenberg.models import MSD, FML_, FMR_, mol_
from mcheisenberg.io import csv
from psutil import BELOW_NORMAL_PRIORITY_CLASS

def config(width):
	return MSD(width, 10, 10, 5, 5, 3, 6, 3, 6)

def task(runtime, width, kT, JmL):
	sim = Simulation(runtime)
	sim.kT = kT
	sim.J[FML_] = JmL
	sim.J[FMR_] = -JmL
	sim.metropolis(50_000_000, progress_bar=True)  # 1~50 million (Section 2)
	sim.metropolis(50_000_000, 10_000, progress_bar=True)  # GUESS?
	return csv(sim, dir="out", prefix=f"tyagi_baker_dangelo_2015, width={width}, kT={kT}, JmL={JmL}", progress_bar=True)

if __name__ == "__main__":
	widths = [3, 7, 11, 15, 19]
	with BalancedRuntimePoolExecutor(config, widths, max_workers=20, tool=VisualStudio(2026), nice=BELOW_NORMAL_PRIORITY_CLASS) as brpe:
		for width in widths:
			for kT in [0.05, 0.10, 0.25]:
				for JmL in [j / 10 for j in range(-10, 11)]:
					brpe.submit(task, width, (kT, JmL), nice=BELOW_NORMAL_PRIORITY_CLASS)
		for path in brpe.as_completed():
			print(path)
