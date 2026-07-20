from mcheisenberg import Simulation, VisualStudio
from mcheisenberg.model import MSD, FML_, FMR_, mol_
from mcheisenberg.io import unique_path
from math import sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm

repetitions = 20
out = unique_path(dir="out", prefix="study-layers-full", suffix=".csv")

# width=7 had 1 anomaly in 10
# width=11 had 1 anomaly in 10
# width=15 appear to enter a frustrainted state in approximently ~< 50% of runs
widths_and_t_eqs = [(3, 10_000_000), (7, 100_000_000), (11, 500_000_000), (15, 1_000_000_000), (19, 1_000_000_000)]
kTs = [0.05, 0.1, 0.25]
J01s = [-1.0, -0.1, 0, 0.1, 1.0]
As = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
JmLs = [j / 10 for j in range(-10, 11, 1)]  # -1.0, -0.9, ..., +1.0

N0 = len(widths_and_t_eqs)
N1 = len(kTs)
N2 = len(J01s)
N3 = len(As)
N = N0 * N1 * N2 * N3  # number of permutations of fixed variables

OuterLayer_ = "Outer"
InnerLayer_ = "Inner"

if __name__ == "__main__":
	columns = ["width", "kT", "J01", "A", "JmL"]
	for region in ["", "0", "1", "L", "R", "m"]:
		for axis in ["", "_x", "_y", "_z"]:
			for sigma in ["", "_sigma", "_error"]:  # sigma is actually s (sample, not population); error is standard error = s / sqrt(n)
				columns.append(f"M{region}{axis}{sigma}")  # e.g. M, M_sigma, M_error, M_x, M_x_sigma, M_error, ..., M1_z_error
	pd.DataFrame(columns=columns).to_csv(out, index=False)
	progress_bar = tqdm(total=(N * len(JmLs) * repetitions), desc="Simulating")
	
	for width, t_eq in widths_and_t_eqs:
		molPos = width // 2
		msd = MSD(
			width   = width,
			height  = 10,
			depth   = 10,
			molPosL = molPos,
			molPosR = molPos,
			topL    = 0,
			bottomL = 9,
			frontR  = 0,
			backR   = 9
		)
		x0 = molPos // 2
		msd.regions[OuterLayer_] = [(x, y, z) for x, y, z in msd.regions[FML_] if x <= x0]
		msd.regions[InnerLayer_] = [(x, y, z) for x, y, z in msd.regions[FML_] if x > x0]
		msd.regionNodeParameters[OuterLayer_] = { "A": (0.0, 0.0, 0.0) }  # variable
		JL = msd.regionEdgeParameters[(FML_, FML_)]["J"]
		del msd.regionEdgeParameters[(FML_, FML_)]["J"]  # this parameter should be defined across sub-regions: OuterLayer_, and InnerLayer_
		msd.regionEdgeParameters[(OuterLayer_, OuterLayer_)] = { "J": JL }
		msd.regionEdgeParameters[(OuterLayer_, InnerLayer_)] = { "J": 0.0 }  # variable
		msd.regionEdgeParameters[(InnerLayer_, InnerLayer_)] = { "J": JL }

		with msd.compile(tool=VisualStudio(year=2022), asm=f"out/study-layers, {width}.asm") as runtime:
			sim = Simulation(runtime)

			for kT in kTs:
				sim.kT = kT
				
				for J01 in J01s:
					sim.J[(OuterLayer_, InnerLayer_)] = J01

					for A in As:
						sim.A[OuterLayer_] = (A, 0, 0)  # x-direction

						for JmL in JmLs:
							sim.J[(FML_, mol_)] = JmL
							sim.J[(mol_, FMR_)] = -JmL

							shape = (repetitions, 3)
							M = np.empty(shape, dtype=float)
							ML = np.empty(shape, dtype=float)
							MR = np.empty(shape, dtype=float)
							Mm = np.empty(shape, dtype=float)
							M0 = np.empty(shape, dtype=float)
							M1 = np.empty(shape, dtype=float)
							for i in range(repetitions):
								sim.randomize()
								sim.metropolis(t_eq)

								M[i] = sim.m.value
								M0[i] = sim.m[OuterLayer_].value
								M1[i] = sim.m[InnerLayer_].value
								ML[i] = sim.m[FML_].value
								MR[i] = sim.m[FMR_].value
								Mm[i] = sim.m[mol_].value

								progress_bar.update(1)

							# save data to CSV
							fact = repetitions ** -0.5  # 1.0 / sqrt(repetitions)
							row = { "width": width, "kT": kT, "J01": J01, "A": A, "JmL": JmL }
							for mat, label in [(M, "M"), (M0, "M0"), (M1, "M1"), (ML, "ML"), (MR, "MR"), (Mm, "Mm")]:
								data = np.linalg.norm(mat, axis=1)  # magnitude (norm) of each vector
								s = np.std(data, ddof=1)  # sample standard deviation (1 degree of freedom)
								row[label] = np.mean(data)  # mean of norms
								row[f"{label}_sigma"] = s
								row[f"{label}_error"] = s * fact

								for column, axis in enumerate(["_x", "_y", "_z"]):
									data = mat[:, column]  # xs = mat[:, 0]; ys = mat[:, 1]; zs = mat[:, 2]
									s = np.std(data, ddof=1)
									row[f"{label}{axis}"] = np.mean(data)  # mean of vector component (xs, ys, or zs)
									row[f"{label}{axis}_sigma"] = s
									row[f"{label}{axis}_error"] = s * fact
							
							pd.DataFrame([row]).to_csv(out, mode="a", header=False, index=False)

	progress_bar.close()

	