from argparse import ArgumentParser
from math import sqrt
from pathlib import Path
from typing import Any
from psutil import HIGH_PRIORITY_CLASS
from tqdm import tqdm
from mcheisenberg import BalancedRuntimePoolExecutor as BRPE, Simulation, VisualStudio, kT_, J_, A_
from mcheisenberg.io import unique_path
from mcheisenberg.models import AsymmetricTwoLayerMSD, FML_, FMR_, mol_, OuterLayer_, InnerLayer_
import logging
import sys
import numpy as np
import pandas as pd

# parameters:
x0s = range(5)
J01s = [-1.0, -0.1, 0, 0.1, 1.0]
A0s = [numerator / 100 for numerator in range(0, 201, 2)]  # e.g. 0, 0.02, 0.04, ..., 2.00

t_eq = 10_000_000  # TODO: determine with iterate
freq = 100_000  # TODO: determin with auto_correlation
sim_count = 100 * freq

max_workers = 20
nice = HIGH_PRIORITY_CLASS

width = 11
height = 10
depth = 10
molPosL = 5
molPosR = 5
topL = 3
bottomL = 6
frontR = 3
backR = 6

REGIONS = [OuterLayer_, InnerLayer_, FML_, FMR_, mol_]

def init_worker():
	np.seterr(divide='ignore', invalid='ignore')

def build_config(x0):
	msd = AsymmetricTwoLayerMSD(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR, x0)
	msd.globalParameters[kT_] = 0.1
	msd.regionNodeParameters[OuterLayer_] = { A_: (0.0, 0.0, 0.0) }
	return msd

def task(runtime, x0, J01, A0):
	sim = Simulation(runtime)
	sim.J[OuterLayer_, InnerLayer_] = J01
	sim.A[OuterLayer_] = (A0, 0.0, 0.0)

	tag = f"J01={J01:.1f}, x0={x0:01}, A0={A0:.2f}"

	sim.randomize()
	sim.metropolis(t_eq,            progress_bar=f"#{BRPE.TASK_ID:04}: 1. Equilibriate {tag}", pbar_args={ "leave": False, "position": BRPE.WORKER_ID })
	sim.metropolis(sim_count, freq, progress_bar=f"#{BRPE.TASK_ID:04}: 2. Record data  {tag}", pbar_args={ "leave": False, "position": BRPE.WORKER_ID })
	# TODO: fix memory error.
	# 	Buffers don't get freed until Runtime is shutdown. Since runtime is cashed, this is never (end of program)
	#	Simple fix: stop caching runtimes.
	#	Maybe better fix, allow each metropolis to use a single shared buffer for all snapshots and only grab needed data??

	progress_bar = tqdm(total=1+7+11+1, desc=f"#{BRPE.TASK_ID:04}: 3. Calc. stats  {tag}", leave=False, position=BRPE.WORKER_ID)

	# TODO: optimize calculating stats. It currently takes ~2hr for ~2500 tasks. Too long!!

	t_history = np.array([t for t in sim.history.keys()])  # shape=(T,) -- All timestamps in order
	progress_bar.update(1)
	t_len = len(sim.history)                               # number of smaples
	T = t_history[-1] - t_history[0]                       # interval length for integration/summation
	dt = t_history[1:] - t_history[:-1]                    # delta time
	
	m_history_        = np.array([ss.m.value for ss in sim.history.values()])[:, None, :]                  # shape=(T, 1, 3) -- Aggrigate state of full system indexed as [time][0][axis]
	progress_bar.update(1)
	m_history_regions = np.array([ [ss.m[r].value for r in REGIONS] for ss in sim.history.values() ])      # shape=(T, R, 3) -- Aggrigate state of each region indexed as [time][region][axis]
	progress_bar.update(1)
	m_history_atoms   = np.array([ss.m.values() for ss in sim.history.values()])                           # shape=(T, N, 3) -- Full state of "atoms" in system indexed as [time][location][axis]	
	progress_bar.update(1)
	m_history = np.concatenate((m_history_, m_history_regions, m_history_atoms), axis=1)                   # shape=(T, 1+R+N, 3)
	progress_bar.update(1)
	m_mean = (0.5 / T) * np.sum((m_history[:-1, :, :] + m_history[1:, :, :]) * dt[:, None, None], axis=0)  # shape=(1+R+N, 3) -- approx. avg. using trapizoidal method
	progress_bar.update(1)
	m_std = np.sqrt((1.0 / (t_len - 1)) * np.sum((m_history - m_mean[None, :, :]) ** 2, axis=0))           # shape=(1+R+N, 3) -- sample std. dev. assuming saples are uncorrelated (i.e. auto-correlation is negligible)
	progress_bar.update(1)
	m_se = m_std / sqrt(t_len)                                                                             # shape=(1+R+N, 3) -- standard error
	progress_bar.update(1)

	def dot_sq(v):
		return np.dot(v, v)

	m_norm2_history_        = np.array([dot_sq(ss.m.value) for ss in sim.history.values()])[:, None]                 # shape=(T, 1)
	progress_bar.update(1)
	m_norm2_history_regions = np.array([ [dot_sq(ss.m[r].value) for r in REGIONS] for ss in sim.history.values() ])  # shape=(T, R)
	progress_bar.update(1)
	m_norm2_history = np.concatenate((m_norm2_history_, m_norm2_history_regions), axis=1)                            # shape=(T, 1+R)
	progress_bar.update(1)
	m_norm_history = np.sqrt(m_norm2_history)                                                                        # shape=(T, 1+R)
	progress_bar.update(1)
	m_norm_mean  = (0.5 / T) * np.sum((m_norm_history[:-1, :] + m_norm_history[1:, :]) * dt[:, None], axis=0)        # shape=(1+R,)
	progress_bar.update(1)
	m_norm_std = np.sqrt((1.0 / (t_len - 1)) * np.sum((m_norm_history - m_norm_mean[None, :]) ** 2, axis=0))         # shape=(1+R,) -- sample standard diviation
	progress_bar.update(1)
	m_norm_se = m_norm_std / sqrt(t_len)
	progress_bar.update(1)
	m_norm2_mean = (0.5 / T) * np.sum((m_norm2_history[:-1, :] + m_norm2_history[1:, :]) * dt[:, None], axis=0)      # shape=(1+R,) -- TODO: should we use trap. method here?
	progress_bar.update(1)
	n = np.array([sim.n.value] + [sim.n[r].value for r in REGIONS])                                                  # shape=(1+R,) -- size of (i.e. atoms in) system and of each region
	progress_bar.update(1)
	m_norm_var = m_norm2_mean - m_norm_mean ** 2                                                                     # shape=(1+R,) -- population variance
	progress_bar.update(1)
	chi        = (n / sim.kT.value) * m_norm_var                                                                     # shape=(1+R,) -- magnetic susceptibility
	chi_atomic = (1 / (sim.kT.value * n)) * m_norm_var                                                               # shape=(1+R,) -- mag. susc. per atom
	progress_bar.update(1)

	nodes = np.array([xyz for xyz in sim.nodes])                                                                     # shape=(N,3) -- (x, y, z) position for each node
	progress_bar.update(1)

	progress_bar.close()
	return x0, J01, A0, m_mean, m_std, m_se, m_norm_mean, m_norm_std, m_norm_se, chi, chi_atomic, nodes, BRPE.WORKER_ID, BRPE.TASK_ID, tag


if __name__ == "__main__":
	parser = ArgumentParser(description="Two-layer anisotropy study")
	parser.add_argument("--out", dest="out", type=str, default="out", help="Ouptut directory")
	parser.add_argument("--year", dest="year", type=int, default=None, help="Select version of Visual Studio (e.g. 2022, 2026)")
	args = parser.parse_args(sys.argv[1:])

	out = unique_path(Path(args.out), "anisotropic_layer_study")
	out.mkdir()
	tool = VisualStudio(year=args.year)

	log_path = out / "ERROR.log"
	handler = logging.FileHandler(log_path, delay=True)
	logging.basicConfig(handlers=[handler], level=logging.ERROR)

	# combinations of config and tast parameters (i.e. args)
	configs: list[tuple] = [(x0,) for x0 in x0s]  # parameter list as tuple
	tasks_per_config: list[tuple] = [(J01, A0) for J01 in J01s for A0 in A0s]  # parameter list as tuple

	with BRPE(build_config, configs, max_workers=max_workers, tool=tool, initializer=init_worker, nice=nice) as exe:
		print("Output directory:", out)

		# submit tasks
		for config_args in configs:
			for task_args in tasks_per_config:
				exe.submit(task, config_args, task_args, use_cache=False)

		dfs: dict[Any, pd.DataFrame] = {}  # (x0, J01) -> pd.DataFrame
		
		# process results
		np.seterr(divide='ignore', invalid='ignore')
		task_count = len(exe.futures)
		failure_count = 0
		for future in tqdm(exe.as_completed(), total=task_count, desc="*** Simulating ***", leave=True, position=0):
			try:
				# task finished: get results, then process and save
				x0, J01, A0, m_mean, m_std, m_se, m_norm_mean, m_norm_std, m_norm_se, chi, chi_atomic, nodes, worker_id, task_id, tag = future.result()
				R = len(REGIONS)
				N = nodes.shape[0]

				# save aggrigate info (i.e. "...A0=(variable).xlsx")
				with tqdm(total=1+R, desc=f"#{task_id:04} 4. Output plots {tag}", leave=False, position=worker_id) as task_pbar:
					data = {}
					for idx, region in enumerate([""] + REGIONS):
						data.update({
							"A0":                  A0,
							f"<M{region}_norm>":    m_norm_mean[idx],
							f"<M{region}_x>":       m_mean[idx][0],
							f"<M{region}_y>":       m_mean[idx][1],
							f"<M{region}_z>":       m_mean[idx][2],
							f"x{region}":           chi[idx],
							f"x{region}_atomic":    chi[idx],
							f"std(M{region}_norm)": m_norm_std[idx],
							f"std(M{region}_x)":    m_std[idx][0],
							f"std(M{region}_y)":    m_std[idx][1],
							f"std(M{region}_z)":    m_std[idx][2],
							f"se(M{region}_norm)":  m_norm_se[idx],
							f"se(M{region}_x)":     m_se[idx][0],
							f"se(M{region}_y)":     m_se[idx][1],
							f"se(M{region}_z)":     m_se[idx][2]
						})
						task_pbar.update(1)
					df_row = pd.DataFrame([data])  # new single row DataFrame
					key = (x0, J01)
					if key not in dfs:
						dfs[key] = df_row
					else:
						dfs[key] = pd.concat([dfs[key], df_row], ignore_index=True)
						dfs[key].sort_values(by="A0")

					# write to .xlsx
					with pd.ExcelWriter(out / f"plots J01={J01:.1f}, x0={x0:01}, A0=(variable).xlsx", "xlsxwriter") as writer:
						dfs[key].to_excel(writer, index=False)
						worksheet = next(iter(writer.sheets.values()))
						# TODO: add sigma bands
						# TODO: add Excel plots
						# TODO: task_pbar.update()

				# save atoms (i.e. "...A0=0.00.xlsx")
				with tqdm(total=N, desc=f"#{task_id:04} 5. Output atoms {tag}", leave=False, position=worker_id) as task_pbar:
					data = []
					for idx, (x, y, z) in enumerate(nodes, start=1+R): 
						data.append({
							"region":
								OuterLayer_ if x < x0 else
								InnerLayer_ if x < molPosL else
								mol_        if x <= molPosR else
								FMR_,
							"x": x,
							"y": y,
							"z": z,
							"<m_x>": m_mean[idx][0],
							"<m_y>": m_mean[idx][1],
							"<m_z>": m_mean[idx][2],
							"std(m_x)": m_std[idx][0],
							"std(m_y)": m_std[idx][1],
							"std(m_z)": m_std[idx][2],
							"se(m_x)": m_se[idx][0],
							"se(m_y)": m_se[idx][1],
							"se(m_z)": m_se[idx][2]
						})
						task_pbar.update(1)
					df = pd.DataFrame(data)

					with pd.ExcelWriter(out / f"atoms J01={J01:.1f}, x0={x0:01}, A0={A0:.2f}.xlsx", "xlsxwriter") as writer:
						df.to_excel(writer, index=False)

			except Exception as ex:
				failure_count += 1
				logging.exception(ex)

	# all tasks are now finished: finallize and clean up
	print(f"Done. ", end="")
	if failure_count == 0:
		print("All simulations completed successfully.")
	else:
		print(f"{failure_count}/{task_count} simulations failed!")
		print(f"See: {log_path}")
