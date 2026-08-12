from math import sqrt
from tqdm import tqdm
from mcheisenberg import BalancedRuntimePoolExecutor as BRPE, Simulation, VisualStudio, kT_, J_, A_
from mcheisenberg.models import AsymmetricTwoLayerMSD, FML_, FMR_, mol_, OuterLayer_, InnerLayer_
import traceback
import numpy as np

t_eq = 10_000_000  # TODO: determine with iterate
freq = 100_000  # TODO: determin with auto_correlation
sim_count = 100 * freq
tool = VisualStudio(2026)
max_workers = 25

def build_config(x0):
	msd = AsymmetricTwoLayerMSD(11, 10, 10, 5, 5, 3, 6, 3, 6, x0)
	msd.globalParameters[kT_] = 0.1
	msd.regionNodeParameters[OuterLayer_] = { A_: (0.0, 0.0, 0.0) }
	return msd

def task(runtime, x0, J01, A0):
	sim = Simulation(runtime)
	sim.J[OuterLayer_, InnerLayer_] = J01
	sim.A[OuterLayer_] = (A0, 0.0, 0.0)

	sim.randomize()
	sim.metropolis(t_eq)             # progress_bar=f"[x0={x0}, J01={J01}, A0={A0}]: Equilibrate")
	sim.metropolis(sim_count, freq)  # progress_bar=f"[x0={x0}, J01={J01}, A0={A0}]: Recording data")
	# TODO: fix memory error.
	# 	Buffers don't get freed until Runtime is shutdown. Since runtime is cashed, this is never (end of program)
	#	Simple fix: stop caching runtimes.
	#	Maybe better fix, allow each metropolis to use a single shared buffer for all snapshots and only grab needed data??

	REGIONS = [OuterLayer_, InnerLayer_, FML_, FMR_, mol_]
	m_history_        = np.array([ss.m.value for ss in sim.history])[:, None, :]                       # shape=(T, 1, 3) -- Aggrigate state of full system indexed as [time][0][axis]
	m_history_regions = np.array([ [ss.m[r].value for ss in sim.history.values()] for r in REGIONS ])  # shape=(T, R, 3) -- Aggrigate state of each region indexed as [time][region][axis]
	m_history_atoms   = np.array([ss.m.values() for ss in sim.history.values()])                       # shape=(T, n, 3) -- Full state of "atoms" in system indexed as [time][location][axis]	
	m_history = np.concatenate((m_history_, m_history_regions, m_history_atoms), axis=0)               # shape=(T, 1+R+n, 3)
	t_history = np.array([t for t in sim.history])  # shape=(T,) -- All timestamps in order
	t_len = len(sim.history)             # number of smaples
	T = t_history[-1] - t_history[0]     # interval length for integration/summation
	dt = t_history[1:] - t_history[:-1]  # delta time
	m_mean = (0.5 / T) * np.sum((m_history[:-1, :, :] + m_history[1:, :, :]) * dt[:, None, None], axis=0)  # shape=(1+R+n, 3) -- approx. avg. using trapizoidal method
	m_std = np.sqrt((1.0 / (t_len - 1)) * np.sum((m_history - m_mean[None, :, :]) ** 2, axis=0))           # shape=(1+R+n, 3) -- sample std. dev. assuming saples are uncorrelated (i.e. auto-correlation is negligible)
	m_se = m_std / sqrt(t_len)                                                                             # shape=(1+R+n, 3) -- standard error

	# TODO: stats on |m|

	return (m_mean, m_std, m_se)


if __name__ == "__main__":
	# combinations of config and tast parameters (i.e. args)
	configs: list[tuple] = []
	for x0 in range(5):
		configs.append((x0,))  # add parameter list as tuple

	tasks_per_config: list[tuple] = []
	for J01 in [-1.0, -0.1, 0, 0.1, 1.0]:
		for _a0 in range(0, 201, 2):
			A0 = _a0 / 100  # e.g. 0, 0.02, 0.04, ..., 2.00
			tasks_per_config.append((J01, A0))  # add parameter list as tuple

	with BRPE(build_config, configs, max_workers=max_workers, tool=tool) as exe:
		# submit tasks
		for config_args in configs:
			for task_args in tasks_per_config:
				exe.submit(task, config_args, task_args, use_cache=False)
		
		# process results
		pbar = tqdm(total=len(exe.futures), desc="Simulating")
		for future in exe.as_completed():
			try:
				_ = future.result()  # TODO: stub
				# TODO: record ...
			except Exception:
				traceback.print_exc()
			pbar.update(1)

	# clean up
	pbar.close()
