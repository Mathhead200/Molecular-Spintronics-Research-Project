from tqdm import tqdm
from scipy import stats
from mcheisenberg import BalancedRuntimePoolExecutor as BRPE, Simulation, VisualStudio, kT_, J_, A_
from mcheisenberg.models import AsymmetricTwoLayerMSD, FML_, FMR_, mol_, OuterLayer_, InnerLayer_
import traceback
import numpy as np
import mcheisenberg as mc

t_eq = 10_000_000  # TODO: determine with iterate
freq = 100_000  # TODO: determin with auto_correlation
sim_count = 100 * freq

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

	# def get_stats(m_history):
	# 	m_x = m_history[:, 0]
	# 	m_y = m_history[:, 1]
	# 	m_z = m_history[:, 2]
	# 	m = np.linalg.norm(m_history, axis=1)
	# 	n = m_history.shape[0]
		
	# 	stats = {}
	# 	stats["m"]

	# 	return stats

	# output = {}
	# m_bar = mc.mean(sim.m)
	# output["M"] = np.linalg.norm(m_bar)
	# output["M_x"] = m_bar[0]
	# output["M_y"] = m_bar[1]
	# output["M_z"] = m_bar[2]
	# for region in [OuterLayer_, InnerLayer_, FML_, FMR_, mol_]:
	# 	m_bar = mc.mean(sim.m[region])
	# 	output[f"M{region}"] = np.linalg.norm(m_bar)
	# 	output[f"M{region}_sigma"] = 
	# 	output[f"M{region}_x"] = m_bar[0]
	# 	output[f"M{region}_y"] = m_bar[1]
	# 	output[f"M{region}_z"] = m_bar[2]

	return (x0, J01, A0, sim.m.value)  # TODO: stub


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

	with BRPE(build_config, configs, max_workers=25, tool=VisualStudio(2022)) as exe:
		# submit tasks
		for config_args in configs:
			for task_args in tasks_per_config:
				exe.submit(task, config_args, task_args, use_cache=False)
		
		# process results
		pbar = tqdm(total=len(exe.futures), desc="Simulating")
		for future in exe.as_completed():
			try:
				x0, J01, A0, m = future.result()
				# print(f"x0={x0}, J01={J01}, A0={A0} -> m={m}")  # TODO: stub
			except:
				traceback.print_exc()
			pbar.update(1)

	# clean up
	pbar.close()
