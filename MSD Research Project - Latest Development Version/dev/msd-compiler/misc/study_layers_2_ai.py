from math import sqrt, ceil
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm
from xlsxwriter.utility import xl_col_to_name, xl_range_formula

from mcheisenberg import BalancedRuntimePoolExecutor as BRPE, Simulation, VisualStudio, kT_, J_, A_
from mcheisenberg.models import AsymmetricTwoLayerMSD, FML_, FMR_, mol_, OuterLayer_, InnerLayer_

# ---------------------------------------------------------------------------
# Changes vs. the version you shared:
#  1. task(): fixed np.concatenate(..., axis=0) -> axis=1 for m_history and
#     m_norm2_history (they were being stacked along the time axis instead
#     of the "location" axis, which either raised ValueError or silently
#     produced garbage shapes).
#  2. task(): m_norm2_history_ reshaped to (T,1) before concatenation.
#  3. task(): fixed a misplaced parenthesis in m_norm_std -- axis=0 was
#     being passed to np.sqrt (which has no such parameter) instead of the
#     inner np.sum.
#  4. task(): now returns x0, J01, A0 directly (not just embedded in the
#     `tag` string) so the driver can route results without parsing text.
#  5. task(): added a sanity-check assertion that sim.nodes count matches
#     the "atoms" dimension of the stats arrays -- protects the snapshot
#     CSV from silently misaligned rows if that assumption ever breaks.
#  6. J01 = 0 -> J01 = 0.0 in the sweep list, so sheet/row labels are
#     consistently formatted.
#  7. tqdm(..., desr=...) typo -> desc=... (silently accepted by tqdm's
#     **kwargs, so it never raised, just showed a blank progress label).
#  8. New: actual output. Writes into a fresh out/study-layers-2[-N]/
#     directory:
#       - snapshot.csv   one row per (x0, J01, A0, node), streamed via
#                        append-mode writes as each task completes
#       - main.xlsx      one sheet per (x0, J01), with the input A0 sweep
#                        and mean/std/se of M_norm, M_x, M_y, M_z at the
#                        global level and each region, plus (bonus, easy
#                        to remove) magnetic susceptibility. Each sheet
#                        gets a layer-comparison chart (M_norm only, no
#                        error bars) and a grid of per-level/per-axis
#                        charts with +/-1 and +/-2 sigma bands.
# ---------------------------------------------------------------------------

t_eq = 10_000_000  # TODO: determine with iterate
freq = 100_000  # TODO: determin with auto_correlation
sim_count = 100 * freq
tool = VisualStudio(2026)
max_workers = 25

REGIONS = [OuterLayer_, InnerLayer_, FML_, FMR_, mol_]
LEVELS = [""] + REGIONS  # "" == whole-system aggregate; index order matches m_mean/m_norm_mean/chi
AXES = ["norm", "x", "y", "z"]
VECTOR_AXES = [("x", 0), ("y", 1), ("z", 2)]  # axis label -> index into the 3-vector arrays


def level_label(level: str) -> str:
	return "M" if level == "" else f"M_{level}"


def excel_safe_sheet_name(name: str) -> str:
	""" Excel sheet names must be <=31 chars and can't contain []:*?/\\ """
	name = re.sub(r'[\[\]:\*\?/\\]', "_", name)
	return name[:31]


def unique_dir(base: Path) -> Path:
	""" Create and return `base`, or `base-1`, `base-2`, ... if `base` already exists. """
	candidate = base
	i = 1
	while candidate.exists():
		candidate = base.with_name(f"{base.name}-{i}")
		i += 1
	candidate.mkdir(parents=True)
	return candidate


def build_config(x0):
	msd = AsymmetricTwoLayerMSD(11, 10, 10, 5, 5, 3, 6, 3, 6, x0)
	msd.globalParameters[kT_] = 0.1
	msd.regionNodeParameters[OuterLayer_] = { A_: (0.0, 0.0, 0.0) }
	return msd


def task(runtime, x0, J01, A0, task_id):
	sim = Simulation(runtime)
	sim.J[OuterLayer_, InnerLayer_] = J01
	sim.A[OuterLayer_] = (A0, 0.0, 0.0)

	tag = f"(x0={x0}, J01={J01}, A0={A0})"

	sim.randomize()
	sim.metropolis(t_eq,            progress_bar=f"#{task_id:03}: 1. Equilibriate {tag}", leave_pbar=False)
	sim.metropolis(sim_count, freq, progress_bar=f"#{task_id:03}: 2. Record data  {tag}", leave_pbar=False)
	# NOTE: use_cache=False means this Runtime is shut down (and its buffers freed) as
	# soon as this task returns -- the memory-leak concern from the old TODO comment
	# doesn't apply here since caching was already turned off.

	progress_bar = tqdm(total=9, desc=f"#{task_id:03}: 3. Calc. stats  {tag}", leave=False)

	t_history = np.array([t for t in sim.history.keys()])  # shape=(T,) -- All timestamps in order
	t_len = len(sim.history)                               # number of samples
	T = t_history[-1] - t_history[0]                        # interval length for integration/summation
	dt = t_history[1:] - t_history[:-1]                     # delta time

	m_history_        = np.array([ss.m.value for ss in sim.history.values()])[:, None, :]               # shape=(T, 1, 3) -- Aggregate state of full system indexed as [time][0][axis]
	m_history_regions = np.array([ [ss.m[r].value for r in REGIONS] for ss in sim.history.values() ])   # shape=(T, R, 3) -- Aggregate state of each region indexed as [time][region][axis]
	m_history_atoms   = np.array([ss.m.values() for ss in sim.history.values()])                        # shape=(T, n, 3) -- Full state of "atoms" in system indexed as [time][location][axis]
	m_history = np.concatenate((m_history_, m_history_regions, m_history_atoms), axis=1)                # shape=(T, 1+R+n, 3)
	m_mean = (0.5 / T) * np.sum((m_history[:-1, :, :] + m_history[1:, :, :]) * dt[:, None, None], axis=0)  # shape=(1+R+n, 3) -- approx. avg. using trapezoidal method
	progress_bar.update(1)
	m_std = np.sqrt((1.0 / (t_len - 1)) * np.sum((m_history - m_mean[None, :, :]) ** 2, axis=0))         # shape=(1+R+n, 3) -- sample std. dev. assuming samples are uncorrelated (i.e. auto-correlation is negligible)
	progress_bar.update(1)
	m_se = m_std / sqrt(t_len)                                                                            # shape=(1+R+n, 3) -- standard error
	progress_bar.update(1)

	def dot_sq(v):
		return np.dot(v, v)

	m_norm2_history_        = np.array([dot_sq(ss.m.value) for ss in sim.history.values()])[:, None]                  # shape=(T, 1)
	m_norm2_history_regions = np.array([ [dot_sq(ss.m[r].value) for r in REGIONS] for ss in sim.history.values() ])   # shape=(T, R)
	m_norm2_history = np.concatenate((m_norm2_history_, m_norm2_history_regions), axis=1)                             # shape=(T, 1+R)
	m_norm_history = np.sqrt(m_norm2_history)                                                                         # shape=(T, 1+R)
	progress_bar.update(1)
	m_norm_mean  = (0.5 / T) * np.sum((m_norm_history[:-1, :]  + m_norm_history[1:, :])  * dt[:, None], axis=0)       # shape=(1+R,)
	progress_bar.update(1)
	m_norm_std = np.sqrt((1.0 / (t_len - 1)) * np.sum((m_norm_history - m_norm_mean[None, :]) ** 2, axis=0))          # shape=(1+R,) -- sample standard deviation
	progress_bar.update(1)
	m_norm_se = m_norm_std / sqrt(t_len)
	progress_bar.update(1)
	m_norm2_mean = (0.5 / T) * np.sum((m_norm2_history[:-1, :] + m_norm2_history[1:, :]) * dt[:, None], axis=0)       # shape=(1+R,)
	chi = (np.array([sim.n] + [sim.n[r] for r in REGIONS]) / sim.kT) * (m_norm2_mean - m_norm_mean ** 2)              # shape=(1+R,) -- magnetic susceptibility
	progress_bar.update(1)

	nodes = np.array([i for i in sim.nodes])
	progress_bar.update(1)
	progress_bar.close()

	# sanity check: the "atoms" slice of m_mean/m_std/m_se must line up 1:1 with `nodes`,
	# since the snapshot CSV zips them together by position.
	n_levels = 1 + len(REGIONS)
	n_atoms_in_stats = m_mean.shape[0] - n_levels
	assert n_atoms_in_stats == len(nodes), (
		f"node count mismatch in {tag}: {n_atoms_in_stats} atom entries in stats arrays "
		f"vs {len(nodes)} from sim.nodes -- the ordering assumption between ss.m.values() "
		f"and sim.nodes is violated"
	)

	return x0, J01, A0, task_id, tag, m_mean, m_std, m_se, m_norm_mean, m_norm_std, m_norm_se, chi, nodes


def write_main_workbook(path: Path, sheet_rows: dict):
	"""
	sheet_rows: dict[(x0, J01), list[row_dict]] -- one row_dict per A0 value.
	Writes one sheet per (x0, J01), each with the aggregate A0 sweep plus two
	kinds of charts: a layer-comparison chart (M_norm, no error bars) and a
	grid of per-level/per-axis charts with +/-1, +/-2 sigma bands.
	"""
	writer = pd.ExcelWriter(path, engine="xlsxwriter")

	default_cell_width = 65
	default_cell_height = 20
	chart_size = { "width": 7 * default_cell_width, "height": 9 * default_cell_height }
	chart_col_width = ceil(chart_size["width"] / default_cell_width)
	chart_row_height = ceil(chart_size["height"] / default_cell_height)

	LEVEL_COLORS = ["#156082", "#E97132", "#196B24", "#0F9ED5", "#A02B93", "#4EA72E"]  # one per LEVELS
	AXIS_SHAPES = { "norm": "circle", "x": "diamond", "y": "triangle", "z": "square" }
	SIGMA_COLORS = { 1: "#A6CAEC", 2: "#4E95D9" }

	for (x0, J01), rows in sorted(sheet_rows.items()):
		df = pd.DataFrame(rows).sort_values("A0").reset_index(drop=True)

		# add +/-1*sigma, +/-2*sigma band columns (computed values, not Excel formulas)
		# immediately after each corresponding *_sigma column
		for level in LEVELS:
			label = level_label(level)
			for axis in AXES:
				base = f"{label}_{axis}"
				loc = df.columns.get_loc(f"{base}_sigma")
				mean_vals = df[base].to_numpy()
				sigma_vals = df[f"{base}_sigma"].to_numpy()
				for offset, (coef, sign) in enumerate([(1, -1), (1, 1), (2, -1), (2, 1)], start=1):
					col_name = f"{base}{'-' if sign < 0 else '+'}{coef}sigma"
					df.insert(loc + offset, col_name, mean_vals + sign * coef * sigma_vals)

		sheet_name = excel_safe_sheet_name(f"x0={x0}, J01={J01}")
		df.to_excel(writer, sheet_name=sheet_name, index=False)
		worksheet = writer.sheets[sheet_name]

		col_A0 = df.columns.get_loc("A0")
		start_row, end_row = 1, len(df)  # +1 for header row baked into start_row already
		x_range = xl_range_formula(sheet_name, start_row, col_A0, end_row, col_A0)
		A0_min, A0_max = df["A0"].min(), df["A0"].max()

		col_charts = len(df.columns) + 1

		# --- Chart: M_norm comparison across all levels (global + regions), no error bars ---
		chart = writer.book.add_chart({ "type": "scatter" })
		for level, color in zip(LEVELS, LEVEL_COLORS):
			label = level_label(level)
			col_M = df.columns.get_loc(f"{label}_norm")
			y_range = xl_range_formula(sheet_name, start_row, col_M, end_row, col_M)
			chart.add_series({
				"name": label,
				"categories": x_range,
				"values": y_range,
				"marker": { "type": "circle", "border": { "color": color }, "fill": { "color": color }, "size": 5 },
				"line": { "dash_type": "solid", "color": color, "width": 1.5 },
			})
		chart.set_title({ "name": f"M_norm vs A0 -- layer comparison (x0={x0}, J01={J01})" })
		chart.set_x_axis({ "name": "A0", "position": "low", "min": A0_min, "max": A0_max })
		chart.set_y_axis({ "name": "M_norm", "position": "low" })
		chart.set_legend({ "position": "bottom" })
		chart.set_size(chart_size)
		worksheet.insert_chart(f"{xl_col_to_name(col_charts)}1", chart, { "x_offset": 0, "y_offset": 0 })

		# --- Chart grid: one chart per (level, axis), mean+SE error bars + sigma bands ---
		for li, level in enumerate(LEVELS):
			label = level_label(level)
			for ai, axis in enumerate(AXES):
				base = f"{label}_{axis}"
				col_mean = df.columns.get_loc(base)
				col_error = df.columns.get_loc(f"{base}_error")

				chart = writer.book.add_chart({ "type": "scatter" })

				y_range = xl_range_formula(sheet_name, start_row, col_mean, end_row, col_mean)
				error_range = xl_range_formula(sheet_name, start_row, col_error, end_row, col_error)
				chart.add_series({
					"name": base,
					"categories": x_range,
					"values": y_range,
					"y_error_bars": { "type": "custom", "plus_values": error_range, "minus_values": error_range },
					"marker": { "type": AXIS_SHAPES[axis], "border": { "color": "#000000" }, "fill": { "color": "#000000" }, "size": 5 },
					"line": { "dash_type": "solid", "color": "#000000", "width": 1.5 },
				})

				for coef, line_color in SIGMA_COLORS.items():
					for sign in ("-", "+"):
						band_col = f"{base}{sign}{coef}sigma"
						col_band = df.columns.get_loc(band_col)
						band_range = xl_range_formula(sheet_name, start_row, col_band, end_row, col_band)
						chart.add_series({
							"name": band_col,
							"categories": x_range,
							"values": band_range,
							"marker": { "type": "none" },
							"line": { "dash_type": "solid", "color": line_color, "width": 1.0 },
						})

				chart.set_title({ "name": f"{base} vs A0 (x0={x0}, J01={J01})" })
				chart.set_x_axis({ "name": "A0", "position": "low", "min": A0_min, "max": A0_max })
				chart.set_y_axis({ "name": base, "position": "low" })
				chart.set_legend({ "position": "bottom" })
				chart.set_size(chart_size)

				xl_col = xl_col_to_name(col_charts + ai * chart_col_width)
				xl_row = 1 + (li + 1) * chart_row_height  # +1 leaves row-slot 0 for the comparison chart
				worksheet.insert_chart(f"{xl_col}{xl_row}", chart, { "x_offset": 0, "y_offset": 0 })

	writer.close()


if __name__ == "__main__":
	out_dir = unique_dir(Path("out") / "study-layers-2")
	snapshot_path = out_dir / "snapshot.csv"
	main_path = out_dir / "main.xlsx"
	print(f"Output directory: {out_dir}")

	# combinations of config and task parameters (i.e. args)
	configs: list[tuple] = [(x0,) for x0 in range(5)]

	tasks_per_config: list[tuple] = []
	for J01 in [-1.0, -0.1, 0.0, 0.1, 1.0]:
		for _a0 in range(0, 201, 2):
			A0 = _a0 / 100  # e.g. 0, 0.02, 0.04, ..., 2.00
			tasks_per_config.append((J01, A0))

	snapshot_columns = [
		"x0", "J01", "A0", "x", "y", "z",
		"M_x", "M_y", "M_z",
		"M_x_sigma", "M_y_sigma", "M_z_sigma",
		"M_x_error", "M_y_error", "M_z_error",
	]
	pd.DataFrame(columns=snapshot_columns).to_csv(snapshot_path, index=False)

	sheet_rows: dict[tuple, list[dict]] = {}
	n_levels = 1 + len(REGIONS)

	with BRPE(build_config, configs, max_workers=max_workers, tool=tool) as exe:
		# submit tasks
		task_id = 0
		for config_args in configs:
			for J01, A0 in tasks_per_config:
				task_id += 1
				exe.submit(task, config_args, (J01, A0, task_id), use_cache=False)

		# process results
		pool_pbar = tqdm(total=len(exe.futures), desc="*** Simulating ***")
		for future in exe.as_completed():
			try:
				x0, J01, A0, task_id, tag, m_mean, m_std, m_se, m_norm_mean, m_norm_std, m_norm_se, chi, nodes = future.result()

				# ---- aggregate row -> main.xlsx sheet (x0, J01) ----
				row = { "A0": A0 }
				for li, level in enumerate(LEVELS):
					label = level_label(level)
					row[f"{label}_norm"] = m_norm_mean[li]
					for axis, ai in VECTOR_AXES:
						row[f"{label}_{axis}"] = m_mean[li, ai]
					row[f"{label}_norm_sigma"] = m_norm_std[li]
					for axis, ai in VECTOR_AXES:
						row[f"{label}_{axis}_sigma"] = m_std[li, ai]
					row[f"{label}_norm_error"] = m_norm_se[li]
					for axis, ai in VECTOR_AXES:
						row[f"{label}_{axis}_error"] = m_se[li, ai]
					row[f"Chi{'' if level == '' else '_' + level}"] = chi[li]  # bonus, remove if unwanted
				sheet_rows.setdefault((x0, J01), []).append(row)

				# ---- snapshot rows -> snapshot.csv (one row per node) ----
				snap_df = pd.DataFrame({
					"x0": x0, "J01": J01, "A0": A0,
					"x": nodes[:, 0], "y": nodes[:, 1], "z": nodes[:, 2],
					"M_x": m_mean[n_levels:, 0], "M_y": m_mean[n_levels:, 1], "M_z": m_mean[n_levels:, 2],
					"M_x_sigma": m_std[n_levels:, 0], "M_y_sigma": m_std[n_levels:, 1], "M_z_sigma": m_std[n_levels:, 2],
					"M_x_error": m_se[n_levels:, 0], "M_y_error": m_se[n_levels:, 1], "M_z_error": m_se[n_levels:, 2],
				})
				snap_df.to_csv(snapshot_path, mode="a", header=False, index=False)

			except Exception:
				traceback.print_exc()
			pool_pbar.update(1)
		pool_pbar.close()

	print("Writing main.xlsx ...")
	write_main_workbook(main_path, sheet_rows)
	print("Done.")
