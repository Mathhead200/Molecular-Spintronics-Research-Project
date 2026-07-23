import sys
import argparse
from pathlib import Path
from math import ceil
import pandas as pd
from xlsxwriter.utility import xl_col_to_name, xl_range_formula
from tqdm import tqdm

regions, region_colors = ["M", "M0", "M1", "ML", "MR", "Mm"], ["#156082", "#E97132", "#196B24", "#0F9ED5", "#A02B93", "#4EA72E"]
axises, axis_shapes = ["", "_x", "_y", "_z"], ["circle", "diamond", "triangle", "square"]

def filter_df(df: pd.DataFrame, col, val):
	return df.loc[ df[col] == val, [c for c in df.columns if c != col] ]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="POST-PROCESSING: dynamically generate Excel plots")
	parser.add_argument("--csv", dest="csv", type=str, help="Output from study_layers")
	args = parser.parse_args(sys.argv[1:])
	
	csv_in = Path(args.csv)
	if not csv_in.is_file():
		raise TypeError(f"CSV path must be file: {csv_in}")
	out_dir = csv_in.parent
	
	df = pd.read_csv(csv_in)
	
	widths = sorted(df["width"].unique())
	kTs = sorted(df["kT"].unique())
	J01s = sorted(df["J01"].unique())
	As = sorted(df["A"].unique())
	N0 = len(widths)
	N1 = len(kTs)
	N2 = len(J01s)
	N3 = len(As)
	N = N0 * N1 * N2 * N3

	JmLs = sorted(df["JmL"].unique())
	default_cell_width = 65
	default_cell_height = 20
	chart_width = 7 * default_cell_width
	chart_height = min(max(len(JmLs) - 1, 9), 16) * default_cell_height
	chart_size = { "width": chart_width, "height": chart_height }
	chart_col_width = ceil(chart_width / default_cell_width)  # approximate number of columns per chart
	chart_row_height = ceil(chart_height / default_cell_height)  # approximate number of rows per chart
	JmL_min = min(JmLs)
	JmL_max = max(JmLs)

	progress_bar = tqdm(total=N + (N // N0) + (N // N1) + (N // N2) + (N // N3), desc="Plotting")

	# 0. (Single) Scatter for each parameter combination with 1*sigma and 2*sigma bars to show variance
	print("0. (Single) Scatter for each parameter combination with 1*sigma and 2*sigma bars to show variance")
	sheet_name = "All data"
	
	out = csv_in.with_suffix(".xlsx")
	writer = pd.ExcelWriter(out, engine="xlsxwriter")
	data = df.copy(deep=True)  # copy data before adding sigma bands
	
	for region in regions:  # add []-2*sigma, []-1*sigma, [mean], []+1*sigma, and []+2*sigma columns inserted after mean and sigma column for M, M_x, M_y, M_z, ML_x, etc.
		for axis in axises:
			region_axis = region + axis
			loc = data.columns.get_loc(f"{region_axis}_sigma")
			data.insert(loc + 1, f"{region_axis}-1*sigma", "")
			data.insert(loc + 2, f"{region_axis}+1*sigma", "")
			data.insert(loc + 3, f"{region_axis}-2*sigma", "")
			data.insert(loc + 4, f"{region_axis}+2*sigma", "")
	
	data.sort_values(["width", "kT", "J01", "A", "JmL"], inplace=True)
	data.reset_index(drop=True, inplace=True)
	
	data.to_excel(writer, sheet_name=sheet_name, index=False)
	worksheet = writer.sheets[sheet_name]

	# add formula for sigma bands
	print("add formula for sigma bands")
	for region in regions:
		for axis in axises:
			region_axis = region + axis
			col_M = data.columns.get_loc(region_axis)
			col_M_sigma = data.columns.get_loc(f"{region_axis}_sigma")
			for coef, op in [(1, ""), (2, "2*")]:
				for sign in ["-", "+"]:
					col_M_sigma_band = data.columns.get_loc(f"{region_axis}{sign}{coef}*sigma")  # destinaion for the sigma band formula, e.g. M-1*sigma, M+2*sigma, etc.
					for pd_row in range(1, len(data) + 1):  # (row index in pandas) skip headers on row 0
						xl_row = pd_row + 1                 # (row # in excel) +1 because pandas is 0 indexed, but excel is 1 indexed
						mean_cell  = f"{xl_col_to_name(col_M      )}{xl_row}"
						sigma_cell = f"{xl_col_to_name(col_M_sigma)}{xl_row}"
						worksheet.write_formula(pd_row, col_M_sigma_band, f"={mean_cell}{sign}{op}{sigma_cell}")  # e.g. =A2-B2, =A5+2*B5

	col_JmL = data.columns.get_loc("JmL")
	col_charts = len(data.columns) + 1  # where to start placing the charts

	for width in widths:
		sub_w = filter_df(data, "width", width)
		for kT in kTs:
			sub_kT = filter_df(sub_w, "kT", kT)
			for J01 in J01s:
				sub_J01 = filter_df(sub_kT, "J01", J01)
				for A in As:
					sub = filter_df(sub_J01, "A", A)
					# assert not sub.empty  # TODO: what if it is?
					if sub.empty:
						continue

					start_row = sub.index.min() + 1
					end_row   = sub.index.max() + 1
					x_range = xl_range_formula(sheet_name, start_row, col_JmL, end_row, col_JmL)
					
					xl_row = start_row + 1  # Excel version of start_row (excel is 1-based, pandas is 0-based)
					chart_count = 0

					# all regions on one graph
					for axis, shape in zip(axises, axis_shapes):
						chart = writer.book.add_chart({ "type": "scatter" })

						for region, color in zip(regions, region_colors):
							region_axis = region + axis
							
							col_M = data.columns.get_loc(region_axis)
							col_M_error = data.columns.get_loc(f"{region_axis}_error")
							y_range = xl_range_formula(sheet_name, start_row, col_M, end_row, col_M)
							error_range = xl_range_formula(sheet_name, start_row, col_M_error, end_row, col_M_error)

							chart.add_series({
								"name": region_axis,
								"categories": x_range,
								"values": y_range,
								"y_error_bars": {
									"type": "custom",
									"plus_values":  error_range,
									"minus_values": error_range
								},
								"marker": {
									"type": shape,
									"border": { "color": color },
									"fill":   { "color": color },
									"size": 5
								},
								"line": {
									"dash_type": "solid",
									"color": color,
									"width": 1.5
								}
							})

						chart.set_title({ "name": f"M{axis} vs. JmL curves by region ({width}x10x10, kT={kT}, J01={J01}, A={A})" })
						chart.set_x_axis({
							"name": "JmL",
							"position": "low",
							"min": JmL_min,
							"max": JmL_max
						})
						chart.set_y_axis({
							"name": f"M{axis}",
							"position": "low"
						})
						chart.set_legend({ "position": "bottom" })
						chart.set_size(chart_size)

						xl_col = xl_col_to_name(col_charts + chart_count * chart_col_width)  # right of data
						chart_count += 1
						worksheet.insert_chart(f"{xl_col}{xl_row}", chart, { "x_offset": 0, "y_offset": 0 })

					# individual axis graphs for each region
					for region, color in zip(regions, region_colors):
						for axis, shape in zip(axises, axis_shapes):
							region_axis = region + axis
							chart = writer.book.add_chart({ "type": "scatter" })

							col_M = data.columns.get_loc(region_axis)
							col_M_error = data.columns.get_loc(f"{region_axis}_error")
							y_range = xl_range_formula(sheet_name, start_row, col_M, end_row, col_M)
							error_range = xl_range_formula(sheet_name, start_row, col_M_error, end_row, col_M_error)

							chart.add_series({
								"name": region_axis,
								"categories": x_range,
								"values": y_range,
								"y_error_bars": {
									"type": "custom",
									"plus_values":  error_range,
									"minus_values": error_range
								},
								"marker": {
									"type": shape,
									"border": { "color": color },
									"fill":   { "color": color },
									"size": 5
								},
								"line": {
									"dash_type": "solid",
									"color": color,
									"width": 1.5
								}
							})

							for sigma, line_color in [("1*sigma", "#A6CAEC"), ("2*sigma", "#4E95D9")]:
								for sign in ["-", "+"]:
									label = f"{region_axis}{sign}{sigma}"
									col_M_sigma = data.columns.get_loc(label)
									chart.add_series({
										"name": label,
										"categories": x_range,
										"values": [sheet_name, start_row, col_M_sigma, end_row, col_M_sigma],  # y_range
										"marker": {
											"type": "none"  # just lines for sigma bands
										},
										"line": {
											"dash_type": "solid",
											"color": line_color,
											"width": 1.5
										}
									})
							
							chart.set_title({ "name": f"{region_axis} vs. JmL ({width}x10x10, kT={kT}, J01={J01}, A={A})" })
							chart.set_x_axis({
								"name": "JmL",
								"position": "low",
								"min": JmL_min,
								"max": JmL_max
							})
							chart.set_y_axis({
								"name": region_axis,
								"position": "low"
							})
							chart.set_legend({ "position": "bottom" })
							chart.set_size(chart_size)

							xl_col = xl_col_to_name(col_charts + chart_count * chart_col_width)
							chart_count += 1
							worksheet.insert_chart(f"{xl_col}{xl_row}", chart, { "x_offset": 0, "y_offset": 0 })

					progress_bar.update(1)
	writer.close()

	# 1. compare M vs. JmL curves for different widths (fix kT, J01, and A)
	print("1. compare M vs. JmL curves for different widths (fix kT, J01, and A)")
	sub_dir = out_dir / f"{csv_in.stem}-more"
	sub_dir.mkdir(exist_ok=True)

	for kT in kTs:
		sub_kT = filter_df(df, "kT", kT)
		for J01 in J01s:
			sub_J01 = filter_df(sub_kT, "J01", J01)
			for A in As:
				data = filter_df(sub_J01, "A", A)
				data.sort_values(["width", "JmL"], inplace=True)
				data.reset_index(drop=True, inplace=True)

				sheet_name = f"kT={kT}, J01={J01}, A={A}"
				writer = pd.ExcelWriter(sub_dir / f"width, {sheet_name}.xlsx", engine="xlsxwriter")
				data.to_excel(writer, sheet_name=sheet_name, index=False)
				worksheet = writer.sheets[sheet_name]

				col_JmL = data.columns.get_loc("JmL")
				col_charts = len(data.columns) + 1  # where to start placing the charts

				for idx_a, axis in enumerate(axises):
					for idx_r, region in enumerate(regions):
						chart = writer.book.add_chart({ "type": "scatter" })  # TODO: connect points with straight lines
						
						col_M = data.columns.get_loc(region)
						col_M_error = data.columns.get_loc(f"{region}_error")

						colors_and_shapes = [("#000000", "square"), ("#C00000", "circle"), ("#00B050", "triangle"), ("#50164A", "star"), ("#00B0F0", "diamond")]
						for width, color, shape in [(w, cs[0], cs[1]) for w, cs in zip(widths, colors_and_shapes)]:
							sub = data[data["width"] == width]
							# assert not sub.empty  # TODO: what if it is?
							if sub.empty:
								continue

							start_row = sub.index.min() + 1  # +1 becasue of header row (TODO: explain we need the +1)
							end_row   = sub.index.max() + 1
							x_range = xl_range_formula(sheet_name, start_row, col_JmL, end_row, col_JmL)

							y_range = xl_range_formula(sheet_name, start_row, col_M, end_row, col_M)
							error_range = xl_range_formula(sheet_name, start_row, col_M_error, end_row, col_M_error)
							
							chart.add_series({
								"name": f"{width}x10x10",
								"categories": x_range,
								"values": y_range,
								"y_error_bars": {
									"type": "custom",
									"plus_values":  error_range,
									"minus_values": error_range
								},
								"marker": {
									"type": shape,
									"border": { "color": color },
									"fill":   { "color": color },
									"size": 5
								},
								"line": {
									"dash_type": "solid",
									"color": color,
									"width": 1.5
								}
							})
						chart.set_title({ "name": f"M{region}{axis} vs. JmL curves for different widths" })
						chart.set_x_axis({
							"name": "JmL",
							"position": "low",
							"min": JmL_min,
							"max": JmL_max
						})
						chart.set_y_axis({
							"name": f"M{region}{axis}",
							"position": "low"
						})
						chart.set_legend({ "position": "bottom" })
						chart.set_size(chart_size)
						xl_col = xl_col_to_name(col_charts + idx_a * chart_col_width)
						xl_row = 1 + idx_r * chart_row_height
						worksheet.insert_chart(f"{xl_col}{xl_row}", chart, { "x_offset": 0, "y_offset": 0 })

				writer.close()
				progress_bar.update(1)
	
	# TODO: 2. compare M vs. JmL curves for different kT (fix width, J01, and A)
	progress_bar.update(N // N1)

	# TODO: 3. compare M vs. JmL curves for different J01 (fix width, kT, and A)
	progress_bar.update(N // N2)

	# TODO: 4. compare M vs. JmL curves for different A (fix width, kT, J01)
	progress_bar.update(N // N3)

	progress_bar.close()
