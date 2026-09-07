# Takes an iterate-style output XLSX (or CSV) file which contains multiple data series.
# Adds an extra sheet (or new CSV file) containing the normalized and non-normalized autocorrelation signals.
#
# Note: This program currently treats the samples as uniform in time (i.e. periodic sampling).
# TODO: add logic to handle non-periodic sampling.

from __future__ import annotations
import os
import sys
import threading
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from mcheisenberg.math import acf, ndindex_as_array
from pandas import DataFrame, ExcelWriter, read_csv, read_excel
from psutil import Process, IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS, ABOVE_NORMAL_PRIORITY_CLASS, HIGH_PRIORITY_CLASS, REALTIME_PRIORITY_CLASS
from tqdm import tqdm

_local = threading.local()
_lock = threading.Lock()
_next_worker_id = 1
_pbar_desc_width = 0

def compute_acf(df: DataFrame, pbar: tqdm):
	pbar_n = pbar.n
	pbar.total += len(df.columns)
	pbar.refresh()

	def col_pos(label: str):
		try:
			return df.columns.get_loc(label)
		except KeyError:
			return None

	acft = None  # output DataFrame
	acfx = None  # output DataFrame
	
	# spacial autocorrelation, e.g. (x, y, z) signals
	x_labels = None
	x_loc = col_pos("x")
	if type(x_loc) is int:
		x_labels =["x"]
		if "y" in df.columns:
			x_labels.append("y")
			if "z" in df.columns:
				x_labels.append("z")
	else:
		x_loc = col_pos("x_0")
		if type(x_loc) is int:
			x_labels =["x_0"]
			while True:
				next_label = f"x_{len(x_labels)}"
				if next_label not in df.columns:
					break
				x_labels.append(next_label)

	if x_labels is not None:
		xs       = df[x_labels].dropna().to_numpy(dtype=int)
		xs_notna = df[x_labels].notna().all(axis=1).to_numpy()  # which rows are not dropped (i.e. fully defined) in xs
		D = len(x_labels)  # number of spatial dimentions, i.e. number of independant spatial coordinates (i.e. "x" columns)
		# n = len(xs)        # expected number of spatial samples in each flattened sequence/signal
		offsets = np.min(xs, axis=0)                # (inclusive) start of sequence in each dimension; e.g. [0, 0, 0]
		lengths = np.max(xs, axis=0) + 1 - offsets  # size/length of sequence/signal in each dimension; e.g. [width, height, depth]
		# Assumption: spacial step size of 1

		# define the (likely irregular) boundaries with a mask, (1.0: defined/in-bounds, 0.0: undefined/out-of-bounds)
		spatial_mask = np.zeros(shape=lengths, dtype=float)
		for x in xs:
			spatial_mask[tuple(x - offsets)] = 1.0

		# get each spatial signals, flattened (as in CSV/Excel), convert to tensor, compute the ACF, then convert back
		flattened_acfxs = {}
		for col in df.columns[x_loc + D:]:
			flattened_signal = df[col]
			if all(flattened_signal.notna().to_numpy() == xs_notna):  # skip columns that are inconsistent with columns "x", ...
				flattened_signal = flattened_signal.dropna().to_numpy()

				# convert to multi-dimensional (e.g. 3D) signal (i.e. n-tensor) for scipy fftconvolve
				signal = np.zeros(shape=lengths, dtype=flattened_signal.dtype)  # pad missing samples with 0's
				for idx, x in enumerate(xs):  # for each batch of samples; i.e. row (idx) in DataFrame associated with spacial coordinates (x)
					signal[tuple(x - offsets)] = flattened_signal[idx]
				del flattened_signal

				# compute normalized autocorrelation signal for this signal
				signal = acf(signal, mask=np.asarray(spatial_mask, dtype=signal.dtype))

				# flatten output
				flattened_acfxs[col] = signal.ravel()  # store flattened signal as series with appropriate header
				del signal
			pbar.update(1)
		del col, idx, x
		del xs, xs_notna, offsets

		# convert to DataFrame where each row is a spacial coordinate, x, followed by an associated batch of samples
		# Note: all possible spatial shifts are represented, not those given as xs, even though we reuse the labels
		xs_all = ndindex_as_array(shape=lengths)
		acfx = DataFrame({
			**{ x_labels[i]: xs_all[:, i] for i in range(D) },
			**flattened_acfxs
		})														
		del xs_all
		pbar.update(D)
	del x_labels

	# temporal autocorrelation, i.e. time series signals
	t_loc = col_pos("t")
	if type(t_loc) is int:  # skip processing time series signals if there is no unambiguous "t" column 
		ts       = df["t"].dropna().to_numpy()
		ts_notna = df["t"].notna().to_numpy()
		n = len(ts)  # expected number of (temporal) samples in each signal/series
		# Assumption: ts and associated signals are in order
		step = ts[1] - ts[0]  # Assumption: regular sampling period

		acfts = {}
		tau_ints = {}  # also compute the \tau_int "integrated autocorrelation" sequence of partial sums for each temporal signal
		for col in df.columns[t_loc + 1:x_loc]:
			signal = df[col]
			if all(signal.notna().to_numpy() == ts_notna):  # skip columns that are inconsistent with column "t"
				signal = signal.dropna().to_numpy()

				# compute normalized autocorrelation signal for this signal
				signal = acf(signal)
				acfts[col] = signal
 
				# compute integrated autocorrelation as sequence of partial sums
				tau_ints[f"int({col})"] = step * (0.5 + np.cumsum(signal) - signal[0])
			pbar.update(1)

		# convert to DataFrame where each row is a temporal coordinate, t, followed by an associated batch of samples
		acft = DataFrame({
			"t": list(range(0, n * step, step)),
			**acfts,
			"i": list(range(n)),
			**tau_ints
		})
		pbar.update(1)

	# sync pbar in case of missing t or x section
	pbar.n = pbar_n + len(df.columns)
	pbar.refresh()

	return acft, acfx

def process_file(in_path: Path) -> str|None:
	""" Returns a str message on error, and None on success. """
	pbar = tqdm(total=2, desc=f"{in_path.name:{_pbar_desc_width}}", position=_local.TQDM_POS, leave=False)
	try:
		file_type = in_path.suffix.lower()

		if file_type == ".csv":
			if in_path.stem[:-1].endswith(".acf"):
				return f"Ignoring .acf file: {in_path.name}."
			
			df = read_csv(in_path)

			acft, acfx = compute_acf(df, pbar)

			if acft is not None:  acft.to_csv(in_path.with_suffix(".acft.csv"), index=False)
			del acft
			pbar.update(1)

			if acfx is not None:  acfx.to_csv(in_path.with_suffix(".acfx.csv"), index=False)
			del acfx
			pbar.update(1)

			return None

		if file_type == ".xlsx":
			df = read_excel(in_path)  # TODO: loop sheets?

			acft, acfx = compute_acf(df, pbar)

			with ExcelWriter(in_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
				if acft is not None:  acft.to_excel(writer, sheet_name="acft (Autocorrelation, Temporal)", index=False)
				del acft
				pbar.update(1)

				if acfx is not None:  acfx.to_excel(writer, sheet_name="acfx (Autocorrelation, Spacial)" , index=False)
				del acfx
				pbar.update(1)
			
			return None

		# else:
		return f"Ignoring unrecognized file: {in_path.name}. Supported types: xlsx, csv."
	
	except Exception as ex:
		# e.g. from pandas.errors import EmptyDataError, ParserError
		return f"Error processing file {in_path.name}: {ex}"

	finally:
		pbar.close()

def set_priority(chill: int) -> None:
	chill = min(max(chill, -3), 2)  # clamp
	Process(os.getpid()).nice({  # NOTE: assumes Windows
		-3: REALTIME_PRIORITY_CLASS,
		-2: HIGH_PRIORITY_CLASS,
		-1: ABOVE_NORMAL_PRIORITY_CLASS,
		0:  NORMAL_PRIORITY_CLASS,
		1:  BELOW_NORMAL_PRIORITY_CLASS,
		2:  IDLE_PRIORITY_CLASS
	}[chill])

def init_worker() -> None:
	global _next_worker_id
	with _lock:
		_local.TQDM_POS = _next_worker_id
		_next_worker_id += 1

if __name__ == "__main__":
	parser = ArgumentParser(description="Add autocorrelation signals to iterate file")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--in", dest="in_file", type=str, default=None, help="Input file (e.g. iteration, 07-31-2026, 1.csv)")
	group.add_argument("--dir", dest="dir", type=str, default=None, help="(Optional) Use instead of --in. Operates on all XLSX/CSV files in the directory.")
	parser.add_argument("--workers", dest="workers", type=int, default=None, help="(Optional) Number of sub-process workers to use when paralell processing a full directory.")
	parser.add_argument("--chill", dest="chill", type=int, nargs="?", const=1, default=None, help="(Optional) Have the script run at a lower priority")
	args = parser.parse_args(sys.argv[1:])
	if args.dir is None and args.workers is not None:       parser.error("--workers can only be used with --dir")

	if args.chill is not None:
		set_priority(args.chill)

	if args.in_file is not None:
		init_worker()
		msg = process_file(Path(args.in_file))
		if msg is not None:
			print(msg)

	elif args.dir is not None:
		files = [
			path
			for path in Path(args.dir).iterdir()
			if path.is_file()
			and path.suffix.lower() in {".csv", ".xlsx"}
			and not path.stem[:-1].endswith(".acf")  # e.g. .acft or .acfx
		]
		if files:
			_pbar_desc_width = max(len(path.name) for path in files)  # global
		with ThreadPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=()) as exe:
			futures = []
			logs = []

			for file in files:
				futures.append(exe.submit(process_file, file))

			for future in tqdm(as_completed(futures), total=len(files), desc=f"Processing files in {args.dir}", position=0):
				msg = future.result()
				if msg is not None:
					logs.append(msg)

			for msg in logs:
				print(msg)
			print(f"{len(files) - len(logs)} file(s) processed.")
