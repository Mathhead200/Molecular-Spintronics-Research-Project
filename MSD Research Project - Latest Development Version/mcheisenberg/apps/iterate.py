import argparse
import sys
from datetime import datetime
from pathlib import Path
from mcheisenberg import VisualStudio
from mcheisenberg.io import IterateParameters, DEFAULT_TOOL
from mcheisenberg.util import report_datetime

def main(argv=sys.argv):
	t = datetime.now()

	parser = argparse.ArgumentParser(description="Runs a sinlge MSD configuration to see how it evolves over time.")
	parser.add_argument("--in", dest="in_file", type=str, default="parameters-iterate.txt", help="Input file containing parameters")
	parser.add_argument("--out", dest="out_dir", type=str, default="out", help="Output directory for CSV")
	parser.add_argument("--temp", dest="temp_dir", type=str, default=None, help="Directory for temp files: .asm, .obj, .def, .dll")
	parser.add_argument("--asm", dest="asm", type=str, nargs="?", default=None, const=True, help="Save .asm file (for debugging)")
	parser.add_argument("--year", dest="year", type=int, default=None, help="Select versio of Visual Studio (e.g. 2022, 2026)")
	parser.add_argument("--edition", dest="edition", type=str, default=None, help="Select edition of Visual Studio (e.g. Community)")
	args = parser.parse_args(argv[1:])
	if args.edition is not None and args.year is None:
		parser.error("Specifying an --edition requires specifying a --year as well")
	if args.edition is None:
		args.edition = "Community"

	print(f"Reading {args.in_file}", flush=True)
	p = IterateParameters.load(args.in_file, verbose=True)

	print("_" * 80)
	print(f"({report_datetime()}) Running iterate...", flush=True)
	print("_" * 80)
	if args.asm is True:
		args.asm = "iterate.asm" if args.temp_dir is None else str(Path(args.temp_dir) / "iterate.asm")
	tool = DEFAULT_TOOL if args.year is None else VisualStudio(year=args.year, edition=args.edition)
	output_filename = p.run(tool=tool, out_dir=args.out_dir, temp_dir=args.temp_dir, asm=args.asm, sim_progress_bar="iterate", out_progress_bar="Writing CSV")

	print(f"({report_datetime()}) Done: {output_filename}")

	t = datetime.now() - t
	print(f"Total time: {t}")

if __name__ == "__main__":
	main()
