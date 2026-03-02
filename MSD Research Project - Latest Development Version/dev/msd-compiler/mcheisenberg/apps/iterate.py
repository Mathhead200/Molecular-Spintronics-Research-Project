import sys
from mcheisenberg.io import IterateParameters
from mcheisenberg.util import report_datetime

def main(argv=sys.argv):
	in_file = "parameters-iterate.txt"
	if len(argv) > 1:
		in_file = argv[1]

	print(f"Reading {in_file}", flush=True)
	p = IterateParameters.load(in_file, verbose=True)

	print("_" * 80)
	print(f"({report_datetime()}) Running iterate...", flush=True)
	print("_" * 80)
	output_filename = p.run(dir=".", asm="iterate.asm", sim_progress_bar="iterate", out_progress_bar="Writing CSV")

	print(f"({report_datetime()}) Done: {output_filename}")

if __name__ == "__main__":
	main()
