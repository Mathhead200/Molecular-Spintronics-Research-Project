from mcheisenberg.io import IterateParameters

if __name__ == "__main__":
	print("Reading parameters-iterate.txt...")
	p = IterateParameters.load("parameters-iterate.txt", verbose=True)

	print("Running iterate...")
	output_filename = p.run(dir=".", asm="iterate.asm")

	print("Done:", output_filename)
