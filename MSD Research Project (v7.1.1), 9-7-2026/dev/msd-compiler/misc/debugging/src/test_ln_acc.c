#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

extern double ln(double w);
// extern double ln2(double w);
// extern double lns(double w, uint64_t iteration);

int main(int argc, char *argv[]) {
	if (argc <= 1) {
		printf("Provide numbers to test as program arguments.\n");
		return 0;
	}

	for (size_t i = 1; i < argc; i++) {
		double w = strtod(argv[i], NULL);
		printf("ln(%g):\n", w);
		if (w < 1.0 || w >= 2.0)
			printf(" !! Warning: out of range [0, 1)\n");
		double y = log(w);
		double y1 = ln(w);
		// double y2 = ln2(w);
		printf("\tC:    %.17g\n", y);
		printf("\tASM:  %.17g, error: %.17g\n", y1, (y - y1));
		// printf("\tASM2: %.17g, error: %.17g\n", y2, (y - y2));
		// printf("\tASMs: %.17g\n", lns(w, 1ull));
	}

	return 0;
}