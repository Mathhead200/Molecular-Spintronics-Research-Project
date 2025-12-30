#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern double ln(double w);

int main(int argc, char *argv[]) {
	if (argc <= 1) {
		printf("Provide numbers to test as program arguments.\n");
		return 0;
	}

	for (size_t i = 1; i < argc; i++) {
		double w = strtod(argv[i], NULL);
		printf("ln(%g):\n", w);
		if (w < 0.0 || w >= 1.0)
			printf(" !! Warning: out of range [0, 1)\n");
		printf("\tC:   %.17g\n", log(w));
		printf("\tASM: %.17g\n", ln(w));
	}

	return 0;
}