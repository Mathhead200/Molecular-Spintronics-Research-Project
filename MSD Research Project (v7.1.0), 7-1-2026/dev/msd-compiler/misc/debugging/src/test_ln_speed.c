#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

extern double lns(double w, uint64_t iterations);

int main(int argc, char *argv[]) {
	uint64_t n = 1234567890ull;
	time_t t1, t2;
	const double x = 1.1;
	volatile double sink;
	
	if (argc > 1)
		n = strtoull(argv[1], NULL, 10);

	time(&t1);
	for (uint64_t i = 0; i < n; i++)
		sink = log(x);
	time(&t2);
	printf("C time: %llu s\n", t2 - t1);

	time(&t1);
	sink = lns(x, n);
	time(&t2);
	printf("ASM time: %llu s\n", t2 - t1);

	return 0;
}