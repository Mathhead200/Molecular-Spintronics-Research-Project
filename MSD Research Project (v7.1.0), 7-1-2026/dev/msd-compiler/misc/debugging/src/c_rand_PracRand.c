// Calls C's stdlib rand and outputs raw bytes
// to be piped to PracRand RNG_test.exe
// https://sourceforge.net/projects/pracrand/
// https://github.com/MartyMacGyver/PractRand

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

const uint64_t N = 1 * 1024 * 1024 * 1024 / sizeof(uint64_t);  // maximum number of numbers

int main(int argc, char* argv[]) {
	unsigned seed = 0;  // default seed = 0

	// seed the PNRG
	if (argc > 1)
		seed = (unsigned) strtoull(argv[1], NULL, 10);
	srand(seed);

	freopen(NULL, "wb", stdout);  // Necessary on Windows. (Harmless on POSIX.)
	FILE *dump = fopen("dump.txt", "w");
	for (uint64_t i = 0; i < N; i++) {
		uint64_t x0 = (uint64_t) rand();
		uint64_t x1 = (uint64_t) rand();
		uint64_t x = (x1 << 32) | x0;
		fwrite(&x, sizeof(x), 1, stdout);
		fprintf(dump, "%016llu\n", x);
	}
	fclose(dump);
	fclose(stdout);

	return 0;
}
