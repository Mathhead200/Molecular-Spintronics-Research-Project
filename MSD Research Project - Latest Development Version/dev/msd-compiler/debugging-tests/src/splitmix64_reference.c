#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t x = 0;  // default seed = 0 (internal state)

// splitmix64 for seeding
uint64_t splitmix64_next() {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void stream() {
	while (1) {
		uint64_t num = splitmix64_next();
		fwrite(&num, sizeof(uint64_t), 1, stdout);
	}
}

void text(size_t N) {
	while(N-- > 0)
		printf("%llu\n", splitmix64_next());
}

int main(int argc, char* argv[]) {
	// seed the PNRG
	if (argc > 1)
		x = strtoull(argv[1], NULL, 10);
	
	// prgm mode
	enum { BIN, TXT } mode = BIN;
	size_t N = 10;  // default value (TXT mode)
	if (argc > 2) {
		const char *arg = argv[2];
		if (strcmp(arg, "bin") == 0 || strcmp(arg, "BIN") == 0) {
			mode = BIN;
		} else if (strcmp(arg, "txt") == 0 || strcmp(arg, "TXT") == 0) {
			mode = TXT;
			if (argc > 3)
				N = (size_t) strtoull(argv[3], NULL, 10);
		}
		else {
			fprintf(stderr, "Unrecognized argument: %s\n", arg);
			return 1;
		}
	}

	// execute
	if (mode == BIN)
		stream();
	else if (mode == TXT)
		text(N);
	else
		return 1; // this shouldn't happen

	return 0;
}