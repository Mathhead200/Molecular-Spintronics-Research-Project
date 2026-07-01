// Calls my ASM implementation of xoshiro256 and outputs raw bytes
// to be piped to PracRand RNG_test.exe
// https://sourceforge.net/projects/pracrand/
// https://github.com/MartyMacGyver/PractRand

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Windows CRLF fix for stdout in text mode
#include <fcntl.h>
#include <io.h>

// extern uint64_t xoshiro256ss(uint64_t state[4]);
extern uint64_t xoshiro256pp(uint64_t state[4]);
// extern uint64_t xoshiro256p(uint64_t state[4]);
extern uint64_t splitmix64(uint64_t seed, uint64_t *state, uint64_t n);

int main(int argc, char* argv[]) {
	uint64_t state[4] = {0};

	enum { BIN, TXT } mode = BIN;
	size_t txt_n = 10;  // number of nums to print in TXT mode
	size_t seed_argv_index = 1;  // seed is assumed to be the 1st arg unless BIN or TXT mode is specified

	if (argc > 1) {
		const char * arg = argv[1];
		if (strcmp(arg, "BIN") == 0 || strcmp(arg, "bin") == 0) {
			mode = BIN;
			seed_argv_index = 2;
		} else if (strcmp(arg, "TXT") == 0 || strcmp(arg, "txt") == 0) {
			mode = TXT;
			if (argc > 2) {
				txt_n = (size_t) strtoull(argv[2], NULL, 10);
				seed_argv_index = 3;
			} else {
				seed_argv_index = 2;  // -> seed_len = 0
			}
		}
	}

	// seed the PNRG
	size_t seed_len = argc - seed_argv_index;  // e.g. prgm BIN 42 -> argc = 3, seed_argv_index = 2 -> seed_len = 1
	if (seed_len >= 4) {
		state[3] = strtoull(argv[seed_argv_index + 0], NULL, 10);
		state[2] = strtoull(argv[seed_argv_index + 1], NULL, 10);
		state[1] = strtoull(argv[seed_argv_index + 2], NULL, 10);
		state[0] = strtoull(argv[seed_argv_index + 3], NULL, 10);
	} else if (seed_len >= 1) {
		uint64_t seed = strtoull(argv[seed_argv_index], NULL, 10);
		splitmix64(seed, state, 4);
	} else {
		splitmix64(0, state, 4);  // default seed = 0
	}

	if (mode == BIN) {
		_setmode(_fileno(stdout), _O_BINARY);  // Windows CRLF fix for stdout in text mode
		while (1) {
			uint64_t x = xoshiro256pp(state);
			fwrite(&x, sizeof(uint64_t), 1, stdout);
		}

	} else if (mode == TXT) {
		printf("Initial state: {%llu, %llu, %llu, %llu}\n", state[0], state[1], state[2], state[3]);
		for (int i = 0; i < txt_n; i++) {
			uint64_t x = xoshiro256pp(state);
			printf("[%2d.] 0x %016llX == %llu\n", i, x, x);
		}
	} else {
		return 1;  // shouldn't happen
	}

	return 0;
}
