#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

extern uint64_t xoshiro256ss(uint64_t state[4]);
extern uint64_t xoshiro256pp(uint64_t state[4]);
extern uint64_t xoshiro256p(uint64_t state[4]);
extern uint64_t* splitmix64(uint64_t seed, uint64_t *state, uint64_t n);

int main(int argc, char* argv[]) {
	uint64_t state[4] = {0};
	uint64_t seed = 0;  // default
	uint64_t n = 10;  // default

	if (argc > 1)
		n = strtoull(argv[1], NULL, 10);

	// seed the PNRG
	if (argc > 5) {
		state[0] = strtoull(argv[2], NULL, 10);
		state[1] = strtoull(argv[3], NULL, 10);
		state[2] = strtoull(argv[4], NULL, 10);
		state[3] = strtoull(argv[5], NULL, 10);
	} else {
		if (argc > 2) {
			seed = strtoull(argv[2], NULL, 10);
			printf("seed = %llu\n", seed);	
		} else {
			printf("(default) seed = %llu\n", 0);
		}
		splitmix64(seed, state, 4);
	}

	printf("Initial state: {%llu, %llu, %llu, %llu}\n", state[0], state[1], state[2], state[3]);
	for (uint64_t i = 0; i < 10; i++) {
		uint64_t x = xoshiro256ss(state);
		printf("[%2llu.] %llu\n", i, x);
	}

	return 0;
}
