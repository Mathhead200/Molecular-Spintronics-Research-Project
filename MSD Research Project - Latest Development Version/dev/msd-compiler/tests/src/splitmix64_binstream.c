// Calls my ASM implementation of xoshiro256 and outputs raw bytes
// to be piped to PracRand RNG_test.exe
// https://sourceforge.net/projects/pracrand/
// https://github.com/MartyMacGyver/PractRand

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// extern uint64_t xoshiro256ss(uint64_t state[4]);
// extern uint64_t xoshiro256pp(uint64_t state[4]);
// extern uint64_t xoshiro256p(uint64_t state[4]);
extern uint64_t splitmix64(uint64_t seed, uint64_t *state, uint64_t n);

#define BUF_SIZE 1048576  // 1 MB
uint64_t nums[BUF_SIZE] = {0};
uint64_t state = 0;  // default seed = 0

void swap(uint64_t *arr, size_t i, size_t j) {
	uint64_t temp = arr[j];
	arr[j] = arr[i];
	arr[i] = temp;
}

void rev_buf() {
	size_t half = BUF_SIZE / 2;
	for (size_t i = 0; i < half; i++) {
		size_t j = BUF_SIZE - 1 - i;
		swap(nums, i, j);
	}
}

int main(int argc, char* argv[]) {
	// seed the PNRG
	if (argc > 1)
		state = strtoull(argv[1], NULL, 10);

	while (1) {
		state = splitmix64(state, nums, BUF_SIZE);
		rev_buf();
		fwrite(nums, sizeof(uint64_t), BUF_SIZE, stdout);
	}

	return 0;
}
