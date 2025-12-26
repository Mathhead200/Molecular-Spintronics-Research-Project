#include <cstdlib>
#include <cstdio>
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
	uint64_t seed = 0;  // default sedd = 0
	if (argc > 1)
		seed = strtoull(argv[1], NULL, 10);
	mt19937_64 mt(seed);
	
	while (1) {
		uint64_t x = mt();
		fwrite(&x, 1, sizeof(uint64_t), stdout);
	}

	return 0;
}