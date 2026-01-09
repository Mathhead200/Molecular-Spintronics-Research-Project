#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <windows.h>

extern double nodes[];  // ASM nodes array

extern void metropolis(uint64_t iterations);
extern void seed();
extern BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

// update
#define OFFSETOF_SPIN (32*0)
#define SIZEOF_NODE (32*1)
#define NODE_COUNT 10

int main(int argc, char *argv[]) {
	uint64_t count = 7;
	if (argc > 1)
		count = strtoull(argv[1], NULL, 10);

	DllMain(NULL, DLL_PROCESS_ATTACH, NULL);
	metropolis(count);

	printf("Nodes:\n");
	for (size_t n = 0; n < NODE_COUNT; n++) {
		size_t i = (n * SIZEOF_NODE + OFFSETOF_SPIN) / sizeof(double);
		printf("%4d: %f, %f, %f\n", n, nodes[i+0], nodes[i+1], nodes[i+2]);
	}
	return 0;
}
