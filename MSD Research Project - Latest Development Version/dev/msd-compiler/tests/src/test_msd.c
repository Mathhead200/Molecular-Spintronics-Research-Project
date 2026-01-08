#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

extern double *nodes;  // ASM nodes array

extern void metropolis(uint64_t iterations);
extern void seed();
extern BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

int main(int argc, char *argv[]) {
	DllMain(NULL, DLL_PROCESS_ATTACH, NULL);
	metropolis(7);
	printf("spin[0] = %f %f %f\n", nodes[0], nodes[1], nodes[2]);
	return 0;
}
