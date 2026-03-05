#include <stdio.h>
#include <immintrin.h>
#include <intrin.h>

static void print_cpuid_and_xcr0(void) {
    int cpuInfo[4];

    // Leaf 1: basic features
    __cpuid(cpuInfo, 1);
    int ecx1 = cpuInfo[2];
    int edx1 = cpuInfo[3];

    int avx_support   = (ecx1 >> 28) & 1; // CPUID.1:ECX[28]
    int osxsave       = (ecx1 >> 27) & 1; // CPUID.1:ECX[27]
    int fma_support   = (ecx1 >> 12) & 1; // CPUID.1:ECX[12]
    int sse_support   = (edx1 >> 25) & 1; // CPUID.1:EDX[25]

    printf("CPUID.1:ECX AVX   = %d\n", avx_support);
    printf("CPUID.1:ECX OSXSAVE = %d\n", osxsave);
    printf("CPUID.1:ECX FMA   = %d\n", fma_support);
    printf("CPUID.1:EDX SSE   = %d\n", sse_support);

    if (!osxsave) {
        printf("OSXSAVE is 0 -> OS is not using XSAVE/XRESTOR; AVX cannot be enabled.\n");
        return;
    }

    // Read XCR0 via XGETBV
    unsigned int eax = 0, edx = 0;
    unsigned long long xcr0 = _xgetbv(0);
    eax = (unsigned int)(xcr0 & 0xFFFFFFFFu);
    edx = (unsigned int)(xcr0 >> 32);

    int xcr0_x87 = (eax >> 0) & 1;
    int xcr0_sse = (eax >> 1) & 1;
    int xcr0_avx = (eax >> 2) & 1;

    printf("XCR0 = 0x%08X%08X\n", edx, eax);
    printf("XCR0[0] x87 = %d\n", xcr0_x87);
    printf("XCR0[1] SSE = %d\n", xcr0_sse);
    printf("XCR0[2] AVX = %d\n", xcr0_avx);
}

static void try_fma(void) {
    printf("\nAttempting FMA...\n");
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_fmadd_pd(a, b, a); // c = a*b + a
    double out[4];
    _mm256_storeu_pd(out, c);
    printf("FMA result: %f %f %f %f\n", out[0], out[1], out[2], out[3]);
}

int main(void) {
    print_cpuid_and_xcr0();
    try_fma();
    return 0;
}
