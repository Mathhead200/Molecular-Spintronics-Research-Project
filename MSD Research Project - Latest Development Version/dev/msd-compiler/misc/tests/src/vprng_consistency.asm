; Tests for consistancy between scalar and vectorized implementations of PRNG algos.
OPTION CASEMAP:NONE

include prng.inc
include vec.inc
include dumpreg.inc

.code
; (scalar) of algo, e.g. _xoshiro256ss
_tests MACRO algo, seed
	; seed -> PRNG state: r11, r12, r13, r14 
	mov rax, seed
	_splitmix64 r11, rax, r15
	_splitmix64 r12, rax, r15
	_splitmix64 r13, rax, r15
	_splitmix64 r14, rax, r15
	
	; generate -> temp: r15
	algo rax, r11, r12, r13, r14, r15  ; [ 0.] rax
	algo rcx, r11, r12, r13, r14, r15  ; [ 1.] rcx
	algo rdx, r11, r12, r13, r14, r15  ; [ 2.] rdx
	algo rbx, r11, r12, r13, r14, r15  ; [ 3.] rbx
	algo rsi, r11, r12, r13, r14, r15  ; [ 4.] rsi
	algo rdi, r11, r12, r13, r14, r15  ; [ 5.] rdi
	algo r8,  r11, r12, r13, r14, r15  ; [ 6.] r8
	algo r9,  r11, r12, r13, r14, r15  ; [ 7.] r9
	algo r10, r11, r12, r13, r14, r15  ; [ 8.] r10
	
	; output
	_dumpregg
ENDM

_testv MACRO algo, seed0, seed1, seed2, seed3
	; seed -> PRNG state: ymm11, ymm12, ymm13, ymm14
	mov rax, seed3
	_splitmix64 r8, rax, r15
	vpxor ymm11, ymm11, ymm11
	movq xmm11, r8
	_vpermj ymm11, ymm11
	_vpermk ymm11, ymm11  ; ymm11 = [0, 0, 0, s0(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm12, ymm12, ymm12
	movq xmm12, r8
	_vpermj ymm12, ymm12
	_vpermk ymm12, ymm12  ; ymm12 = [0, 0, 0, s1(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm13, ymm13, ymm13
	movq xmm13, r8
	_vpermj ymm13, ymm13
	_vpermk ymm13, ymm13  ; ymm13 = [0, 0, 0, s2(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm14, ymm14, ymm14
	movq xmm14, r8
	_vpermj ymm14, ymm14
	_vpermk ymm14, ymm14  ; ymm14 = [0, 0, 0, s3(seed3)]

	mov rax, seed2
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermk ymm0, ymm0       ; ymm0 = [0, 0, s0(seed2), 0]
	vpor ymm11, ymm11, ymm0  ; ymm11 = [0, 0, s0(seed2), s0(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermk ymm0, ymm0       ; ymm0 = [0, 0, s1(seed2), 0]
	vpor ymm12, ymm12, ymm0  ; ymm12 = [0, 0, s1(seed2), s1(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermk ymm0, ymm0       ; ymm0 = [0, 0, s2(seed2), 0]
	vpor ymm13, ymm13, ymm0  ; ymm13 = [0, 0, s2(seed2), s2(seed3)]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermk ymm0, ymm0     ; ymm0 = [0, 0, s3(seed2), 0]
	vpor ymm14, ymm14, ymm0  ; ymm14 = [0, 0, s3(seed2), s3(seed3)]
	
	mov rax, seed1
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermj ymm0, ymm0     ; ymm0 = [0, s0(seed1), 0, 0]
	vpor ymm11, ymm11, ymm0  ; ymm11 = [0, s0(seed1), ..., ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermj ymm0, ymm0     ; ymm0 = [0, s1(seed1), 0, 0]
	vpor ymm12, ymm12, ymm0  ; ymm12 = [0, s1(seed1), ..., ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermj ymm0, ymm0     ; ymm0 = [0, s2(seed1), 0, 0]
	vpor ymm13, ymm13, ymm0  ; ymm13 = [0, s2(seed1), ..., ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8
	_vpermj ymm0, ymm0     ; ymm0 = [0, s3(seed1), 0, 0]
	vpor ymm14, ymm14, ymm0  ; ymm14 = [0, s3(seed1), ..., ...]

	mov rax, seed0
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8          ; ymm0 = [s0(seed0), 0, 0, 0]
	vpor ymm11, ymm11, ymm0  ; ymm11 = [s0(seed0), ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8          ; ymm0 = [s1(seed0), 0, 0, 0]
	vpor ymm12, ymm12, ymm0  ; ymm12 = [s1(seed0), ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8          ; ymm0 = [s2(seed0), 0, 0, 0]
	vpor ymm13, ymm13, ymm0  ; ymm13 = [s2(seed0), ...]
	_splitmix64 r8, rax, r15
	vpxor ymm0, ymm0, ymm0
	movq xmm0, r8          ; ymm0 = [s3(seed0), 0, 0, 0]
	vpor ymm14, ymm14, ymm0  ; ymm14 = [s3(seed0), ...]

	; generate -> temp: ymm15
	algo ymm0,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 0.] ymm0
	algo ymm1,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 1.] ymm1
	algo ymm2,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 2.] ymm2
	algo ymm3,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 3.] ymm3
	algo ymm4,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 4.] ymm4
	algo ymm5,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 5.] ymm5
	algo ymm6,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 6.] ymm6
	algo ymm7,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 7.] ymm7
	algo ymm8,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 8.] ymm8
	algo ymm9,  ymm11, ymm12, ymm13, ymm14, ymm15  ; [ 9.] ymm9
	algo ymm10, ymm11, ymm12, ymm13, ymm14, ymm15  ; [10.] ymm10

	; output
	_dumpregv
ENDM

PUBLIC main
main PROC
	_tests _xoshiro256ss, 0
	_tests _xoshiro256ss, 42
	_tests _xoshiro256ss, 1234567
	_tests _xoshiro256ss, 200
	_testv _vxoshiro256ss, 0, 42, 1234567, 200

	_tests _xoshiro256pp, 0
	_tests _xoshiro256pp, 42
	_tests _xoshiro256pp, 1234567
	_tests _xoshiro256pp, 200
	_testv _vxoshiro256pp, 0, 42, 1234567, 200

	_tests _xoshiro256p, 0
	_tests _xoshiro256p, 42
	_tests _xoshiro256p, 1234567
	_tests _xoshiro256p, 200
	_testv _vxoshiro256p, 0, 42, 1234567, 200

	xor rax, rax  ; return 0
	ret
main ENDP
END