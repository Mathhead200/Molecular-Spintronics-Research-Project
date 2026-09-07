; Look-up table + Taylor series interpolation to approximate ln(w) with 0 <= w < 1
OPTION CASEMAP:NONE

include vec.inc
include dumpreg.inc

M_TRUNC		EQU 2  ; Mantissa trancation length (in bits)
TAYLOR_ITER	EQU 3  ; Taylor seriers iterations

.data
TAYLOR SEGMENT ALIGN(32) 
				; a, f(a)=ln(a), f'(a)=1/a, f''(a)/2!=-1/(2a^2)
taylor_table dq	0.25, -1.38629436111989061883, 4.0, -8.0,
				0.5, -0.69314718055994530941, 2.0, -2.0,
				0.75, -0.28768207245178092743, 1.33333333333333333333, -0.88888888888888888889,
				1.0, 0.0, 1.0, -0.5
TAYLOR ENDS

.code
; @param (xmm0) - double w in [0, 1)
; @return (xmm0)
PUBLIC ln
ln PROC
	; w = (bits) sxxx xxxx xxxx mmmm, mmmm mmmm mmmm mmmm, ...
	;   =        0011 1111 1111 mmmm, mmmm mmmm mmmm mmmm, ...
	;           MSb of frac. -> ^^^^, ^^^^ ...
	; Exact number of MSb (most sig. bits) will depend on look-up table size
;	_dumpreg

	; compute address offset for look-up table
	vmovq rax, xmm0
	shl rax, 12  ; mask sign (1 bit) + exponent (11 bits)
	shr rax, 64 - M_TRUNC  ; mask all but MSb of mantissa and move into low position
	shl rax, 5  ; rax *= 32 (sizeof ymmword ptr)
	_dumpreg
	
	; load pre-computed Taylor terms: a, f(a), f'(a), etc.
	lea rcx, taylor_table
	vmovapd ymm1, ymmword ptr [rcx + rax]  ; [a, f(a), f'(a), f''(a)/2!]
	_dumpreg

	vsubsd xmm2, xmm0, xmm1  ; xmm2 = w - a
	_dumpreg

	_vpermj ymm0, ymm1  ; ymm0 = [f(a), a, f'(a), f''(a)/2!] -> xmm0 = f(a)
	_dumpreg
	
	_vpermk ymm1, ymm0  ; ymm1 = [f'(a), f''(a)/2!, f(a), a] -> xmm1 = f'(a)
	vfmadd231sd xmm0, xmm1, xmm2  ; xmm0 += xmm1 * xmm2 -> xmm0 = f(a) + f'(a)*(w-a)
	_dumpreg

	
	vmul_vpermj ymm1, ymm1  ; ymm1 = [f''(a)/2!, f'(a), f(a), a] -> xmm1 = f''(a)/2!sd xmm3, xmm2, xmm2  ; xmm3 *= xmm2 -> xmm2 = (w - a)^2
	vfmadd231sd xmm0, xmm1, xmm3
	_dumpreg

	ret
ln ENDP
END
