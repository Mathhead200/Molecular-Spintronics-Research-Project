
; Vectorized Rotate Left on 64 bit elements (i.e. packed quad-words, or double)
; dest, src, temp are all XMMn, YMMn, or ZMMn registers
; imm is an immediate, the shift amount
; (NOTE: AVX512 has VPROLQ)
; Preconditions:
;	1. src and dest can be the same,
;		but temp must be unique from both.
; Postconditions:
;	1. temp is clobbered!
_vrol MACRO dest, src, imm, temp
	vpsllq temp, src, imm  ; left shift (vector packed shift left literal quadword)
	vpsrlq dest, src, 64 - imm  ; right shift
	vpor dest, dest, temp
ENDM

; (Fig. 4, p. 5, Blackman & Vigna)
; S = 5, R = 7, T = 9 (Table 3, p. 10, Blackman & Vigna)
_xoshiro256ssscrambler MACRO dest, s0, s1, s2, s3
	; const uint_t result_starstar = rotl(s[1] * S, R) * T;
	lea dest, [s1 + s1*4]      ; dest = s1 * 5
	rol dest, 7
	lea dest, [dest + dest*8]  ; dest *= 9
ENDM

; Vectorized version of _xoshiro256ssscrambler
_vxoshiro256ssscrambler MACRO dest, s0, s1, s2, s3, t
	vpsllq dest, s1, 2     ; dest = s[1] * 4;
	vpaddq dest, dest, s1  ; dest += s[1];  // dest == s[1] * 5
	_vrol dest, dest, 7, t
	vpsllq t, dest, 3      ; t = dest * 8;
	vpaddq dest, dest, t   ; dest' = t + dest;  // dest' == dest * 9
ENDM

; (Fig. 4, p. 5, Blackman & Vigna)
; R = 23 (Table 3, p. 10, Blackman & Vigna)
_xoshiro256ppscrambler MACRO dest, s0, s1, s2, s3
	; const uint_t result_plusplus = rotl(s[0] + s[3], R) + s[0];
	mov dest, s0
	add dest, s3
	rol dest, 23
	add dest, s0
ENDM

; Vectorized version of _xoshiro256ppscrambler
_vxoshiro256ppscrambler MACRO dest, s0, s1, s2, s3, t
	vpaddq dest, s0, s3
	_vrol dest, dest, 23, t
	vpaddq dest, dest, s0
ENDM

; (Fig. 4, p. 5, Blackman & Vigna)
_xoshiro256pscrambler MACRO dest, s0, s1, s2, s3
	; const uint_t result_plus = s[0] + s[3];
	mov dest, s0
	add dest, s3
ENDM

; Vectorized version of _xoshiro256pscrambler
_vxoshiro256pscrambler MACRO dest, s0, s1, s2, s3
	vpaddq dest, s0, s3
ENDM

; (Fig. 4, p. 5, Blackman & Vigna)
; A = 17, B = 45 (Table 2, p. 10, Blackman & Vigna)
_xoshiro256engine MACRO s0, s1, s2, s3, t
	mov t, s1   ; const uint64_t t = s[1] << A;
	shl t, 17
	xor s2, s0  ; s[2] ^= s[0];
	xor s3, s1  ; s[3] ^= s[1];
	xor s1, s2  ; s[1] ^= s[2];
	xor s0, s3  ; s[0] ^= s[3];
	xor s2, t   ; s[2] ^= t;
	rol s3, 45  ; s[3] = rotl(s[3], B);
ENDM

; Vectorized version of _xoshiro256engine
_vxoshiro256engine MACRO s0, s1, s2, s3, t
	vpsllq t, s1, 17       ; const uint64_t t = s[1] << A;
	vpxor s2, s2, s0       ; s[2] ^= s[0];
	vpxor s3, s3, s1       ; s[3] ^= s[1];
	vpxor s1, s1, s2       ; s[1] ^= s[2];
	vpxor s0, s0, s3       ; s[0] ^= s[3];
	vpxor s2, s2, t        ; s[2] ^= t;
	_vrol s3, s3, 45, t    ; s[3] = rotl(s[3], B);
ENDM

; For uniform uint64 (p. 11, Blackman & Vigna)
; dest, s0, s1, s2, s3, t are all general purpose 64-bit registers.
; s0, s1, s2, s3 store the state of the previous call to xshiro256ss,
; or the seed initially.
; t gets clobbered!
; Preconditions:
;	* TODO: ...
_xoshiro256ss MACRO dest, s0, s1, s2, s3, t
	_xoshiro256ssscrambler dest, s0, s1, s2, s3
	_xoshiro256engine s0, s1, s2, s3, t
ENDM

; For uniform uint64 (p. 11, Blackman & Vigna)
_xoshiro256pp MACRO dest, s0, s1, s2, s3, t
	_xoshiro256ppscrambler dest, s0, s1, s2, s3
	_xoshiro256engine s0, s1, s2, s3, t
ENDM

; For uniform doubles in [0, 1) (p. 11, Blackman & Vigna)
_xoshiro256p MACRO dest, s0, s1, s2, s3, t
	_xoshiro256pscrambler dest, s0, s1, s2, s3
	_xoshiro256engine s0, s1, s2, s3, t
ENDM

; Vectorized verion of _xoshiro256ss
;
; Each AVX register holds 2 (AVX), 4 (AVX2), or 8 (AVX512) uint64 numbers
; 	and represents a independant PRNG stream. As such, they should be seeded
;	differently. i.e. for AVX2 (YMMn registers):
;	s0   = [s0[0], s0[1], s0[2], s0[3]]
;	s1   = [s1[0], s1[1], s1[2], s1[3]]
;	...
;	dest = [dest[0], dest[1], dest[2], dest[3]]
;	
;	s0[0], s1[0], s2[0], s3[0] (first column) represent the state of the first stream.
;	s0[1], s1[1], s2[1], s3[1] (second column) represet th state of the decond stream.
;	etc.
;
;	dest[0] is the psuedo-random result from the first stream,
;	dest[1] is the psuedo-random result from the second stream,
;	etc.
;
; dest, s0, s1, s2, s3, t are AVX registers (XMMn/YMMn/ZMMn)
_vxoshiro256ss MACRO dest, s0, s1, s2, s3, t
	_vxoshiro256ssscrambler dest, s0, s1, s2, s3, t
	_vxoshiro256engine s0, s1, s2, s3, t
ENDM

; Vectorized verion of _xoshiro256pp
; dest, s0, s1, s2, s3, t are AVX registers (XMMn/YMMn/ZMMn)
_vxoshiro256pp MACRO dest, s0, s1, s2, s3, t
	_vxoshiro256ppscrambler dest, s0, s1, s2, s3, t
	_vxoshiro256engine s0, s1, s2, s3, t
ENDM

; Vectorized verion of _xoshiro256p
; dest, s0, s1, s2, s3, t are AVX registers (XMMn/YMMn/ZMMn)
_vxoshiro256p MACRO dest, s0, s1, s2, s3, t
	_vxoshiro256pscrambler dest, s0, s1, s2, s3
	_vxoshiro256engine s0, s1, s2, s3, t
ENDM

; https://prng.di.unimi.it/splitmix64.c
; x, z, and temp are general purpose 64-bit registers.
; z stores the result of split mix, and
; x stores the state, or the seed initially.
; temp is clobbered!
; Preconditions:
;	* temp must not be x nor z.
;	* x stores the state for the next call to _splitmix64. It should not be the
;		same as z unless this is the last call to  _splitmix64 for this chain.
_splitmix64 MACRO z, x, temp
	mov temp, 9E3779B97F4A7C15h  ; uint64_t z = (x += 0x9e3779b97f4a7c15);
	add x, temp
	mov z, x
	
	mov temp, z                  ; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	shr temp, 30
	xor z, temp
	mov temp, 0BF58476D1CE4E5B9h
	imul z, temp
	
	mov temp, z                  ; z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	shr temp, 27
	xor z, temp
	mov temp, 94D049BB133111EBh
	imul z, temp
	
	mov temp, z                  ; return z ^ (z >> 31);
	shr temp, 31
	xor z, temp
ENDM
; NOTE: Tried to vectorize _splitmix64, but there is no vectorized 64-bit
; unsigned multiplication in AVX2. (There is vpmullq in AVX512*.)
; Could not come up with a replacment more efficient than just using
; _splitmix64, movq, and _vpermj/_vpremk three/four times.
; * See: https://uops.info/html-instr/VPMULLQ_ZMM_ZMM_M512.html

; Maps a uniform random uint64 into a double, \omega ~ Uniform[0, 1)
; dest and src can overlap.
; ones must be YMMn initialized to exactly [1.0, 1.0, 1.0, 1.0]
_vomega MACRO dest, src, ones
	vpsrlq dest, src, 12
	vpor dest, dest, ones
	vsubpd dest, dest, ones
ENDM

; **References**
; Blackman, D., & Vigna, S. (2022, March 28). Scrambled linear pseudorandom number generators (v3). arXiv.
; Vigna, S. (2015). SplitMix64. https://prng.di.unimi.it/splitmix64.c
