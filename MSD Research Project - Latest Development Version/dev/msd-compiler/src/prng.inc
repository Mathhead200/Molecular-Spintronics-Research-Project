; (Fig. 4, p. 5, Blackman & Vigna)
; S = 5, R = 7, T = 9 (Table 3, p. 10, Blackman & Vigna)
_xoshiro256ssscrambler MACRO dest, s0, s1, s2, s3
	; const uint_t result_starstar = rotl(s[1] * S, R) * T;
	lea dest, [s1 + s1*4]      ; dest = s1 * 5
	rol dest, 7
	lea dest, [dest + dest*8]  ; dest *= 9
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

; (Fig. 4, p. 5, Blackman & Vigna)
_xoshiro256pscrambler MACRO dest, s0, s1, s2, s3
	; const uint_t result_plus = s[0] + s[3];
	mov dest, s0
	add dest, s3
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

; **References**
; Blackman, D., & Vigna, S. (2022, March 28). Scrambled linear pseudorandom number generators (v3). arXiv.
; Vigna, S. (2015). SplitMix64. https://prng.di.unimi.it/splitmix64.c
