OPTION CASEMAP:NONE

include ../../prng.inc

.code
; @param seed (rcx) - uint64
; @param state (rdx) - output array with space for n uint64
; @param n (r8) - uint64 length of state array
; @return (rcx) new internal state ("x") of SplitMix64 algo. for subsequent calls.
PUBLIC splitmix64
splitmix64 PROC
	LOOP_START:
		cmp r8, 0
		jz LOOP_END
		dec r8

		_splitmix64 r9, rcx, rax  ; generate 4 states for xoshiro PRNG
		mov [rdx + r8*8], r9
		
		jmp LOOP_START
	LOOP_END:
	mov rax, rcx
	ret
splitmix64 ENDP

; @param state (rcx) - an initialized array containing PRNG state
; @return (rax) a psuedo-random uint64
PUBLIC xoshiro256ss
xoshiro256ss PROC
	mov r8, qword ptr [rcx + (3)*8]
	mov r9, qword ptr [rcx + (2)*8]
	mov r10, qword ptr  [rcx + (1)*8]
	mov r11, qword ptr [rcx + (0)*8]
	_xoshiro256ss rax, r8, r9, r10, r11, r15
	mov qword ptr [rcx + (3)*8], r8
	mov qword ptr [rcx + (2)*8], r9
	mov qword ptr [rcx + (1)*8], r10
	mov qword ptr  [rcx + (0)*8], r11
	ret
xoshiro256ss ENDP

; @param state (rcx) - an initialized array containing PRNG state
; @return (rax) a psuedo-random uint64
PUBLIC xoshiro256pp
xoshiro256pp PROC
	mov r8, qword ptr [rcx + (3)*8]
	mov r9, qword ptr [rcx + (2)*8]
	mov r10, qword ptr [rcx + (1)*8]
	mov r11, qword ptr [rcx + (0)*8]
	_xoshiro256pp rax, r8, r9, r10, r11, r15
	mov qword ptr  [rcx + (3)*8], r8
	mov qword ptr [rcx + (2)*8], r9
	mov qword ptr [rcx + (1)*8], r10
	mov qword ptr [rcx + (0)*8], r11
	ret
xoshiro256pp ENDP

; @param state (rcx) - an initialized array containing PRNG state
; @return (rax) a psuedo-random uint64
PUBLIC xoshiro256p
xoshiro256p PROC
	mov r8, qword ptr [rcx + (3)*8]
	mov r9, qword ptr [rcx + (2)*8]
	mov r10, qword ptr [rcx + (1)*8]
	mov r11, qword ptr [rcx + (0)*8]
	_xoshiro256p rax, r8, r9, r10, r11, r15
	mov qword ptr [rcx + (3)*8], r8
	mov qword ptr [rcx + (2)*8], r9
	mov qword ptr [rcx + (1)*8], r10
	mov qword ptr [rcx + (0)*8], r11
	ret
xoshiro256p ENDP
END