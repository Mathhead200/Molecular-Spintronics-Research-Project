
.code
PUBLIC main
main PROC
	mov rcx, 1  ; number of iterations
	call metropolis
	_dumpreg

	xor rax, rax  ; return 0
	ret
main ENDP
