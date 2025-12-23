.code
PUBLIC main
main PROC
	vfnmaddpd ymm0, ymm0, ymm0, ymm0
	xor rax, rax
	ret
main ENDP
END