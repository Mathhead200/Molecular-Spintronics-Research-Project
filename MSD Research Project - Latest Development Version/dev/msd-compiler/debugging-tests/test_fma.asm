.code
PUBLIC main
main PROC
	vfnmadd213pd ymm0, ymm0, ymm0
	xor rax, rax
	ret
main ENDP
END