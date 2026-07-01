OPTION CASEMAP:NONE

include vec.inc
include dumpreg.inc

.code
PUBLIC main
main PROC
	_vputi xmm13, rax
	_vputj xmm14, rax
	_vputk ymm15, xmm15, rax

	_vput0 ymm0
	vaddpd ymm0, ymm0, ymm13  ; +i
	vaddpd ymm0, ymm0, ymm14  ; +j
	vaddpd ymm0, ymm0, ymm14  ; +j
	vaddpd ymm0, ymm0, ymm15  ; +k
	vaddpd ymm0, ymm0, ymm15  ; +k
	vaddpd ymm0, ymm0, ymm15  ; +k -> (1, 2, 3, 0)

	_vput0 ymm1
	vsubpd ymm1, ymm1, ymm13  ; -i
	vsubpd ymm1, ymm1, ymm13  ; -i
	vaddpd ymm1, ymm1, ymm14  ; +j
	vaddpd ymm1, ymm1, ymm14  ; +j
	vsubpd ymm1, ymm1, ymm15  ; -k
	vsubpd ymm1, ymm1, ymm15  ; -k
	vsubpd ymm1, ymm1, ymm15  ; -k
	vsubpd ymm1, ymm1, ymm15  ; -k -> (-2, 2, -4, 0)

	_vdotp xmm2, ymm2, ymm0, ymm1, xmm12     ; -10 = -2 + 4 -12
	_vndotp xmm3, ymm3, ymm0, ymm1, xmm12    ; 10 = -(-10)
	vmovapd ymm4, ymm2
	_vdotadd xmm4, ymm4, ymm0, ymm1, xmm12   ; -20 = -10 + -10
	vmovapd ymm5, ymm2
	_vndotadd xmm5, ymm5, ymm0, ymm1, xmm12  ; 0 = 10 - 10
	_dumpregv

	xor rax, rax  ; return 0
	ret
main ENDP
END