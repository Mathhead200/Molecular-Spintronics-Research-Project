OPTION CASEMAP:NONE

include vec.inc
include dumpreg.inc

.code
PUBLIC main
main PROC
	; _vputi xmm0, rax
	; vaddsd xmm0, xmm0, xmm0  ; ymm0 =  (2.0, 0.0,    0.0, 0.0)
	; _vputj xmm1, rax
	; vaddpd ymm1, ymm1, ymm0  ; ymm1 =  (2.0, 1.0,    0.0, 0.0)
	; vaddsd xmm2, xmm1, xmm1  ; ymm2 =? (4.0, 1.0(?), 0.0, 0.0)
	; vaddsd xmm3, xmm1, xmm0  ; ymm3 =? (4.0, 1.0,    0.0, 0.0)
	; vaddsd xmm4, xmm0, xmm1  ; ymm4 =? (4.0, 0.0,    0.0, 0.0)
	; _vputk ymm5, xmm5, rax   ; ymm5 =  (0.0, 0.0,    1.0, 0.0)
	; vaddpd ymm6, ymm5, ymm1  ; ymm6 =  (2.0, 1.0,    1.0, 0.0)
	; vaddsd xmm7, xmm5, xmm1  ; ymm7 =  (2.0, 0.0 (?), 0.0, 0.0)
	; _dumpregv

	_vputi xmm1, rax        ; [1, 0, 0, 0]
	_vputj xmm2, rax        ; [0, 1, 0, 0]
	_vputk ymm3, xmm3, rax  ; [0, 0, 1, 0]
	vmovapd ymm4, ymm1
	vaddpd ymm4, ymm4, ymm2
	vaddpd ymm4, ymm4, ymm2
	vaddpd ymm4, ymm4, ymm3
	vaddpd ymm4, ymm4, ymm3
	vaddpd ymm4, ymm4, ymm3 ; [1, 2, 3, 0]
	_vneg ymm6, ymm1, ymm15
	_vneg ymm7, ymm3, ymm15
	vaddpd ymm5, ymm6, ymm7
	vaddpd ymm5, ymm5, ymm7
	vaddpd ymm5, ymm5, ymm2
	vaddpd ymm5, ymm5, ymm2
	vaddpd ymm5, ymm5, ymm2
	vaddpd ymm5, ymm5, ymm2
	vaddpd ymm5, ymm5, ymm2 ; [-1, 5, -2, 0]

	; _vdotp
	vmulpd ymm0, ymm4, ymm5      ; ymm0  = (-1, 10, -6,  0)
	vhaddpd ymm0, ymm0, ymm0     ; ymm0  = ( 9,  9, -6, -6)
	vextractf128 xmm15, ymm0, 1  ; ymm15 = (-6, -6,  0,  0)
	vaddsd xmm0, xmm0, xmm15     ; ymm0  = ( 3,  9,  0,  0)
	
	; _vdotadd (Method 1)
	vxorpd xmm15, xmm15, xmm15
	vaddsd xmm0, xmm15, xmm0
	vfmadd231pd ymm0, ymm4, ymm5
	vhaddpd ymm0, ymm0, ymm0 
	vextractf128 xmm15, ymm0, 1
	vaddsd xmm0, xmm0, xmm15
	_dumpregv

	; _vdotadd (Method 2)
	vmulpd ymm14, ymm4, ymm5
	vhaddpd ymm14, ymm14, ymm14
	vextractf128 xmm15, ymm14, 1
	vaddsd xmm14, xmm14, xmm15
	vaddsd xmm0, xmm0, xmm14
	_dumpregv

	vbroadcastsd ymm0, xmm0
	; _vdotadd (Method 3)
	vmulpd ymm14, ymm4, ymm5
	vhaddpd ymm14, ymm14, ymm14
	vperm2f128 ymm15, ymm14, ymm14, 01h
	vaddpd ymm14, ymm14, ymm15
	vaddpd ymm0, ymm0, ymm14

	xor rax, rax  ; return 0
	ret
main ENDP
END