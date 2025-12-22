; Vector macros for working with AVX-256 vectors

.data
VEC SEGMENT ALIGN(32)
VEC_I		dq 1.0, 0.0, 0.0, 0.0
VEC ENDS

; Change VEC_I=(1.0, 0.0, 0.0, 0.0) at src
;	into VEC_J=(0.0, 1.0, 0.0, 0.0) at dest
_vpermj MACRO dest, src
	vpermilpd dest, src, 01h  ; swap doubles in low 128-bit lane
ENDM

; Change VEC_I=(1.0, 0.0, 0.0, 0.0) at src
;	into VEC_K=(0.0, 0.0, 1.0, 0.0) at dest
_vpermk MACRO dest, src
	; vperm2f128 dest, src0, src1, (imm8) ctrl
	; Explaination of 4th arg., "ctrl": 0b00HH00LL
	;	Think of the source as having 4 128-bit possabilities:
	;		[src0.low, src0.high, src1.low, src1.high]
	;	"HH" selects one of these sources to be copied into dest.high, and
	;	"LL" selects one of these sources to be copied into dest.low 
	; e.g.
	;	vperm2f128 ymm0, ymm1 = (a,b,c,d), ymm2 = (x,y,z,w), 0x12
	;	sources = [
	;		ymm1.low = (a,b),
	;		ymm1.high = (c,d),
	;		ymm2.low = (x,y),
	;		ymm2.high = (z,w)
	;	]
	;	HH = 1 -> dest.high = sources[1] = ymm1.high = (c,d)
	;	LL = 2 -> dest.low  = sources[2] = ymm2.low  = (x,y)
	;	=> dest = ymm1.high | (ymm2.low << 128/8) = (x,y,c,d)
	vperm2f128 dest, src, src, 01h  ; swaps low and high 128-bit lanes
ENDM

; macros to load unit vectors
_vput0 MACRO dest
	vxorpd dest, dest, dest
ENDM

_vputi MACRO dest
	vmovapd dest, ymmword ptr [VEC_I]
ENDM

_vputj MACRO dest
	_vputi dest
	_vpermj dest, dest
ENDM

_vputk MACRO dest
	_vputi dest
	_vpermk dest, dest
ENDM

_vneg MACRO dest, src, temp
	; (vpcmpeqq) Vector Packed Compare Equal Quadword (ints):
	;	Compares the 64-bit packed ints for equality, i.e. src0[i] == src1[i].
	;	If equal, dest[i] = -1 = 1111...1111b, else dest[i] = 0 = 0000....0000b
	vpcmpeqq temp, temp, temp  ; temp = (1111...1111b, 1111...1111b, ...)
	; (psllq) Packed Shift-left (Logical) Quadword (ints):
	;	Vectorized long left-shift operation
	vpsllq temp, temp, 63  ; now only highest ("sign") bit (in each quadword) is 1: 1000...0000b = -0.0

	vxorpd dest, src, temp  ; XOR: dest ^ (-0.0, ...) = -dest
ENDM

; Computes the dot product of src1 and scr2, and stores in (scalar) dest.
; Types: XMMn, YMMn, YMMn, XMMn, YMMn
; Preconditions:
;	1. temp can NOT overlap with dest.
;	2. tempX is XMMn version of tempY.
; Postconditions:
;	1. dest is modified and scalar, i.e. dest = (result, 0, 0, 0).
;	2. src1 and src2 are not modifed unless they overlap with dest or temp.
;	1. temp is clobbered!
; Notes:
;	* temp can overlap src1 or src2 since they are only used once at the top.
;	* dest can be the same as one or both of src1 and src2.
_vdotp MACRO destX, src1Y, src2Y, tempX, tempY
	vmulpd tempY, src1Y, src2Y    ; (Hadamard product) B .* \Delta s := (x, y, z, w)
	vhaddpd tempY, tempY, tempY   ; (x+y, x+y, z+w, z+w)
	vextractf128 destX, tempY, 1  ; extract high 128-bit lane to new register; dest = (z+w, z+w)
	vaddsd destX, destX, tempX    ; z+w+x+y
	; NOTE: using vextractf128 over vperm2f128 (i.e. _vpermk) since it may have lower latency.
	;	A version using _vperk could compute 4 dot products and would only require YMMn registers.
	;	This may make a programatically cleaner macro.
	; See: _vdotp4
ENDM

; Computes the dot product of scr1 and scr2, and adds it to (scalar) dest.
; Types: XMMn, YMMn, YMMn, YMMn, XMMn
; Preconditions:
;	1. temp can NOT overlap with dest.
;	2. destY is YMMn version of destX.
; Postconditions:
;	1. dest is modified and scalar, i.e. dest = (result, 0, 0, 0).
;	2. src1 and src2 are not modified unless they overlap with dest or temp.
;	3. temp is clobbered!
; Notes:
;	* temp can overlap with src1 or src2 since they are only used once at the top.
;	* dest can be the same as one or both of src1 and src2.
_vdotadd MACRO destX, destY, src1Y, src2Y, tempX
	; Inputs:	destY = (s, 0, 0, 0)
	;			src1  = (a, b, c, d)
	;			src2  = (x, y, z, w)
	vfmaddpd destY, src1Y, src2Y, destY  ; destY = (s + ax, by, cz, dw)
	vhaddpd destY, destY, destY          ; destY = (s + ax + by, ..., cz + dw, ...)
	vextractf128 tempX, destY, 1         ; tempX = (cz + dw, ...)
	vaddsd destX, destX, tempX           ; destX = (s + ax + by + cz + dw, 0, 0, 0)
ENDM

; Computes the negative dot product of src1 and src2, and add it to (scalar) dest.
; i.e. Subtracts the dot product of src1 and src2 from (scalar) dest.
; See: _vdotadd
_vndotadd MACRO destX, destY, src1Y, src2Y, tempX
	; Inputs:	destY = (s, 0, 0, 0)
	;			src1  = (a, b, c, d)
	;			src2  = (x, y, z, w)
	vfnmaddpd destY, src1Y, src2Y, destY  ; destY = (s - ax, -by, -cz, -dw)
	vhaddpd destY, destY, destY           ; destY = (s - ax - by, ..., -cz - dw, ...)
	vextractf128 tempX, destY, 1          ; tempX = (-cz - dw, ...)
	vaddsd destX, destX, tempX            ; destX = (s - ax - by - cz - dw, 0, 0, 0)
ENDM

_vdotp4 MACRO dest, src1, src2, temp
	vmulpd temp, src1, src2   ; (Hadamard product) B .* \Delta s := (x, y, z, w)
	vhaddpd temp, temp, temp  ; (x+y, x+y, z+w, z+w)
	_vpermk dest, temp        ; (z+w, z+w, x+y, x+y)
	vaddpd dest, dest, temp   ; (z+w+x+y, ...)
	; See: _vdotp
ENDM
