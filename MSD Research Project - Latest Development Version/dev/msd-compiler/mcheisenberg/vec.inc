; Vector macros for working with AVX-256 vectors

; Change VEC_I=(1.0, 0.0, 0.0, 0.0) at src
;	into VEC_J=(0.0, 1.0, 0.0, 0.0) at dest
_vpermj MACRO dest, src
	vpermilpd dest, src, 09h  ; swap doubles in low 128-bit lane
ENDM

; Change VEC_I=(1.0, 0.0, 0.0, 0.0) at src
;	into VEC_K=(0.0, 0.0, 1.0, 0.0) at dest
_vpermk MACRO dest, src
	; vperm2f128 dest, src0, src1, (imm8) ctrl
	; Explaination of 4th arg., "ctrl": 0b00HH00LL
	;	Think of the (combined) source as having 4 128-bit possibilities:
	;		[src0.low, src0.high, src1.low, src1.high]
	;	"HH" selects one of these sources to be copied into dest.high, and
	;	"LL" selects one of these sources to be copied into dest.low 
	; e.g.
	;	vperm2f128 ymm0, ymm1 = (a,b,c,d), ymm2 = (x,y,z,w), 12h
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

; macros to load common vector, e.g. zero vector, unit vectors
_vput0 MACRO dest
	vxorpd dest, dest, dest
ENDM

; dest (XMMn), temp (general purpose reg.)
_vputi MACRO destX, temp
	mov temp, 03FF0000000000000h  ; (double) 1.0
	vmovq destX, temp
ENDM

_vputj MACRO destX, temp
	_vputi destX, temp
	_vpermj destX, destX
ENDM

; Precondition: destY and destX are the same AVX reg.
_vputk MACRO destY, destX, temp
	_vputi destX, temp
	_vpermk destY, destY
ENDM

; negate the given vector
_vneg MACRO dest, src, temp
	; This method is faster if the mask, temp, is already in memory.
	; Could also do _vput0 + vsubpd if dest != src

	; (vpcmpeqq) Vector Packed Compare Equal Quadword (ints):
	;	Compares the 64-bit packed ints for equality, i.e. src0[i] == src1[i].
	;	If equal, dest[i] = -1 = 1111...1111b, else dest[i] = 0 = 0000....0000b
	vpcmpeqq temp, temp, temp  ; temp = (1111...1111b, 1111...1111b, ...)
	; (psllq) Packed Shift-left (Logical) Quadword (ints):
	;	Vectorized long left-shift operation
	vpsllq temp, temp, 63  ; now only highest ("sign") bit (in each quadword) is 1: 1000...0000b = -0.0

	vxorpd dest, src, temp  ; XOR: dest ^ (-0.0, ...) = -dest
ENDM

; Build (1.0, 1.0, 1.0, 1.0) -> dest
; in-place without memory access.
_vones MACRO dest
	vpcmpeqq dest, dest, dest  ; dest = (1111...1111b, 1111...1111b, ...)
	vpsrlq dest, dest, 54      ; dest = (0...11 1111 1111b, ...)
	vpsllq dest, dest, 52      ; dest = (0011 1111 1111 0..., ...) = (1.0, ...)
ENDM

; clears dest[1:3] = 0
; input:       XMMn src  = (s, ?, ?, ?)
; output:      XMMn dest = (s, 0, 0, 0)
; side-effect: XMMn temp = (0, 0, 0, 0)
_sfilter MACRO dest, src, temp
	_vput0 temp
	vaddsd dest, temp, src  ; Exploits a quirk in scalar add where temp[1] falls through to dest[1]
	; dest = (temp[0] + src[0], temp[0], 0, 0) = (0 + src[0], 0, 0, 0) = (src[0], 0, 0, 0)
ENDM

; Adds all lanes into scalar, i.e. (s, ?, 0, 0)
_vreduce MACRO destX, destY, srcY, tempX
	; input: src = (a, b, c, d)
	vhaddpd destY, srcY, srcY     ; dest = (a+b, a+b, c+d, d+d)
	vextractf128 tempX, destY, 1  ; extract high 128-bit lane to new register; temp = (c+d, c+d, 0, 0)
	vaddsd destX, destX, tempX    ; dest = (a+b+c+d, a+b, 0, 0) <- Answer is in scalar position, dest[0]
	; NOTE: using vextractf128 over vperm2f128 (i.e. _vpermk).
	;	A version using _vpermk could compute 4 sums and would only require YMMn registers.
	;	This may make a programatically cleaner macro.
	; See: _vreduce4
ENDM

; Computes the dot product of src1 and scr2, and stores in (scalar) dest.
; Types: XMMn, YMMn, YMMn, YMMn, XMMn
; Preconditions:
;	1. temp can NOT overlap with dest.
;	2. destX is XMMn version of destY.
; Postconditions:
;	1. dest is modified and scalar, i.e. dest = (result, ?, 0, 0).
;	2. src1 and src2 are not modifed unless they overlap with dest or temp.
;	3. temp is clobbered!
;	4. dest[1] is clobbered! And dest[2:3] = 0
; Notes:
;	* temp can overlap src1 or src2 since they are only used once at the top.
;	* dest can be the same as one or both of src1 and src2.
_vdotp MACRO destX, destY, src1Y, src2Y, tempX
	vmulpd destY, src1Y, src2Y    ; (Hadamard product) B .* \Delta s := (x, y, z, w)
	_vreduce destX, destY, destY, tempX
ENDM

; Assumes dest is pure scalar, i.e. (s, 0, 0, 0)
; Notes:
;	* temp can overlap with src1 or src2 (unlike _vdotadd) since they are only used once at the top.
; See: _vdotadd
_vdotadds MACRO destX, destY, src1Y, src2Y, tempX
	; input src1 = (a, b, c, d); src2 = (x, y, z, w)
	vfmadd231pd destY, src1Y, src2Y   ; destY += src1Y * src2Y -> dest = (s + ax, by, cz, dw)
	_vreduce destX, destY, destY, tempX
ENDM

; Assumes dest is pure scalar, i.e. (s, 0, 0, 0)
; Notes:
;	* temp can overlap with src1 or src2 (unlike _vndotadd) since they are only used once at the top.
; See: _vndotadd
_vndotadds MACRO destX, destY, src1Y, src2Y, tempX
	; input src1 = (a, b, c, d); src2 = (x, y, z, w)
	vfnmadd231pd destY, src1Y, src2Y  ; destY -= src1Y * src2Y -> dest = (-ax, -by, -cz, -dw)
	_vreduce destX, destY, destY, tempX
ENDM

; Computes the dot product of scr1 and scr2, and adds it to (scalar) dest.
; Types: XMMn, YMMn, YMMn, YMMn, XMMn
; Preconditions:
;	1. temp can NOT overlap with dest NOR src.
;	2. destY is YMMn version of destX.
; Postconditions:
;	1. dest is modified and scalar, i.e. dest = (result, ?, 0, 0).
;	2. src1 and src2 are not modified.
;	3. temp is clobbered!
;	4. dest[1] is clobbered! And dest[2:3] = 0
; Notes:
;	* dest can be the same as one or both of src1 and src2.
_vdotadd MACRO destX, destY, src1Y, src2Y, tempX
	; Inputs:	destY = (s, ?, ?, ?)
	;			src1  = (a, b, c, d)
	;			src2  = (x, y, z, w)
	_sfilter destX, destX, tempX
	_vdotadds destX, destY, src1Y, src2Y, tempX
	; NOTE: may be 1 vxorpd instruction slower than _vdotp + vaddsd, but requires one less register.
ENDM

; Computes the negative dot product of src1 and src2, and add it to (scalar) dest.
; i.e. Subtracts the dot product of src1 and src2 from (scalar) dest.
; See: _vdotadd
_vndotadd MACRO destX, destY, src1Y, src2Y, tempX
	; Inputs:	destY = (s, ?, ?, ?)
	;			src1  = (a, b, c, d)
	;			src2  = (x, y, z, w)
	_sfilter destX, destX, tempX
	_vndotadds destX, destY, src1Y, src2Y, tempX
ENDM

; Negative dot product
_vndotp MACRO destX, destY, src1Y, src2Y, tempX
	_vput0 destX
	_vndotadd destX, destY, src1Y, src2Y, tempX
ENDM

; Adds all lanes into duplicated scalar, i.e. (s, s, s, s)
_vreduce4 MACRO dest, src, temp
	; input src = (a, b, c, d)
	vhaddpd dest, src, src   ; dest = (a+b, a+b, c+d, c+d)
	_vpermk temp, dest       ; temp = (c+d, c+d, a+b, a+b)
	vaddpd dest, dest, temp  ; dest = (a+b+c+d, ...)
	; See: _vreduce
ENDM

_vdotp4 MACRO dest, src1, src2, temp
	vmulpd dest, src1, src2   ; (Hadamard product) B .* \Delta s := (x, y, z, w)
	_vreduce4 dest, dest, temp
	; See: _vdotp
ENDM
