; link ucrt.lib
EXTERN: malloc:PROC
EXTERN: calloc:PROC
EXTERN: realloc:PROC
EXTERN: free:PROC

; Pre-conditions:
;	1. RCX contains the number of bytes to be allocated.
;	2. Stack is 16-byte aligned. (Windows ABI)
; Post-conditions:
;	RAX will contain the address of newly the allocated block.
_malloc MACRO
	sub rsp, 32  ; shadow space
	call malloc
	add rsp, 32
ENDM

; Pre-conditions:
;	1. RCX contains the number of element to be allocated.
;	2. RDX contains the size of each element.
;	3. Stack is 16-byte aligned. (Windows ABI)
; Post-conditions:
;	RAX will contain the address of newly the allocated array.
_calloc MACRO
	sub rsp, 32
	call calloc
	add rsp, 32
ENDM

; Pre-conditions:
;	1. RCX contains the address of a previously allocated block.
;	2. RDX contains the number of bytes to reallocate.
;	3. Stack is 16-byte aligned. (Windows ABI)
; Post-conditions:
;	RAX will contain the address of the reallocated block.
_realloc MACRO
	sub rsp, 32
	call realloc
	add rsp, 32
ENDM

; Pre-conditions:
;	1. RCX contains the address of a previously allocated block.
;	2. Stack is 16-byte aligned. (Windows ABI)
_free MACRO
	sub rsp, 32  ; shadow space
	call free
	add rsp, 32
ENDM
