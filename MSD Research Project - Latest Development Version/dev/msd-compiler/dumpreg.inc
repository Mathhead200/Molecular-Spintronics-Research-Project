; link ucrt.lib legacy_stdio_definitions.lib
EXTERN printf:PROC
EXTERN fflush:PROC

ASCII_TAB	EQU 09h
ASCII_LF	EQU 0Ah
ASCII_CR	EQU 0Dh

.data
; debug counter
dumpreg_count	dq 0

; general purpose registers: rax, rcx, ..., r15
regg	dq 16 DUP(0)

REGV SEGMENT ALIGN(32)
; SIMD registers: ymm0, ymm1, ..., ymm15
regv	dq 16 DUP(0.0, 0.0, 0.0, 0.0)
REGV ENDS

fmt_count db "[count=%d]", ASCII_LF, 0
fmt_rax	db "RAX=%016llx", ASCII_TAB, 0
fmt_rcx	db "RCX=%016llx", ASCII_TAB, 0
fmt_rdx	db "RDX=%016llx", ASCII_TAB, 0
fmt_rbx	db "RBX=%016llx", ASCII_LF, 0
fmt_rsp	db "RSP=%016llx", ASCII_TAB, 0
fmt_rbp	db "RBP=%016llx", ASCII_TAB, 0
fmt_rsi	db "RSI=%016llx", ASCII_TAB, 0
fmt_rdi	db "RDI=%016llx", ASCII_LF, 0
fmt_r8	db "R8 =%016llx", ASCII_TAB, 0
fmt_r9	db "R9 =%016llx", ASCII_TAB, 0
fmt_r10	db "R10=%016llx", ASCII_TAB, 0
fmt_r11	db "R11=%016llx", ASCII_LF, 0
fmt_r12	db "R12=%016llx", ASCII_TAB, 0
fmt_r13	db "R13=%016llx", ASCII_TAB, 0
fmt_r14	db "R14=%016llx", ASCII_TAB, 0
fmt_r15	db "R15=%016llx", ASCII_LF, 0
fmt_ymmnlbl		db "YMM%c:  ", 0
fmt_ymmnlbl1	db "YMM1%c: ", 0
fmt_fs		db "%f, ", 0
fmt_f		db "%f", 0
fmt_endl	db ASCII_LF, 0

_print_and_inc_count MACRO
	lea rcx, fmt_count
	mov rdx, dumpreg_count
	inc rdx
	mov dumpreg_count, rdx
	call printf
ENDM

_printregg MACRO fmt, offset
	lea rcx, fmt  ; one of fmt_rax, fmt_rcx, ..., fmt_r15
	mov rdx, qword ptr [regg + 8*offset]
	call printf
ENDM

_printregv MACRO char, offset
	lea rcx, fmt_ymmnlbl  ; 1st arg: format string
	mov rdx, char         ; 2nd arg: %c
	call printf
	
	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(0)]  ; 2nd arg: YMMn[0]
	call printf
	
	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(1)] ; 2nd arg: YMMn[0]
	call printf

	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(2)]  ; 2nd arg: YMMn[0]
	call printf

	lea rcx, fmt_f        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(3)]  ; 2nd arg: YMMn[0]
	call printf

	_endl
ENDM

_printregv1 MACRO char, offset
	lea rcx, fmt_ymmnlbl1  ; 1st arg: format string
	mov rdx, char          ; 2nd arg: %c (one's place)
	call printf
	
	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(0)]  ; 2nd arg: YMMn[0]
	call printf
	
	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(1)] ; 2nd arg: YMMn[0]
	call printf

	lea rcx, fmt_fs        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(2)]  ; 2nd arg: YMMn[0]
	call printf

	lea rcx, fmt_f        ; 1st arg: format string
	mov rdx, qword ptr [regv + 32*offset + 8*(3)]  ; 2nd arg: YMMn[0]
	call printf

	_endl
ENDM

_flush MACRO
	; flush (all)
	xor rcx, rcx  ; rcx = nullptr: flushes all open output streams
	call fflush
ENDM

_endl MACRO
	lea rcx, fmt_endl
	call printf
	_flush
ENDM

_dumpreg MACRO
	mov qword ptr [regg + 8*(0) ], rax
	mov qword ptr [regg + 8*(1) ], rcx
	mov qword ptr [regg + 8*(2) ], rdx
	mov qword ptr [regg + 8*(3) ], rbx
	mov qword ptr [regg + 8*(4) ], rsp
	mov qword ptr [regg + 8*(5) ], rbp
	mov qword ptr [regg + 8*(6) ], rsi
	mov qword ptr [regg + 8*(7) ], rdi
	mov qword ptr [regg + 8*(8) ], r8
	mov qword ptr [regg + 8*(9) ], r9
	mov qword ptr [regg + 8*(10)], r10
	mov qword ptr [regg + 8*(11)], r11
	mov qword ptr [regg + 8*(12)], r12
	mov qword ptr [regg + 8*(13)], r13
	mov qword ptr [regg + 8*(14)], r14
	mov qword ptr [regg + 8*(15)], r15

	vmovapd ymmword ptr [regv + 32*(0) ], ymm0
	vmovapd ymmword ptr [regv + 32*(1) ], ymm1
	vmovapd ymmword ptr [regv + 32*(2) ], ymm2
	vmovapd ymmword ptr [regv + 32*(3) ], ymm3
	vmovapd ymmword ptr [regv + 32*(4) ], ymm4
	vmovapd ymmword ptr [regv + 32*(5) ], ymm5
	vmovapd ymmword ptr [regv + 32*(6) ], ymm6
	vmovapd ymmword ptr [regv + 32*(7) ], ymm7
	vmovapd ymmword ptr [regv + 32*(8) ], ymm8
	vmovapd ymmword ptr [regv + 32*(9) ], ymm9
	vmovapd ymmword ptr [regv + 32*(10)], ymm10
	vmovapd ymmword ptr [regv + 32*(11)], ymm11
	vmovapd ymmword ptr [regv + 32*(12)], ymm12
	vmovapd ymmword ptr [regv + 32*(13)], ymm13
	vmovapd ymmword ptr [regv + 32*(14)], ymm14
	vmovapd ymmword ptr [regv + 32*(15)], ymm15

	sub rsp, 32  ; push shadow space
	and rsp, -16  ; force 16-bit alignment of stack
	_print_and_inc_count
	_printregg fmt_rax, 0
	_printregg fmt_rcx, 1
	_printregg fmt_rdx, 2
	_printregg fmt_rbx, 3
	_printregg fmt_rsp, 4
	_printregg fmt_rbp, 5
	_printregg fmt_rsi, 6
	_printregg fmt_rdi, 7
	_printregg fmt_r8,  8
	_printregg fmt_r9,  9
	_printregg fmt_r10, 10
	_printregg fmt_r11, 11
	_printregg fmt_r12, 12
	_printregg fmt_r13, 13
	_printregg fmt_r14, 14
	_printregg fmt_r15, 15
	_printregv '0', 0
	_printregv '1', 1
	_printregv '2', 2
	_printregv '3', 3
	_printregv '4', 4
	_printregv '5', 5
	_printregv '6', 6
	_printregv '7', 7
	_printregv '8', 8
	_printregv1 '9', 9
	_printregv1 '0', 10
	_printregv1 '1', 11
	_printregv1 '2', 12
	_printregv1 '3', 13
	_printregv1 '4', 14
	_printregv1 '5', 15
	_endl

	; restore register values since printf may mutable volitile registers
	mov rax, qword ptr [regg + 8*(0)]
	mov rcx, qword ptr [regg + 8*(1)]
	mov rdx, qword ptr [regg + 8*(2)]
	mov rbx, qword ptr [regg + 8*(3)]
	mov rsp, qword ptr [regg + 8*(4)]
	mov rbp, qword ptr [regg + 8*(5)]
	mov rsi, qword ptr [regg + 8*(6)]
	mov rdi, qword ptr [regg + 8*(7)]
	mov r8,  qword ptr [regg + 8*(8)]
	mov r9,  qword ptr [regg + 8*(9)]
	mov r10, qword ptr [regg + 8*(10)]
	mov r11, qword ptr [regg + 8*(11)]
	mov r12, qword ptr [regg + 8*(12)]
	mov r13, qword ptr [regg + 8*(13)]
	mov r14, qword ptr [regg + 8*(14)]
	mov r15, qword ptr [regg + 8*(15)]

	vmovapd ymm0,  ymmword ptr [regv + 32*(0)]
	vmovapd ymm1,  ymmword ptr [regv + 32*(1)]
	vmovapd ymm2,  ymmword ptr [regv + 32*(2)]
	vmovapd ymm3,  ymmword ptr [regv + 32*(3)]
	vmovapd ymm4,  ymmword ptr [regv + 32*(4)]
	vmovapd ymm5,  ymmword ptr [regv + 32*(5)]
	vmovapd ymm6,  ymmword ptr [regv + 32*(6)]
	vmovapd ymm7,  ymmword ptr [regv + 32*(7)]
	vmovapd ymm8,  ymmword ptr [regv + 32*(8)]
	vmovapd ymm9,  ymmword ptr [regv + 32*(9)]
	vmovapd ymm10, ymmword ptr [regv + 32*(10)]
	vmovapd ymm11, ymmword ptr [regv + 32*(11)]
	vmovapd ymm12, ymmword ptr [regv + 32*(12)]
	vmovapd ymm13, ymmword ptr [regv + 32*(13)]
	vmovapd ymm14, ymmword ptr [regv + 32*(14)]
	vmovapd ymm15, ymmword ptr [regv + 32*(15)]
ENDM

_dumpregg MACRO
	mov qword ptr [regg + 8*(0) ], rax
	mov qword ptr [regg + 8*(1) ], rcx
	mov qword ptr [regg + 8*(2) ], rdx
	mov qword ptr [regg + 8*(3) ], rbx
	mov qword ptr [regg + 8*(4) ], rsp
	mov qword ptr [regg + 8*(5) ], rbp
	mov qword ptr [regg + 8*(6) ], rsi
	mov qword ptr [regg + 8*(7) ], rdi
	mov qword ptr [regg + 8*(8) ], r8
	mov qword ptr [regg + 8*(9) ], r9
	mov qword ptr [regg + 8*(10)], r10
	mov qword ptr [regg + 8*(11)], r11
	mov qword ptr [regg + 8*(12)], r12
	mov qword ptr [regg + 8*(13)], r13
	mov qword ptr [regg + 8*(14)], r14
	mov qword ptr [regg + 8*(15)], r15

	sub rsp, 32  ; push shadow space
	and rsp, -16  ; force 16-bit alignment of stack
	_print_and_inc_count
	_printregg fmt_rax, 0
	_printregg fmt_rcx, 1
	_printregg fmt_rdx, 2
	_printregg fmt_rbx, 3
	_printregg fmt_rsp, 4
	_printregg fmt_rbp, 5
	_printregg fmt_rsi, 6
	_printregg fmt_rdi, 7
	_printregg fmt_r8,  8
	_printregg fmt_r9,  9
	_printregg fmt_r10, 10
	_printregg fmt_r11, 11
	_printregg fmt_r12, 12
	_printregg fmt_r13, 13
	_printregg fmt_r14, 14
	_printregg fmt_r15, 15
	_endl

	; restore register values since printf may mutable volitile registers
	mov rax, qword ptr [regg + 8*(0)]
	mov rcx, qword ptr [regg + 8*(1)]
	mov rdx, qword ptr [regg + 8*(2)]
	mov rbx, qword ptr [regg + 8*(3)]
	mov rsp, qword ptr [regg + 8*(4)]
	mov rbp, qword ptr [regg + 8*(5)]
	mov rsi, qword ptr [regg + 8*(6)]
	mov rdi, qword ptr [regg + 8*(7)]
	mov r8,  qword ptr [regg + 8*(8)]
	mov r9,  qword ptr [regg + 8*(9)]
	mov r10, qword ptr [regg + 8*(10)]
	mov r11, qword ptr [regg + 8*(11)]
	mov r12, qword ptr [regg + 8*(12)]
	mov r13, qword ptr [regg + 8*(13)]
	mov r14, qword ptr [regg + 8*(14)]
	mov r15, qword ptr [regg + 8*(15)]
ENDM

_dumpregv MACRO
	vmovapd ymmword ptr [regv + 32*(0) ], ymm0
	vmovapd ymmword ptr [regv + 32*(1) ], ymm1
	vmovapd ymmword ptr [regv + 32*(2) ], ymm2
	vmovapd ymmword ptr [regv + 32*(3) ], ymm3
	vmovapd ymmword ptr [regv + 32*(4) ], ymm4
	vmovapd ymmword ptr [regv + 32*(5) ], ymm5
	vmovapd ymmword ptr [regv + 32*(6) ], ymm6
	vmovapd ymmword ptr [regv + 32*(7) ], ymm7
	vmovapd ymmword ptr [regv + 32*(8) ], ymm8
	vmovapd ymmword ptr [regv + 32*(9) ], ymm9
	vmovapd ymmword ptr [regv + 32*(10)], ymm10
	vmovapd ymmword ptr [regv + 32*(11)], ymm11
	vmovapd ymmword ptr [regv + 32*(12)], ymm12
	vmovapd ymmword ptr [regv + 32*(13)], ymm13
	vmovapd ymmword ptr [regv + 32*(14)], ymm14
	vmovapd ymmword ptr [regv + 32*(15)], ymm15

	sub rsp, 32  ; push shadow space
	and rsp, -16  ; force 16-bit alignment of stack
	_printregv '0', 0
	_printregv '1', 1
	_printregv '2', 2
	_printregv '3', 3
	_printregv '4', 4
	_printregv '5', 5
	_printregv '6', 6
	_printregv '7', 7
	_printregv '8', 8
	_printregv1 '9', 9
	_printregv1 '0', 10
	_printregv1 '1', 11
	_printregv1 '2', 12
	_printregv1 '3', 13
	_printregv1 '4', 14
	_printregv1 '5', 15
	_endl

	; restore register values since printf may mutable volitile registers
	vmovapd ymm0,  ymmword ptr [regv + 32*(0)]
	vmovapd ymm1,  ymmword ptr [regv + 32*(1)]
	vmovapd ymm2,  ymmword ptr [regv + 32*(2)]
	vmovapd ymm3,  ymmword ptr [regv + 32*(3)]
	vmovapd ymm4,  ymmword ptr [regv + 32*(4)]
	vmovapd ymm5,  ymmword ptr [regv + 32*(5)]
	vmovapd ymm6,  ymmword ptr [regv + 32*(6)]
	vmovapd ymm7,  ymmword ptr [regv + 32*(7)]
	vmovapd ymm8,  ymmword ptr [regv + 32*(8)]
	vmovapd ymm9,  ymmword ptr [regv + 32*(9)]
	vmovapd ymm10, ymmword ptr [regv + 32*(10)]
	vmovapd ymm11, ymmword ptr [regv + 32*(11)]
	vmovapd ymm12, ymmword ptr [regv + 32*(12)]
	vmovapd ymm13, ymmword ptr [regv + 32*(13)]
	vmovapd ymm14, ymmword ptr [regv + 32*(14)]
	vmovapd ymm15, ymmword ptr [regv + 32*(15)]
ENDM
