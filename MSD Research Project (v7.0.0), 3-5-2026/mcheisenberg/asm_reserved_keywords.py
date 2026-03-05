# Source: https://pic.hallikainen.org/techref/language/masm/masmAPD.htm
# Source: https://faculty.kfupm.edu.sa/coe/mudawar/coe205/manuals/MASMProgrGuide/D-ReservedWords.pdf
MASM_RESERVED_KEYWORDS = {
	# Operands and Symbols
	"ABS", "ALL", "ASSUMES", "AT", "CASEMAP", "COMMON", "COMPACT", "CPU", "DOTNAME",
	"EMULATOR", "EPILOGUE", "ERROR", "EXPORT", "EXPR16", "EXPR32", "FARSTACK", "FLAT", "FORCEFRAME",
	"HUGE", "LANGUAGE", "LARGE", "LISTING", "LJMP", "LOADDS", "M510", "MEDIUM", "MEMORY", "NEARSTACK",
	"NODOTNAME", "NOEMULATOR", "NOKEYWORD", "NOLJMP", "NOM510", "NONE", "NONUNIQUE",
	"NOOLDMACROS", "NOOLDSTRUCTS", "NOREADONLY", "NOSCOPED", "NOSIGNEXTEND", "NOTHING", "NOTPUBLIC",
	"OLDMACROS", "OLDSTRUCTS", "OS_DOS", "PARA", "PRIVATE", "PROLOGUE", "RADIX", "READONLY", "REQ",
	"SCOPED", "SETIF2", "SMALL", "STACK", "TINY", "USE16", "USE32", "USES",

	"$", "?", "@B", "@F", "ADDR", "BASIC", "BYTE", "C", "CARRY?", "DWORD", "FAR", "FAR16", "FORTRAN",
	"FWORD", "NEAR", "NEAR16", "OVERFLOW?", "PARITY?", "PASCAL", "QWORD", "REAL4", "REAL8", "REAL10",
	"SBYTE", "SDWORD", "SIGN?", "STDCALL", "SWORD", "SYSCALL", "TBYTE", "VARARG", "WORD", "ZERO?",
	
	# Special Operands for the 80386/486
	"FLAT", "NEAR32", "FAR32",
	
	# Predefined Symbols
	"@CATSTR", "@CODE", "@CODESIZE", "@CPU", "@CURSEG", "@DATA", "@DATASIZE", "@DATE", "@ENVIRON",
	"@FARDATA", "@FARDATA?", "@FILECUR", "@FILENAME", "@INSTR", "@INTERFACE", "@LINE", "@MODEL",
	"@SIZESTR", "@STACK", "@SUBSTR", "@TIME", "@VERSION", "@WORDSIZE",
	
	# Registers
	"AH", "AL", "AX", "BH", "BL", "BP", "BX", "CH", "CL", "CR0", "CR2", "CR3", "CS", "CX",
	"DH", "DI", "DL", "DR0", "DR1", "DR2", "DR3", "DR6", "DR7", "DS", "DX",
	"EAX", "EBP", "EBX", "ECX", "EDI", "EDX", "ES", "ESI", "ESP", "FS", "GS",
	"SI", "SP", "SS", "ST", "TR3", "TR4", "TR5", "TR6", "TR7",

	# Operators and Directives
	".186", ".286", ".286C", ".286P", ".287", ".386", ".386C", ".386P", ".387", ".486", ".486P", ".8086", ".8087",
	".ALPHA", ".BREAK", ".CODE", ".CONST", ".CONTINUE", ".CREF", ".DATA", ".DATA?", ".DOSSEG",
	".ELSE", ".ELSEIF", ".ENDIF", ".ENDW", ".ERR", ".ERR1", ".ERR2", ".ERRB",
	".ERRDEF", ".ERRDIF", ".ERRDIFI", ".ERRE", ".ERRIDN", ".ERRIDNI", ".ERRNB", ".ERRNDEF", ".ERRNZ",
	".EXIT", ".FARDATA", ".FARDATA?", ".IF", ".LALL", ".LFCOND",
	".LIST", ".LISTALL", ".LISTIF", ".LISTMACRO", ".LISTMACROALL", ".MODEL",
	".NO87", ".NOCREF", ".NOLIST", ".NOLISTIF", ".NOLISTMACRO", ".RADIX", ".REPEAT",
	".SALL", ".SEQ", ".SFCOND", ".STACK", ".STARTUP", ".TFCOND", ".TYPE", ".UNTIL", ".UNTILCXZ",
	".WHILE", ".XALL", ".XCREF", ".XLIST",
	"ALIAS", "ALIGN", "ASSUME", "CATSTR", "COMM", "COMMENT",
	"DB", "DD", "DF", "DOSSEG", "DQ", "DT", "DUP", "DW",
	"ECHO", "ELSE", "ELSEIF", "ELSEIF1", "ELSEIF2", "ELSEIFB", "ELSEIFDEF", "ELSEIFDIF",
	"ELSEIFDIFI", "ELSEIFE", "ELSEIFIDN", "ELSEIFIDNI", "ELSEIFNB", "ELSEIFNDEF",
	"END", "ENDIF", "ENDM", "ENDP", "ENDS", "EQ", "EQU", "EVEN", "EXITM", "EXTERN", "EXTERNDEF", "EXTRN",
	"FOR", "FORC", "GE", "GOTO", "GROUP", "GT", "HIGH", "HIGHWORD",
	"IF", "IF1", "IF2", "IFB", "IFDEF", "IFDIF", "IFDIFI", "IFE", "IFIDN", "IFIDNI", "IFNB", "IFNDEF",
	"INCLUDE", "INCLUDELIB", "INSTR", "INVOKE", "IRP", "IRPC",
	"LABEL", "LE", "LENGTH", "LENGTHOF", "LOCAL", "LOW", "LOWWORD", "LROFFSET", "LT",
	"MACRO", "MASK", "MOD", ".MSFLOAT", "NAME", "NE", "OFFSET", "OPATTR", "OPTION", "ORG", "%OUT",
	"PAGE", "POPCONTEXT", "PROC", "PROTO", "PTR", "PUBLIC", "PURGE", "PUSHCONTEXT",
	"RECORD", "REPEAT", "REPT", "SEG", "SEGMENT", "SHORT", "SIZE", "SIZEOF", "SIZESTR",
	"STRUC", "STRUCT", "SUBSTR", "SUBTITLE", "SUBTTL",
	"TEXTEQU", "THIS", "TITLE", "TYPE", "TYPEDEF", "UNION", "WHILE", "WIDTH",
	
	# 8086/8088 Processor Instructions
	"AAA", "AAD", "AAM", "AAS", "ADC", "ADD", "AND",
	"CALL", "CBW", "CLC", "CLD", "CLI", "CMC", "CMP", "CMPS", "CMPSB", "CMPSW", "CWD",
	"DAA", "DAS", "DEC", "DIV", "ESC", "HLT", "IDIV", "IMUL", "IN", "INC", "INT", "INTO", "IRET",
	"JA", "JAE", "JB", "JBE", "JC", "JCXZ", "JE", "JG", "JGE", "JL", "JLE", "JMP",
	"JNA", "JNAE", "JNB", "JNBE", "JNC", "JNE", "JNG", "JNGE", "JNL", "JNLE",
	"JNO", "JNP", "JNS", "JNZ", "JO", "JP", "JPE", "JPO", "JS", "JZ",
	"LAHF", "LDS", "LEA", "LES", "LODS", "LODSB", "LODSW",
	"LOOP", "LOOPE", "LOOPEW", "LOOPNE", "LOOPNEW", "LOOPNZ", "LOOPNZW", "LOOPW", "LOOPZ", "LOOPZW",
	"MOV", "MOVS", "MOVSB", "MOVSW", "MUL", "NEG", "NOP", "NOT",
	"OR", "OUT", "POP", "POPF", "PUSH", "PUSHF",
	"RCL", "RCR", "RET", "RETF", "RETN", "ROL", "ROR",
	"SAHF", "SAL", "SAR", "SBB", "SCAS", "SCASB", "SCASW", "SHL", "SHR", "STC",
	"STD", "STI", "STOS", "STOSB", "STOSW", "SUB", "TEST", "WAIT", "XCHG", "XLAT", "XLATB", "XOR",

	# 80186 Processor Instructions
	"BOUND", "ENTER", "INS", "INSB", "INSW", "LEAVE", "OUTS", "OUTSB", "OUTSW",
	"POPA", "PUSHA", "PUSHW",
	
	# 80286 Processor Instructions
	"ARPL", "LAR", "LSL", "SGDT", "SIDT", "SLDT", "SMSW", "STR", "VERR", "VERW",
	
	# 80286 and 80386 Privileged-Mode Instructions
	"CLTS", "LGDT", "LIDT", "LLDT", "LMSW", "LTR",
	
	# 80386 Processor Instructions
	"BSF", "BSR", "BT", "BTC", "BTR", "BTS", "CDQ", "CMPSD", "CWDE",
	"INSD", "IRETD", "IRETDF", "IRETF",
	"JECXZ", "LFS", "LGS", "LODSD", "LOOPD", "LOOPED", "LOOPNED", "LOOPNZD", "LOOPZD", "LSS",
	"MOVSD", "MOVSX", "MOVZX", "OUTSD", "POPAD", "POPFD", "PUSHAD", "PUSHD", "PUSHFD",
	"SCASD", "SETA", "SETAE", "SETB", "SETBE", "SETC", "SETE", "SETG", "SETGE",
	"SETL", "SETLE", "SETNA", "SETNAE", "SETNB", "SETNBE", "SETNC", "SETNE",
	"SETNG", "SETNGE", "SETNL", "SETNLE", "SETNO", "SETNP", "SETNS", "SETNZ",
	"SETO", "SETP", "SETPE", "SETPO", "SETS", "SETZ", "SHLD", "SHRD", "STOSD",
	
	# 80486 Processor Instructions
	"BSWAP", "CMPXCHG", "INVD", "INVLPG", "WBINVD", "XADD",

	#Instruction Prefixes
	"LOCK", "REP", "REPE", "REPNE", "REPNZ", "REPZ",

	# Coprocessor Instructions
	"F2XM1", "FABS", "FADD", "FADDP", "FBLD", "FBSTP",
	"FCHS", "FCLEX", "FCOM", "FCOMP", "FCOMPP",
	"FDECSTP", "FDISI", "FDIV", "FDIVP", "FDIVR", "FDIVRP", "FENI", "FFREE",
	"FIADD", "FICOM", "FICOMP", "FIDIV", "FIDIVR", "FILD", "FIMUL", "FINCSTP",
	"FINIT", "FIST", "FISTP", "FISUB", "FISUBR",
	"FLD", "FLD1", "FLDCW", "FLDENV", "FLDENVW", "FLDL2E", "FLDL2T", "FLDLG2", "FLDLN2", "FLDPI", "FLDZ",
	"FMUL", "FMULP", "FNCLEX", "FNDISI", "FNENI", "FNINIT", "FNOP",
	"FNSAVE", "FNSAVEW", "FNSTCW", "FNSTENV", "FNSTENVW", "FNSTSW",
	"FPATAN", "FPREM", "FPTAN", "FRNDINT", "FRSTOR", "FRSTORW",
	"FSAVE", "FSAVEW", "FSCALE", "FSQRT", "FST", "FSTCW", "FSTENV", "FSTENVW",
	"FSTP", "FSTSW", "FSUB", "FSUBP", "FSUBR", "FSUBRP", "FTST", "FWAIT",
	"FXAM", "FXCH", "FXTRACT", "FYL2X", "FYL2XP1",
	
	# 80287 Privileged-Mode Instruction
	"FSETPM",
	
	# 80387 Instructions
	"FCOS", "FLDENVD", "FNSAVED", "FNSTENVD", "FPREM1", "FRSTORD", "FSAVED",
	"FSIN", "FSINCOS", "FSTENVD", "FUCOM", "FUCOMP", "FUCOMPP"

	# TODO: Add AVX/AVX2/AVX512/FMA insn
}
