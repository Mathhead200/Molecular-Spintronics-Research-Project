from math import log, isinf
from time import time

M_TRUNC = 9

class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)

if __name__ == "__main__":
	t = time()
	src = StrJoiner()

	src += "OPTION CASEMAP:NONE\n\n"
	
	src += "include vec.inc\n"
	src += "include dumpreg.inc\n\n"

	src += f"M_TRUNC EQU {M_TRUNC}   ; Mantissa trancation length (in bits)\n\n"

	src += ".data\n"
	src += "TAYLOR SEGMENT ALIGN(32)\n"
	src += "\t\t\t\t; a, f(a)=ln(a), f'(a)=1/a, f''(a)/2!=1/(2*a^2)\n"
	src += "taylor_table\t"
	first_line = True
	M = 2**M_TRUNC
	for x in range(1023):  # [0, 1022] all possible exponent values
		for m in range(0, M):  # [0, 2**n - 1] mantissa high-bit possibile values
			pow2_ceil = 2**(x + 1 - 1023)
			pow2_floor = 2**(x - 1023) if x != 0 else pow2_ceil  # handel denormalized case
			a = pow2_ceil - (M - (m + 1)) / M * pow2_floor  # e.g. M_TRUNC=2
				# (DENORMALIZED) LIMIT: w = 0
				# (DENORMALIZED) x = 0, m = 0 -> w = (1/4) * 2**(-1022) = ...
				# (DENORMALIZED) x = 0, m = 1 -> w = (2/4) * 2**(-1022) = ...
				# (DENORMALIZED) x = 0, m = 2 -> w = (3/4) * 2**(-1022) = 2**(-1022) - (1/4)*(2**(-1022))
				# (DENORMALIZED) x = 0, m = 3 -> w = 2**(-1022)
				# ...
				# x = 1020, m = 3 -> w = 0.25 = 2**(-2)
				# x = 1021, m = 0 -> w = 0.3125 = 0.5 - (3/4)*(0.25)
				# x = 1021, m = 1 -> w = 0.375  = 0.5 - (2/4)*(0.25)
				# x = 1021, m = 2 -> w = 0.4375 = 0.5 - (1/4)*(0.25)
				# x = 1021, m = 3 -> w = 0.5 = 2**(-1)
				# x = 1022, m = 0 -> w = 0.625 = 1.0 - (3/4)*(0.5)
				# x = 1022, m = 1 -> w = 0.75  = 1.0 - (2/4)*(0.5)
				# x = 1022, m = 2 -> w = 0.875 = 1.0 - (1/4)*(0.5)
				# x = 1022, m = 3 -> w = 1.0 = 2**(0)
			if first_line:
				first_line = False
			else:
				src += "\t\t\t\t"
			
			def asm_dq(x: float) -> str:
				if isinf(x):
					if x > 0:  x =  "7FF0000000000000h"  # (double) +inf
					else:      x = "0FFF0000000000000h"  # (double) -inf
				return x
			
			f = log(a)
			f1 = asm_dq(1 / a)
			try:
				f2 = asm_dq(1/(2 * a**2))
			except ZeroDivisionError:
				f2 = "7FF0000000000000h"  # (double) +inf
			src += f"dq {a}, {f}, {f1}, {f2}"

			if x == 0:
				src += "  ; denormalized"
			if m == 0:
				src += f"  ; exp. = {x}"
			src += "\n"

	src += "TAYLOR ENDS\n\n"

	src += ".code\n"
	src += "; @param (xmm0) - double w in [0, 1)\n"
	src += "; @return (xmm0)\n"
	src += "PUBLIC ln\n"
	src += "ln PROC\n"
	src += "\t; compute address offset for look-up table\n"
	src += "\tvpsrlq ymm1, ymm0, 52  ; ymm1: x-index (from exp.)\n"
	src += "\tvpsllq ymm1, ymm1, M_TRUNC  ; mul by 2^M_TRUNC length inner array\n"
	src += "\tvpsllq ymm2, ymm0, 12  ; ymm2: m-index (from mantissa M_TRUNC high-bits)\n"
	src += "\tvpsrlq ymm2, ymm2, 64 - M_TRUNC\n"
	src += "\tvpaddq ymm1, ymm1, ymm2\n"
	src += "\tvpsllq ymm1, ymm1, 5  ; mul 32 (ymmword ptr offset)\n"
	src += "\tvmovq rax, xmm1\n\n"
	
	src += "\t; load pre-computed Taylor terms: a, f(a), f'(a), etc.\n"
	src += "\tlea rcx, taylor_table\n"
	src += "\tvmovapd ymm1, ymmword ptr [rcx + rax]  ; [a, f(a), f'(a), f''(a)/2!]\n\n"
	
	src += "\tvsubsd xmm2, xmm0, xmm1  ; xmm2 = w - a\n\n"

	src += "\t_vpermj ymm0, ymm1  ; ymm0 = [f(a), a, f'(a), f''(a)/2!] -> xmm0 = f(a)\n\n"

	src += "\t_vpermk ymm1, ymm0  ; ymm1 = [f'(a), f''(a)/2!, f(a), a] -> xmm1 = f'(a)\n"
	src += "\tvfmadd231sd xmm0, xmm1, xmm2  ; xmm0 += xmm1 * xmm2 -> xmm0 = f(a) + f'(a)*(w-a)\n"

	src += "\t_vpermj ymm1, ymm1  ; ymm1 = [f''(a)/2!, f'(a), f(a), a] -> xmm1 = f''(a)/2!\n"
	src += "\tvfmadd231sd xmm0, xmm1, xmm3\n\n"

	src += "\tret\n"
	src += "ln ENDP\n"
	src += "END\n"

	with open(f"src/ln{M_TRUNC}.asm", "w", encoding="utf-8") as file:
		file.write(str(src))
	
	t = time() - t
	print(f"Done. time: {t:.3f} s")