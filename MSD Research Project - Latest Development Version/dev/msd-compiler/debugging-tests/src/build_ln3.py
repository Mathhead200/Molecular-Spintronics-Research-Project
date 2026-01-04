# vectorized

from math import log, ceil
from time import time
from remez import remez
from struct import pack

M_TRUNC = 10
MINIMAX_DEG = 4

class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)
	
def minimax_coefs(delta = 1 / 2**(M_TRUNC + 1), deg = MINIMAX_DEG):
	fx = "np.log1p(x)"
	fx_der = "1.0/(1.0 + x)"
	px, _, _ = remez(fx, fx_der, [-delta, delta], deg)
	print(px)  # e.g. 1.0001 * x**2 - 15.2 * x + 1.002e-3
	
	# parse string result, px
	px = px.replace("- ", "-")
	px = px.replace("+ ", "+")
	px = px.replace(" * ", " ")  # 1.0001 x**2 -15.2 x +1.002e-3
	print("px:", px)
	px = px.split(" ") # ["1.0001", "x**2", "-15.2", "x", "+1.002e-3"]
	coefs = [0.0 for _ in range(deg + 1)]
	i = 0
	while i < len(px):
		try:
			c = float(px[i])  # monomial coeficient
			k = px[i+1] if i+1 < len(px) else "x**0"  # monomial degree
		except ValueError:
			c = 1.0
			px.insert(i+1, None)
			k = px[i]
		k = int(k[3:]) if len(k) >= 3 else 1
		coefs[k] = c
		i += 2
	coefs.reverse()
	print("coefs:", coefs)

	# DEBUG for Desmos
	des = StrJoiner()
	for i, c in enumerate(coefs):
		k = MINIMAX_DEG - i
		if (c >= 0):  des += "+"
		des += f"{c:.17}x^{k}"
	print("Desmos:", str(des))

	return coefs

if __name__ == "__main__":
	t = time()

	coefs = minimax_coefs()  # floats
	# coefs = [f"0{int.from_bytes(pack('>d', c), 'big'):016X}h" if c != 0.0 else None for c in coefs]
	while coefs[0] is None:  coefs.pop(0)  # reduce polynomial degree
	
	src = StrJoiner()

	src += "OPTION CASEMAP:NONE\n\n"
	
	src += "include vec.inc\n"
	src += "include dumpreg.inc\n\n"

	src += f"M_TRUNC EQU {M_TRUNC}   ; Mantissa trancation length (in bits)\n\n"

	src += ".data\n"
	a_table = "a_table\t"
	a_inv_table = "a_inv_table\t"
	ln_a_table = "ln_a_table\t"
	first_line = True
	M = 2**M_TRUNC
	for m in range(M):
		a = 1 + (m + 0.5)/M  # reference point on [1, 2)
		log_a = log(a)
		if first_line:
			first_line = False
		else:
			a_table += "\t\t"
			a_inv_table += "\t\t\t"
			ln_a_table += "\t\t\t"
		a_table += f"dq {a}\n"
		a_inv_table += f"dq {1/a}  ; 1/{a}\n"
		ln_a_table += f"dq {log_a}  ; ln({a})\n"
	src += f"{a_table}\n{a_inv_table}\n{ln_a_table}\n"

	src += "coefs\t"
	first_line = True
	for i, c in enumerate(coefs):
		if first_line:
			first_line = False
		else:
			src += "\t\t"
		src += f"dq {c}  ; c_{MINIMAX_DEG - i}\n"
	src += "\n"

	src += "EXP_BIAS dq -1023\n"
	src += "LN_2 dq 0.69314718055994530942\n"
	src += "EXP_0 dq 3FF0000000000000h  ; boased exp. 2^0 for doubles in [1, 2)\n\n"
	src += "SLN1 SEGMENT ALIGN(32)\n"
	src += "VPERMD_INDICES dd 0,2, 4,6, 0,0, 0,0\n"
	src += "SLN1 ENDS\n\n"

	src += ".code\n"
	src += "; @param (ymm0) - four packed doubles\n"
	src += "; @return (ymm0)\n"
	src += "; Preconditions:\n"
	src += ";\t1. ymm0 is not negative, denormalized, infinite, nor NaN\n"
	src += "PUBLIC ln\n"
	src += "ln PROC\n"

	src += "\t; ymm0[i] -> 0|xxx xxxx xxxx|mmmm mmmm mmmm ...\n"
	src += "\t;   sign bit ^|^^ exponent ^|^^ mantissa ^^ ...\n"
	src += "\t; ln(m*2^x) = ln(m) + x*ln(2)\n\n"

	src += "\t; ymm4: calculate x*ln(2)\n"
	src += "\tvpsrlq ymm4, ymm0, 52  ; (biased) exponent, (uint64) x\n"
	src += "\tvpbroadcastq ymm1, qword ptr EXP_BIAS\n"
	src += "\tvpaddq ymm4, ymm4, ymm1  ; (unbiased) exponent, (sint64) x - 1023\n"
	src += "\tvmovdqa ymm1, ymmword ptr VPERMD_INDICES\n"
	src += "\tvpermd ymm4, ymm1, ymm4  ; cast (sint64) ymm4 to (sint32) xmm4\n"
	src += "\tvcvtdq2pd ymm4, xmm4  ; cast (sint32) xmm4 to (double)\n"
	src += "\tvbroadcastsd ymm1, LN_2\n"
	src += "\tvmulpd ymm4, ymm4, ymm1  ; x*ln(2)\n\n"

	src += "\t; ymm1: (parallel) table indices\n"
	src += "\tvpsllq ymm0, ymm0, 12  ; remove sign (1 bit) + exponent (11 bits)\n"
	src += "\tvpsrlq ymm1, ymm0, 64 - M_TRUNC  ; most significant bits of mantissa -> indices in (paralell) tables\n\n"

	src += "\t; ymm0: m \\in [1, 2)\n"
	src += "\tvpsrlq ymm0, ymm0, 12  ; move mantissa back in place\n"
	src += "\tvbroadcastsd ymm3, EXP_0\n"
	src += "\tvpor ymm0, ymm0, ymm3  ; set exponent to 2^0\n\n"

	src += "\t; ymm2: calculate residual, r = m/a - 1.0 = (m - a)/a\n"
	src += "\tlea rax, a_table\n"
	src += "\tvpcmpeqq ymm3, ymm3, ymm3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd ymm2, [rax + ymm1*8], ymm3\n"
	src += "\tvsubpd ymm0, ymm0, ymm2  ; m - a\n"
	src += "\tlea rax, a_inv_table\n"
	src += "\tvpcmpeqq ymm3, ymm3, ymm3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd ymm2, [rax + ymm1*8], ymm3\n"
	src += "\tvmulpd ymm2, ymm2, ymm0  ; 1/a * (m - a)\n\n"

	src += "\t; ymm0: ln(a)\n"
	src += "\tlea rax, ln_a_table\n"
	src += "\tvpcmpeqq ymm3, ymm3, ymm3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd ymm0, [rax + ymm1*8], ymm3\n\n"

	src += "\t; ymm3: P(r) ~= ln(1 + r);\n"
	src += f"\t;\tHorner's method: degree {MINIMAX_DEG} minimax polynomial\n"
	for i, c in enumerate(coefs):
		k = MINIMAX_DEG - i
		if i == 0:
			src += f"\tvbroadcastsd ymm3, [coefs + ({i})*8]  ; c_{k} -> ymm3\n"
		else:
			src += f"\tvbroadcastsd ymm1, [coefs + ({i})*8]  ; c_{k}\n"
			src += f"\tvfmadd132pd ymm3, ymm1, ymm2  ; ymm3 * r + c_{k} -> ymm3\n"
	src += "\n"

	src += "\t; return ln(...)\n"
	src += "\tvaddpd ymm0, ymm0, ymm3  ; ln(a) + P(r) != ln(m)\n"
	src += "\tvaddpd ymm0, ymm0, ymm4  ; ln(m) + x*ln(2)\n"
	src += "\tret\n"
	
	src += "ln ENDP\n"
	src += "END\n"

	with open(f"src/ln{M_TRUNC}v3.asm", "w", encoding="utf-8") as file:
		file.write(str(src))
	
	t = time() - t
	print(f"Done. time: {t:.3f} s")