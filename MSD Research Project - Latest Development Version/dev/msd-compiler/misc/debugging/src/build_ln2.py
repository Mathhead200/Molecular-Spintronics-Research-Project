from math import log, ceil
from time import time
from remez import remez
from struct import pack

M_TRUNC = 10
MINIMAX_DEG = 3

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
	cmem = int(4 * ceil(len(coefs) / 4))  # round to nearest multiple of 4

	src = StrJoiner()

	src += "OPTION CASEMAP:NONE\n\n"
	
	src += "include vec.inc\n"
	src += "include dumpreg.inc\n\n"

	src += f"M_TRUNC EQU {M_TRUNC}   ; Mantissa trancation length (in bits)\n\n"

	src += ".data\n"
	src += "TABLE SEGMENT ALIGN(32)\n"
	src += "\t\t\t; a, 1/a, log(a)\n"
	src += "log_table\t"
	first_line = True
	M = 2**M_TRUNC
	for m in range(M):
		a = 1 + (m + 0.5)/M  # reference point on [1, 2)
		log_a = log(a)
		if first_line:
			first_line = False
		else:
			src += "\t\t\t"
		src += f"dq {a}, {1/a}, {log_a}, 0.0\n"
	src += "TABLE ENDS\n\n"

	src += "COEFS SEGMENT ALIGN(32)\n"
	src += "coefs\t"
	first_line = True
	for i in range(cmem):
		if first_line:
			first_line = False
		else:
			src += "\t\t"
		if i < len(coefs):
			src += f"dq {coefs[i]}  ; c_{MINIMAX_DEG - i}\n"
		else:
			src += "dq 0.0  ; padding (unused)\n"
	src += "COEFS ENDS\n\n"

	src += ".code\n"
	src += "; @param (xmm0) - double x in [0, 1)\n"
	src += "; @return (xmm0)\n"
	src += "PUBLIC ln\n"
	src += "ln PROC\n"

	src += "\t; Assume xmm0[i] in [1,2) -> 0|111 1111 1111|mmmm mmmm mmmm ...\n"
	src += "\t;                   sign bit ^|^^ exponent ^|^^ mantissa ^^ ...\n"
	src += "\tvmovq rax, xmm0\n"
	src += "\tshl rax, 12  ; remove sign (1 bit) + exponent (11 bits)\n"
	src += "\tshr rax, 64 - M_TRUNC  ; most significant bits of mantissa -> index in table\n"
	src += "\tshl rax, 5  ; mul 32 (sizeof ymmword ptr)\n"
	src += "\tlea rcx, log_table\n"
	src += "\tvmovapd ymm1, ymmword ptr [rcx + rax]  ; [a, 1/a, ln(a), 0.0]\n\n"
	
	src += "\t; xmm2: calculate residual, r = x/a - 1.0 = (x - a)/a\n"
	src += "\tvsubsd xmm2, xmm0, xmm1  ; x - a\n"
	src += "\t_vpermj ymm1, ymm1  ; [1/a, a, ln(a), 0.0]\n"
	src += "\tvmulsd xmm2, xmm2, xmm1  ; xmm0 *= 1/a\n\n"
	# src += "\tmov rax, 0BFF0000000000000h  ; -1.0\n"
	# src += "\tvmovq xmm2, rax\n"
	# src += "\tvfmadd231sd xmm2, xmm0, xmm1  ; x*(1/a) + (-1.0)\n\n"

	src += "\t; xmm0: ln(a)\n"
	src += "\t_vpermk ymm0, ymm1  ; [ln(a), 0.0, 1/a, a]\n\n"

	src += "\t; xmm3: P(r) ~= ln(1 + r);\n"
	src += "\t;\tHorner's method: degree 5 minimax polynomial\n"
	for i in range(0, cmem, 4):
		if i == 0:
			src += f"\tvmovapd ymm3, ymmword ptr [coefs + ({i // 4})*32]\n"
			if (i+1 >= len(coefs)):  break
			src += "\t_vpermj ymm1, ymm3\n"
		else:
			src += f"\tvmovapd ymm1, ymmword ptr [coefs + ({i // 4})*32]\n"
			src += "\tvfmadd132sd xmm3, xmm1, xmm2\n"
			if (i+1 >= len(coefs)): break
			src += "\t_vpermj ymm1, ymm1\n"
		src += "\tvfmadd132sd xmm3, xmm1, xmm2\n"
		if (i+2 >= len(coefs)):  break
		src += "\t_vpermk ymm1, ymm1\n"
		src += "\tvfmadd132sd xmm3, xmm1, xmm2\n"
		if (i+3 >= len(coefs)):  break
		src += "\t_vpermj ymm1, ymm1\n"
		src += "\tvfmadd132sd xmm3, xmm1, xmm2\n"
	# src += f"\tmov rax, {coefs[0]}  ; c_{MINIMAX_DEG} -> xmm3\n"
	# src += "\tvmovq xmm3, rax\n"
	# for i in range(1, MINIMAX_DEG + 1):
	# 	k = MINIMAX_DEG - i
	# 	if c[i] is not None:
	# 		src += f"\tmov rax, {c[i]}  ; c_{k}\n"
	# 		src += "\tvmovq xmm1, rax\n"
	# 		src += f"\tvfmadd132sd xmm3, xmm1, xmm2  ; xmm3 * r + c_{k} -> xmm3\n"
	# 	else:
	# 		src += f"\tvmulsd xmm3, xmm3, xmm2  ; xmm3 * r + (c_{k}=0) -> xmm3\n"
	src += "\n"

	src += "\t; return ln(a) + P(r) ~= ln(x)\n"
	src += "\tvaddsd xmm0, xmm0, xmm3\n"
	src += "\tret\n"
	
	src += "ln ENDP\n"
	src += "END\n"

	with open(f"src/ln{M_TRUNC}v2.asm", "w", encoding="utf-8") as file:
		file.write(str(src))
	
	t = time() - t
	print(f"Done. time: {t:.3f} s")