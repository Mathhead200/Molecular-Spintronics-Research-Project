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

	src += f"M_TRUNC     EQU {M_TRUNC}  ; Mantissa trancation length (in bits)\n"
	src += f"MINIMAX_DEG EQU {MINIMAX_DEG}  ; Degree of minimax polynomial\n\n"

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

	src += "; src (YMMn): four packed doubles in [1, 2)\n"
	src += "; dest (YMMn)\n"
	src += "; temp1, temp2, temp3 (YMMn)\n"
	src += "; tempG (general purpose register, e.g. rax)\n"
	src += "; PRECONDITIONS:\n"
	src += ";\t1. temp1, temp2, temp3 must not overlap with each other, dest, nor src.\n"
	src += ";\t2. dest and src can overlap.\n"
	src += "; POSTCONDITIONS:\n"
	src += ";\t1. All temps cobbered!\n"
	src += "_vln MACRO dest, src, temp1, temp2, temp3, tempG\n"

	src += "\t; Assume src[i] in [1,2) -> 0|111 1111 1111|mmmm mmmm mmmm ...\n"
	src += "\t;                   sign bit ^|^^ exponent ^|^^ mantissa ^^ ...\n"
	src += "\tvpsllq temp1, src, 12  ; remove sign (1 bit) + exponent (11 bits)\n"
	src += "\tvpsrlq temp1, temp1, 64 - M_TRUNC  ; most significant bits of mantissa -> indices in (paralell) tables\n\n"

	src += "\t; temp2: calculate residual, r = x/a - 1.0 = (x - a)/a\n"
	src += "\tlea rax, a_table\n"
	src += "\tvpcmpeqq temp3, temp3, temp3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd temp2, [rax + temp1*8], temp3\n"
	src += "\tvsubpd dest, src, temp2  ; x - a\n"
	src += "\tlea rax, a_inv_table\n"
	src += "\tvpcmpeqq temp3, temp3, temp3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd temp2, [rax + temp1*8], temp3\n"
	src += "\tvmulpd temp2, temp2, dest  ; 1/a * (x - a)\n\n"

	src += "\t; dest: ln(a)\n"
	src += "\tlea rax, ln_a_table\n"
	src += "\tvpcmpeqq temp3, temp3, temp3  ; load mask (all 1's)\n"
	src += "\tvgatherqpd dest, [rax + temp1*8], temp3\n\n"

	src += "\t; temp3: P(r) ~= ln(1 + r);\n"
	src += f"\t;\tHorner's method: degree {MINIMAX_DEG} minimax polynomial\n"
	for i, c in enumerate(coefs):
		k = MINIMAX_DEG - i
		if i == 0:
			src += f"\tvbroadcastsd temp3, [coefs + ({i})*8]  ; c_{k} -> temp3\n"
		else:
			src += f"\tvbroadcastsd temp1, [coefs + ({i})*8]  ; c_{k}\n"
			src += f"\tvfmadd132pd temp3, temp1, temp2  ; temp3 * r + c_{k} -> temp3\n"
	src += "\n"

	src += "\t; return ln(a) + P(r) ~= ln(x)\n"
	src += "\tvaddpd dest, dest, temp3\n"

	src +="ENDM\n"

	with open(f"src/ln.inc", "w", encoding="utf-8") as file:
		file.write(str(src))
	
	t = time() - t
	print(f"Done. time: {t:.3f} s")