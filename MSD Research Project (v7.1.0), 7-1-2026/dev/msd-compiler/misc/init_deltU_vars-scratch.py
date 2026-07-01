reg_m1 = "ymm2"  # or ymm0 if m' == s'
reg_m  = "ymm5"  # or ymm3 if m  == s
reg_dm = "ymm8"  # or ymm6 if Δm == Δs
src += "\t; (param) ymm0: s' (new)\n"
if flux_mode:
	src += "\t; (param) ymm1: f' (new)\n"
	src += "\tvaddpd ymm2, ymm0, ymm1  ; m' (new)\n"
else:                # Not in flux mode:
	reg_m1 = "ymm0"  # m' == s'; just use the same register for both symbols
	reg_m  = "ymm3"  # m  == s ; just use the same register for both symbols
	reg_dm = "ymm6"  # Δm == Δs; just use the same register for both symbols
src += f"\tvmovapd ymm3, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_SPIN]  ; s (current)\n"
if flux_mode:
	src += f"\tvmovapd ymm4, ymmword ptr [nodes + ({index})*SIZEOF_NODE + OFFSETOF_FLUX]  ; f (current)\n"
	src += "\tvaddpd ymm5, ymm3, ymm4  ; m (current)\n"
src += "\tvsubpd ymm6, ymm0, ymm3  ; \\Delta s\n"
if flux_mode:
	src += "\tvsubpd ymm7, ymm1, ymm4  ; \\Delta f\n"
	src += "\tvsubpd ymm8, ymm2, ymm5  ; \\Delta m\n"
src += "\n"