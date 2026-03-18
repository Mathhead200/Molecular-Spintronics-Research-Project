S_   = "S"    # Magnitude of spin vectors
F_   = "F"    # Supremum of flux vector magnitudes
kT_  = "kT"   # Temperature
J_   = "J"    # Heisenburg exchange coupling parameter (spin, neighboring spin)
Je0_ = "Je0"  # Exchange coupling parameter (spin, local flux)
Je1_ = "Je1"  # Exchange coupling parameter (spin, neighboring flux) & (flux, neighboring spin)
Jee_ = "Jee"  # Exchange coupling parameter (flux, neighboring flux)
B_   = "B"    # External (i.e. applied) magnetic field
A_   = "A"    # Anisotropy
b_   = "b"    # Biquadratic coupling
D_   = "D"    # Dzyaloshinskii–Moriya interaction (DMI), i.e. magnetic skyrmions

NODES_ = "__NODES__"  # enum
EDGES_ = "__EDGES__"  # enum

NODE_PARAMETERS = [S_, F_, kT_, B_, A_, Je0_]
EDGE_PARAMETERS = [J_, Je1_, Jee_,  b_, D_]
SCALAR_PARAMETERS = [S_, F_, kT_, Je0_, J_, Je1_, Jee_, b_]
VECTOR_PARAMETERS = [B_, A_, D_]
PARAMETERS = [S_, F_, kT_, J_, Je0_, Je1_, Jee_, B_, A_, b_, D_]
