__S__   = "S"    # Magnitude of spin vectors
__F__   = "F"    # Supremum of flux vector magnitudes
__kT__  = "kT"   # Temperature
__J__   = "J"    # Heisenburg exchange coupling parameter (spin, neighboring spin)
__Je0__ = "Je0"  # Exchange coupling parameter (spin, local flux)
__Je1__ = "Je1"  # Exchange coupling parameter (spin, neighboring flux) & (flux, neighboring spin)
__Jee__ = "Jee"  # Exchange coupling parameter (flux, neighboring flux)
__B__   = "B"    # External (i.e. applied) magnetic field
__A__   = "A"    # Anisotropy
__b__   = "b"    # Biquadratic coupling
__D__   = "D"    # Dzyaloshinskii–Moriya interaction (DMI), i.e. magnetic skyrmions

__NODES__ = "__NODES__"  # enum
__EDGES__ = "__EDGES__"  # enum

NODE_PARAMETERS = [__S__, __F__, __kT__, __B__, __A__, __Je0__]
EDGE_PARAMETERS = [__J__, __Je1__, __Jee__,  __b__, __D__]
PARAMETERS = [__S__, __F__, __kT__, __J__, __Je0__, __Je1__, __Jee__, __B__, __A__, __b__, __D__]
