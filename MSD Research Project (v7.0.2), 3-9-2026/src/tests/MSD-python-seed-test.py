import sys
import os

# import MSD library from correct path: ~/lib/python
MSD_LIB_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../lib/python")
sys.path.append(MSD_LIB_PATH)
from MSD import *


msd = MSD(
	width = 11, height = 10, depth = 10,
	molPosL = 5, molPosR = 5,
	topL = 0, bottomL = 9,
	frontR = 0, backR = 9 )

msd.setParameters(
	MSD.Parameters(
		# Tempurature
		kT = 0.3,

		# External Magnetic Field
		B = Vector(0, 0, 0),

		# Magnetude of spin vectors
		SL = 1,
		SR = 1,

		# Maximum Magnetude of spin fluctuation ("flux") vectors
		FL = 0,
		FR = 0,

		# Heisenberg exchange coupling between two neighboring spins
		JL = 1,
		JR = 1,
		JmL = 0.75,
		JmR = -0.75,
		JLR = 0,

		# exchange coupling between a spin and its local flux
		Je0L = 0,
		Je0R = 0,

		# exchange coupling between a spin and its neighboring flux (and vice versa)
		Je1L = 0,
		Je1R = 0,
		Je1mL = 0,
		Je1mR = 0,
		Je1LR = 0,

		# exchange coupling between two neighboring fluxes
		JeeL = 0,
		JeeR = 0,
		JeemL = 0,
		JeemR = 0,
		JeeLR = 0,

		# Anisotropy constant(s), as vectors 
		AL = Vector(0, 0, 0),
		AR = Vector(0, 0, 0),

		# Biquadratic coupling
		bL = 0,
		bR = 0,
		bmL = 0,
		bmR = 0,
		bLR = 0,

		# Dzyaloshinskii-Moriya (i.e. Skyrmion) interaction, as vectors
		DL = Vector(0, 0, 0),
		DR = Vector(0, 0, 0),
		DmL = Vector(0, 0, 0),
		DmR = Vector(0, 0, 0),
		DLR = Vector(0, 0, 0) ))

msd.setMolParameters(
	Molecule.NodeParameters(
		Sm = 1,
		Fm = 0,
		Je0m = 0,
		Am = Vector(0, 0, 0) ),
	Molecule.EdgeParameters(
		Jm = 1,
		Je1m = 0,
		Jeem = 0,
		bm = 0,
		Dm = Vector(0, 0,0 ) ))

msd.seed = 0
print("seed:", msd.seed)
msd.randomize(reseed = False)  # do NOT forget about reseed arg
print("M:", msd.results.M)
print("ML:", msd.results.M)
print("MR:", msd.results.M)
print("Mm:", msd.results.M)
print("U:", msd.results.U)
