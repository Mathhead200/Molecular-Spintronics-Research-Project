import sys
import os

# import MSD library from correct path: ~/lib/python
MSD_LIB_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../lib/python")
sys.path.append(MSD_LIB_PATH)
from MSD import *


def main():
	print("Under construction. (11-23-2022)")
	print()

	print("---- Molecule, i.e. MSD.MolProto test ----")
	cube_mol = MSD.MolProto(8)
	vertPairs = [
		(0, 1), (1, 2), (2, 3), (3, 0),
		(4, 5), (5, 6), (6, 7), (7, 0),
		(0, 4), (1, 5), (2, 6), (3, 7)
		]
	edgeIdx = []
	for a, b in vertPairs:
		edgeIdx.append(cube_mol.connectNodes(a, b))
	print("edgeIdx:", edgeIdx)
	for i in range(0, 4):
		cube_mol.setNodeParamters(i, Molecule.NodeParameters(Sm = 2, Fm = 0.1))
	for i in range(4, 8):
		cube_mol.setNodeParamters(i, Molecule.NodeParameters(Sm = 5, Fm = 0.25))
	cube_mol.leads = (0, 6)
	print("cube_mol:", cube_mol, sep = "\n")
	print()

	print("---- Serialization/Deserialization test ----")
	buf_size = cube_mol.serializationSize
	buffer = create_string_buffer(buf_size)
	print("buffer (empty):", buffer)
	cube_mol.serialize(buffer)
	print("buffer (full):", buffer)
	copy_mol = MSD.MolProto()
	print("copy_mol (empty):", copy_mol, sep = "\n")
	copy_mol.deserialize(buffer)
	print("copy_mol (full):", copy_mol, sep = "\n")
	print()

	print("---- MSD constructor test ----")
	msd = MSD(13, 10, 10, molPosL = 5, molPosR = 7, molType = MSD.CIRCULAR_MOL, topL = 3, bottomL = 7, frontR = 3, backR = 7)
	msd = MSD(18, 10, 10, molPosL = 5, molProto = copy_mol, topL = 3, bottomL = 7, frontR = 3, backR = 7)
	msd.flippingAlgorithm = MSD.UP_DOWN_MODEL
	print()

	print("---- MolProto and Molecule Iterators tests ----")
	mol = msd.getMolProto()
	print("Node count:", len(mol.nodes))
	print("True Edge count:", len(mol.edges))
	print("Unique Edge count:", len(mol.edgesUnique))
	print()

	print("---- seed test ----")
	print("seed (pre):", msd.seed)
	msd.seed = 0
	print("seed (post):", msd.seed)
	print()

	print("--- parameters/results test ----")
	p = MSD.Parameters(FL = 0.1, FR = 0.1)
	msd.setParameters(p)
	msd.kT = 0.5
	msd.B = Vector(0.2, 0, 0)
	print("Parameters:", msd.getParameters())
	msd.metropolis(100000)
	print("Results:", msd.results)
	print()

	print("---- spin/flux/localM test ----")
	x, y, z = 0, 3, 0
	a = x + msd.width * (y + msd.height * z)
	msd.setSpin(x, y, z, Vector(10, 0, 0))
	msd.setFlux(a, Vector(0, 0, -1))
	msd.setLocalM(a, Vector.I, Vector.J)
	print("spin:", msd.getSpin(x, y, z), msd.getSpin(a))
	print("flux:", msd.getFlux(x, y, z), msd.getFlux(a))
	print("localM:", msd.getLocalM(x, y, z), msd.getLocalM(a))
	print()

	print("---- dimensions and other properties test ----")
	print("n:", msd.n, msd.nL, msd.nR, msd.n_m, msd.n_mL, msd.n_mR, msd.nLR)
	print("dimensions:", msd.width, msd.height, msd.depth, msd.dimensions)
	print("molPos:", msd.molPosL, msd.molPosR, msd.molPos)
	print("innerBounds:", msd.topL, msd.bottomL, msd.frontR, msd.backR, msd.innerBounds)
	print("regions:", msd.FM_L_exists, msd.FM_R_exists, msd.mol_exists, msd.regions)
	print()
	
	print("---- reinitialize/randomize test ----")
	msd.reinitialize()
	print("Parameters (post-reinitialize):", msd.getParameters())
	print("Results (post-reinitialize):", msd.getResults())
	print("seed (post-reinitialize):", msd.seed)
	msd.randomize(reseed = False)
	print("Parameters (post-randomize):", msd.getParameters())
	print("Results (post-randomize):", msd.getResults())
	print("seed (post-randomize):", msd.seed)
	print()

	print("---- record test ----")
	msd.metropolis(int(2e6), freq = int(500e3))
	print(msd.record)
	msd.record = []
	print(msd.record)
	msd.record = [MSD.Results(t = 200)]
	print(msd.record)
	print()

	print("---- __getitem__/__setitem__ test")
	msd[x,y,z] = (Vector.ZERO, Vector.K)
	msd[a] = msd[x,y,z]
	print("__getitem__:", msd[x,y,z], msd[a])
	print()

	print("---- MSD iterator ----")
	for atom in msd:
		print(f"{atom.metaIndex}. {atom.pos} = {atom.index}: {atom.spin} + {atom.flux} = {atom.localM}")
	print("assert n == len: ", msd.n, len(msd))
	print()

	print("---- __del__ test ----")
	del msd
	print()

	print("---- Vector Math test ----")
	from math import pi
	v1 = Vector.I * 10
	v2 = Vector.ZERO + Vector.J + Vector.K
	print(v1 + v2)
	v3 = Vector.sphericalForm(1, 0, pi / 4)
	v4 = Vector.sphericalForm(1, 0, PI / 4.0)
	print(v4 - v3)
	v2.normalize()
	print(v2)


if __name__ == "__main__":
	main()
