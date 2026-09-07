import os
import sys
import traceback
import json

# import MSD library from correct path: ~/lib/python
MSD_LIB_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../lib/python")
sys.path.append(MSD_LIB_PATH)
from MSD import *


def molType(t: str) -> c_void_p:
	''' Convert str to MSD MolType '''
	return {
		"LINEAR": MSD.LINEAR_MOL,
		"CIRCULAR": MSD.CIRCULAR_MOL
	}[t.upper()]

def flippingAlgorithm(algo: str) -> c_void_p:
	''' Convert str to MSD FlippingAlgorithm'''
	return {
		"UP_DOWN_MODEL": MSD.UP_DOWN_MODEL,
		"CONTINUOUS_SPIN_MODEL": MSD.CONTINUOUS_SPIN_MODEL
	}[algo.upper()]

def vec(v: list) -> Vector:
	''' Convert list to MSD Vector '''
	return Vector(*[float(x) for x in v])

# JSON keys mapped to their respective parse functions
MSD_INIT_ARGS = {
	"width": int,
	"height": int,
	"depth": int,
	
	"molType": molType,
	# TODO: molProto
	"molPosL": int,
	"molPosR": int,
	
	"topL": int,
	"bottomL": int,
	"frontR": int ,
	"backR": int
}
CONFIG_PARAMS = {
	"seed": int,
	"randomize": bool,
	"flippingAlgorithm": flippingAlgorithm,
	"reseed": bool
}
MSD_PARAMS = {
	"kT": float,
	"B": vec,
	"SL": float, "SR": float,
	"FL": float, "FR": float,
	"JL": float, "JR": float, "JmL": float, "JmR": float, "JLR": float,
	"Je0L": float, "Je0R": float,
	"Je1L": float, "Je1R": float, "Je1mL": float, "Je1mR": float, "Je1LR": float,
	"JeeL": float, "JeeR": float, "JeemL": float, "JeemR": float, "JeeLR": float,
	"bL": float, "bR": float, "bmL": float, "bmR": float, "bLR": float,
	"AL": vec, "AR": vec,
	"DL": vec, "DR": vec, "DmL": vec, "DmR": vec, "DLR": vec
}
MOL_NODE_PARAMS = {
	"Sm": float,
	"Fm": float,
	"Je0m": float,
	"Am": vec
}
MOL_EDGE_PARAMS = {
	"Jm": float,
	"Je1m": float,
	"Jeem": float,
	"bm": float,
	"Dm": vec
}
SIM_ARGS = {
	"simCount": int,
	"freq": int,
	"dkT": float,
	"dB": vec
}
ALL_TYPES = {
	**MSD_INIT_ARGS,
	**CONFIG_PARAMS,
	**MSD_PARAMS,
	**MOL_NODE_PARAMS,
	**MOL_EDGE_PARAMS,
	**SIM_ARGS
}

def readJSON() -> dict:
	''' Load and parse JSON from given input stream as a dictionary,
		converting data types where appropriate, and return.
	'''
	kw = json.loads(input())
	for k,v in kw.items():
		if k in ALL_TYPES:
			kw[k] = ALL_TYPES[k](v)
	return kw

def setParameters(msd: MSD, kw: dict):
	''' Updates the msd parameters (including molProto parameters)
	    as efficiently as possible. Ignores any keys in given dict that aren't
	    MSD.Parameters, Molecule.NodeParameters, or Molecule.EdgeParameters.
		Only updates the parameters given. Any excluded parameters will be left
		unchanged.
	'''
	if {"kT", "B"}.issuperset(kw.keys()):
		# Efficient update?:
		if "B" in kw:
			msd.B = kw["B"]
		if "kT" in kw:
			msd.kT = kw["kT"]
	else:
		# Filter dict
		kw_p = { k: v for k,v in kw.items() if k in MSD_PARAMS }
		kw_node_p = { k: v for k,v in kw.items() if k in MOL_NODE_PARAMS }
		kw_edge_p = { k: v for k,v in kw.items() if k in MOL_EDGE_PARAMS }
		
		# Update parameters
		if (kw_p):
			msd.parameters = MSD.Parameters(**{
				**msd.getParameters().__dict__,
				**kw_p })
		if kw_node_p or kw_edge_p:
			# update nodes and edges one at a time incase their parameters are
			# not uniform. This will only set node/edge parameters given in kw.
			# The rest will be unchanged, and may remain non-uniform.
			molProto = msd.getMolProto()
			if kw_node_p:
				for node in molProto.nodes:
					molProto.setNodeParamters(node.index, Molecule.NodeParameters(**{
						**node.getParameters().__dict__,
						**kw_node_p }))
			if kw_edge_p:
				for edge in molProto.edgesUnique:
					molProto.setEdgeParamters(edge.index, Molecule.EdgeParameters(**{
						**edge.getParameters().__dict__,
						**kw_edge_p }))
			msd.molProto = molProto

def maybeSeed(msd: MSD, kw: dict):
	''' Possibly seed the MSD based on the given JSON:
		{ "seed": int }
	'''
	if "seed" in kw:
		msd.seed = kw["seed"]
		return True
	return False

def maybeRandomize(msd: MSD, kw: dict, reseed: bool):
	''' Possibly randomize the MSD based on the given JSON
	    { "randomize": bool }
	'''
	if kw.get("randomize", False):
		msd.randomize(reseed)
		return True
	return False

def reset(msd: MSD, kw: dict):
	''' Either randomize or reinitialize the system based on the given JSON
	    { "randomize": bool, "reseed": bool, "seed": int }
		By default, MSD is not randomized, but is reseeded.
		If a custom seed is given, the system will use that seed.
	'''
	reseed = not maybeSeed(msd, kw) and kw.get("reseed", True)
	if not maybeRandomize(msd, kw, reseed):
		msd.reinitialize(reseed)

S_RESULTS = "results"
S_PARAMETERS = "parameters"
S_SEED = "seed"
S_MSD = "msd"
S_MOL = "mol"
S_ALL = {S_RESULTS, S_PARAMETERS, S_SEED, S_MSD, S_MOL}
def state(msd: MSD, what: set = S_ALL) -> str:
	''' Gets the state of the current MSD and converts it into JSON. '''

	def encodeMSD(obj):
		TYPES = {
			Vector: list,
			c_ulonglong: int,
			c_double: float
		}
		for pyType, encType in TYPES.items():
			if isinstance(obj, pyType):
				return encType(obj)
		raise TypeError("Can't encode Unrecognized type: " + str(type(obj)))
	
	state = {}

	if S_RESULTS in what:
		state[S_RESULTS] = msd.getResults().__dict__
	
	if S_PARAMETERS in what:
		state[S_PARAMETERS] = msd.getParameters().__dict__
	
	if S_SEED in what:
		state[S_SEED] = msd.seed
	
	if S_MSD in what:
		state[S_MSD] = [{
			"index": a.index,
			"pos": a.pos,
			"spin": a.spin,
			"flux": a.flux,
			"localM": a.localM
		} for a in msd]

	if S_MOL in what:
		molProto = msd.getMolProto()
		state[S_MOL] = {
			"nodes": [{
				"index": node.index,
				"parameters": node.getParameters().__dict__
			} for node in molProto.nodes],

			"edges": [{
				"index": edge.index,
				"parameters": edge.getParameters().__dict__,
				"src": edge.src,
				"dest": edge.dest,
				"direction": edge.direction
			} for edge in molProto.edgesUnique]
		}
	
	return json.dumps(state, default=encodeMSD, indent=None)

def sim(msd: MSD, n: int, dkT = 0., dB = 0.):
	# TODO NEW FEATURE: Allow for non-linear changes to kT and B by passing
	# 	either dKt and dB, or kT and B as functions of time (iterations), t,
	# 	instead of just constanst.
	if dkT == 0. and dB == 0.:
		msd.metropolis(n)
	else:
		p = msd.getParameters()
		for _ in range(n):
			msd.metropolis(1)
			# increment kT and B
			p.kT += dkT
			p.B += dB
			# update MSD object
			msd.kT = p.kT
			msd.B = p.B

def main():
	try:
		# 1. Read init parameters from stdin as single line JSON
		kw = readJSON()

		# Construct MSD object and set it up based on input
		msd = MSD(**{k: v for k,v in kw.items() if k in MSD_INIT_ARGS})
		if "flippingAlgorithm" in kw:
			msd.flippingAlgorithm = kw["flippingAlgorithm"]
		setParameters(msd, kw)
		# TODO NEW FEATURE: set molProto if one is given
		maybeSeed(msd, kw)
		maybeRandomize(msd, kw, reseed = False)
		
		print("READY")  # Confirm worker process is crrectly setup and ready for job requests

		# 2. Read a command, then execute
		while True:
			match(input().upper()):
				case "EXIT":
					# Gracefuly destroy this simulation and subprocess.
					break  # stop loop

				case "SET":
					# Change the parameters to the given values (efficiently).
					setParameters(msd, readJSON())
					print("DONE")  # parameters are set
				
				case "RUN":
					# Run metropolis() with the given "simCount" and "freq".
					# MSD state will be periodically sent to stdout.
					# After each state is printed, an input is expected before the simulation can continue.
					# This allows the user to prematurely end the program by sending "CANCEL".
					# All other messages will be treated as "CONTINUE", and the simulation will continue.
					kw = readJSON()
					simCount = kw["simCount"]
					freq = kw.get("freq", 0)
					dkT = kw.get("dkT", 0.)
					dB = kw.get("dB", 0.)
					if freq <= 0:
						freq = simCount
					cont = True
					print(state(msd))  # initial state
					cont = input().upper() != "CANCEL"  # should we continue?
					while cont and simCount >= freq:
						sim(msd, freq, dkT, dB)
						simCount -= freq
						print(state(msd))
						cont = input().upper() != "CANCEL"  # should we continue?
					if cont and simCount > 0:
						sim(msd, simCount, dkT, dB)
						print(state(msd))
						input()  # capture and ignore the last CANCEL? input
					print("DONE")  # signals end of new data

				case "GET":
					# Get the current state of the MSD without running.
					what = set(json.loads(input()))
					if len(what) == 0:
						what = S_ALL
					print(state(msd, what))
				
				case "RESET":
					reset(msd, readJSON())
					print(state(msd, what = {S_SEED}))

				case cmd:
					print("Unrecognized command: " + cmd, file=sys.stderr)
	
		# after loop
		print("GOODBYE")

	except EOFError:
		pass  # do nothing, proceed with shutdown

	except:
		print(traceback.format_exc(), file=sys.stderr)
		exit(1)
	
	finally:
		pass  # shutdown

if __name__ == "__main__":
	main()
