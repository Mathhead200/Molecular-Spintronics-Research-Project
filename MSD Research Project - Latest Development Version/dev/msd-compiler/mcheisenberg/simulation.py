from __future__ import annotations
from .runtime import Runtime
import numpy

class Simulation:
	def __init__(self, rt: Runtime):
		self.rt = rt
	
	# TODO: wrapper which converts everything to Numpy arrays.ndarray
	#	and adds simulation logic like record, aggregates (e.g. mag, M, MS, MF, etc.), and other math stuff
