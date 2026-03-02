from . import csv
from ..build import VisualStudio
from ..config import Config
from ..model import MSD
from ..runtime import Runtime
from ..simulation import Simulation
from ..util import report_date, TypeCheckedAny as Any, TypeCheckedSequence as Sequence, TypeCheckedTuple as TTuple

real = int|float
vec = TTuple[real, real, real]  # class

CONTINUOUS_SPIN_MODEL = "CONTINUOUS_SPIN_MODEL"  # enum

LINEAR = "LINEAR"  # special values

DEFAULT_TOOL = VisualStudio()  # for assembling and linking

class IterateParameters:
	_fields = {
		# batch file meta-parameters
		"model":     (CONTINUOUS_SPIN_MODEL, Any),
		"mol_type":  (LINEAR, str),
		"randomize": (True, bool),
		"seed":      (None, Sequence[int]|int|None),  # i.e. unique

		"simCount": (0, int),  # required
		"freq":     (0, int),  # required

		"width":   (1, int),  # required
		"height":  (1, int),  # required
		"depth":   (1, int),  # required

		"molPosL": (None, int|None),
		"molPosR": (None, int|None),
		"topL":    (None, int|None),
		"bottomL": (None, int|None),
		"frontR":  (None, int|None),
		"backR":   (None, int|None),

		"kT": (0.1, real),  # required

		"B": (None, vec|None),

		"SL": (1.0, real),  # required
		"SR": (1.0, real),  # required
		"Sm": (1.0, real),  # required

		"FL": (None, real|None),
		"FR": (None, real|None),
		"Fm": (None, real|None),

		"JL": (None, real|None),
		"JR": (None, real|None),
		"Jm": (None, real|None),
		"JmL": (None, real|None),
		"JmR": (None, real|None),
		"JLR": (None, real|None),

		"Je0L": (None, real|None),
		"Je0R": (None, real|None),
		"Je0m": (None, real|None),

		"Je1L": (None, real|None),
		"Je1R": (None, real|None),
		"Je1m": (None, real|None),
		"Je1mL": (None, real|None),
		"Je1mR": (None, real|None),
		"Je1LR": (None, real|None),

		"JeeL": (None, real|None),
		"JeeR": (None, real|None),
		"Jeem": (None, real|None),
		"JeemL": (None, real|None),
		"JeemR": (None, real|None),
		"JeeLR": (None, real|None),

		"AL": (None, vec|None),
		"AR": (None, vec|None),
		"Am": (None, vec|None),

		"bL": (None, real|None),
		"bR": (None, real|None),
		"bm": (None, real|None),
		"bmL": (None, real|None),
		"bmR": (None, real|None),
		"bLR": (None, real|None),

		"DL": (None, vec|None),
		"DR": (None, vec|None),
		"Dm": (None, vec|None),
		"DmL": (None, vec|None),
		"DmR": (None, vec|None),
		"DLR": (None, vec|None)
	}

	__slots__ = tuple(_fields)

	def __init__(self):
		for field, (default, T) in self._fields.items():
			setattr(self, field, default)

	def __setattr__(self, key, value):
		T = IterateParameters._fields[key][1]
		if not isinstance(value, T):
			raise TypeError(f"{key} must be type {T}, got {type(value)}")
		# print(f"DEBUG: {T}; {type(key)} {key}; {type(value)} {value}")
		object.__setattr__(self, key, value)

	@staticmethod
	def load(file: str, verbose: bool=False) -> IterateParameters:
		obj = IterateParameters()
		with open(file, "r") as f:
			for line in f:
				line = line.strip()
				if line.startswith("#") or len(line) == 0:
					continue
				key, value = line.split("=", 2)
				key = key.strip()
				value = value.strip().split(" ", 3)
				if len(value) > 1:
					if verbose:  print(f"Parsing vector, {key} = {value}")
					value = tuple(float(v) for v in value)
				else:
					if verbose:  print(f"Parsing scalar, {key} = {value}")
					if value[0].lower() == "true":
						value = True
					elif value[0].lower() == "false":
						value = False
					else:
						try:
							value = int(value[0])
						except ValueError:
							try:
								value = float(value[0])
							except ValueError:
								value = value[0]  # keep value[0] as str
				setattr(obj, key, value)
		return obj
	
	def save(self, file: str) -> None:
		with open(file, "w", encoding="utf-8") as f:
			for key in self._fields:
				value = getattr(self, key)
				if value is None:
					continue
				if isinstance(value, tuple):
					value = " ".join(str(v) for v in value)
				f.write(f"{key} = {value}\n")

	def config(self) -> Config:
		msd = MSD(self.width, self.height, self.depth, \
			self.molPosL, self.molPosR, \
			self.topL, self.bottomL, self.frontR, self.backR)
		msd.programParameters = { "seed": self.seed }
		# TODO: parse self.mol_type
		return msd
	
	def compile(self, tool=DEFAULT_TOOL, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True) -> Runtime:
		return self.config().compile(tool, asm, _def, obj, dll, dir, copy_config)

	def sim(self, tool=DEFAULT_TOOL, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True, progress_bar: str=None) -> Simulation:
		rt = self.compile(tool, asm, _def, obj, dll, dir, copy_config)
		sim = Simulation(rt)
		if self.randomize:
			sim.randomize()
		sim.metropolis(self.simCount, self.freq, progress_bar=progress_bar)
		return sim

	def run(self, tool=DEFAULT_TOOL, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True, out: str=None, prefix: str=None, sim_progress_bar: str=None, out_progress_bar: str=None) -> str:
		if prefix is None:
			prefix = f"iteration, {report_date()}, "
		sim = None
		try:
			sim = self.sim(tool, asm, _def, obj, dll, dir, copy_config, sim_progress_bar)
			return csv(sim, dir=dir, out=out, prefix=prefix, params={ p: getattr(self, p) for p in IterateParameters._fields }, progress_bar=out_progress_bar)
		finally:
			if sim is not None:
				sim.rt.shutdown()
