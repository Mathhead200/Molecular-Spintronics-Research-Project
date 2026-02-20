from . import csv
from ..config import Config
from ..model import MSD
from ..runtime import Runtime
from ..simulation import Simulation
from ..util import TypeCheckedSequence as Sequence, TypeCheckedAny as Any
from datetime import date

type vec = tuple[float, float, float]

CONTINUOUS_SPIN_MODEL = "CONTINUOUS_SPIN_MODEL"  # enum

LINEAR = "LINEAR"  # special values

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

		"kT": (0.1, float),  # required

		"B": (None, vec|None),

		"SL": (1.0, float),  # required
		"SR": (1.0, float),  # required
		"Sm": (1.0, float),  # required

		"FL": (None, float|None),
		"FR": (None, float|None),
		"Fm": (None, float|None),

		"JL": (None, float|None),
		"JR": (None, float|None),
		"Jm": (None, float|None),
		"JmL": (None, float|None),
		"JmR": (None, float|None),
		"JLR": (None, float|None),

		"Je0L": (None, float|None),
		"Je0R": (None, float|None),
		"Je0m": (None, float|None),

		"Je1L": (None, float|None),
		"Je1R": (None, float|None),
		"Je1m": (None, float|None),
		"Je1mL": (None, float|None),
		"Je1mR": (None, float|None),
		"Je1LR": (None, float|None),

		"JeeL": (None, float|None),
		"JeeR": (None, float|None),
		"Jeem": (None, float|None),
		"JeemL": (None, float|None),
		"JeemR": (None, float|None),
		"JeeLR": (None, float|None),

		"AL": (None, vec|None),
		"AR": (None, vec|None),
		"Am": (None, vec|None),

		"bL": (None, float|None),
		"bR": (None, float|None),
		"bm": (None, float|None),
		"bmL": (None, float|None),
		"bmR": (None, float|None),
		"bLR": (None, float|None),

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
		print(f"DEBUG: {T}; {type(key)} {key}; {type(value)} {value}")
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
					value = float(value[0])
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
	
	def compile(self, tool=None, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=None) -> Runtime:
		return self.config().compile(tool, asm, _def, obj, dll, dir, copy_config)

	def sim(self, tool=None, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=None) -> Simulation:
		rt = self.compile(tool, asm, _def, obj, dll, dir, copy_config)
		sim = Simulation(rt)
		if self.randomize:
			sim.randomize()
		sim.metropolis(self.simCount, self.freq)
		return sim

	def run(self, tool=None, asm: str=None, _def: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=None, out: str=None, prefix: str=None) -> str:
		if prefix is None:
			d = date.today().strftime("%m-%d-%Y")
			prefix = f"iteration, {d}, "
		sim = None
		try:
			sim = self.sim(tool, asm, _def, obj, dll, dir, copy_config)
			return csv(sim, dir=dir, out=out, prefix=prefix, params={ p: getattr(self, p) for p in IterateParameters._fields })
		finally:
			if sim is not None:
				sim.rt.shutdown()
