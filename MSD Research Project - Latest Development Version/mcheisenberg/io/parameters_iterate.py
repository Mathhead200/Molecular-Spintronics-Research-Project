from __future__ import annotations
from . import CSV
from ..build import VisualStudio
from ..config import Config
from ..model import MSD, __FML__, __FMR__, __mol__
from ..runtime import Runtime
from ..simulation import Simulation, Snapshot
from ..util import report_date, TypeCheckedAny as Any, TypeCheckedSequence as Sequence, TypeCheckedTuple as TTuple
from collections import defaultdict
from typing import Callable

real = int|float
vec = TTuple[real, real, real]  # class

CONTINUOUS_SPIN_MODEL = "CONTINUOUS_SPIN_MODEL"  # enum
UP_DOWN_MODEL = "UP_DOWN_MODEL"  # enum

LINEAR = "LINEAR"  # special values
CIRCULAR = "CIRCULAR"

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
					value = tuple(float(v) for v in value)
					if verbose:  print(f"Parsed vector ({type(value).__name__}), {key} = {value}")
				else:
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
					if verbose:  print(f"Parsed scalar ({type(value).__name__}), {key} = {value}")
				setattr(obj, key, value)
		return obj
	
	# TODO: test
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
		if self.model != CONTINUOUS_SPIN_MODEL:
			if self.model == UP_DOWN_MODEL:  raise NotImplementedError(f"{UP_DOWN_MODEL} is not yet implemented")
			else:                            raise ValueError(f"Unsupported model: {self.model}")
		
		if self.mol_type != LINEAR:
			if self.mol_type == CIRCULAR:  raise NotImplementedError(f"{CIRCULAR} mol. type is not yet implemented")
			else:                          raise NotImplementedError(f"Custom .mmt files not yet implemented")

		msd = MSD(self.width, self.height, self.depth, \
			self.molPosL, self.molPosR, \
			self.topL, self.bottomL, self.frontR, self.backR)
		msd.programParameters = { "seed": self.seed }

		# Required parameters:
		msd.globalParameters = { "kT": self.kT }      # override MSD default global parameters
		msd.regionNodeParameters = {                  # override MSD default region node parameters (default is empty {})
			__FML__: { "S": self.SL },
			__FMR__: { "S": self.SR },
			__mol__: { "S": self.Sm }
		}
		msd.regionEdgeParameters = defaultdict(dict)  # override MSD default edge parameters

		# Optional parameters:
		if self.B  is not None:
			msd.globalParameters["B"] = self.B

		for region_suffix, region in [("L", __FML__), ("R", __FMR__), ("m", __mol__)]:
			for param_prefix in ["F", "A", "Je0"]:
				# e.g. if self.AL is not None:  msd.regionNodeParameters["FML"]["A"] = self.AL
				value = getattr(self, f"{param_prefix}{region_suffix}")
				if value is not None:
					msd.regionNodeParameters[region][param_prefix] = value
		
		for region_suffix, region0, region1 in [("L", __FML__, __FML__), ("R", __FMR__, __FMR__), ("m", __mol__, __mol__), ("mL", __FML__, __mol__, ), ("mR", __mol__, __FMR__), ("LR", __FML__, __FMR__)]:
			for param_prefix in ["J", "Je1", "Jee", "b", "D"]:
				# e.g. if self.JmL is not None:  msd.regionEdgeParameters["FML", "mol"]["J"] = self.JmL
				value = getattr(self, f"{param_prefix}{region_suffix}")
				if value is not None:
					msd.regionEdgeParameters[region0, region1][param_prefix] = value

		return msd
	
	def compile(self, tool=DEFAULT_TOOL, asm: str=None, def_: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True) -> Runtime:
		return self.config().compile(tool, asm, def_, obj, dll, dir, copy_config)

	def sim(self, tool=DEFAULT_TOOL, asm: str=None, def_: str=None, obj: str=None, dll: str=None, dir: str=None, copy_config: bool=True, callback: Callable[[Snapshot], None]=None, progress_bar: str|bool=None) -> Simulation:
		rt = self.compile(tool, asm, def_, obj, dll, dir, copy_config)
		sim = Simulation(rt)
		if self.randomize:
			sim.randomize()
		sim.metropolis(self.simCount, self.freq, callback=callback, progress_bar=progress_bar)
		return sim

	def run(self, tool=DEFAULT_TOOL, asm: str=None, def_: str=None, obj: str=None, dll: str=None, temp_dir: str=None, copy_config: bool=True, out: str=None, out_dir: str=None, prefix: str=None, sim_progress_bar: str|bool=None, out_progress_bar: str|bool=None) -> str:
		if prefix is None:
			prefix = f"iteration, {report_date()}"
		with self.compile(tool, asm, def_, obj, dll, temp_dir, copy_config) as rt:
			sim = Simulation(rt)
			csv = CSV(sim, data_count=(self.simCount // self.freq) + 1)
			with csv.open(out, out_dir, prefix, out_progress_bar):
				csv.write_header(params={ p: getattr(self, p) for p in IterateParameters._fields })
				if self.randomize:  sim.randomize()
				def callback(snapshot):  csv.add_data(snapshot)
				sim.metropolis(self.simCount, self.freq, callback, reuse_buffer=True, progress_bar=sim_progress_bar)
				csv.write_data()
				return csv.file.name
