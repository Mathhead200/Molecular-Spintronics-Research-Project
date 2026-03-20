import __future__
from .util import unique_path
from ..__init__ import __version__
from ..simulation import Simulation
from ..util import StrJoiner
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, TYPE_CHECKING
from tqdm import tqdm
import numpy as np
if TYPE_CHECKING:
	from ..simulation import Snapshot
	from pathlib import Path

class CSV:
	file = None
	out = None
	data = None
	progress_bar = None
	n_index_len = None

	def __init__(self, sim: Simulation, data_count: int):
		self.sim = sim
		self.data_count = data_count
	
	def _progress_bar_update(self, n=1) -> None:
		progress_bar = self.progress_bar
		if progress_bar is not None:
			progress_bar.update(n)

	def open(self, out: str=None, dir: Path|str=".", prefix: str=None, progress_bar: bool|str=None) -> CSV:
		sim = self.sim
		if progress_bar is not None and progress_bar is not False:
			if progress_bar is True:  progress_bar = "Writing CSV"
			# header (1 row) + data (c rows) + write (max(n, c) rows)
			self.progress_bar = tqdm(total=1 + self.data_count + max(len(sim.nodes), self.data_count), desc=progress_bar)
		
		self.data: list[str] = []  # Data rows. Each element is one line (no \n). These get merged with final snapshot on write.

		if out is None:
			out = unique_path(dir, prefix, suffix=".csv")
		self.out = out
		self.file = open(out, "w", encoding="utf-8")
		return self

	def close(self) -> None:
		self._n_index_len = None
		if self.file is not None:
			self.file.close()
			self.file = None
		self.out = None
		self.data = []
		if self.progress_bar is not None:
			self.progress_bar.close()
			self.progress_bar = None

	def __enter__(self):
		if self.file is None:
			self.open()
		return self
	
	def __exit__(self, ex_type, ex_value, traceback):
		if self.file is not None:
			self.close()
		return False

	def write_header(self, params: Mapping[str, Any]=None):
		sim = self.sim

		# header row, (old) e.g. t,,M_x,M_y,M_z,M_norm,M_theta,M_phi,,ML_x,ML_y,ML_z,ML_norm,ML_theta,ML_phi,,MR_x,MR_y,MR_z,MR_norm,MR_theta,MR_phi,,Mm_x,Mm_y,Mm_z,Mm_norm,Mm_theta,Mm_phi,,MS_x,MS_y,MS_z,MS_norm,MS_theta,MS_phi,,MSL_x,MSL_y,MSL_z,MSL_norm,MSL_theta,MSL_phi,,MSR_x,MSR_y,MSR_z,MSR_norm,MSR_theta,MSR_phi,,MSm_x,MSm_y,MSm_z,MSm_norm,MSm_theta,MSm_phi,,MF_x,MF_y,MF_z,MF_norm,MF_theta,MF_phi,,MFL_x,MFL_y,MFL_z,MFL_norm,MFL_theta,MFL_phi,,MFR_x,MFR_y,MFR_z,MFR_norm,MFR_theta,MFR_phi,,MFm_x,MFm_y,MFm_z,MFm_norm,MFm_theta,MFm_phi,,U,UL,UR,Um,UmL,UmR,ULR,,,x,y,z,m_x,m_y,m_z,s_x,s_y,s_z,f_x,f_y,f_z,,,width = 11,height = 25,depth = 25,molPosL = 5,molPosR = 5,topL = 10,bottomL = 14,frontR = 10,backR = 14,simCount = 1000000000,freq = 1000000,kT = 0.1,"B = <0, 0, 0>",SL = 1,SR = 1,Sm = 1,FL = 0,FR = 0,Fm = 0,JL = 1,JR = 1,Jm = 1,JmL = 1,JmR = -1,JLR = 0,Je0L = 0,Je0R = 0,Je0m = 0,Je1L = 0,Je1R = 0,Je1m = 0,Je1mL = 0,Je1mR = 0,Je1LR = 0,JeeL = 0,JeeR = 0,Jeem = 0,JeemL = 0,JeemR = 0,JeeLR = 0,"AL = <0, 0, 0>","AR = <0, 0, 0>","Am = <0, 0, 0>",bL = 0,bR = 0,bm = 0,bmL = 0,bmR = 0,bLR = 0,"DL = <0, 0, 0>","DR = <0, 0, 0>","Dm = <0, 0, 0>","DmL = <0, 0, 0>","DmR = <0, 0, 0>","DLR = <0, 0, 0>",molType = LINEAR,randomize = 1,seed = 2064714695,,msd_version = 6.2a\n
		headers = StrJoiner()
		# magnetization
		headers += 't,,m_x,m_y,m_z,m_norm,m_theta,m_phi,,'
		for r in sim.regions:  headers += f'm_{r}_x,m_{r}_y,m_{r}_z,m_{r}_norm,m_{r}_theta,m_{r}_phi,,'
		# magnetization due to spins
		headers += 's_x,s_y,s_z,s_norm,s_theta,s_phi,,'
		for r in sim.regions:  headers += f's_{r}_x,s_{r}_y,s_{r}_z,s_{r}_norm,s_{r}_theta,s_{r}_phi,,'
		# magnetic due to fluxes
		headers += 'f_x,f_y,f_z,f_norm,f_theta,f_phi,,'
		for r in sim.regions:  headers += f'f_{r}_x,f_{r}_y,f_{r}_z,f_{r}_norm,f_{r}_theta,f_{r}_phi,,'
		# energy
		headers += 'U,'
		for r in sim.regions:   headers += f'U_{r},'
		for e in sim.eregions:  headers += f'U_{e[0]}_{e[1]},'
		headers += ',,'
		# final snapshot (state)
		n_index_len = 1
		for n in sim.nodes:
			if isinstance(n, Sequence):
				n_index_len = max(n_index_len, len(n))
		coordinate_labels = ["x", "y", "z"]
		if n_index_len <= len(coordinate_labels):
			for i in range(n_index_len):  headers += coordinate_labels[i] + ','
		else:
			for i in range(n_index_len):  headers += f'x_{i},'
		self.n_index_len = n_index_len
		headers += f'm_x,m_y,m_z,s_x,s_y,s_z,f_x,f_y,f_z,,,'
		# parameters
		config = sim.rt.config
		for p, v in config.globalParameters.items():  headers += f'"{p} = {v}",'
		for r, map in config.regionNodeParameters.items():
			for p, v in map.items():
				headers += f'"{p}_{r} = {v}",'
		for r, map in config.regionEdgeParameters.items():
			for p, v in map.items():
				headers += f'"{p}_{r[0]}_{r[1]} = {v}",'
		# TODO: output local parameters?
		headers += ','
		# program parameters
		for p, v in config.programParameters.items():
			headers += f'"{p} = {v}",'
		headers += ','
		# optional custom user params
		if params is not None:
			for p, v in params.items():
				headers += f'"{p} = {v}",'
		# version
		headers += f',"mcheisenberg_version = {__version__}"\n'
		self.file.write(str(headers))
		self._progress_bar_update()
	
	def add_data(self, ss: Snapshot):
		sim = self.sim
		ss.ready("m", "s", "f", "u")
		line = StrJoiner()
		line += f'{ss.t},,'
		m = ss.m.value
		m_norm  = np.linalg.norm(m)
		m_theta = np.arccos(m[2]/m_norm) if m_norm != 0 else 0.0  # TODO: double check calculations
		m_phi   = np.arctan2(m[1], m[0]) if m_norm != 0 else 0.0
		line += f'{m[0]},{m[1]},{m[2]},{m_norm},{m_theta},{m_phi},,'
		for r in sim.regions:
			m = ss.m[r].value
			m_norm  = np.linalg.norm(m)
			m_theta = np.arccos(m[2]/m_norm) if m_norm != 0 else 0.0
			m_phi   = np.arctan2(m[1], m[0]) if m_norm != 0 else 0.0
			line += f'{m[0]},{m[1]},{m[2]},{m_norm},{m_theta},{m_phi},,'
		s = ss.s.value
		s_norm  = np.linalg.norm(s)
		s_theta = np.arccos(s[2]/s_norm) if s_norm != 0 else 0.0
		s_phi   = np.arctan2(s[1], s[0]) if s_norm != 0 else 0.0
		line += f'{s[0]},{s[1]},{s[2]},{s_norm},{s_theta},{s_phi},,'
		for r in sim.regions:
			s = ss.s[r].value
			s_norm  = np.linalg.norm(s)
			s_theta = np.arccos(s[2]/s_norm) if s_norm != 0 else 0.0
			s_phi   = np.arctan2(s[1], s[0]) if s_norm != 0 else 0.0
			line += f'{s[0]},{s[1]},{s[2]},{s_norm},{s_theta},{s_phi},,'
		f = ss.f.value
		f_norm  = np.linalg.norm(f)
		f_theta = np.arccos(f[2]/f_norm) if f_norm != 0 else 0.0
		f_phi   = np.arctan2(f[1], f[0]) if f_norm != 0 else 0.0
		line += f'{f[0]},{f[1]},{f[2]},{f_norm},{f_theta},{f_phi},,'
		for r in sim.regions:
			f = ss.f[r].value
			f_norm  = np.linalg.norm(f)
			f_theta = np.arccos(f[2]/f_norm) if f_norm != 0 else 0.0
			f_phi   = np.arctan2(f[1], f[0]) if f_norm != 0 else 0.0
			line += f'{f[0]},{f[1]},{f[2]},{f_norm},{f_theta},{f_phi},,'
		u = ss.u.value
		line += f'{u},'
		for r in chain(sim.regions, sim.eregions):
			u = ss.u[r].value
			line += f'{u},'
		line += ',,'
		self.data.append(line)
		self._progress_bar_update()
	
	def write_data(self) -> None:
		sim = self.sim
		n_index_len = self.n_index_len
		node_iter = iter(sim.nodes)
		sim.ready("m")
		for line in self.data:
			try:
				n = next(node_iter)
				if isinstance(n, Sequence):
					for coord in n:  line += f'{coord},'
					line += ',' * (n_index_len - len(n))
				else:
					line += f'{n},'
					line += ',' * (n_index_len - 1)
				m = sim.m[n].value
				line += f'{m[0]},{m[1]},{m[2]},'
				s = sim.s[n].value
				line += f'{s[0]},{s[1]},{s[2]},'
				f = sim.f[n].value
				line += f'{f[0]},{f[1]},{f[2]},,,'
			except StopIteration:
				line += ',' * n_index_len  # blank lines
			line += '\n'
			self.file.write(str(line))
			self._progress_bar_update()
		
		# any remaining state rows
		while True:
			try:
				line = StrJoiner()
				n = next(node_iter)
				line += "," * (2 + 3 * (7 + 7 * len(sim.regions)) + 3 + len(sim.regions) + len(sim.eregions))  # t + M,S,F + U
				if isinstance(n, Sequence):
					for coord in n:  line += f'{coord},'
					line += ',' * (n_index_len - len(n))
				else:
					line += f'{n},'
					line += ',' * (n_index_len - 1)
				m = sim.m[n].value
				line += f'{m[0]},{m[1]},{m[2]},'
				s = sim.s[n].value
				line += f'{s[0]},{s[1]},{s[2]},'
				f = sim.f[n].value
				line += f'{f[0]},{f[1]},{f[2]}\n'
				self.file.write(str(line))
				self._progress_bar_update()
			except StopIteration:
				break


def csv(sim: Simulation, out: str=None, dir: Path|str=".", prefix: str=None, params: Mapping[str, Any]=None, progress_bar: bool|str=None) -> str:
	csv = CSV(sim, data_count=len(sim.history))
	with csv.open(out, dir, prefix, progress_bar):
		csv.write_header(params)
		for t in sim.history:
			csv.add_data(sim.history[t])
		csv.write_data()
		return csv.file.name
