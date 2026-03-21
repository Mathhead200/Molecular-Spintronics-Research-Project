from __future__ import annotations
from ..runtime import Runtime
from .simulation_proxies import ChiProxy, CProxy
from .simulation_util import VEC_J, VEC_ZERO, rtvec
from .snapshot import Snapshot
from .data_view_wraper import DataViewWrapper
from typing import TYPE_CHECKING
from tqdm import tqdm
if TYPE_CHECKING:
	from ..runtime import MutableStateBuffer
	from .simulation_util import numpy_vec
	from typing import Callable


# Runtime wrapper which converts everything to numpy float arrays and adds
#	simulation logic like recording snapshots, aggregates (e.g. m, U, n, etc.), etc.
class Simulation(DataViewWrapper):
	x: ChiProxy
	c: CProxy

	def __init__(self, rt: Runtime):
		super().__init__(rt)
		self.rt: Runtime = rt
		self.t: int = 0  # current simulation time since last restart (i.e. reinitialization, or randomization)
		self.history: dict[int, Snapshot] = {}

		self._x_proxy = ChiProxy(self)
		self._c_proxy = CProxy(self)

	def seed(self, *seed: int) -> None:
		self.rt.seed(seed)
	
	def reinitialize(self, initSpin: numpy_vec=VEC_J, initFlux: numpy_vec=VEC_ZERO, clear_history: bool=True) -> None:
		self.rt.reinitialize(rtvec(initSpin), rtvec(initFlux))
		if clear_history:  self.clear_history()
	
	def randomize(self, *seed: int, clear_history: bool=True) -> None:
		self.rt.randomize(*seed)
		if clear_history:  self.clear_history()

	def metropolis(self, iterations: int, freq: int=0, callback: Callable[[Snapshot], None]=None, bookend: bool=True, reuse_buffer: MutableStateBuffer|bool=False, progress_bar: bool|str=None) -> None:
		"""
		Run the metropolis algorithm on this model for the given number of iterations.
		May specify a recording/sampling period (freq), i.e. record a snapshot every freq iterations.
		If bookend, a final snapshot will be recorded after all iterations are completed. E.g.
			sim.metropolis(100, freq=10, bookend=True) will generate 11 snapshopts at times, t=0, 10, 20, ..., 100.
			sim.metropolis(100, freq=10, bookend=False) will generate 10 snapshots at times, t=0, 10, 20, ..., 90.
				The algorithm will still run for 100 iterations.
			sim.metropolis(100, freq=11, bookend=True) will generate 11 snapshots at times, t=0, 11, 22, ..., 99, 100.
		"""
		if not reuse_buffer:        reuse_buffer = None                       # new buffer each time
		elif reuse_buffer is True:  reuse_buffer = self.rt.allocate_buffer()  # allocate a single buffer to reuse for each

		if progress_bar is not None and progress_bar is not False:
			if progress_bar is True:  progress_bar = "Metropolis"
			progress_bar = tqdm(total=iterations, desc=progress_bar)
		
		if not freq:
			self.rt.metropolis(iterations)
			self.t += iterations
			if progress_bar is not None:  progress_bar.update(iterations)
		
		else:
			if callback is None:  callback = self.record  # default action is to record in Simulation.history
			
			callback(self.snapshot(reuse_buffer))
			while iterations > freq:
				self.rt.metropolis(freq)
				self.t += freq
				callback(self.snapshot(reuse_buffer))
				iterations -= freq
				if progress_bar is not None:  progress_bar.update(freq)
			if iterations != 0:
				self.rt.metropolis(iterations)
				self.t += iterations
				if progress_bar is not None:  progress_bar.update(iterations)
			if bookend:
				callback(self.snapshot(reuse_buffer))
		if progress_bar is not None:  progress_bar.close()
	
	def snapshot(self, buffer=None) -> Snapshot:
		if buffer is None:
			buffer = self.rt.allocate_buffer()
		data = self.rt.snapshot(buffer)
		return Snapshot(data, self.t)

	def record(self, snapshot: Snapshot) -> None:
		self.history[snapshot.t] = snapshot

	def clear_history(self) -> None:
		self.t = 0
		self.history = {}

for param in ["x", "c"]:
	setattr(DataViewWrapper, param, property(
		fget=lambda self,        _p=param: getattr(self, f"_{_p}_proxy"),
		fset=lambda self, value, _p=param: setattr(getattr(self, f"_{_p}_proxy"), "value", value)
	))
