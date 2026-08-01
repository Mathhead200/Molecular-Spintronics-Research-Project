from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from ..build import VisualStudio
from ..runtime import Runtime
from typing import TYPE_CHECKING, override
import os
if TYPE_CHECKING:
	from typing import Any, Callable
	from ..config import Config

_dll_map: dict = {}

def _build_config(fn, args) -> Config:
	if   type(args) is tuple:  return fn(*args)
	elif type(args) is dict:   return fn(**args)
	else:                      return fn(args)

def _key(args):
	if type(args) is dict:  return frozenset(args.items())
	else:                   return args

class RuntimePool:
	@staticmethod
	def _initialize(config_fn, args, dll_map, tool, asm, def_, obj, dll, dir, progress_bars):
		pass  # TODO

	def __init__(self, config_fn: Callable[..., Config], config_args: list=[()], max_workers: int=None, tool=VisualStudio(), asm: str=None, def_: str=None, obj: str=None, dll: str=None, dir: str=None, progress_bars: bool=False):
		"""
		@param config_fn: MUST BE PICKLEABLE! This function returns a configuration.
		@param config_args: (optional) MUST BE PICKLEABLE! A list specifying different configurations to be monitored by this pool.
			Each configuration is represented as a tuple of arguments that will be passed to config_fn.
		"""
		global _dll_map

		if max_workers is None:
			max_workers = cpu_count()

		for args in config_args:
			config = _build_config(config_fn, args)
			if asm is not None:   asm  = str(Path(asm).with_suffix(f"{args}.asm"))
			if def_ is not None:  def_ = str(Path(def_).with_suffix(f"{args}.def"))
			if obj is not None:   obj  = str(Path(obj).with_suffix(f"{args}.obj"))
			if dll is not None:   dll  = str(Path(dll).with_suffix(f"{args}.dll"))
			runtime = config.compile(tool=tool, asm=asm, def_=def_, obj=obj, dll=dll, dir=dir, progress_bars=progress_bars)
			_dll_map[_key(args)] = (runtime.dll, runtime._delete)
		
		self.config_fn = config_fn
		self.executor = ProcessPoolExecutor(max_workers=max_workers, initializer=RuntimePool._initialize, initargs=(config_fn, config_args, _dll_map, tool, asm, def_, obj, dll, dir, progress_bars))
		self.futures = []

	def submit(self, task: Callable[..., Any], config_args, task_args):
		config = _build_config(self.config_fn, config_args)
		config.precompile(progress_bars=False)  # to initialize fields like SIZEOF_NODE for runtime validation
		dll, _ = _dll_map[_key(config_args)]
		runtime = Runtime(config=config, dll=dll, delete=False)
		if type(config_args) is tuple and type(task_args) is tuple:
			future = self.executor.submit(task, runtime, *config_args, *task_args)  # Can't send Runtime thru pipe (pickleing) Move to _initialize
		elif type(config_args) is dict and type(task_args) is dict:
			future = self.executor.submit(task, runtime, **(config_args | task_args))
		elif type(config_args) is tuple and type(task_args) is dict:
			future = self.executor.submit(task, runtime, *config_args, **task_args)
		else:
			future = self.executor.submit(task, runtime, config_args, task_args)
		self.futures.append(future)
		return future

	def as_completed(self):
		for future in as_completed(self.futures):
			yield future

	def __enter__(self):
		return super().__enter__()

	def __exit__(self, exc_type, exc_value, traceback):
		for _, (dll, dll_temp) in _dll_map:
			if dll_temp:
				os.remove(dll)
		return super().__exit__(exc_type, exc_value, traceback)
