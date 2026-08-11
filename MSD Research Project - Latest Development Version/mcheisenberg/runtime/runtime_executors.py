from __future__ import annotations
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from ..build import VisualStudio
from ..runtime import Runtime
import atexit
import os
if TYPE_CHECKING:
	from concurrent.futures import Future
	from typing import Any, Callable
	from ..config import Config

class RuntimePoolExecutor:
	"""
	Allows multiple tasks, each with their own independant Runtime, to run in parallel.
	These tasks will be deligated to worker sub-Processes, each spinning up a new Runtime
	when needed and handling cleanup as well.
	"""

	# static:
	_dll_paths: dict = None      # worker input: contains .dll paths
	_runtime_cache: dict = None  # worker state

	@staticmethod
	def _initialize_worker(dll_paths):
		RuntimePoolExecutor._dll_paths = dll_paths
		RuntimePoolExecutor._runtime_cache = {}
		atexit.register(RuntimePoolExecutor._destruct_worker)

	@staticmethod
	def _destruct_worker():
		for runtime in RuntimePoolExecutor._runtime_cache.values():
			runtime.shutdown()

	@staticmethod
	def _build_config(fn, args) -> Config:
		if   type(args) is tuple:  return fn(*args)
		elif type(args) is dict:   return fn(**args)
		else:                      return fn(args)

	@staticmethod
	def _key(args):
		""" key-ify, i.e. make args hashable, in case they are a mutable dict. """
		if type(args) is dict:  return frozenset(args.items())
		else:                   return args

	@staticmethod
	def _task_wrapper(task: Callable[..., Any], config_fn, config_args, task_args, use_cache, runtime_initializer, progress_bars):
		key = RuntimePoolExecutor._key(config_args)

		def new_runtime():
			config = RuntimePoolExecutor._build_config(config_fn, config_args)
			config.precompile(progress_bars=progress_bars)  # to initialize fields like SIZEOF_NODE for runtime validation
			dll = RuntimePoolExecutor._dll_paths[key]
			runtime = Runtime(config, dll, delete=False)
			if runtime_initializer is not None:
				if   type(config_args) is tuple:  runtime_initializer(runtime,  *config_args)
				elif type(config_args) is dict:   runtime_initializer(runtime, **config_args)
				else:                             runtime_initializer(runtime,   config_args)
			return runtime
		
		if use_cache:
			if key in RuntimePoolExecutor._runtime_cache:
				runtime = RuntimePoolExecutor._runtime_cache[key]
			else:
				runtime = new_runtime()
				RuntimePoolExecutor._runtime_cache[key] = runtime
		else:
			runtime = new_runtime()

		try:
			if type(config_args) is tuple and type(task_args) is tuple:
				result = task(runtime, *config_args, *task_args)
			elif type(config_args) is dict and type(task_args) is dict:
				result = task(runtime, **(config_args | task_args))
			elif type(config_args) is tuple and type(task_args) is dict:
				result = task(runtime, *config_args, **task_args)
			else:
				result = task(runtime, config_args, task_args)
		
		finally:
			if not use_cache:
				runtime.shutdown()

		return result

	def __init__(self, config_fn: Callable[..., Config], config_list: list=[()], max_workers: int=None, tool=VisualStudio(), asm: str=None, def_: str=None, obj: str=None, dll: str=None, dir: str=None, progress_bars: bool=False):
		"""
		@param config_fn: This function returns a configuration.
			Preconditions: picklable, and pure function
		@param config_list: (optional) A list specifying different configurations to be monitored by this pool.
			Each configuration (i.e. list element) is represented as an args object that will be passed to the config_fn.
			1. If type is tuple, args are *unpacked in orer.
			2. If type is dict, args are **unpacked as keyword args.
			3. Else, args are treated as a single object ans passed as the sole argument.
			Prconditions: picklable, and (if not dict) should be a valid dict key.
		"""
		self.config_fn = config_fn

		self.runtimes: list[Runtime] = []
		dll_paths: dict[Any, str] = {}
		for n, args in enumerate(config_list, start=1):
			config = RuntimePoolExecutor._build_config(config_fn, args)
			_asm = None if asm  is None else str(Path(asm).with_suffix(f".{n}.asm"))
			_def = None if def_ is None else str(Path(def_).with_suffix(f".{n}.def"))
			_obj = None if obj  is None else str(Path(obj).with_suffix(f".{n}.obj"))
			_dll = None if dll  is None else str(Path(dll).with_suffix(f".{n}.dll"))
			runtime = config.compile(tool=tool, asm=_asm, def_=_def, obj=_obj, dll=_dll, dir=dir, copy_config=False, progress_bars=progress_bars)
			self.runtimes.append(runtime)
			dll_paths[RuntimePoolExecutor._key(args)] = runtime.dll

		self.executor = ProcessPoolExecutor(max_workers=max_workers, initializer=RuntimePoolExecutor._initialize_worker, initargs=(dll_paths,))

		self.futures: list[Future] = []

	def submit(self, task: Callable[..., Any], config_args=(), task_args=(), use_cache: bool=False, runtime_initializer: Callable[..., None]=None, progress_bars: bool=False):
		"""
		Submit a task to the underlying process pool.
		
		@param config_args must match one of the elements of config_list.
			If config_list was omitted, the empty tuple () represents the single config availble to the pool.

		@param task_args will be passed after config_args to the task.
			Similar to config_args (see #__init__), task_args can be tuple, dict, or object.
			
			Easiest if task_args follows a scheme (tuple, dict, or simple object) compatable with config_args.
			Compatable schemes include (type(config_args), type(taskargs)) is
			1. (tuple, tuple)
			2. (dict, dict)
			3. (tuple, dict)
			4. (object, object)  <-- default
			If none of these schemes are used config_args, and task_args are passed as (object, object) and must be explicitlly destructured.
			
			Preconditions: picklable
		
		@param use_cache should this task use the runtime cached.
			Using the runtime cache allows a subprocess to avoid recreating new runtimes for the same config.
			When using the runtime cache, the runtime must be explicitlly reinitialized or randomized at the start of each task,
			otherwise the stale state from the previous task will still be present.
		
		@param runtime_initializer called when a new runtime must be created.
			Allows for any configurating required on a per-runtime basis as opposed to a per-task basis.
			Precondition: picklable
		"""
		future = self.executor.submit(RuntimePoolExecutor._task_wrapper, task, self.config_fn, config_args, task_args, use_cache, runtime_initializer, progress_bars)
		self.futures.append(future)
		return future

	def as_completed(self):
		for future in as_completed(self.futures):
			yield future

	def __enter__(self):
		for i in range(len(self.runtimes)):
			self.runtimes[i] = self.runtimes[i].__enter__()
		self.executor = self.executor.__enter__()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		result = self.executor.__exit__(exc_type, exc_value, traceback)  # invokes executor.shutdown(wait=True)
		for runtime in self.runtimes:
			runtime.__exit__(exc_type, exc_value, traceback)  # deletes .dll files
		return result

	def free_runtimes(self):
		for runtime in self.runtimes:
			runtime.shutdown()  # deletes .dll files
		self.runtimes = []  # to avoid redundant frees

	def shutdown(self, wait=True, *args, cancel_futures=False):
		self.executor.shutdown(wait, *args, cancel_futures=cancel_futures)
		if wait:  # Can't free runtimes if workers are still executing
			self.free_runtimes()

	def kill_workers(self):
		self.executor.kill_workers()
		self.free_runtimes()
	
	def terminate_workers(self):
		self.executor.terminate_workers()


class BalancedRuntimePoolExecutor:
	# static:
	_config_fn = None    # worker input 
	_config_args = None  # worker input
	_dll = None          # worker input
	_runtime = None      # worker state

	@staticmethod
	def _initialize_worker(config_fn, config_args, dll):
		# TODO: set sub-process priority (pip psutil)
		BRPE._config_fn = config_fn
		BRPE._config_args = config_args
		BRPE._dll = dll
		BRPE._runtime = None  # lazy constructed
		atexit.register(BRPE._destruct_worker)

	@staticmethod
	def _destruct_worker():
		if BRPE._runtime is not None:
			BRPE._runtime.shutdown()

	@staticmethod
	def _build_config(fn, args) -> Config:
		if   type(args) is tuple:  return fn(*args)
		elif type(args) is dict:   return fn(**args)
		else:                      return fn(args)

	@staticmethod
	def _key(args):
		""" key-ify, i.e. make args hashable, in case they are a mutable dict. """
		if type(args) is dict:  return frozenset(args.items())
		else:                   return args

	def __init__(self, config_fn: Callable[..., Config], config_list: list=[()], max_workers: int|Sequence=None, tool=VisualStudio(), asm: str=None, def_: str=None, obj: str=None, dll: str=None, dir: str=None, progress_bars: bool=False):
		"""
		@param config_fn: This function returns a configuration.
			Preconditions: picklable, and pure function
		@param config_list: (optional) A list specifying different configurations to be monitored by this pool.
			Each configuration (i.e. list element) is represented as an args object that will be passed to the config_fn.
			1. If type is tuple, args are *unpacked in orer.
			2. If type is dict, args are **unpacked as keyword args.
			3. Else, args are treated as a single object ans passed as the sole argument.
			Prconditions: picklable, and (if not dict) should be a valid dict key.
		"""
		if max_workers is None:
			max_workers = os.cpu_count() or 1

		# convert max_workers to a sequence (if not already)
		N = len(config_list)
		if isinstance(max_workers, Sequence):
			if len(max_workers) != N:
				raise ValueError(f"length mismatch: len(config_list) ({N}) != len(max_workers) ({len(max_workers)})")
		else:
			quotient = max_workers // N
			remainder = max_workers % N
			max_workers = [quotient + 1 if i < remainder else quotient for i in range(N)]

		self.runtimes: list[Runtime] = []
		self.executors: dict[Any, ProcessPoolExecutor] = {}
		for n, (args, w) in enumerate(zip(config_list, max_workers), start=1):
			config = BRPE._build_config(config_fn, args)
			_asm = None if asm   is None else str(Path(asm ).with_suffix(f".{n}.asm"))
			_def = None if def_  is None else str(Path(def_).with_suffix(f".{n}.def"))
			_obj = None if obj   is None else str(Path(obj ).with_suffix(f".{n}.obj"))
			_dll = None if dll   is None else str(Path(dll ).with_suffix(f".{n}.dll"))
			runtime = config.compile(tool=tool, asm=_asm, def_=_def, obj=_obj, dll=_dll, dir=dir, copy_config=False, progress_bars=progress_bars)
			self.runtimes.append(runtime)  # for cleanup

			key = BRPE._key(args)
			self.executors[key] = ProcessPoolExecutor(max_workers=w, initializer=BRPE._initialize_worker, initargs=(config_fn, args, runtime.dll))

		self.futures: list[Future] = []

	@staticmethod
	def _task_wrapper(task: Callable[..., Any], task_args, use_cache, runtime_initializer, progress_bars):
		config_args = BRPE._config_args

		def initialize_runtime(runtime):
			if runtime_initializer is not None:
				if   type(config_args) is tuple:  runtime_initializer(runtime,  *config_args)
				elif type(config_args) is dict:   runtime_initializer(runtime, **config_args)
				else:                             runtime_initializer(runtime,   config_args)

		# first time (lazy build)
		if BRPE._runtime is None:
			config = BRPE._build_config(BRPE._config_fn, config_args)
			config.precompile(progress_bars=progress_bars)  # to initialize fields like SIZEOF_NODE for runtime validation
			BRPE._runtime = Runtime(config, BRPE._dll, delete=False)
			initialize_runtime(BRPE._runtime)
		
		if use_cache:
			runtime = BRPE._runtime
		else:
			runtime = Runtime(deepcopy(BRPE._runtime.config), BRPE._dll, delete=False)
			initialize_runtime(runtime)

		try:
			if type(config_args) is tuple and type(task_args) is tuple:
				result = task(runtime, *config_args, *task_args)
			elif type(config_args) is dict and type(task_args) is dict:
				result = task(runtime, **(config_args | task_args))
			elif type(config_args) is tuple and type(task_args) is dict:
				result = task(runtime, *config_args, **task_args)
			else:
				result = task(runtime, config_args, task_args)
		
		finally:
			if not use_cache:
				runtime.shutdown()

		return result

	def submit(self, task: Callable[..., Any], config_args=(), task_args=(), use_cache: bool=False, runtime_initializer: Callable[..., None]=None, progress_bars: bool=False):
		"""
		Submit a task to the underlying process pool.
		
		@param config_args must match one of the elements of config_list.
			If config_list was omitted, the empty tuple () represents the single config availble to the pool.

		@param task_args will be passed after config_args to the task.
			Similar to config_args (see #__init__), task_args can be tuple, dict, or object.
			
			Easiest if task_args follows a scheme (tuple, dict, or simple object) compatable with config_args.
			Compatable schemes include (type(config_args), type(taskargs)) is
			1. (tuple, tuple)
			2. (dict, dict)
			3. (tuple, dict)
			4. (object, object)  <-- default
			If none of these schemes are used config_args, and task_args are passed as (object, object) and must be explicitlly destructured.
			
			Preconditions: picklable
		
		@param use_cache should this task use the runtime cached.
			Using the runtime cache allows a subprocess to avoid recreating new runtimes for the same config.
			When using the runtime cache, the runtime must be explicitlly reinitialized or randomized at the start of each task,
			otherwise the stale state from the previous task will still be present.
		
		@param runtime_initializer called when a new runtime must be created.
			Allows for any configurating required on a per-runtime basis as opposed to a per-task basis.
			Precondition: picklable
		"""
		key = BRPE._key(config_args)
		future = self.executors[key].submit(BRPE._task_wrapper, task, task_args, use_cache, runtime_initializer, progress_bars)
		self.futures.append(future)
		return future

	def as_completed(self):
		for future in as_completed(self.futures):
			yield future

	def __enter__(self):
		for i in range(len(self.runtimes)):
			self.runtimes[i] = self.runtimes[i].__enter__()
		for key in self.executors:
			self.executors[key] = self.executors[key].__enter__()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		result = None
		for executor in self.executors.values():
			result = result or executor.__exit__(exc_type, exc_value, traceback)  # invokes executor.shutdown(wait=True)

		for runtime in self.runtimes:
			runtime.__exit__(exc_type, exc_value, traceback)  # deletes .dll files
		
		return result

	def free_runtimes(self):
		for runtime in self.runtimes:
			runtime.shutdown()  # deletes .dll files
		self.runtimes = []  # to avoid redundant frees

	def shutdown(self, wait=True, *args, cancel_futures=False):
		for executor in self.executors.values():
			executor.shutdown(wait, *args, cancel_futures=cancel_futures)
		
		if wait:  # Can't free runtimes if workers are still executing
			self.free_runtimes()

	def kill_workers(self):
		for executor in self.executors.values():
			executor.kill_workers()
		self.free_runtimes()
	
	def terminate_workers(self):
		for executor in self.executors.values():
			executor.terminate_workers()

BRPE = BalancedRuntimePoolExecutor
