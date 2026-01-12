from __future__ import annotations
from .driver import Driver
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from .config import Config

class Runtime:
	def __init__(self, config: Config, dll: str, delete: bool=False):
		self.dll = dll
		self.delete = delete  # delete on exit?
		self.driver = Driver(config, dll)

	def __enter__(self):
		return self
	
	def shutdown(self):
		self.driver.free()
		if self.delete:
			os.remove(self.dll)

	def __exit__(self, ex_type, ex_value, traceback):
		self.shutdown()
		return False
