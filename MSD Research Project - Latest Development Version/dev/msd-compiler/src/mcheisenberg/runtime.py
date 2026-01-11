from .driver import Driver
import os

class Runtime(Driver):
	def __init__(self, dll: str, delete: bool=False):
		super().__init__(dll)
		self.dll = dll
		self.delete = delete  # delete on exit?

	def __enter__(self):
		pass  # TODO: (stub)

	def __exit__(self):
		if self.delete:
			os.remove(self.dll)
