from .driver import Driver
import os

class Runtime:
	def __init__(self, dll: str, delete: bool=False):
		self.dll = dll
		self.delete = delete  # delete on exit?
		self.driver = Driver(dll)

	def __enter__(self):
		pass  # TODO: (stub)

	def __exit__(self):
		if self.delete:
			os.remove(self.dll)
