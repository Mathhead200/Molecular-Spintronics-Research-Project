from __future__ import annotations
from ..runtime import MutableStateBuffer
from .data_view_wraper import DataViewWrapper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..runtime import DataView


# The full state of the Simulation at some simulation time, t.
class Snapshot(DataViewWrapper[MutableStateBuffer]):
	def __init__(self, view: DataView, t: int):
		assert isinstance(view.source, MutableStateBuffer)
		super().__init__(view)
		self.t = t
	
	def free(self) -> None:
		self.view.source.free()

	def __enter__(self):
		return self

	def __exit__(self, ex_type, ex_value, traceback):
		self.free()
		return False
