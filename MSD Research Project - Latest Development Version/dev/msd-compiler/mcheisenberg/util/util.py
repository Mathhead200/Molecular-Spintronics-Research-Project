from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from collections.abc import Iterable


def floats(values: Iterable) -> tuple[float]:
	return (float(x) for x in values)

def is_pow2(n: int) -> bool:
	return n > 0 and (n & (n - 1)) == 0

def div8(n: int) -> int:
	return int(n // 8) if n is not None else None


class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)
