from __future__ import annotations
from datetime import date, datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from collections.abc import Iterable


class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)


def floats(values: Iterable) -> tuple[float]:
	return (float(x) for x in values)

def is_pow2(n: int) -> bool:
	return n > 0 and (n & (n - 1)) == 0

def div8(n: int) -> int:
	return int(n // 8) if n is not None else None


def report_date(t: date=None) -> str:
	if t is None:
		t = date.today()
	return t.strftime("%m-%d-%Y")

def report_time(t: date=None) -> str:
	if t is None:
		t = datetime.now()
	return t.strftime("%I:%M:%S %p").strip("0").casefold()

def report_datetime(t: date=None) -> str:
	if t is None:
		t = datetime.now()
	return f"{report_date(t)} @ {report_time(t)}"
