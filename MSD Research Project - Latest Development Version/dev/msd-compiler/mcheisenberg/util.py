from __future__ import annotations
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from typing import Any


def floats(values: Iterable) -> tuple[float]:
	return (float(x) for x in values)

def is_pow2(n: int) -> bool:
	return n > 0 and (n & (n - 1)) == 0

def div8(n: int) -> int:
	return int(n // 8) if n is not None else None

def ordered_set(xs: Iterable) -> dict[Any, None]:
	return { x: None for x in xs }


class StrJoiner:
	def __init__(self):
		self.pieces = []

	def __iadd__(self, value):
		self.pieces.append(value)
		return self

	def __str__(self):
		return "".join(self.pieces)


class ReadOnlyCollection(Collection):
	def __init__(self, c: Collection):  self._obj = c

	def __len__(self):          return len(self._obj)
	def __iter__(self):         return iter(self._obj)
	def __contains__(self, v):  return v in self._obj

	def __repr__(self):  return repr(self._obj)
	def __str__(self):   return str(self._obj)

# Wrapper allowing read-only access to underlying list/Sequence
class ReadOnlyList(ReadOnlyCollection):
	def __init__(self, lst: Sequence):  super().__init__(lst)
	
	def __getitem__(self, i):   return self._obj[i]

# Parent to be extended/derived from
class AbstractReadableDict(Mapping):
	def __getitem__(self, key):   raise NotImplementedError()  # abstract
	def __iter__(self):           raise NotImplementedError()  # abstract
	def __contains__(self, key):  raise NotImplementedError()  # abstract

	# def keys(self):    return iter(self)
	# def values(self):  return (self[k] for k in self)
	# def items(self):   return ((k, self[k]) for k in self)
	# def get(self, key, default=None):  return self[key] if key in self else default
	def __or__(self, other):   return dict(self) | dict(other)
	def __ror__(self, other):  return dict(other) | dict(self)

	# TODO: __eq__, __ne__, __str__, __hash__, __reeduce(_ex)__,

# Wrapper allowing read-only access to underlying dict
class ReadOnlyDict(AbstractReadableDict):
	def __init__(self, d: Mapping):  self._obj = d

	def __getitem__(self, key):   return self._obj[key]    # override
	def __iter__(self):           return iter(self._obj)   # override
	def __len__(self):            return len(self._obj)
	def __contains__(self, key):  return key in self._obj  # override

	def get(self, key, default=None):  return self._obj.get(key, default)  # override

	def __str__(self):   return str(self._obj)
	def __repr__(self):  return repr(self._obj)
