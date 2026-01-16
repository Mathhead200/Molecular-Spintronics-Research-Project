from typing import Iterable, Mapping, Sequence

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

# Wrapper allowing read-only access to underlying list/Sequence
class ReadOnlyList(Sequence):
	def __init__(self, lst: Sequence):  self._lst = lst

	def __len__(self):          return len(self._lst)	
	def __getitem__(self, i):   return self._lst[i]
	def __iter__(self):         return iter(self._lst)
	def __contains__(self, v):  return v in self._lst

# Parent to be extended/derived from
class AbstractReadableDict(Mapping):
	def __getitem__(self, key):   raise NotImplementedError()  # abstract
	def __iter__(self):           raise NotImplementedError()  # abstract
	def __contains__(self, key):  raise NotImplementedError()  # abstract

	def keys(self):    return iter(self)
	def values(self):  return (self[k] for k in self)
	def items(self):   return ((k, self[k]) for k in self)
	def get(self, key, default=None):  return self[key] if key in self else default

# Wrapper allowing read-only access to underlying dict
class ReadOnlyDict(AbstractReadableDict):
	def __init__(self, dct: dict):  self._dct = dct

	def __getitem__(self, key):   return self._dct[key]
	def __iter__(self):           return iter(self._dct)
	def __len__(self):            return len(self._dct)
	def __contains__(self, key):  return key in self._dct

	def get(self, key, default=None):  return self._dct.get(key, default)
