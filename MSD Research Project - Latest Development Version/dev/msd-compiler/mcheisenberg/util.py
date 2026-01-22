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
	def __len__(self, key):       raise NotImplementedError()  # abstract

	# def __contains__(self, key):  # defined in Mapping
	# def keys(self):    ... # defined in Mapping
	# def values(self):  ... # defined in Mapping
	# def items(self):   ... # defined in Mapping
	# def get(self, key, default=None):  ... # defined in Mapping
	# def __eq__(self, other):  ... # defined in Mapping
	# def __ne__(self, other):  ... # defined in Mapping
	def __or__(self, other)  -> dict:  return dict(self) | dict(other)
	def __ror__(self, other) -> dict:  return dict(other) | dict(self)

	# TODO: __str__, __hash__, __reeduce(_ex)__,

# Wrapper allowing read-only access to underlying dict
class ReadOnlyDict(AbstractReadableDict):
	def __init__(self, d: Mapping):  self._obj = d

	def __getitem__(self, key):   return self._obj[key]    # override
	def __iter__(self):           return iter(self._obj)   # override
	def __len__(self):            return len(self._obj)    # override

	def get(self, key, default=None):  return self._obj.get(key, default)  # override

	def __str__(self):   return str(self._obj)
	def __repr__(self):  return repr(self._obj)

# Just changes the defalt view behaviour. Otherwise still a Mapping.
class ReadOnlyOrderedSet(ReadOnlyDict):
	def __str__(self):   return str(self._obj.keys())
	def __repr__(self):  return repr(self._obj.keys())


# Interface for a numerical/mathamatical object, e.g. numpy ndarray, or float.
#	Any object which can be used in mathamatical expressions, namely
#	parameters proxies and result proxies can inherit from this class.
class Numeric:
	@property
	def value(self) -> Any:  # return somthing "numerical"
		raise NotImplementedError()  # abstract

	def __add__(self, addend):       return self.value + addend
	def __sub__(self, subtrahend):   return self.value - subtrahend
	def __mul__(self, multiplier):   return self.value * multiplier
	def __truediv__(self, divisor):  return self.value / divisor
	def __floordiv__(self, divisor): return self.value // divisor
	def __mod__(self, modulus):      return self.value % modulus
	def __pow__(self, exponent):     return self.value ** exponent
	def __neg__(self):               return -self.value
	def __pos__(self):               return +self.value
	def __abs__(self):               return abs(self.value)

	def __radd__(self, augend):         return augend + self.value
	def __rsub__(self, minuend):        return minuend - self.value
	def __rmul__(self, multiplicand):   return multiplicand * self.value
	def __rtruediv__(self, dividend):   return dividend / self.value
	def __rfloordiv__(self, dividend):  return dividend // self.value
	def __rmod__(self, dividend):       return dividend % self.value
	def __rpow__(self, base):           return base ** self.value

	def __iadd__(self, addend):        self.value = self + addend;     return self
	def __isub__(self, subtrahend):    self.value = self - subtrahend; return self
	def __imul__(self, multiplier):    self.value = self * multiplier; return self
	def __itruediv__(self, divisor):   self.value = self / divisor;    return self
	def __ifloordiv__(self, divisor):  self.value = self // divisor;   return self
	def __imod__(self, modulus):       self.value = self % modulus;    return self
	def __ipow__(self, exponent):      self.value = self ** exponent;  return self

	def __eq__(self, other):  return self.value == other
	def __ne__(self, other):  return self.value != other
	def __lt__(self, other):  return self.value < other
	def __le__(self, other):  return self.value <= other
	def __gt__(self, other):  return self.value > other
	def __ge__(self, other):  return self.value >= other

	def __str__(self):   return str(self.value)
	def __repr__(self):  return repr(self.value)
	def __hash__(self):  return hash(self.value)

# [inteface] Behaves as int
class IInt(Numeric):
	def __int__(self) -> int:    return self.value
	def __index__(self) -> int:  return self.value
	def __bool__(self) -> bool:  return bool(self.value)

	def __or__(self, other):      return self.value | other
	def __xor__(self, other):     return self.value ^ other
	def __and__(self, other):     return self.value & other
	def __lshift__(self, shift):  return self.value << shift
	def __rshift__(self, shift):  return self.value >> shift
	def __invert__(self):         return ~self.value

	def __ror__(self, other):      return other | self.value
	def __rxor__(self, other):     return other ^ self.value
	def __rand__(self, other):     return other & self.value
	def __rlshift__(self, value):  return value << self.value
	def __rrshift__(self, value):  return value << self.value
