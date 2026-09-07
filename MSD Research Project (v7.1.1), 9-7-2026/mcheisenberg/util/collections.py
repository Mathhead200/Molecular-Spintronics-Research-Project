from __future__ import annotations
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from collections.abc import Iterable, Sequence
	from typing import Any


type OrderedSet[T] = Mapping[T, None]

def ordered_set(xs: Iterable) -> dict[Any, None]:
	return { x: None for x in xs }


class ReadOnlyCollection(Collection):
	def __init__(self, c: Collection):  self._obj = c

	def __len__(self):          return len(self._obj)
	def __iter__(self):         return iter(self._obj)
	def __contains__(self, v):  return v in self._obj

	def __repr__(self):  return repr(self._obj)
	def __str__(self):   return str(self._obj)

# Wrapper allowing read-only access to underlying list/Sequence
class ReadOnlyList[T](ReadOnlyCollection[T]):
	def __init__(self, lst: Sequence):  super().__init__(lst)
	
	def __getitem__(self, i):   return self._obj[i]

# Parent to be extended/derived from
class AbstractReadableDict[K, V](Mapping[K, V]):
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
	def __or__(self, other)   -> dict:  return dict(self) | dict(other)
	def __ror__(self, other)  -> dict:  return dict(other) | dict(self)
	def __and__(self, other)  -> dict:  return { k: self[k] for k in self if k in other }
	def __rand__(self, other) -> dict:  return { k: other[k] for k in other if k in self }
	def __sub__(self, other)  -> dict:  return { k: self[k] for k in self if k not in other }
	def __rsub__(self, other) -> dict:  return { k: other[k] for k in other if k not in self }

	# TODO: __str__, __hash__, __reeduce(_ex)__,


# Wrapper allowing read-only access to underlying dict
class ReadOnlyDict[K, V](AbstractReadableDict[K, V]):
	def __init__(self, d: Mapping):  self._obj = d

	def __getitem__(self, key):   return self._obj[key]    # override
	def __iter__(self):           return iter(self._obj)   # override
	def __len__(self):            return len(self._obj)    # override

	def get(self, key, default=None):  return self._obj.get(key, default)  # override

	def __str__(self):   return str(self._obj)
	def __repr__(self):  return repr(self._obj)


# Changes the defalt view behaviour. Otherwise still a Mapping.
class ReadOnlyOrderedSet[T](ReadOnlyDict[T, None]):
	def __str__(self):   return str(self._obj.keys())
	def __repr__(self):  return repr(self._obj.keys())
