from collections.abc import Iterable, Collection, Sequence, Set, Mapping
from typing import Any, get_origin

class TypeCheckedAny(Any):
	def __instancecheck__(cls, obj):
		return True

class TypeCheckedIterable[E](Iterable[E]):
	def __instancecheck__(cls, obj):
		return isinstance(obj, get_origin(cls)) and all(isinstance(ele, E) for ele in obj)

class TypeCheckedCollection[E](Collection[E], TypeCheckedIterable[E]):  pass
class TypeCheckedSequence[E](Sequence[E], TypeCheckedIterable[E]):  pass
class TypeCheckedSet[E](Set[E], TypeCheckedIterable[E]):  pass
class TypeCheckedMapping[E](Mapping[E], TypeCheckedIterable[E]):  pass
