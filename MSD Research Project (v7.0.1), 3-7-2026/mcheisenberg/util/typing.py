from collections.abc import Iterable, Collection, Sequence, Set, Mapping
from typing import Any, get_origin

# metaclass
class _AnyMeta(type(Any)):
	def __instancecheck__(cls, obj):
		return True

class TypeCheckedAny(metaclass=_AnyMeta):  pass

# metaclass
class _IterableMeta(type(Iterable)):
	def __instancecheck__(cls, obj):
		assert cls._origin is not None
		if not isinstance(obj, cls._origin):
			return False
		if cls._E is None:
			return True  # raw type, can't check elements
		return all(isinstance(ele, cls._E) for ele in obj)

class _TypeCheckedIterable(metaclass=_IterableMeta):
	""" Type constructor """

	_origin = None
	_E = None

	@classmethod
	def for_origin(cls, origin):
		return type(f"TypeChecked{origin.__name__}", (cls,), {"_origin": origin})

	@classmethod
	def __class_getitem__(cls, E):
		return type(f"TypeChecked{cls._origin.__name__}[{E.__name__}]", (cls,), {"_origin": cls._origin, "_E": E})

# template/generic classes 
TypeCheckedIterable = _TypeCheckedIterable.for_origin(Iterable)
TypeCheckedCollection = _TypeCheckedIterable.for_origin(Collection)
TypeCheckedSequence = _TypeCheckedIterable.for_origin(Sequence)
TypeCheckedSet = _TypeCheckedIterable.for_origin(Set)
TypeCheckedMapping = _TypeCheckedIterable.for_origin(Mapping)

# metaclass
class _TupleMeta(type(tuple)):
	def __instancecheck__(cls, obj):
		assert cls._E is not None
		if not isinstance(obj, tuple):
			return False
		if isinstance(cls._E, tuple):
			return len(obj) == len(cls._E) and all(isinstance(ele, E) for ele, E in zip(obj, cls._E))  # tuple of types given
		else:
			return all(isinstance(ele, cls._E) for ele in obj)  # only one type was given

class TypeCheckedTuple(metaclass=_TupleMeta):
	""" Type constructor """

	_E = None

	@classmethod
	def __class_getitem__(cls, E):
		if not isinstance(E, tuple):  subscript = E.__name__                        # single type given. all elements must be of this type
		elif len(E) == 1:             subscript = f"{E[0].__name__}, "              # len==1 tuple has trailing comma
		else:                         subscript = ", ".join(X.__name__ for X in E)  # explict tuple of types given
		return type(f"{cls.__name__}[{subscript}]", (cls,), {"_E": E})
