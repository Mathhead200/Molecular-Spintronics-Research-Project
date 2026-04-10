from __future__ import annotations
from ctypes import Structure, c_double, sizeof
from typing import TYPE_CHECKING
from ..util import Incrementer
if TYPE_CHECKING:
	from ..config import Config

c_double_3 = c_double * 3

def _f(pad: Incrementer) -> str:
	return f"_pad{pad()}"

def _1(pad: Incrementer) -> tuple[str, type]:
	return (_f(pad), c_double)

def Node(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	if config.OFFSETOF_SPIN is not None:  fields.extend([ ("spin", c_double_3), _1(pad) ])
	if config.OFFSETOF_FLUX is not None:  fields.extend([ ("flux", c_double_3), _1(pad) ])
	if config.OFFSETOF_B    is not None:  fields.extend([ ("B",    c_double_3), _1(pad) ])
	if config.OFFSETOF_dB   is not None:  fields.extend([ ("dB",   c_double_3), _1(pad) ])
	if config.OFFSETOF_A    is not None:  fields.extend([ ("A",    c_double_3), _1(pad) ])
	block32 = [config.OFFSETOF_S, config.OFFSETOF_F, config.OFFSETOF_kT, config.OFFSETOF_Je0]
	if any(p is not None for p in block32):
		fields.append(("S"   if config.OFFSETOF_S   is not None else _f(pad), c_double))
		fields.append(("F"   if config.OFFSETOF_F   is not None else _f(pad), c_double))
		fields.append(("kT"  if config.OFFSETOF_kT  is not None else _f(pad), c_double))
		fields.append(("Je0" if config.OFFSETOF_Je0 is not None else _f(pad), c_double))
	if config.OFFSETOF_dkT is not None:
		fields.append(_1(pad))
		fields.append(_1(pad))
		fields.append(("dkT", c_double))
		fields.append(_1(pad))

	class _Node(Structure):
		_fields_ = fields
	
	assert sizeof(_Node) == config.SIZEOF_NODE  # DEBUG
	return _Node

def Region(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	if config.OFFSETOF_REGION_B  is not None:  fields.extend([ ("B",  c_double_3), _1(pad) ])
	if config.OFFSETOF_REGION_dB is not None:  fields.extend([ ("dB", c_double_3), _1(pad) ])
	if config.OFFSETOF_REGION_A  is not None:  fields.extend([ ("A",  c_double_3), _1(pad) ])
	block32 = [config.OFFSETOF_REGION_S, config.OFFSETOF_REGION_F, config.OFFSETOF_REGION_kT, config.OFFSETOF_REGION_Je0]
	if any(p is not None for p in block32):
		fields.append(("S"   if config.OFFSETOF_REGION_S   is not None else _f(pad), c_double))
		fields.append(("F"   if config.OFFSETOF_REGION_F   is not None else _f(pad), c_double))
		fields.append(("kT"  if config.OFFSETOF_REGION_kT  is not None else _f(pad), c_double))
		fields.append(("Je0" if config.OFFSETOF_REGION_Je0 is not None else _f(pad), c_double))
	if config.OFFSETOF_REGION_dkT is not None:
		fields.append(_1(pad))
		fields.append(_1(pad))
		fields.append(("dkT", c_double))
		fields.append(_1(pad))
	
	class _Region(Structure):
		_fields_ = fields
	
	assert sizeof(_Region) == config.SIZEOF_REGION  # DEBUG
	return _Region

def GlobalNode(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	if "B"   in config.globalKeys:  fields.extend([ ("B",  c_double_3), _1(pad) ])
	if "dkT" in config.globalKeys:  fields.extend([ ("dB", c_double_3), _1(pad) ])
	if "A"   in config.globalKeys:  fields.extend([ ("A",  c_double_3), _1(pad) ])
	if any(p in config.globalKeys for p in ["S", "F", "kT", "Je0"]):
		fields.append(("S"   if "S"   in config.globalKeys else _f(pad), c_double))
		fields.append(("F"   if "F"   in config.globalKeys else _f(pad), c_double))
		fields.append(("kT"  if "kT"  in config.globalKeys else _f(pad), c_double))
		fields.append(("Je0" if "Je0" in config.globalKeys else _f(pad), c_double))
	if "dkT" in config.globalKeys:
		fields.append(_1(pad))
		fields.append(_1(pad))
		fields.append(("dkT", c_double))
		fields.append(_1(pad))
	
	class _GlobalNode(Structure):
		_fields_ = fields
	
	return _GlobalNode

def Edge(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	block32 = [config.OFFSETOF_J, config.OFFSETOF_Je1, config.OFFSETOF_Jee, config.OFFSETOF_b]
	if any(p is not None for p in block32):
		fields.append(("J"   if config.OFFSETOF_J   is not None else _f(pad), c_double))
		fields.append(("Je1" if config.OFFSETOF_Je1 is not None else _f(pad), c_double))
		fields.append(("Jee" if config.OFFSETOF_Jee is not None else _f(pad), c_double))
		fields.append(("b"   if config.OFFSETOF_b   is not None else _f(pad), c_double))
	if config.OFFSETOF_D is not None:  fields.extend([ ("D", c_double_3), _1(pad) ])

	class _Edge(Structure):
		_fields_ = fields
	
	assert sizeof(_Edge) == config.SIZEOF_EDGE  # DEBUG
	return _Edge

def EdgeRegion(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	block32 = [config.OFFSETOF_REGION_J, config.OFFSETOF_REGION_Je1, config.OFFSETOF_REGION_Jee, config.OFFSETOF_REGION_b]
	if any(p is not None for p in block32):
		fields.append(("J"   if config.OFFSETOF_REGION_J   is not None else _f(pad), c_double))
		fields.append(("Je1" if config.OFFSETOF_REGION_Je1 is not None else _f(pad), c_double))
		fields.append(("Jee" if config.OFFSETOF_REGION_Jee is not None else _f(pad), c_double))
		fields.append(("b"   if config.OFFSETOF_REGION_b   is not None else _f(pad), c_double))
	if config.OFFSETOF_REGION_D is not None:  fields.extend([ ("D", c_double_3), _1(pad) ])

	class _EdgeRegion(Structure):
		_fields_ = fields
	
	assert sizeof(_EdgeRegion) == config.SIZEOF_EDGE_REGION  # DEBUG
	return _EdgeRegion

def GlobalEdge(config: Config) -> Structure:
	fields = []
	pad = Incrementer()
	if any(p in config.globalKeys for p in ["J", "Je1", "Jee", "b"]):
		fields.append(("J"   if "J"   in config.globalKeys else _f(pad), c_double))
		fields.append(("Je1" if "Je1" in config.globalKeys else _f(pad), c_double))
		fields.append(("Jee" if "Jee" in config.globalKeys else _f(pad), c_double))
		fields.append(("b"   if "b"   in config.globalKeys else _f(pad), c_double))
	if "D" in config.globalKeys:  fields.extend([ ("D", c_double_3), _1(pad) ])

	class _GlobalEdge(Structure):
		_fields_ = fields
	
	return _GlobalEdge
