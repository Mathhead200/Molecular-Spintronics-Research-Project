from __future__ import annotations
from ..util import PARAMETERS
from ctypes import addressof, c_double
from itertools import chain
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ctypes import Array, Structure
	from ..runtime import MutableStateBuffer
	from .simulation import Simulation
	from .simulation_util import Node, Edge, numpy_vec

def wrap_vec(arr: Array[c_double]) -> numpy_vec:
	return np.ctypeslib.as_array((c_double * 3).from_address(addressof(arr)))

class BufferView:
	def __init__(self, section: Array[Structure], attr: str):
		self._section = section
		self._attr = attr

class VecView[K=Node|Edge](BufferView):
	def __getitem__(self, key: K) -> numpy_vec:
		return wrap_vec(getattr(self._section[key], self._attr))
	
	def __setitem__(self, key: Node|Edge, value: numpy_vec) -> None:
		self[key][:] = value

class ScalView[K=Node|Edge](BufferView):
	def __getitem__(self, key: K) -> float:
		return getattr(self._section[key], self._attr)
	
	def __setitem__(self, key: K, value: float):
		setattr(self._section[key], self._attr, value)

# The full state of the Simulation at some simulation time, t.
class Snapshot:
	def __init__(self, sim: Simulation, buffer: MutableStateBuffer):
		self._s_view   = VecView[ Node](buffer.nodes, "spin")
		self._f_view   = VecView[ Node](buffer.nodes, "flux")
		self._B_view   = VecView[ Node](buffer.nodes, "B")
		self._A_view   = VecView[ Node](buffer.nodes, "A")
		self._S_view   = ScalView[Node](buffer.nodes, "S")
		self._F_view   = ScalView[Node](buffer.nodes, "F")
		self._kT_view  = ScalView[Node](buffer.nodes, "kT")
		self._Je0_view = ScalView[Node](buffer.nodes, "Je0")
		self._J_view   = ScalView[Edge](buffer.edges, "J")
		self._Je1_view = ScalView[Edge](buffer.edges, "Je1")
		self._Jee_view = ScalView[Edge](buffer.edges, "Jee")
		self._b_view   = ScalView[Edge](buffer.edges, "b")
		self._D_view   = VecView[ Edge](buffer.edges, "D")
	
	def __init__OLD(self, sim: Simulation):
		mat_n, mat_n2, list_n = sim._buf_mat_node, sim._buf_mat_node2, sim._buf_list_node
		mat_e, list_e = sim._buf_mat_edge, sim._buf_list_edge
		s_i, s_j = sim._buf_s_i, sim._buf_s_j
		f_i, f_j = sim._buf_f_i, sim._buf_f_j
		m_i, m_j = sim._buf_m_i, sim._buf_m_j
		Je0, J, Je1, Jee, b = sim._buf_Je0, sim._buf_J, sim._buf_Je1, sim._buf_Jee, sim._buf_b
		
		self.t = sim.t

		nodes = sim.nodes
		B = sim.B.values(None)  # must make copy to avoid shallow copying rows
		A = sim.A.values(None)  # must make copy to avoid shallow copying rows
		self.B   = dict(zip(nodes, B))
		self.A   = dict(zip(nodes, A))
		self.S   = dict(zip(nodes, sim.S.values(list_n)))   # don't need to save NDArray, not used in u calc. & don't need copy since is 1D and iter -> scalars
		self.F   = dict(zip(nodes, sim.F.values(list_n)))   # don't need to save NDArray, not used in u calc. & don't need copy since is 1D and iter -> scalars
		self.kT  = dict(zip(nodes, sim.kT.values(list_n)))  # don't need to save NDArray, not used in u calc. & don't need copy since is 1D and iter -> scalars
		self.Je0 = dict(zip(nodes, sim.Je0.values(Je0)))    # don't need copy since is 1D and iter -> scalars

		edges = sim.edges
		D   = sim.D.values(None)  # must make copy to avoid shallow copying rows
		self.J   = dict(zip(edges, sim.J.values(J)))      # don't need copy since is 1D and iter -> scalars
		self.Je1 = dict(zip(edges, sim.Je1.values(Je1)))  # don't need copy since is 1D and iter -> scalars
		self.Jee = dict(zip(edges, sim.Jee.values(Jee)))  # don't need copy since is 1D and iter -> scalars
		self.b   = dict(zip(edges, sim.b.values(b)))      # don't need copy since is 1D and iter -> scalars
		self.D   = dict(zip(edges, D))
		
		s = sim.s.values(None)  # must make copy to avoid shallow copying rows
		f = sim.f.values(None)  # must make copy to avoid shallow copying rows
		self.s = dict(zip(nodes, s))
		self.f = dict(zip(nodes, f))
		
		# compute magnetizations in paralell with numpy
		m = sim.m.values(out=None, s=s, f=f, buf_mat=mat_n)  # must make copy to avoid shallow copying rows
		self.m = dict(zip(nodes, m))

		# compute energies in paralell with numpy
		u = sim.u.values(out=None,
			buf_mat_node=mat_n, buf_mat_node2=mat_n2, buf_list_node=list_n, buf_mat_edge=mat_e, buf_list_edge=list_e,
			buf_s_i=s_i, buf_s_j=s_j, buf_f_i=f_i, buf_f_j=f_j, buf_m_i=m_i, buf_m_j=m_j,
			s=s, f=f, m=m, B=B, A=A, Je0=Je0, J=J, Je1=Je1, Jee=Jee, b=b, D=D)  # copy m again since it may be clobbered by anisotropy calc.
		self.u = dict(zip(chain(nodes, edges), u))

for p in ["s", "f", *PARAMETERS]:
	setattr(Snapshot, p, property(
		fget=lambda self, _p=p: getattr(self, f"_{_p}_view")
	))
