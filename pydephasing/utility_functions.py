import numpy as np
from math import exp
from pydephasing.phys_constants import eps, kb, hbar
import sys
#
# utility functions module
#
# 1) function computing the norm of a vector
#    input : n-dimensional vector
# 2) function computing the norm of a complex vector
#    input : n-dimensional vector
# 3) function computing the bose occupation number
#    input : energy (eV), temperature (K)
# 4) delta function
#    input : x, y
# 5) solve ODE dy/dt = Fy
#    input : y0, F, dt
# 6) set cross product matrix
#    input : a
#    output : [a] : axv = [a]v
# 7) triplet state evolution
#    input : Ht, psi0, dt
#    output : psit
# 8) compute index to ia map
# 9) compute index to idx map
# 10)compute T2
#    input : t, Ct, D2
#    output: tauc, T2_inv, [exp. fit]
# 11)compute T2
#    input : t, Ct, D2
#    output: tauc, T2_inv, [expsin, fit]
#
#  function 1)
#
def norm_realv(v):
	nrm = np.sqrt(sum(v[:]*v[:]))
	return nrm
#
#  function 2)
#
def norm_cmplv(v):
	nrm = np.sqrt(sum(v[:]*v[:].conjugate()).real)
	return nrm
#
#  function 3)
#
def bose_occup(E, T):
	if T < eps:
		n = 0.
	else:
		try:
			n = 1./(exp(E / (kb * T)) - 1.)
		except ZeroDivisionError as err:
			print('Run-time error:', err)
			sys.exit(1)
	return n
#
#  function 4)
#
def delta(x, y):
	if x == y:
		return 1.
	else:
		return 0.
#
#  function 5)
#
def ODE_solver(y0, F, dt):
	# this routine solves
	# dy/dt = F(t) y -> y real 3d vector
	# F(t) is 3 x 3 matrix
	# using RK4 algorithm
	nt = F.shape[2]
	yt = np.zeros((3,int(nt/2)))
	yt[:,0] = y0[:]
	# iterate over t
	for i in range(int(nt/2)-1):
		y = np.zeros(3)
		y[:] = yt[:,i]
		# K1
		K1 = dt * np.matmul(F[:,:,2*i], y)
		y1 = y + K1 / 2.
		# K2
		K2 = dt * np.matmul(F[:,:,2*i+1], y1)
		y2 = y + K2 / 2.
		# K3
		K3 = dt * np.matmul(F[:,:,2*i+1], y2)
		y3 = y + K3
		# K4
		K4 = dt * np.matmul(F[:,:,2*i+2], y3)
		#
		yt[:,i+1] = y[:] + (K1[:] + 2.*K2[:] + 2.*K3[:] + K4[:]) / 6.
	return yt
#
#   function 6)
#
def set_cross_prod_matrix(a):
	A = np.zeros((3,3))
	A[0,1] = -a[2]
	A[0,2] =  a[1]
	A[1,0] =  a[2]
	A[1,2] = -a[0]
	A[2,0] = -a[1]
	A[2,1] =  a[0]
	return A
#
#   function 7)
#
def triplet_evolution(Ht, psi0, dt):
	# this routine solves
	# dpsi/dt = -i/hbar Ht psi -> psi triplet wfc
	# H(t) is 3X3
	nt = Ht.shape[2]
	psit = np.zeros((3,int(nt/2)), dtype=np.complex128)
	psit[:,0] = psi0[:]
	# iterate over time
	for i in range(int(nt/2)-1):
		v = np.zeros(3, dtype=np.complex128)
		v[:] = psit[:,i]
		# K1
		F1 = -1j / hbar * np.matmul(Ht[:,:,2*i], v)
		K1 = dt * F1
		v1 = v + K1 / 2.
		# K2
		F2 = -1j / hbar * np.matmul(Ht[:,:,2*i+1], v1)
		K2 = dt * F2
		v2 = v + K2 / 2.
		# K3
		F3 = -1j / hbar * np.matmul(Ht[:,:,2*i+1], v2)
		K3 = dt * F3
		v3 = v + K3
		# K4
		F4 = -1j / hbar * np.matmul(Ht[:,:,2*i+2], v3)
		K4 = dt * F4
		psit[:,i+1] = v[:] + (K1[:] + 2.*K2[:] + 2.*K3[:] + K4[:]) / 6.
	return psit
#
# function 8)
#
def compute_index_to_ia_map(nat):
	index_to_ia_map = np.zeros(3*nat, dtype=int)
	for jax in range(3*nat):
		index_to_ia_map[jax] = int(jax/3) + 1
	return index_to_ia_map
#
# function 9)
#
def compute_index_to_idx_map(nat):
	index_to_idx_map = np.zeros(3*nat, dtype=int)
	for jax in range(3*nat):
		index_to_idx_map[jax] = jax%3
	return index_to_idx_map
#
# function 10)
#
def extract_T2_Exp(t, Ct, D2):
	# t : time array
	# Ct : acf
	# D2 : Delta^2
	#
	# fit over exp. function
	def Exp(x, c):
		return np.exp(-c * x)
	# e^-g(t) -> g(t)=D2*tau_c^2[e^(-t/tau_c)+t/tau_c-1]
	# D2 -> eV^2
	# tau_c -> ps
	def exp_gt(x, D2, tau_c):
		r = np.exp(-gt(x, D2, tau_c))
		return r
	# gt function
