#
#   This module defines
#   the spin Hamiltonian for spin triplet
#   ground states
#   
#   Hss = D [Sz^2 - S(S+1)/3] + E (Sx^2 - Sy^2) + \sum_i^N I_i Ahfi(i) S
#
import numpy as np
import scipy.linalg as la
from pydephasing.utility_functions import delta
#
class spin_hamiltonian:
	#
	def __init__(self):
		self.s = 1.
		self.basis_vectors = [[self.s,1.], [self.s,0.], [self.s,-1.]]
		self.Splus = np.zeros((3,3), dtype=np.complex128)
		self.Sminus = np.zeros((3,3), dtype=np.complex128)
		self.Sx = np.zeros((3,3), dtype=np.complex128)
		self.Sy = np.zeros((3,3), dtype=np.complex128)
		self.Sz = np.zeros((3,3), dtype=np.complex128)
		# set spin matrices
		self.set_Sz()
		self.set_Splus()
		self.set_Sminus()
		self.set_Sx()
		self.set_Sy()
	def Splus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 - m2) * (j2 + m2 + 1)) * delta(j1, j2) * delta(m1, m2+1)
		return r
	def Sminus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 + m2) * (j2 - m2 + 1)) * delta(j1, j2) * delta(m1, m2-1)
		return r
	def set_Splus(self):
		#
		#    S+  ->
		#    <j1,m1|S+|j2,m2> = sqrt((j2-m2)(j2+m2+1)) delta(j1,j2) delta(m1,m2+1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Splus[r,c] = self.Splus_mtxel(j1, m1, j2, m2)
	def set_Sminus(self):
		#
		#    S-   ->
		#    <j1,m1|S-|j2,m2> = sqrt((j2+m2)(j2-m2+1)) delta(j1,j2) delta(m1,m2-1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Sminus[r,c] = self.Sminus_mtxel(j1, m1, j2, m2)
	def set_Sz(self):
		self.Sz[0,0] = 1.
		self.Sz[2,2] = -1.
		self.Sz[:,:] = self.Sz[:,:]
	def set_Sx(self):
		self.Sx[:,:] = (self.Splus[:,:] + self.Sminus[:,:]) / 2.
	def set_Sy(self):
		self.Sy[:,:] = (self.Splus[:,:] - self.Sminus[:,:]) / (2.*1j)
	def set_D_coef(self, qs):
		I = np.identity(3)
		A = np.matmul(self.Sz, self.Sz) - self.s * (self.s + 1) / 3 * I
		r = np.dot(A, qs)
		E_D = np.dot(qs.conjugate(), r)
		return E_D.real
	def set_E_coef(self, qs):
		A = np.matmul(self.Sx, self.Sx) - np.matmul(self.Sy, self.Sy)
		r = np.dot(A, qs)
		E_E = np.dot(qs.conjugate(), r)
		return E_E.real
	def set_gD_coef(self, gradZFS, nat, qs):
		# Hss = S gradD S
		E_gD = np.zeros(3*nat)
		jax = 0
		for ia in range(nat):
			for idx in range(3):
				E_O = np.zeros((3,3), dtype=np.complex128)
				gD = np.zeros((3,3))
				gD[:,:] = gradZFS.U_gradD_U[jax,:,:]
				E_O[:,:] = E_O[:,:] + gD[0,0] * np.matmul(self.Sx, self.Sx)
				E_O[:,:] = E_O[:,:] + gD[0,1] * np.matmul(self.Sx, self.Sy)
				E_O[:,:] = E_O[:,:] + gD[1,0] * np.matmul(self.Sy, self.Sx)
				E_O[:,:] = E_O[:,:] + gD[1,1] * np.matmul(self.Sy, self.Sy)
				E_O[:,:] = E_O[:,:] + gD[0,2] * np.matmul(self.Sx, self.Sz)
				E_O[:,:] = E_O[:,:] + gD[2,0] * np.matmul(self.Sz, self.Sx)
				E_O[:,:] = E_O[:,:] + gD[1,2] * np.matmul(self.Sy, self.Sz)
				E_O[:,:] = E_O[:,:] + gD[2,1] * np.matmul(self.Sz, self.Sy)
				E_O[:,:] = E_O[:,:] + gD[2,2] * np.matmul(self.Sz, self.Sz)
				r = np.dot(E_O, qs)
				E_gD[jax] = np.dot(qs.conjugate(), r).real
				#
				jax = jax+1
		return E_gD
	def set_SDS(self, unprt_struct):
		Ddiag = unprt_struct.Ddiag*2.*np.pi*1.E-6
		#
		#  THz units
		#
		self.SDS = Ddiag[0]*np.matmul(self.Sx, self.Sx)
		self.SDS = self.SDS + Ddiag[1]*np.matmul(self.Sy, self.Sy)
		self.SDS = self.SDS + Ddiag[2]*np.matmul(self.Sz, self.Sz)
	def set_A_coef(self, qs):
		E_A = np.zeros(3)
		r = np.dot(self.Sx, qs)
		E_A[0] = np.dot(qs.conjugate(), r).real
		#
		r = np.dot(self.Sy, qs)
		E_A[1] = np.dot(qs.conjugate(), r).real
		#
		r = np.dot(self.Sz, qs)
		E_A[2] = np.dot(qs.conjugate(), r).real
		return E_A
	def set_grad_Ess(self, gradZFS, gradHFI, spin_config, qs, nat, lambda_coef):
		# compute grad Ess
		# gradient of the energy of the spin system
		self.grad_Ess = np.zeros(3*nat)
		# compute : grad D < qs | [Sz^2 - S(S+1)/3] | qs >
		#E_D = self.set_D_coef(qs)
		#self.grad_Ess[:] = lambda_coef[0] * E_D * gradZFS.gradD[:]
		# compute : grad E < qs |(Sx^2 - Sy^2)| qs >
		#E_E = self.set_E_coef(qs)
		#self.grad_Ess[:] = self.grad_Ess[:] + lambda_coef[0] * E_E * gradZFS.gradE[:]
		E_gD = self.set_gD_coef(gradZFS, nat, qs)
		self.grad_Ess[:] = self.grad_Ess[:] + lambda_coef[0] * E_gD[:]
		# compute : \sum_i^N I_i grad Ahfi(i) < qs | S | qs >
		E_A = self.set_A_coef(qs)
		for isp in range(spin_config.nsp):
			Ii = spin_config.nuclear_spins[isp]['I']
			aa = spin_config.nuclear_spins[isp]['site']
			for jax in range(3*nat):
				# gradient Ahfi
				gax_Ahfi = np.zeros((3,3))
				gax_Ahfi[:,:] = gradHFI.gradAhfi[jax,aa-1,:,:]
				# matrix product
				gAS = np.dot(gax_Ahfi, E_A)
				self.grad_Ess[jax] = self.grad_Ess[jax] + lambda_coef[1] * np.dot(Ii, gAS)
	def set_grad_deltaEss(self, gradZFS, gradHFI, spin_config, qs1, qs2, nat, lambda_coef):
		# compute grad delta Ess
		# gradient of the spin state energy difference
		self.grad_deltaEss = np.zeros(3*nat)
		# compute : grad D < qs | [Sz^2 - S(S+1)/3] | qs >
		#E_D1 = self.set_D_coef(qs1)
		#E_D2 = self.set_D_coef(qs2)
		#dE_D = E_D1 - E_D2
		#self.grad_deltaEss[:] = lambda_coef[0] * dE_D * gradZFS.gradD[:]
		# compute : grad E < qs |(Sx^2 - Sy^2)| qs >
		#E_E1 = self.set_E_coef(qs1)
		#E_E2 = self.set_E_coef(qs2)
		#dE_E = E_E1 - E_E2
		E_gD1 = self.set_gD_coef(gradZFS, nat, qs1)
		E_gD2 = self.set_gD_coef(gradZFS, nat, qs2)
		self.grad_deltaEss[:] = self.grad_deltaEss[:] + lambda_coef[0] * (E_gD1[:] - E_gD2[:])
		#self.grad_deltaEss[:] = self.grad_deltaEss[:] + lambda_coef[0] * dE_E * gradZFS.gradE[:]
		# compute : \sum_i^N I_i grad Ahfi(i) < qs | S | qs >
		E_A1 = self.set_A_coef(qs1)
		E_A2 = self.set_A_coef(qs2)
		dE_A = E_A1 - E_A2
		for isp in range(spin_config.nsp):
			Ii = spin_config.nuclear_spins[isp]['I']
			aa = spin_config.nuclear_spins[isp]['site']
			for jax in range(3*nat):
				# gradient Ahfi
				gax_Ahfi = np.zeros((3,3))
				gax_Ahfi[:,:] = gradHFI.gradAhfi[jax,aa-1,:,:]
				# product
				gAS = np.dot(gax_Ahfi, dE_A)
				self.grad_deltaEss[jax] = self.grad_deltaEss[jax] + lambda_coef[1] * np.dot(Ii, gAS)
	def set_time(self, dt, T):
		# set time in ps units
		# for spin vector evolution
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		#
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
	def set_magnetization(self):
		# compute magnet. expect. value
		nt = len(self.time)
		self.Mt = np.zeros((3,nt))
		# run on t
		for i in range(nt):
			vx = np.dot(self.Sx, self.psit[:,i])
			self.Mt[0,i] = np.dot(self.psit[:,i].conjugate(), vx).real
			#
			vy = np.dot(self.Sy, self.psit[:,i])
			self.Mt[1,i] = np.dot(self.psit[:,i].conjugate(), vy).real
			#
			vz = np.dot(self.Sz, self.psit[:,i])
			self.Mt[2,i] = np.dot(self.psit[:,i].conjugate(), vz).real
	def compute_spin_vector_evol(self, psi0, B):
		# initial state : psi0
		# magnetic field : B (gauss)
		# H = SDS + gamma_e B S
		# 1) compute SDS
		self.set_SDS()
		# n. time steps
		nt = len(self.time_dense)
		Ht = np.zeros((3,3,nt), dtype=np.complex128)
		# 2) set Ht in eV units
		for i in range(nt):
			Ht[:,:,i] = Ht[:,:,i] + hbar * self.SDS[:,:]
			# eV
			# add B field
			Ht[:,:,i] = Ht[:,:,i] + gamma_e * hbar * (B[0] * self.Sx[:,:] + B[1] * self.Sy[:,:] + B[2] * self.Sz[:,:])
		dt = self.time[1]-self.time[0]
		# ps units
		# triplet wave function evolution
		self.tripl_psit = triplet_evolution(Ht, psi0, dt)
		# set magnetization vector Mt
		self.set_magnetization()
