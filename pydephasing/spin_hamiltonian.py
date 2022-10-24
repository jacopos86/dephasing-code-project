#
#   This module defines
#   the spin Hamiltonian for spin triplet
#   ground states
#   
#   Hss = D [Sz^2 - S(S+1)/3] + E (Sx^2 - Sy^2) + \sum_i^N I_i Ahfi(i) S
#
import numpy as np
import yaml
from pydephasing.utility_functions import delta
from pydephasing.phys_constants import hbar, gamma_e
from pydephasing.utility_functions import triplet_evolution
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
		# THz / Ang units
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
	# set ZFS energy gradient
	def set_grad_deltaEzfs(self, gradZFS, qs1, qs2, nat):
		# compute grad delta Ezfs
		# gradient of the spin state energy difference
		grad_deltaEzfs = np.zeros(3*nat)
		#
		# compute : < qs1 | S gradD S | qs1 > - < qs2 | S gradD S | qs2 >
		#
		E_gD1 = self.set_gD_coef(gradZFS, nat, qs1)
		E_gD2 = self.set_gD_coef(gradZFS, nat, qs2)
		grad_deltaEzfs[:] = grad_deltaEzfs[:] + (E_gD1[:] - E_gD2[:])
		# THz / Ang units (energy)
		# add 2pi factor
		grad_deltaEzfs[:] = grad_deltaEzfs[:] * 2.*np.pi
		return grad_deltaEzfs
	# set HFI energy gradient
	def set_grad_deltaEhfi(self, gradHFI, nucl_spins_conf, qs1, qs2, nat):
		# compute grad delta Ehfi
		# gradient of spin states energy diff.
		grad_deltaEhfi = np.zeros(3*nat)
		#
		# compute : < qs1 | S gradA_HFI I | qs1 > - < qs2 | S gradA_HFI I | qs2 >
		#
		E_gA1 = self.set_A_coef(qs1)
		E_gA2 = self.set_A_coef(qs2)
		dE_gA = E_gA1 - E_gA2
		# run over spin index
		for isp in range(len(nucl_spins_conf)):
			Ii = nucl_spins_conf[isp]['I']
			aa = nucl_spins_conf[isp]['site']
			# run over (ax) index
			for jax in range(3*nat):
				# gradient Ahfi
				gax_Ahfi = np.zeros((3,3))
				gax_Ahfi[:,:] = gradHFI.gradAhfi[jax,aa-1,:,:]
				# THz / ang units
				gAS = np.dot(gax_Ahfi, dE_gA)
				grad_deltaEhfi[jax] = grad_deltaEhfi[jax] + np.dot(Ii, gAS)
		return grad_deltaEhfi
	# set time array
	def set_time(self, dt, T):
		# set time in ps units
		# for spin vector evolution
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		#
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
	# compute magnetization array
	def set_magnetization(self):
		# compute magnet. expect. value
		nt = len(self.time)
		self.Mt = np.zeros((3,nt))
		# run on t
		for i in range(nt):
			vx = np.dot(self.Sx, self.tripl_psit[:,i])
			self.Mt[0,i] = np.dot(self.tripl_psit[:,i].conjugate(), vx).real
			#
			vy = np.dot(self.Sy, self.tripl_psit[:,i])
			self.Mt[1,i] = np.dot(self.tripl_psit[:,i].conjugate(), vy).real
			#
			vz = np.dot(self.Sz, self.tripl_psit[:,i])
			self.Mt[2,i] = np.dot(self.tripl_psit[:,i].conjugate(), vz).real
	# compute spin vector
	def compute_spin_vector_evol(self, struct0, psi0, B):
		# initial state : psi0
		# magnetic field : B (gauss)
		# H = SDS + gamma_e B S
		# 1) compute SDS
		self.set_SDS(struct0)
		# n. time steps
		nt = len(self.time_dense)
		Ht = np.zeros((3,3,nt), dtype=np.complex128)
		# 2) set Ht in eV units
		# run over time steps
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
	# write spin vector on file
	def write_spin_vector_on_file(self, out_dir):
		# time steps
		nt = len(self.time)
		# write on file
		namef = out_dir + "/spin-vector.yml"
		# set dictionary
		dict = {'time' : 0, 'Mt' : 0}
		dict['time'] = self.time
		dict['Mt'] = self.Mt
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict, out_file)
		# mag. vector
		namef = out_dir + "/occup-prob"
		# set dictionary
		dict2 = {'time' : 0, 'occup' : 0}
		occup = np.zeros((3,nt))
		# run over t
		for i in range(nt):
			occup[:,i] = np.dot(self.tripl_psit[:,i].conjugate(), self.tripl_psit[:,i]).real
		dict2['time'] = self.time
		dict2['occup']= occup
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict2, out_file)