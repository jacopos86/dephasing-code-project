#
#   This module defines
#   the nuclear spin configuration class
#   - number of non zero nuclear spins
#   - atomic sites with non zero spin
#   - nuclear spin value
#
import numpy as np
import random
from pydephasing.utility_functions import norm_realv
from pydephasing.utility_functions import set_cross_prod_matrix
#
class nuclear_spins_config:
	#
	def __init__(self, nsp):
		self.nsp = nsp
	def set_time(self, dt, T):
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		# finer array
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
		# micro sec units
	# set nuclear configuration method
	def set_nuclear_spins(self, nat):
		self.nuclear_spins = []
		Ilist = []
		for isp in range(self.nsp):
			# set spin vector
			I = np.zeros(3)
			for j in range(3):
				I[j] = random.random()
			I = I / norm_realv(I) / 2.
			Ilist.append(I)
		# set atom's site
		sites = np.random.randint(1, nat+1, self.nsp)
		# define dictionary
		keys = ['site', 'I']
		for isp in range(self.nsp):
			self.nuclear_spins.append(dict(zip(keys, [sites[isp], Ilist[isp]])))
	# set spin vector evolution
	# method to time evolve I
	def set_nuclear_spin_evol(self, B, spin_hamilt, unprt_struct):
		# dIa/dt = gamma_n B X Ia + (A_hf(a) M(t)) X Ia
		# B : applied magnetic field (Gauss)
		# spin_hamilt : spin Hamiltonian object
		# unprt_struct : unperturbed atomic structure
		#Mt = spin_hamilt.Mt
		# ps units
		#t = spin_hamilt.time
		#T = t[-1]
		# compute <Mt> -> spin magnetization
		#rx = integrate.simpson(Mt[0,:], t) / T
		#ry = integrate.simpson(Mt[1,:], t) / T
		#rz = integrate.simpson(Mt[2,:], t) / T
		#M = np.array([rx, ry, rz])
		M = np.array([0.,0.,1.])
		# n. time steps integ.
		nt = len(self.time_dense)
		# time interv. (micro sec)
		dt = self.time[1]-self.time[0]
		# set [B] matrix
		Btilde = set_cross_prod_matrix(B)
		print(Btilde)
		# run over the spins active
		# in the configuration
		for isp in range(self.nsp):
			site = self.nuclear_spins[isp]['site']
			print(site)
