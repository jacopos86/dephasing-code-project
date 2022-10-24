#
#   This module defines
#   the nuclear spin configuration class
#   - number of non zero nuclear spins
#   - atomic sites with non zero spin
#   - nuclear spin value
#
import numpy as np
from scipy import integrate
import math
import yaml
from pydephasing.utility_functions import set_cross_prod_matrix, norm_realv, cart2sph, ODE_solver
from pydephasing.phys_constants import gamma_n
#
class nuclear_spins_config:
	#
	def __init__(self, nsp, B0):
		self.nsp = nsp
		self.B0 = np.array(B0)
		# applied mag. field (G) units
		self.nuclear_spins = []
		# I spins list
	def set_time(self, dt, T):
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		# finer array
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
		# micro sec units
	# set nuclear configuration method
	def set_nuclear_spins(self, nat, T_K):
		# produce a number of possible orientations
		# from a gaussian distr. centered on B0 direction
		# and sigma given by the temperature
		v = self.B0 / norm_realv(self.B0)
		# cart to spherical
		r = cart2sph(v)
		th = r[1]
		phi= r[2]
		# set 2D gaussian distr.
		mean = np.array([th, phi])
		cov = np.zeros((2,2))
		cov[0,0] = T_K * np.pi / 180.
		cov[1,1] = T_K * np.pi / 180.
		p = np.random.multivariate_normal(mean, cov, self.nsp)
		# set spin list
		Ilist = []
		for isp in range(self.nsp):
			# set spin vector
			I = np.zeros(3)
			# theta / phi
			th = p[isp,0]
			phi= p[isp,1]
			# compute components
			# in cart. coordinates
			I[0] = 0.5 * math.cos(th) * math.cos(phi)
			I[1] = 0.5 * math.cos(th) * math.sin(phi)
			I[2] = 0.5 * math.sin(th)
			Ilist.append(I)
		# set atom's site
		sites = np.random.randint(1, nat+1, self.nsp)
		# define dictionary
		keys = ['site', 'I']
		for isp in range(self.nsp):
			self.nuclear_spins.append(dict(zip(keys, [sites[isp], Ilist[isp]])))
	# compute force grad HFI
	def compute_force_gHFI(self, gradHFI, Hss, qs1, qs2, nat):
		# set grad AHFI force
		nucl_spins_conf = self.nuclear_spins
		grad_deltaEhfi = Hss.set_grad_deltaEhfi(gradHFI, nucl_spins_conf, qs1, qs2, nat)
		# THz / ang units
		self.Fax = np.zeros(3*nat)
		self.Fax[:] = 2. * np.pi * grad_deltaEhfi[:]
	# set spin vector evolution
	# method to time evolve I
	def set_nuclear_spin_evol(self, Hss, unprt_struct):
		# dIa/dt = gamma_n B X Ia + (A_hf(a) M(t)) X Ia
		# B : applied magnetic field (Gauss)
		# spin_hamilt : spin Hamiltonian object
		# unprt_struct : unperturbed atomic structure
		Mt = Hss.Mt
		# ps units
		t = Hss.time
		T = t[-1]
		# compute <Mt> -> spin magnetization
		rx = integrate.simpson(Mt[0,:], t) / T
		ry = integrate.simpson(Mt[1,:], t) / T
		rz = integrate.simpson(Mt[2,:], t) / T
		M = np.array([rx, ry, rz])
		# n. time steps integ.
		nt = len(self.time_dense)
		# time interv. (micro sec)
		dt = self.time[1]-self.time[0]
		# set [B] matrix
		Btilde = set_cross_prod_matrix(self.B0)
		# run over the spins active
		# in the configuration
		for isp in range(self.nsp):
			site = self.nuclear_spins[isp]['site']
			# set HFI matrix (MHz)
			A = np.zeros((3,3))
			A[:,:] = 2.*np.pi*unprt_struct.Ahfi[site-1,:,:]
			# set F(t) = gamma_n B + Ahf(a) M
			Ft = np.zeros((3,3,nt))
			# A(a) M
			AM = np.matmul(A, M)
			AM_tilde = set_cross_prod_matrix(AM)
			for i in range(nt):
				Ft[:,:,i] = gamma_n * Btilde[:,:]
				Ft[:,:,i] = Ft[:,:,i] + AM_tilde[:,:]
				# MHz units
			I0 = self.nuclear_spins[isp]['I']
			It = ODE_solver(I0, Ft, dt)
			self.nuclear_spins[isp]['It'] = It
	# set nuclear spin time fluct.
	def set_nuclear_spin_time_fluct(self):
		# time steps
		nt = len(self.time)
		# run over different active spins
		# in the config.
		for isp in range(self.nsp):
			It = self.nuclear_spins[isp]['It']
			dIt = np.zeros((3,nt))
			dIt[0,:] = It[0,:] - It[0,0]
			dIt[1,:] = It[1,:] - It[1,0]
			dIt[2,:] = It[2,:] - It[2,0]
			self.nuclear_spins[isp]['dIt'] = dIt
	# write I(t) on ext. file
	def write_It_on_file(self, out_dir, ic):
		# write file name
		name_file = out_dir + "/config-sp" + str(self.nsp) + "-" + str(ic+1) + ".yml"
		# set dictionary
		dict = {'time' : 0, 'nuclear spins' : []}
		dict['time'] = self.time
		dict['nuclear spins'] = self.nuclear_spins
		# save data
		with open(name_file, 'w') as out_file:
			yaml.dump(dict, out_file)
	# compute spin fluct. forces
	def compute_force_HFS(self, Hss, unprt_struct, qs1, qs2):
		# compute spin coeffs.
		S_1 = Hss.set_A_coef(qs1)
		S_2 = Hss.set_A_coef(qs2)
		DS  = S_1 - S_2
		# run over active spins
		for isp in range(self.nsp):
			site = self.nuclear_spins[isp]['site']
			# set HFI matrix (THz)
			Ahf = np.zeros((3,3))
			Ahf[:,:] = 2.*np.pi*unprt_struct.Ahfi[site-1,:,:]*1.E-6
			# force vector (THz)
			F = np.zeros(3)
			F = np.matmul(Ahf, DS)
			self.nuclear_spins[isp]['F'] = F