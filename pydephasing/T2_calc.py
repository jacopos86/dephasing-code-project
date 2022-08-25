#
# This module sets up the method
# needed to compute T2 / T2*
# given the energy fluct. auto correlation function
#
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from pydephasing.phys_constants import hbar
#
# functions : Exp ExpSin exp_gt gt
#
def Exp(x, c):
	return np.exp(-c * x)
#
def ExpSin(x, a, b, c):
	r = np.exp(-c * x) * np.sin(a * x + b)
	return r
#
# e^-g(t) -> g(t)=D2*tau_c^2[e^(-t/tau_c)+t/tau_c-1]
# D2 -> eV^2
# tau_c -> ps
def exp_gt(x, D2, tau_c):
	r = np.exp(-gt(x, D2, tau_c))
	return r
#
def gt(x, D2, tau_c):
	r = np.zeros(len(x))
	r[:] = D2 / hbar ** 2 * tau_c ** 2 * (np.exp(-x[:]/tau_c) + x[:]/tau_c - 1)
	return r
#
# class T2_eval definition
#
class T2_eval:
	# initialization calculation
	# parameters
	def __init__(self, input_params):
		# fft sample points
		self.N = input_params.N_fft
        # sample spacing -> ps
		self.T = input_params.T_fft
	#
	# T2 calculation methods
	#
	# 1)  compute T2
	# input : t, Ct, D2
	# output: tauc, T2_inv, [exp. fit]
	def extract_T2_Exp(self, t, Ct, D2):
		# t : time array
		# Ct : acf
		# D2 : Delta^2
		#
		# fit over exp. function
		p0 = 1    # start with values near those we expect
		p, cv = scipy.optimize.curve_fit(Exp, t, Ct, p0)
		# p = 1/tau_c (ps^-1)
		tau_c = 1./p
		# ps units
		r = np.sqrt(D2) / hbar * tau_c
		# check r size
		# see Mukamel book
		if r < 1.E-4:
			T2_inv = D2 / hbar ** 2 * tau_c
			# ps^-1
		elif r > 1.E4:
			T2_inv = np.sqrt(D2) / hbar
			# ps^-1
		else:
			N = self.N
			T = self.T
			x = np.linspace(0.0, N*T, N, endpoint=False)
			y = exp_gt(x, D2, tau_c)
			# fft
			yf = fft(y)
			xf = fftfreq(N, T)[:N//2]
			yff = 2./N*np.abs(yf[0:N//2])
			ymax = np.max(yff)
			for i in range(len(yff)):
				if yff[i] < ymax/2.:
					hlw = xf[i]    # psec^-1
					break
			T2_inv = 2.*hlw    # ps^-1
            #
		return tau_c, T2_inv, Exp(t, p)
	#
	# function 2 -> compute T2
	#
	# input : t, Ct, D2
	# output: tauc, T2_inv, [expsin, fit]
	def extract_T2_ExpSin(self, t, Ct, D2):
		# t : time array
		# Ct : acf
		# D2 : Delta^2
		#
		# fit over exp. function
		p0 = [1., 1., 1.]      # start with values near those we expect
		p, cv = scipy.optimize.curve_fit(ExpSin, t, Ct, p0)
		# p = 1/tau_c (ps^-1)
		tau_c = 1./p[2]        # ps
		# r -> Mukamel book
		r = np.sqrt(D2) / hbar * tau_c
		#
		if r < 1.E-4:
			T2_inv = D2 / hbar ** 2 * tau_c
			# ps^-1
		elif r > 1.E4:
			T2_inv = np.sqrt(D2) / hbar
		else:
			N = self.N
			T = self.T
			x = np.linspace(0.0, N*T, N, endpoint=False)
			y = exp_gt(x, D2, tau_c)
			# fft
			yf = fft(y)
			xf = fftfreq(N, T)[:N//2]
			yff = 2./N*np.abs(yf[0:N//2])
			ymax = np.max(yff)
			for i in range(len(yff)):
				if yff[i] < ymax/2:
					hlw = xf[i]    # ps^-1
					break
			T2_inv = 2.*hlw
			# psec^-1
		return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])
	#
	# 3)   compute T2*
	# input : t, Ct, D2
	# output: tauc, T2_inv, [expsin, fit]
	def extract_T2_star(self, t, Ct, D2):
		# check non Nan
		if not np.isfinite(Ct).all():
			return [None, None, None]
		# perform the fit
		p0 = [1., 1., 1.]
		p, cv = scipy.optimize.curve_fit(ExpSin, t, Ct, p0)
		# p = 1/tau_c (mus^-1)
		tau_c = 1./p[2]
		# tau_c (mu sec)
		r = np.sqrt(D2) / hbar * tau_c * 1.E+6
		#
		# check limit r conditions
		if r < 1.E-4:
			T2_inv = D2 / hbar ** 2 * tau_c * 1.E+6
			# ps^-1
		elif r > 1.E+4:
			T2_inv = np.sqrt(D2) / hbar * 1.E+6
			# ps^-1
		else:
			tauc_ps = tau_c * 1.E+6
			# fft sample points
			N = self.N
			T = self.T
			x = np.linspace(0.0, N*T, N, endpoint=False)
			x_ps = x * 1.E+6
			y = exp_gt(x_ps, D2, tauc_ps)
			# fft
			yf = fft(y)
			xf = fftfreq(N, T)[:N//2]
			yff = 2./N*np.abs(yf[0:N//2])
			ymax = np.max(yff)
			for i in range(len(yff)):
				if yff[i] < ymax/2.:
					hlw = xf[i]     # mu sec^-1
					break
			T2_inv = 2.*hlw * 1.E-6
			# ps^-1
		return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])