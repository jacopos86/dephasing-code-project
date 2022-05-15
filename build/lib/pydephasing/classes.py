import numpy as np
#
#  This module contains the classes for :
#  1) temporal list
#  2) temperature list
#  3) zero field splitting gradient
#
class TimeList:
	def __init__(self, T, dt):
		self.T = T            # ps
		self.dt = dt          # ps
		self.Nt = int(T / dt)
	def set_time_array(self):
		self.t = np.linspace(0., self.T, self.Nt)
#
class TemperatureList:
	def __init__(self, Tin, Tfin, dT):
		self.Tin = Tin        # K
		self.Tfin= Tfin       # K
		self.dT = dT          # K
	def set_temperature_array(self):
		self.temperatures = np.arange(self.Tin, self.Tfin+self.dT, self.dT)
		self.N = len(self.temperatures)
#
#class GradZFS:
#def __init__(self):
#self.gradD = np.zeros(
#
class EnergyFluct:
	def __init__(self, N):
		self.deltaE = np.zeros(N, dtype=np.double)
		self.N = N
	# compute t=0 autocorrelation
	def time0_autocorrelation(self):
		C = sum(self.deltaE[:] * self.deltaE[:]) / self.N
		return C
