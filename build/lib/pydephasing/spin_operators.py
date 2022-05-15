#
#   This module defines
#   the spin operators for spin triplets
#
import numpy as np
#
class spin_operators:
	def __init__(self):
		self.Sx = np.zeros((3,3), dtype=np.complex128)
		self.Sy = np.zeros((3,3), dtype=np.complex128)
		self.Sz = np.zeros((3,3), dtype=np.complex128)
#
