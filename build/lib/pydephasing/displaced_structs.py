#
#   This module defines the displaced atomic
#   coordinates for the gradient calculation
#
import numpy as np
import os
from pymatgen.io.vasp.inputs import Poscar
#
class DisplacedStructs:
	def __init__(self, ipath, out_dir):
		self.ipath = ipath
		self.out_dir = out_dir
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
	### read POSCAR file
	def read_poscar(self):
		poscar = Poscar.from_file("{}".format(self.ipath+"POSCAR"))
		struct = poscar.structure
		return struct
#
