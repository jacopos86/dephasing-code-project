#
#   This module defines the displaced atomic
#   coordinates for the gradient calculation
#
import numpy as np
import os
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
import scipy.linalg as la
#
#  ground state structure
#  acquires ground state data
#
class UnpertStruct:
	def __init__(self, ipath):
		self.ipath = ipath + "/"
	### read POSCAR file
	def read_poscar(self):
		poscar = Poscar.from_file("{}".format(self.ipath+"POSCAR"), read_velocities=False)
		struct = poscar.structure
		self.struct = struct
		# set number of atoms
		struct_dict = struct.as_dict()
		atoms_key = list(struct_dict.keys())[4]
		self.nat = len(list(struct_dict[atoms_key]))
	### read ZFS from OUTCAR
	def read_zfs(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 3 and l[0] == "D_diag" and l[1] == "eigenvector":
				j = i+2
				l2 = lines[j].split()
				D1 = float(l2[0])     # MHz
				#
				l2 = lines[j+1].split()
				D2 = float(l2[0])     # MHz
				#
				l2 = lines[j+2].split()
				D3 = float(l2[0])     # MHz
		# set D tensor
		self.Ddiag = np.array([D1, D2, D3])
		# MHz
	### read full ZFS tensor
	def read_zfs_tensor(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "D_xx" and l[5] == "D_yz":
				j = i+2
				l2 = lines[j].split()
				# D tensor (MHz)
				Dxx = float(l2[0])
				Dyy = float(l2[1])
				Dzz = float(l2[2])
				Dxy = float(l2[3])
				Dxz = float(l2[4])
				Dyz = float(l2[5])
		self.Dtensor = np.zeros((3,3))
		self.Dtensor[0,0] = Dxx
		self.Dtensor[1,1] = Dyy
		self.Dtensor[2,2] = Dzz
		self.Dtensor[0,1] = Dxy
		self.Dtensor[1,0] = Dxy
		self.Dtensor[0,2] = Dxz
		self.Dtensor[2,0] = Dxz
		self.Dtensor[1,2] = Dyz
		self.Dtensor[2,1] = Dyz
		# set diagonalization
		# basis vectors
		[eig, eigv] = la.eig(self.Dtensor)
		eig = eig.real
		# set D spin quant. axis coordinates
		self.read_zfs()
		self.Deigv = np.zeros((3,3))
		for i in range(3):
			d = self.Ddiag[i]
			for j in range(3):
				if abs(d-eig[j]) < 1.E-2:
					self.Ddiag[i] = eig[j]
					self.Deigv[:,i] = eigv[:,j]
		# D units -> MHz
	### read HFI from OUTCAR
	def read_hfi(self):
		# read file
		f = open(self.ipath+"OUTCAR", 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 7 and l[0] == "ion" and l[6] == "A_yz":
				j = i+2
				A = np.zeros((self.nat,6))
				for k in range(j, j+self.nat):
					l2 = lines[k].split()
					for u in range(6):
						A[k-j,u] = float(l2[1+u])
		f.close()
		return A
	### set HFI in D diag. basis
	def set_hfi_Dbasis(self):
		self.Ahfi = np.zeros((self.nat,3,3))
		# set transf. matrix U
		U = self.Deigv
		# read Ahfi from outcar
		ahf = self.read_hfi()
		# run over atoms
		for aa in range(self.nat):
			A = np.zeros((3,3))
			# set HF tensor
			A[0,0] = ahf[aa,0]
			A[1,1] = ahf[aa,1]
			A[2,2] = ahf[aa,2]
			A[0,1] = ahf[aa,3]
			A[1,0] = A[0,1]
			A[0,2] = ahf[aa,4]
			A[2,0] = A[0,2]
			A[1,2] = ahf[aa,5]
			A[2,1] = A[1,2]
			#
			AU = np.matmul(A, U)
			r = np.matmul(U.transpose(), AU)
			self.Ahfi[aa,:,:] = r[:,:]
			# MHz units
	### read free energy from OUTCAR
	def read_free_energy(self):
		# read file
		f = open(self.ipath+"OUTCAR")
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "free" and l[1] == "energy" and l[2] == "TOTEN":
				E = float(l[4])      # eV
		self.E = E
#
#  displaced structures class
#  prepare the VASP displacement calculation
#
class DisplacedStructs:
	def __init__(self, out_dir, outcars_dir):
		self.out_dir = out_dir + "/"
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		self.outcars_dir = outcars_dir
	# set atomic displacement (angstrom)
	def atom_displ(self, dr=np.array([0.1,0.1,0.1])):
		self.dx = dr[0]
		self.dy = dr[1]
		self.dz = dr[2]
	# generate new displaced structures
	def build_atom_displ_structs(self, struct_unprt):
		# struct -> unperturbed atomic structure
		struct_dict = struct_unprt.struct.as_dict()
		atoms_key = list(struct_dict.keys())[4]
		lattice_key = list(struct_dict.keys())[3]
		charge_key = list(struct_dict.keys())[2]
		unit_cell_key = list(struct_dict[lattice_key].keys())[0]
		# unit cell -> angstrom units
		self.unit_cell = np.array(struct_dict[lattice_key][unit_cell_key])
		# charge
		self.charge = struct_dict[charge_key]
		# list of atoms dictionary
		atoms = list(struct_dict[atoms_key])
		# set species list
		self.species = []
		species_key = list(atoms[0].keys())[0]
		element_key = list(atoms[0][species_key][0].keys())[0]
		for ia in range(struct_unprt.nat):
			self.species.append(atoms[ia][species_key][0][element_key])
		# extract atomic cartesian coordinates
		coord_xyz_keys = list(atoms[0].keys())[2]
		coord_abc_keys = list(atoms[0].keys())[1]
		atoms_cart_coords = np.zeros((struct_unprt.nat, 3))
		for ia in range(struct_unprt.nat):
			cart_coord_ia = atoms[ia][coord_xyz_keys]
			atoms_cart_coords[ia,:] = cart_coord_ia[:]
		# build perturbed structures
		displ_struct_list = []
		for ia in range(struct_unprt.nat):
			# x - displ 1
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,0] = atoms_cart_coords[ia,0] + self.dx
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '1', '1', struct])
			# x - displ 2
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,0] = atoms_cart_coords[ia,0] - self.dx
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '1', '2', struct])
			# y - displ 1
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,1] = atoms_cart_coords[ia,1] + self.dy
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '2', '1', struct])
			# y - displ 2
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,1] = atoms_cart_coords[ia,1] - self.dy
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '2', '2', struct])
			# z - displ 1
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,2] = atoms_cart_coords[ia,2] + self.dz
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '3', '1', struct])
			# z - displ 2
			atoms_cart_displ = np.zeros((struct_unprt.nat,3))
			atoms_cart_displ[:,:] = atoms_cart_coords[:,:]
			atoms_cart_displ[ia,2] = atoms_cart_coords[ia,2] - self.dz
			struct = Structure(lattice=self.unit_cell, species=self.species, coords=atoms_cart_displ,
				charge=self.charge, validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)
			displ_struct_list.append([str(ia+1), '3', '2', struct])
		# set up dictionary
		self.displ_structs = []
		keys = ['atom', 'x', 'sign', 'structure']
		for displ_struct in displ_struct_list:
			self.displ_structs.append(dict(zip(keys, displ_struct)))
	# write structures on file
	def write_structs_on_file(self, significant_figures=16, direct=True, vasp4_compatible=False):
		for displ_struct in self.displ_structs:
			struct = displ_struct['structure']
			at_index = displ_struct['atom']
			x = displ_struct['x']
			sgn = displ_struct['sign']
			# prepare file
			file_name = "POSCAR-" + at_index + "-" + x + "-" + sgn
			poscar = Poscar(struct)
			poscar.write_file(filename="{}".format(self.out_dir+file_name), direct=direct,
				vasp4_compatible=vasp4_compatible, significant_figures=significant_figures)
#