#
#   This module defines the gradient
#   of the different interaction terms
#   1) class: gradient ZFS
#   2) class: gradient Hyperfine interaction tensor
#   3) class: gradient ZPL
#
import numpy as np
import math
from pydephasing.phys_constants import eps
import matplotlib.pyplot as plt
#
class gradient_ZFS:
	# initialization
	def __init__(self, nat):
		self.gradDtensor = np.zeros((3*nat,3,3))
		self.gradD = np.zeros(3*nat)
		self.gradE = np.zeros(3*nat)
	# read outcar file
	def read_outcar_diag(self, outcar):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) > 0 and l[0] == "D_diag":
				# Dxx
				j = i+2
				l2 = lines[j].split()
				Dxx = float(l2[0])
				# Dyy
				j = j+1
				l2 = lines[j].split()
				Dyy = float(l2[0])
				# Dzz
				j = j+1
				l2 = lines[j].split()
				Dzz = float(l2[0])
				#
				D = np.array([Dxx, Dyy, Dzz])
		return D
	# read full tensor from outcar
	def read_outcar_full(self, outcar):
		# read file
		print(outcar)
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "D_xx" and l[5] == "D_yz":
				j = i+2
				l2 = lines[j].split()
				D = np.zeros((3,3))
				# D tensor components
				Dxx = float(l2[0])
				Dyy = float(l2[1])
				Dzz = float(l2[2])
				Dxy = float(l2[3])
				Dxz = float(l2[4])
				Dyz = float(l2[5])
				D[0,0] = Dxx
				D[1,1] = Dyy
				D[2,2] = Dzz
				D[0,1] = Dxy
				D[1,0] = Dxy
				D[0,2] = Dxz
				D[2,0] = Dxz
				D[1,2] = Dyz
				D[2,1] = Dyz
		return D
	# set ZFS gradients
	def set_tensor_gradient(self, displ_structs, unpert_struct, atoms_info, out_dir):
		# read data from atoms_info
		f = open(atoms_info, 'r')
		lines = f.readlines()
		self.pert_dict = {}
		for line in lines:
			l = line.split()
			if int(l[0]) == 0 and int(l[1]) == 0:
				self.default_dir = l[2]
			else:
				self.pert_dict[(int(l[0]), int(l[1]))] = l[2]
		jax = 0
		# run over atoms
		for ia in range(unpert_struct.nat):
			for idx in range(3):
				if (ia+1, idx+1) in self.pert_dict.keys():
					outcars_dir = self.pert_dict[(ia+1, idx+1)]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = out_dir + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						# Ang units
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# write file name
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc1 = self.read_outcar_full(outcar)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc2 = self.read_outcar_full(outcar)
				#
				self.gradDtensor[jax,0,0] = (Dc1[0,0] - Dc2[0,0]) / (2.*dr[idx])
				self.gradDtensor[jax,1,1] = (Dc1[1,1] - Dc2[1,1]) / (2.*dr[idx])
				self.gradDtensor[jax,2,2] = (Dc1[2,2] - Dc2[2,2]) / (2.*dr[idx])
				self.gradDtensor[jax,0,1] = (Dc1[0,1] - Dc2[0,1]) / (2.*dr[idx])
				self.gradDtensor[jax,1,0] = (Dc1[1,0] - Dc2[1,0]) / (2.*dr[idx])
				self.gradDtensor[jax,0,2] = (Dc1[0,2] - Dc2[0,2]) / (2.*dr[idx])
				self.gradDtensor[jax,2,0] = (Dc1[2,0] - Dc2[2,0]) / (2.*dr[idx])
				self.gradDtensor[jax,1,2] = (Dc1[1,2] - Dc2[1,2]) / (2.*dr[idx])
				self.gradDtensor[jax,2,1] = (Dc1[2,1] - Dc2[2,1]) / (2.*dr[idx])
				#
				# MHz / Ang units
				#
				jax = jax + 1
	def plot_tensor_grad_component(self, displ_structs, unpert_struct, out_dir):
		Dc0= unpert_struct.Dtensor
		# run over atoms
		for ia in range(unpert_struct.nat):
			for idx in range(3):
				if (ia+1, idx+1) in self.pert_dict.keys():
					outcars_dir = self.pert_dict[(ia+1, idx+1)]
				else:
					outcars_dir = self.default_dir
				out_dir_full = ''
				out_dir_full = out_dir + outcars_dir
				# look for right displaced struct
				for displ_struct in displ_structs:
					if displ_struct.outcars_dir == out_dir_full:
						dr = np.array([displ_struct.dx, displ_struct.dy, displ_struct.dz])
						d = [-dr[idx], 0., dr[idx]]
					else:
						pass
				out_dir_full = out_dir_full + "/"
				# write file name
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc1 = self.read_outcar_full(outcar)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir_full + file_name)
				Dc2 = self.read_outcar_full(outcar)
				#
				print(ia+1, idx+1)
				Dxy = [Dc2[0,1], Dc0[0,1], Dc1[0,1]]
				Dxz = [Dc2[0,2], Dc0[0,2], Dc1[0,2]]
				Dyz = [Dc2[1,2], Dc0[1,2], Dc1[1,2]]
				modelxy = np.polyfit(d, Dxy, 2)
				modelxz = np.polyfit(d, Dxz, 2)
				modelyz = np.polyfit(d, Dyz, 2)
				ffitxy = np.poly1d(modelxy)
				ffitxz = np.poly1d(modelxz)
				ffityz = np.poly1d(modelyz)
				x_s = np.arange(d[0], d[2]+d[2], d[2])
				plt.scatter(d, Dxy, c='k')
				plt.plot(x_s, ffitxy(x_s), color="k")
				plt.show()
				#
				plt.scatter(d, Dyz, c='r')
				plt.plot(x_s, ffityz(x_s), color="r")
				plt.show()
				#
				plt.scatter(d, Dxz, c='b')
				plt.plot(x_s, ffitxz(x_s), color="b")
				plt.show()
	def write_Dtensor_to_file(self, out_dir):
		# n. dof
		nm = self.gradDtensor.shape[0]
		# write data on file
		file_name = "Dtensor_xx"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,0,0])
		f.close()
		# write data on file
		file_name = "Dtensor_xy"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,0,1])
		f.close()
		# write data on file
		file_name = "Dtensor_yy"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,1,1])
		f.close()
		# write data on file
		file_name = "Dtensor_xz"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,0,2])
		f.close()
		# write data on file
		file_name = "Dtensor_yz"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,1,2])
		f.close()
		# write data on file
		file_name = "Dtensor_zz"
		out_file = "{}".format(out_dir + '/' + file_name)
		f = open(out_file, 'w')
		f.write("# jax          THz/Ang\n")
		for jax in range(nm):
			f.write("%d          " % (jax+1) + "%.10f\n" % self.U_gradD_U[jax,2,2])
		f.close()
	#
	# method
	# set grad D
	def set_grad_D(self):
		# D = 3/2 * Dzz
		self.gradD[:] = 3./2 * self.U_gradD_U[:,2,2]
		#
		#  (THz/Ang) units
		#
	# set grad E
	def set_grad_E(self, unpert_struct):
		# E = |Dxx - Dyy|/2
		# gradE = grad(Dxx - Dyy) * (Dxx - Dyy)/2/|Dxx - Dyy|
		D = unpert_struct.Ddiag
		if abs(D[0] - D[1]) < eps:
			sgn = math.copysign(1, D[0]-D[1])
			self.gradE[:] = sgn * (self.U_gradD_U[:,0,0] - self.U_gradD_U[:,1,1]) / 2.
		else:
			self.gradE[:] = (D[0]-D[1]) / (2.*abs(D[0]-D[1])) * (self.U_gradD_U[:,0,0] - self.U_gradD_U[:,1,1])
		#
		#  (THz/Ang) units
		#
	# set grad D tensor
	def set_grad_D_tensor(self, unpert_struct):
		# gradD = U^+ gD U
		self.U_gradD_U = np.zeros((3*unpert_struct.nat, 3, 3))
		U = unpert_struct.Deigv
		D = np.zeros((3,3))
		# iterate over jax
		jax = 0
		for ia in range(unpert_struct.nat):
			for idx in range(3):
				D[:,:] = 0.
				D[:,:] = self.gradDtensor[jax,:,:]
				DU = np.matmul(D, U)
				Dt = np.matmul(U.transpose(), DU)
				self.U_gradD_U[jax,:,:] = Dt[:,:]
				jax = jax+1
		self.U_gradD_U[:,:,:] = self.U_gradD_U[:,:,:] * 1.E-6
		#
		#   (THz/Ang)  units
		#
#
#   class :
#   gradient hyperfine interaction
#
class gradient_HFI:
	def __init__(self, nat):
		self.gradAhfi = np.zeros((3*nat,nat,3,3))
		# local basis array
		self.gAhfi_xx = []
		self.gAhfi_yy = []
		self.gAhfi_zz = []
		self.gAhfi_xy = []
		self.gAhfi_xz = []
		self.gAhfi_yz = []
	# read outcar file
	def read_outcar(self, outcar, nat):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 7:
				if l[0] == "ion" and l[6] == "A_yz":
					j = i+2
					A = np.zeros((nat,6))
					for k in range(j, j+nat):
						l2 = lines[k].split()
						for u in range(6):
							A[k-j,u] = float(l2[1+u])
		f.close()
		return A
	# set Ahfi gradient
	def set_grad_Ahfi(self, displ_structs, nat):
		dr = np.array([displ_structs.dx, displ_structs.dy, displ_structs.dz])
		out_dir = displ_structs.outcars_dir + "/"
		# run over atoms
		jax = 0
		for ia in range(nat):
			gradAxx = np.zeros((nat,3))
			gradAyy = np.zeros((nat,3))
			gradAzz = np.zeros((nat,3))
			gradAxy = np.zeros((nat,3))
			gradAxz = np.zeros((nat,3))
			gradAyz = np.zeros((nat,3))
			for idx in range(3):
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir + file_name)
				As1 = self.read_outcar(outcar, nat)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir + file_name)
				As2 = self.read_outcar(outcar, nat)
				#
				gradAxx[:,idx] = (As1[:,0] - As2[:,0]) / (2.*dr[idx])
				gradAyy[:,idx] = (As1[:,1] - As2[:,1]) / (2.*dr[idx])
				gradAzz[:,idx] = (As1[:,2] - As2[:,2]) / (2.*dr[idx])
				gradAxy[:,idx] = (As1[:,3] - As2[:,3]) / (2.*dr[idx])
				gradAxz[:,idx] = (As1[:,4] - As2[:,4]) / (2.*dr[idx])
				gradAyz[:,idx] = (As1[:,5] - As2[:,5]) / (2.*dr[idx])
				#
				# MHz/Ang units
				#
			self.gAhfi_xx.append(gradAxx)
			self.gAhfi_yy.append(gradAyy)
			self.gAhfi_zz.append(gradAzz)
			self.gAhfi_xy.append(gradAxy)
			self.gAhfi_xz.append(gradAxz)
			self.gAhfi_yz.append(gradAyz)
	# set grad Ahfi D diag basis set
	def set_grad_Ahfi_Ddiag_basis(self, unprt_struct):
		# set transf. matrix U
		U = unprt_struct.Deigv
		# start iterations over atoms
		jax = 0
		for ia in range(unprt_struct.nat):
			gradAxx = self.gAhfi_xx[ia]
			gradAyy = self.gAhfi_yy[ia]
			gradAzz = self.gAhfi_zz[ia]
			gradAxy = self.gAhfi_xy[ia]
			gradAxz = self.gAhfi_xz[ia]
			gradAyz = self.gAhfi_yz[ia]
			#
			gradA = np.zeros((unprt_struct.nat,3,3,3))
			gradA[:,:,0,0] = gradAxx[:,:]
			gradA[:,:,1,1] = gradAyy[:,:]
			gradA[:,:,2,2] = gradAzz[:,:]
			gradA[:,:,0,1] = gradAxy[:,:]
			gradA[:,:,1,0] = gradAxy[:,:]
			gradA[:,:,0,2] = gradAxz[:,:]
			gradA[:,:,2,0] = gradAxz[:,:]
			gradA[:,:,1,2] = gradAyz[:,:]
			gradA[:,:,2,1] = gradAyz[:,:]
			for idx in range(3):
				# run over every atoms
				for aa in range(unprt_struct.nat):
					ga = np.zeros((3,3))
					ga[:,:] = gradA[aa,idx,:,:]
					# transform over new basis
					gaU = np.matmul(ga, U)
					ga = np.matmul(U.transpose(), gaU)
					self.gradAhfi[jax,aa,:,:] = ga[:,:]
					#
					#  MHz / Ang units
					#
				jax = jax + 1
		#
		#  THz / Ang units
		#
		self.gradAhfi[:,:,:,:] = self.gradAhfi[:,:,:,:] * 1.E-6
#
#  class 
#  gradient ZPL
#
class gradient_Eg:
	def __init__(self,nat):
		self.gradE = np.zeros(3*nat)
		self.forces = np.zeros(3*nat)
		self.force_const = np.zeros((3*nat,3*nat))
	# read outcar file
	def read_outcar(self, outcar):
		# read file
		f = open(outcar, 'r')
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].split()
			if len(l) == 6 and l[0] == "free" and l[1] == "energy" and l[2] == "TOTEN":
				E = float(l[4])   # eV
		return E
	# compute grad E
	def set_gradE(self, displ_structs, nat):
		dr = np.array([displ_structs.dx, displ_structs.dy, displ_structs.dz])
		# OUTCAR directory
		out_dir = displ_structs.outcars_dir
		# compute grad E
		gE = np.zeros(3*nat)
		jax = 0
		for ia in range(nat):
			for idx in range(3):
				file_name = str(ia+1) + "-" + str(idx+1) + "-1/OUTCAR"
				outcar = "{}".format(out_dir + '/' + file_name)
				E_1 = self.read_outcar(outcar)
				#
				file_name = str(ia+1) + "-" + str(idx+1) + "-2/OUTCAR"
				outcar = "{}".format(out_dir + '/' + file_name)
				E_2 = self.read_outcar(outcar)
				#
				gE[jax] = (E_1 - E_2) / (2.*dr[idx])
				#
				# eV / Ang units
				#
				jax = jax + 1
		self.gradE[:] = gE[:]
	# compute forces method
	def set_forces(self, struct0):
		# read forces
		F = struct0.read_forces()
		# set up local force array
		nat = struct0.nat
		# run over atoms
		for ia in range(nat):
			self.forces[3*ia]   = F[ia,0]
			self.forces[3*ia+1] = F[ia,1]
			self.forces[3*ia+2] = F[ia,2]
			# eV / ang 
			# units
	# extract force constants
	def set_force_constants(self, struct0, fc_file):
		# read force constants
		Fc = struct0.read_force_const(fc_file)
		# iterate over atomic index
		for ia in range(struct0.nat):
			for ix in range(3):
				jax = ia*3 + ix
				for ib in range(struct0.nat):
					for iy in range(3):
						jby = ib*3 + iy
						self.force_const[jax,jby] = Fc[ia,ib,ix,iy]
		# eV / ang^2
		# units