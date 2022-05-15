#
#  test script
#
from pydephasing.set_structs import DisplacedStructs, UnpertStruct
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.gradient_interactions import gradient_ZFS, gradient_HFI, gradient_Eg
import numpy as np
import sys
import scipy.linalg as la
#
atoms_unprt = UnpertStruct("./examples/NV-diamond/GS")
atoms_unprt.read_poscar()
atoms_unprt.read_zfs_tensor()
U = atoms_unprt.Deigv
DU = np.matmul(atoms_unprt.Dtensor, U)
D = np.matmul(U.transpose(), DU)
struct_displ = DisplacedStructs("./examples/NV-diamond/DISPLACEMENT-FILES")
struct_displ.atom_displ(np.array([0.1, 0.1, 0.1]))
struct_displ.build_atom_displ_structs(atoms_unprt)
struct_displ.write_structs_on_file()
#
gradHFI = gradient_HFI(atoms_unprt.nat)
gradHFI.set_grad_Ahfi(struct_displ, atoms_unprt.nat, "./examples/NV-diamond/DISPL/")
gradHFI.set_grad_Ahfi_Ddiag_basis(atoms_unprt)
gradZFS = gradient_ZFS(atoms_unprt.nat)
gradZFS.set_tensor_gradient(struct_displ, atoms_unprt, "./examples/NV-diamond/DISPL/")
#gradZFS.set_grad_D()
#gradZFS.set_grad_E(atoms_unprt)
#gradEg = gradient_Eg(atoms_unprt.nat)
#gradEg.set_gradEg(displ_structs, nat, "./examples/GCB/GS/DISPL/", "./examples/GCB/EXC/DISPL/")
#gradZFS.set_grad_D_tensor(atoms_unprt)
#
Hss = spin_hamiltonian()
spin_config = nuclear_spins_config(10)
spin_config.set_time(0.01, 300.)
spin_config.set_nuclear_spins(atoms_unprt.nat)
spin_config.set_nuclear_spin_evol(np.array([0.,0.,100.]), Hss, atoms_unprt)
sys.exit()
#
qs1 = np.array([1.,0.,0.])
qs2 = np.array([0.,1.,0.])
lambda_coef = np.array([1.,0.])
#Hss.set_grad_Ess(gradZFS, gradHFI, spin_config, qs, atoms_unprt.nat, lambda_coef)
Hss.set_grad_deltaEss(gradZFS, gradHFI, spin_config, qs1, qs2, atoms_unprt.nat, lambda_coef)
#Hss.diagonalize_Hss(atoms_unprt.Ddiag)
sys.exit()
#
E_S1 = Hss.set_E_coef(qs)
E_A1 = Hss.set_A_coef(qs)
#qs = np.array([1.,-1j,0.])/np.sqrt(2.)
qs = np.array([0.,1.,0.])
E_S2 = Hss.set_E_coef(qs)
E_A2 = Hss.set_A_coef(qs)
print(E_A1-E_A2)
qs1 = qs
qs2 = np.array([0.,1.,0.])
Hss.set_grad_deltaEss(gradZFS, gradHFI, spin_config, qs1, qs2, atoms_unprt.nat, lambda_coef)
