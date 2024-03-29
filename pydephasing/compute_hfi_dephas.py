# This is the main subroutine used
# for the calculation of the HFI component
# of the dephasing time
# it computes the relative autocorrelation function
# and returns it for further processing
import numpy as np
import sys
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.utility_functions import compute_index_to_ia_map, compute_index_to_idx_map
from pydephasing.gradient_interactions import gradient_HFI
#
def compute_hfi_autocorrel_func(input_params, at_resolved, ph_resolved):
    # input_params -> input parameters object
    # at_resolved : True -> compute atom resolved auto correlation
    # ph_resolved : True -> compute ph resolved auto correlation
    #
    # first set unperturbed structure
    struct0 = UnpertStruct(input_params.unpert_dir)
    struct0.read_poscar()
    # number of atoms in simulation
    nat = struct0.nat
    # set hyperfine interaction
    # first get ZFS tensor
    struct0.read_zfs_tensor()
    # create perturbed atomic structures
    struct_list = []
    for i in range(len(input_params.pert_dirs)):
        displ_struct = DisplacedStructs(input_params.pert_dirs[i], input_params.outcar_dirs[i])
        # set atomic displacements
        displ_struct.atom_displ(input_params.atoms_displ[i])    # ang
        # append to list
        struct_list.append(displ_struct)
    # set spin Hamiltonian
    Hss = spin_hamiltonian()
    # set index maps
    index_to_ia_map = compute_index_to_ia_map(nat)
    index_to_idx_map = compute_index_to_idx_map(nat)
    # compute grad Ahfi
    gradHFI = gradient_HFI(nat)
    # set the gradient
    gradHFI.set_grad_Ahfi(struct_list, nat)
    # set gradient in quant. vector basis
    gradHFI.set_grad_Ahfi_Ddiag_basis(struct0)