# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import numpy as np
import sys
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.gradient_interactions import gradient_ZFS
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.utility_functions import compute_index_to_ia_map, compute_index_to_idx_map
#
def compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved, debug, print_data):
    # input_params: input parameters object
    # at_resolved : True -> compute atom resolved auto correlation
    # ph_resolved : True -> compute ph resolved auto correlation
    #
    # first : extract ZFS from unperturbed structure calculation
    struct0 = UnpertStruct(input_params.unpert_dir)
    struct0.read_poscar()
    # atoms number in simulation
    nat = struct0.nat
    # get zfs tensor
    struct0.read_zfs_tensor()
    # create displaced structures
    struct_list = []
    for i in range(len(input_params.pert_dirs)):
        displ_struct = DisplacedStructs(input_params.pert_dirs[i], input_params.outcar_dirs[i])
        # set atomic displ. in the structure
        displ_struct.atom_displ(input_params.atoms_displ[i])      # Ang
        # append to list
        struct_list.append(displ_struct)
    # set ZFS gradient
    gradZFS = gradient_ZFS(nat)
    gradZFS.set_tensor_gradient(struct_list, struct0, input_params.grad_info, input_params.out_dir)
    # set ZFS gradient in quant. axis coordinates
    gradZFS.set_grad_D_tensor(struct0)
    # debug mode
    if debug:
        gradZFS.plot_tensor_grad_component(struct_list, struct0, input_params.out_dir)
        sys.exit()
    # print data
    if print_data:
        gradZFS.write_Dtensor_to_file(input_params.out_dir)
    # set up the spin Hamiltonian
    Hss = spin_hamiltonian()
    # compute index maps
    index_to_ia_map = compute_index_to_ia_map(nat)
    index_to_idx_map = compute_index_to_idx_map(nat)
    # set the energy difference gradient 
    # F = <1|S Grad D S|1> - <0|S Grad D S|0>
    F_ax = Hss.set_grad_deltaEzfs(gradZFS, input_params.qs1, input_params.qs2, nat)
    #
    # compute auto-correlation function
    # procedure