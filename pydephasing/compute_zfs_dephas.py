# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import numpy as np
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.gradient_interactions import gradient_ZFS
#
def compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved):
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
    # compute energy fluctuations