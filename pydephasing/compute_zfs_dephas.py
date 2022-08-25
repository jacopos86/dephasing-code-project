# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import sys
import numpy as np
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.gradient_interactions import gradient_ZFS
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.utility_functions import compute_index_to_ia_map, extract_atoms_coords
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.ph_ampl_module import PhononAmplitude
from pydephasing.energy_fluct_mod import spin_level_fluctuations_ofq, spin_level_fluctuations
#
def compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved, debug, print_data, comm, rank, size):
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
    # set the energy difference gradient 
    # F = <1|S Grad D S|1> - <0|S Grad D S|0>
    F_ax = Hss.set_grad_deltaEzfs(gradZFS, input_params.qs1, input_params.qs2, nat)
    # set atoms dict
    atoms_dict = extract_atoms_coords(input_params, nat)
    #
    # extract phonon data
    #
    u, wu, nq, qpts, wq, mesh = extract_ph_data(input_params)
    if rank == 0:
        print("nq: ", nq)
        print("mesh: ", mesh)
    # initialize phonon amplitudes
    ph_ampl = PhononAmplitude(nat, input_params)
    #
    comm.Barrier()
    #
    # prepare calculation over q pts.
    #
    qirr_list = range(nq)
    qirr_loc_list = np.array_split(qirr_list, size)
    qirr_list_loc_proc = qirr_loc_list[rank]
    # set dE(t) sum over q pts
    dE_oft = spin_level_fluctuations(at_resolved, ph_resolved, nat, input_params)
    #
    # run over q pts
    #
    for iq in qirr_list_loc_proc:
        # w(q) data (THz)
        wuq = wu[iq]
        # eu(q) data
        euq = u[iq]
        # q vector
        qv = qpts[iq]
        # compute atoms displ. first
        ph_ampl.compute_ph_amplq(euq, wuq, nat, input_params, index_to_ia_map, atoms_dict)
        # compute now deltaE(q,t)
        dEq_oft = spin_level_fluctuations_ofq(at_resolved, ph_resolved, nat, input_params)
        dEq_oft.compute_deltaEq_oft(at_resolved, ph_resolved, qv, wuq, nat, F_ax, input_params, index_to_ia_map, atoms_dict, ph_ampl)
        # compute average deltaE(t)
        dE_oft.compute_deltaE_oft(at_resolved, ph_resolved, wq[iq], dEq_oft)
        print("iq: " + str(iq) + " -> completed")
    #
    comm.Barrier()
    # run over temperature list
    for iT in range(input_params.ntmp):
        # collect data from different processors
        if ph_resolved:
            deltaE_phm_oft = np.zeros((input_params.nt2,3*nat))
            for im in range(3*nat):
                deltaE_phm_list = comm.gather(dE_oft.deltaE_phm[:,im,iT], root=0)
                if rank == 0:
                    for dE in deltaE_phm_list:
                        deltaE_phm_oft[:,im] = deltaE_phm_oft[:,im] + dE[:]
        if at_resolved:
            deltaE_atr_oft = np.zeros((input_params.nt2,nat))
            for ia in range(nat):
                deltaE_atr_list = comm.gather(dE_oft.deltaE_atr[:,ia,iT], root=0)
                if rank == 0:
                    for dE in deltaE_atr_list:
                        deltaE_atr_oft[:,ia] = deltaE_atr_oft[:,ia] + dE[:]
        deltaE_oft = np.zeros(input_params.nt)
        deltaE_oft_list = comm.gather(dE_oft.deltaE_oft[:,iT], root=0)
        if rank == 0:
            for dE in deltaE_oft_list:
                deltaE_oft[:] = deltaE_oft[:] + dE[:]
        # compute auto-correlation function
        # procedure