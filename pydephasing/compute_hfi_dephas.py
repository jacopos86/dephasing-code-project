# This is the main subroutine used
# for the calculation of the HFI component
# of the dephasing time
# it computes the relative autocorrelation function
# and returns it for further processing
import numpy as np
from itertools import product
import sys
import yaml
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.utility_functions import compute_index_to_ia_map, extract_atoms_coords
from pydephasing.gradient_interactions import gradient_HFI
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.ph_ampl_module import PhononAmplitude
from pydephasing.energy_fluct_mod import spin_level_fluctuations_ofq, spin_level_fluctuations
from pydephasing.auto_correlation_module import autocorrel_func_hfi_dyn
#
def compute_hfi_autocorrel_func(input_params, at_resolved, ph_resolved, comm, rank, size):
    # input_params -> input parameters object
    # at_resolved : True -> compute atom resolved auto correlation
    # ph_resolved : True -> compute ph resolved auto correlation
    #
    # first set unperturbed structure
    struct0 = UnpertStruct(input_params.unpert_dir)
    struct0.read_poscar()
    # number of atoms in simulation
    nat = struct0.nat
    # first get ZFS tensor
    struct0.read_zfs_tensor()
    # set hyperfine interaction
    if not input_params.fc_core:
        struct0.set_hfi_Dbasis(input_params.fc_core)
    else:
        struct0.set_hfi_Dbasis()
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
    # compute grad Ahfi
    gradHFI = gradient_HFI(nat)
    # set the gradient
    if not input_params.fc_core:
        gradHFI.set_grad_Ahfi(struct_list, nat, input_params.grad_info, input_params.out_dir, input_params.fc_core)
    else:
        gradHFI.set_grad_Ahfi(struct_list, nat, input_params.grad_info, input_params.out_dir)
    # set gradient in quant. vector basis
    gradHFI.set_grad_Ahfi_Ddiag_basis(struct0)
    # extract atom positions
    atoms_dict = extract_atoms_coords(input_params, nat)
    #
    # set up the spin
    # configurations
    #
    spin_config_list = []
    if rank == 0:
        print("n. config: ", input_params.nconf)
        print("n. spins: ", input_params.nsp)
    for ic in range(input_params.nconf):
        # set spin config.
        config = nuclear_spins_config(input_params.nsp, input_params.B0)
        config.set_nuclear_spins(nat, input_params.temperatures[0])
        # compute the forces
        config.compute_force_gHFI(gradHFI, Hss, input_params.qs1, input_params.qs2, nat)
        # add to list
        spin_config_list.append(config)
    conf_rng_list = range(len(spin_config_list))
    #
    # extract phonon data
    #
    u, wu, nq, qpts, wq, mesh = extract_ph_data(input_params)
    if rank == 0:
        print("nq: ", nq)
        print("mesh: ", mesh)
    # initialize ph. ampl.
    ph_ampl = PhononAmplitude(nat, input_params)
    #
    comm.Barrier()
    #
    # prepare calculation
    # over q pts.
    #
    qirr_list = range(nq)
    # parallelization sec.
    full_calc_list = list(product(qirr_list, conf_rng_list))
    calc_loc_list = np.array_split(full_calc_list, size)
    calc_loc_sp = calc_loc_list[rank]
    # set dE(t) class
    dE_oft = spin_level_fluctuations(at_resolved, ph_resolved, nat, input_params, False, input_params.nconf)
    #
    # run over (ic,isp) pairs
    #
    for ic in calc_loc_sp:
        # spin config.
        icl = ic[1]
        config = spin_config_list[icl]
        # q pt. index
        iq = ic[0]
        print("iq= ", iq, "ic= ", icl)
        # w(q) data (THz)
        wuq = wu[iq]
        # eu(q) data
        euq = u[iq]
        # q vector
        qv = qpts[iq]
        # compute atoms displ.
        ph_ampl.compute_ph_amplq(euq, wuq, nat, input_params, index_to_ia_map, atoms_dict)
        # initialize deltaEq(t)
        dEq_oft = spin_level_fluctuations_ofq(at_resolved, ph_resolved, nat, input_params, False, input_params.nconf)
        # compute deltaEq(t)
        dEq_oft.compute_deltaEq_hfi_oft(at_resolved, ph_resolved, qv, wuq, nat, config, input_params, index_to_ia_map, atoms_dict, ph_ampl, icl)
        # compute average deltaE(t)
        dE_oft.compute_deltaE_hfi_oft(at_resolved, ph_resolved, wq[iq], dEq_oft)
        print("iq: " + str(iq) + "- ic: " + str(icl) + " -> completed")
    #
    comm.Barrier()
    # collect data from different processors
    if ph_resolved:
        deltaE_phm_oft = np.zeros((input_params.nt2,3*nat))
        for im in range(3*nat):
            deltaE_phm_list = comm.gather(dE_oft.deltaE_hfi_phm[:,im], root=0)
            if rank == 0:
                for dE in deltaE_phm_list:
                    deltaE_phm_oft[:,im] = deltaE_phm_oft[:,im] + dE[:]
        deltaE_phm_oft = comm.bcast(deltaE_phm_oft, root=0)
        deltaE_phm_oft[:,:] = deltaE_phm_oft[:,:] / input_params.nconf
    # atom resolved
    if at_resolved:
        deltaE_atr_oft = np.zeros((input_params.nt2,nat))
        for ia in range(nat):
            deltaE_atr_list = comm.gather(dE_oft.deltaE_hfi_atr[:,ia], root=0)
            if rank == 0:
                for dE in deltaE_atr_list:
                    deltaE_atr_oft[:,ia] = deltaE_atr_oft[:,ia] + dE[:]
        deltaE_atr_oft = comm.bcast(deltaE_atr_oft, root=0)
        deltaE_atr_oft[:,:] = deltaE_atr_oft[:,:] / input_params.nconf
    # full
    deltaE_oft = np.zeros((input_params.nt,input_params.nconf))
    for ic in range(input_params.nconf):
        deltaE_oft_list = comm.gather(dE_oft.deltaE_hfi_oft[:,ic], root=0)
        if rank == 0:
            for dE in deltaE_oft_list:
                deltaE_oft[:,ic] = deltaE_oft[:,ic] + dE[:]
    # bcast to other processes
    deltaE_oft = comm.bcast(deltaE_oft, root=0)
    comm.Barrier()
    #
    # share config. among processes
    #
    conf_rng_list = range(len(spin_config_list)+1)
    calc_loc_list = np.array_split(conf_rng_list, size)
    calc_loc_sp = calc_loc_list[rank]
    # initialize acf class
    acf = autocorrel_func_hfi_dyn(nat, input_params)
    # run over each separate configuration
    for ic in calc_loc_sp:
        # init dE array
        deltaEt = np.zeros(input_params.nt)
        # if ic = nconf compute average
        if ic == 0:
            # compute average
            for icl in range(input_params.nconf):
                deltaEt[:] = deltaEt[:] + deltaE_oft[:,icl]
            deltaEt[:] = deltaEt[:] / input_params.nconf
        else:
            deltaEt[:] = deltaE_oft[:,ic-1]
        # compute acf + T2 times + print acf data
        acf.compute_acf(deltaEt, input_params, ic)
    # T2 inv.
    T2i   = acf.T2i_ofT
    # D2
    D2    = acf.D2_ofT
    # tauc
    tau_c = acf.tauc_ofT
    # collect results
    T2i_list = comm.gather(T2i, root=0)
    D2_list  = comm.gather(D2, root=0)
    tauc_list= comm.gather(tau_c, root=0)
    # write data on file
    if rank == 0:
        # init. arrays
        T2i[:,:]   = 0.
        D2[:]      = 0.
        tau_c[:,:] = 0.
        # iterate over proc.
        for x in T2i_list:
            T2i[:,:] = T2i[:,:] + x[:,:]
        # D2
        for x in D2_list:
            D2[:] = D2[:] + x[:]
        # tau_c
        for x in tauc_list:
            tau_c[:,:] = tau_c[:,:] + x[:,:]
        # write data on file
        T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0}
        T2_dict['T2_inv'] = T2i
        T2_dict['D2'] = D2
        T2_dict['tau_c'] = tau_c
        #
        # save dicts on file
        #
        namef = "T2-data.yml"
        with open(input_params.write_dir+namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    #
    # atom resolved section
    #
    if at_resolved:
        # make list atoms
        conf_rng_list = range(nat)
        calc_loc_list = np.array_split(conf_rng_list, size)
        calc_loc_sp = calc_loc_list[rank]
        # run over atoms
        for ia in calc_loc_sp:
            # init. deltaE
            deltaEt = np.zeros(input_params.nt2)
            deltaEt[:] = deltaE_atr_oft[:,ia]
            # compute acf + T2 times + print acf data
            acf.compute_acf_atr(deltaEt, input_params, ia)
        # T2 inv
        T2i_atr  = acf.T2i_atr_ofT
        # D2
        D2_atr   = acf.D2_atr_ofT
        # tau_c
        tauc_atr = acf.tauc_atr_ofT
        # collect data
        T2i_list = comm.gather(T2i_atr, root=0)
        D2_list  = comm.gather(D2_atr, root=0)
        tauc_list= comm.gather(tauc_atr, root=0)
        # write data on file
        if rank == 0:
            # init arrays
            T2i_atr[:,:] = 0.
            D2_atr[:]    = 0.
            tauc_atr[:,:]= 0.
            # iterate over proc.
            for x in T2i_list:
                T2i_atr[:,:] = T2i_atr[:,:] + x[:,:]
            # D2
            for x in D2_list:
                D2_atr[:] = D2_atr[:] + x[:]
            # tau_c
            for x in tauc_list:
                tauc_atr[:,:] = tauc_atr[:,:] + x[:,:]
            # write data on file
            T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0}
            T2_dict['T2_inv'] = T2i_atr
            T2_dict['D2'] = D2_atr
            T2_dict['tau_c'] = tauc_atr
            #
            # save dicts on file
            #
            namef = "T2-atr-data.yml"
            with open(input_params.write_dir+namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)
    #
    # phonon resolved section
    #
    if ph_resolved:
        # make modes list
        conf_rng_list = range(3*nat)
        calc_loc_list = np.array_split(conf_rng_list, size)
        calc_loc_sp   = calc_loc_list[rank]
        # run over modes
        for im in calc_loc_sp:
            # init deltaE
            deltaEt = np.zeros(input_params.nt2)
            deltaEt[:] = deltaE_phm_oft[:,im]
            # compute acf + T2 times + print acf data
            acf.compute_acf_phm(deltaEt, input_params, im)
        # T2 inv
        T2i_phm = acf.T2i_phm_ofT
        # D2
        D2_phm  = acf.D2_phm_ofT
        # tau_c
        tauc_phm= acf.tauc_phm_ofT
        # collect data
        T2i_list = comm.gather(T2i_phm, root=0)
        D2_list  = comm.gather(D2_phm, root=0)
        tauc_list= comm.gather(tauc_phm, root=0)
        # write data on file
        if rank == 0:
            # init. arrays
            T2i_phm[:,:] = 0.
            D2_phm[:]    = 0.
            tauc_phm[:,:]= 0.
            # iterate over proc.
            for x in T2i_list:
                T2i_phm[:,:] = T2i_phm[:,:] + x[:,:]
            # D2
            for x in D2_list:
                D2_phm[:] = D2_phm[:] + x[:]
            # tau_c
            for x in tauc_list:
                tauc_phm[:,:] = tauc_phm[:,:] + x[:,:]
            # write data on file
            T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0}
            T2_dict['T2_inv'] = T2i_phm
            T2_dict['D2'] = D2_phm
            T2_dict['tau_c'] = tauc_phm
            #
            # save dicts on file
            #
            namef = "T2-phr-data.yml"
            with open(input_params.write_dir+namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)
    comm.Barrier()
    if rank == 0:
        print("-----------------------------------------------")
        print("---------------- JOB FINISHED -----------------")
        print("-----------------------------------------------")
    sys.exit()