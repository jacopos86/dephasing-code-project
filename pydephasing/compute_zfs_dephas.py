# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import sys
import numpy as np
import yaml
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.gradient_interactions import gradient_ZFS
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.utility_functions import compute_index_to_ia_map, extract_atoms_coords
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.ph_ampl_module import PhononAmplitude
from pydephasing.energy_fluct_mod import spin_level_fluctuations_ofq, spin_level_fluctuations
from pydephasing.auto_correlation_module import autocorrel_func
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
    # prepare data arrays
    T2i_ofT = np.zeros((2,input_params.ntmp))
    D2_ofT  = np.zeros(input_params.ntmp)
    tauc_ofT= np.zeros((2,input_params.ntmp))
    # if ph res.
    if ph_resolved:
        T2i_phr_ofT = np.zeros((2,3*nat,input_params.ntmp))
        D2_phr_ofT = np.zeros((3*nat,input_params.ntmp))
        tauc_phr_ofT = np.zeros((2,3*nat,input_params.ntmp))
    # at. res.
    if at_resolved:
        T2i_atr_ofT = np.zeros((2,nat,input_params.ntmp))
        D2_atr_ofT = np.zeros((nat,input_params.ntmp))
        tauc_atr_ofT = np.zeros((2,nat,input_params.ntmp))
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
            deltaE_phm_oft = comm.bcast(deltaE_phm_oft, root=0)
        if at_resolved:
            deltaE_atr_oft = np.zeros((input_params.nt2,nat))
            for ia in range(nat):
                deltaE_atr_list = comm.gather(dE_oft.deltaE_atr[:,ia,iT], root=0)
                if rank == 0:
                    for dE in deltaE_atr_list:
                        deltaE_atr_oft[:,ia] = deltaE_atr_oft[:,ia] + dE[:]
            deltaE_atr_oft = comm.bcast(deltaE_atr_oft, root=0)
        deltaE_oft = np.zeros(input_params.nt)
        deltaE_oft_list = comm.gather(dE_oft.deltaE_oft[:,iT], root=0)
        if rank == 0:
            for dE in deltaE_oft_list:
                deltaE_oft[:] = deltaE_oft[:] + dE[:]
        deltaE_oft = comm.bcast(deltaE_oft, root=0)
        comm.Barrier()
        # compute auto-correlation function
        # procedure
        # initialize acf class
        acf = autocorrel_func(nat, input_params)
        # compmute acf
        acf.compute_acf(deltaE_oft, input_params)
        # T2 inv.
        T2i   = acf.T2i_ofT
        # D2
        D2    = acf.D2_ofT
        # tauc
        tau_c = acf.tauc_ofT
        # store data in arrays
        T2i_ofT[:,iT] = T2i[:]
        D2_ofT[iT] = D2
        tauc_ofT[:,iT] = tau_c[:]
        #
        # if atom resolved calc.
        #
        if at_resolved:
            # make list of atoms
            conf_rng_list = range(nat)
            calc_loc_list = np.array_split(conf_rng_list, size)
            calc_loc_sp   = calc_loc_list[rank]
            # run over separated atoms
            for ia in calc_loc_sp:
                # init. deltaE
                deltaEt = np.zeros(input_params.nt2)
                deltaEt[:] = deltaE_atr_oft[:,ia]
                # compute acf + T2 times + print acf data
                acf.compute_acf_atr(deltaEt, input_params, ia)
            # T2 inv
            T2i_atr = acf.T2i_atr_ofT
            # D2
            D2_atr  = acf.D2_atr_ofT
            # tau_c
            tauc_atr= acf.tauc_atr_ofT
            # collect data in single proc.
            T2i_list = comm.gather(T2i_atr, root=0)
            D2_list  = comm.gather(D2_atr, root=0)
            tauc_list= comm.gather(tauc_atr, root=0)
            comm.Barrier()
            # write results on array
            if rank == 0:
                # init. arrays
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
            # bcast data
            T2i_atr = comm.bcast(T2i_atr, root=0)
            D2_atr = comm.bcast(D2_atr, root=0)
            tauc_atr = comm.bcast(tauc_atr, root=0)
            # store data into arrays
            T2i_atr_ofT[:,:,iT] = T2i_atr[:,:]
            D2_atr_ofT[:,iT] = D2_atr[:]
            tauc_atr_ofT[:,:,iT] = tauc_atr[:,:]
        #
        # if ph. resolved calc.
        #  
        if ph_resolved:
            # make list of modes
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
            T2i_phr = acf.T2i_phm_ofT
            # D2
            D2_phr  = acf.D2_phm_ofT
            # tau_c
            tauc_phr= acf.tauc_phm_ofT
            # collect data into single proc.
            T2i_list = comm.gather(T2i_phr, root=0)
            D2_list  = comm.gather(D2_phr, root=0)
            tauc_list= comm.gather(tauc_phr, root=0)
            comm.Barrier()
            # write final results in arrays
            if rank == 0:
                # init. arrays
                T2i_phr[:,:] = 0.
                D2_phr[:]    = 0.
                tauc_phr[:,:]= 0.
                # iterate over proc.
                for x in T2i_list:
                    T2i_phr[:,:] = T2i_phr[:,:] + x[:,:]
                # D2
                for x in D2_list:
                    D2_phr[:] = D2_phr[:] + x[:]
                # tau_c
                for x in tauc_list:
                    tauc_phr[:,:] = tauc_phr[:,:] + x[:,:]
            # bcast data
            T2i_phr = comm.bcast(T2i_phr, root=0)
            D2_phr  = comm.bcast(D2_phr, root=0)
            tauc_phr= comm.bcast(tauc_phr, root=0)
            # store data into arrays
            T2i_phr_ofT[:,:,iT] = T2i_phr[:,:]
            D2_phr_ofT[:,iT] = D2_phr[:]
            tauc_phr_ofT[:,:,iT] = tauc_phr[:,:]
    #
    # save results on file
    #
    if rank == 0:
        T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0, 'temperature' : 0}
        T2_dict['T2_inv'] = T2i_ofT
        T2_dict['D2'] = D2_ofT
        T2_dict['tau_c'] = tauc_ofT
        T2_dict['temperature'] = input_params.temperatures
        # write T2 yaml file
        namef = "T2-data.yml"
        with open(input_params.write_dir+namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
        #
        # if atom resolved
        #
        if at_resolved:
            T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0, 'temperature' : 0}
            T2_dict['T2_inv'] = T2i_atr_ofT
            T2_dict['D2'] = D2_atr_ofT
            T2_dict['tau_c'] = tauc_atr_ofT
            T2_dict['temperature'] = input_params.temperatures
            # write T2 yaml file
            namef = "T2-atr-data.yml"
            with open(input_params.write_dir+namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)
        #
        # if phonon resolved
        #
        if ph_resolved:
            T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0, 'temperature' : 0, 'w_lambda' : 0}
            # compute average phonon freq.
            wl = np.zeros(3*nat)
            for iq in range(nq):
                wuq = wu[iq]
                wl[:] = wl[:] + wq[iq] * wuq[:]
            T2_dict['T2_inv'] = T2i_phr_ofT
            T2_dict['D2'] = D2_phr_ofT
            T2_dict['tau_c'] = tauc_phr_ofT
            T2_dict['temperature'] = input_params.temperatures
            T2_dict['w_lambda'] = wl
            # write T2 yaml file
            namef = "T2-phr-data.yml"
            with open(input_params.write_dir+namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)