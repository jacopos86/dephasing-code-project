# This is the driving subroutine for
# the calculation of the static dephasing
# it computes the energy auto-correlation
# and it returns it for further processing
import numpy as np
import yaml
import sys
from pydephasing.set_structs import UnpertStruct
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.energy_fluct_mod import spin_level_static_fluctuations
from pydephasing.auto_correlation_module import autocorrel_func_hfi_stat
#
def compute_hfi_stat_dephas(input_params, comm, rank, size):
    # input_params -> input parameters object
    # at_resolved : true -> compute atom res. auto-correlation
    # ph_resolved : true -> compute ph. resolved auto-correlation
    #
    # first : set calculation list
    calc_list = range(input_params.nconf)
    # parallelization section
    calc_loc_list = np.array_split(calc_list, size)
    calc_loc_sp = calc_loc_list[rank]
    # print calc. data
    if rank == 0:
        print("n. config: ", input_params.nconf)
        print("n. spins: ", input_params.nsp)
    comm.Barrier()
    # unpert. structure
    struct0 = UnpertStruct(input_params.unpert_dir)
    struct0.read_poscar()
    # n. atoms in the simulation
    nat = struct0.nat
    # get ZFS tensor
    struct0.read_zfs_tensor()
    # set HF interaction
    if not input_params.fc_core:
        struct0.set_hfi_Dbasis(input_params.fc_core)
    else:
        struct0.set_hfi_Dbasis()
    # set spin hamiltonian
    Hss = spin_hamiltonian()
    # set time spinor evol.
    Hss.set_time(input_params.dt, input_params.T)
    # set up the spin vector evolution
    Hss.compute_spin_vector_evol(struct0, input_params.psi0, input_params.B0)
    # write data on file
    if rank == 0:
        Hss.write_spin_vector_on_file(input_params.write_dir)
    comm.Barrier()
    # set average fluct. energy
    nt = int(input_params.T_mus/input_params.dt_mus)
    deltaE_aver_oft = np.zeros(nt)
    #
    # set up nuclear spins
    #
    for ic in calc_loc_sp:
        print("compute It - conf: ", str(ic+1))
        # set up spin config.
        config = nuclear_spins_config(input_params.nsp, input_params.B0)
        # set up time in (mu sec)
        config.set_time(input_params.dt_mus, input_params.T_mus)
        # set initial spins
        config.set_nuclear_spins(nat, input_params.temperatures[0])
        # compute dynamical evolution
        config.set_nuclear_spin_evol(Hss, struct0)
        # set temporal fluctuations
        config.set_nuclear_spin_time_fluct()
        # write data on file
        config.write_It_on_file(input_params.write_dir, ic)
        # compute eff. forces 
        # for each spin
        config.compute_force_HFS(Hss, struct0, input_params.qs1, input_params.qs2)
        # initialize energy fluct.
        E_fluct = spin_level_static_fluctuations(config)
        # set spin fluct. energy
        # extract delta I(t)
        E_fluct.compute_deltaE_oft(config)
        # compute average fluct.
        deltaE_aver_oft[:] = deltaE_aver_oft[:] + E_fluct.deltaE_oft[:]
        # eV units
        # init. ACF
        acf = autocorrel_func_hfi_stat(input_params)
        # compute ACF
        acf.compute_acf(E_fluct.deltaE_oft, input_params, ic)
        print("end It calculation - conf: ", str(ic+1))
    # wait processes
    comm.Barrier()
    #
    # gather arrays on a single
    # processor
    #
    deltaEt_aver_list = comm.gather(deltaE_aver_oft, root=0)
    #
    if rank == 0:
        # set total time arrays
        deltaE_aver_oft[:] = 0.
        # compute deltaE
        # divide by nconf
        for dE in deltaEt_aver_list:
            deltaE_aver_oft[:] = deltaE_aver_oft[:] + dE[:]
        deltaE_aver_oft[:] = deltaE_aver_oft[:] / input_params.nconf
        # compute acf
        acf.compute_acf(deltaE_aver_oft, input_params, -1)
    comm.Barrier()
    # extract data
    T2i_dat = acf.T2i_dat
    # D^2
    D2_dat = acf.D2_dat
    # tau_c
    tauc_dat = acf.tauc_dat
    # gather data into lists
    T2i_sp_list = comm.gather(T2i_dat, root=0)
    D2_sp_list  = comm.gather(D2_dat, root=0)
    tauc_sp_list= comm.gather(tauc_dat, root=0)
    #
    if rank == 0:
        # sec. units
        T2i_dat[:]  = 0.
        # eV^2 units
        D2_dat[:]   = 0.
        # ps units
        tauc_dat[:] = 0.
        # compute T2i
        for T2i in T2i_sp_list:
            T2i_dat[:] = T2i_dat[:] + T2i[:]
        # compute D2
        for D2 in D2_sp_list:
            D2_dat[:]  = D2_dat[:] + D2[:]
        # compute tau_c
        for tau_c in tauc_sp_list:
            tauc_dat[:] = tauc_dat[:] + tau_c[:]
        # write data on file
        T2_dict = {'T2_inv' : 0, 'D2' : 0, 'tau_c' : 0}
        T2_dict["T2_inv"] = T2i_dat
        T2_dict["D2"] = D2_dat
        T2_dict["tau_c"] = tauc_dat
        #
        # save dicts on file
        #
        namef = "T2-data.yml"
        with open(input_params.write_dir+namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    comm.Barrier()
    if rank == 0:
        print("-----------------------------------------------")
        print("---------------- JOB FINISHED -----------------")
        print("-----------------------------------------------")
    sys.exit()