# This is the main subroutine
# for the homogeneous exciton dephasing calculation
# it computes the energy auto-correlation function
# and it returns it for further processing
import numpy as np
from pydephasing.set_structs import UnpertStruct
from pydephasing.gradient_interactions import gradient_Eg
from pydephasing.utility_functions import extract_atoms_coords, compute_index_to_ia_map
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.ph_ampl_module import PhononAmplitude
from pydephasing.energy_fluct_mod import energy_level_fluctuations, energy_level_fluctuations_ofq
#
def compute_hom_exc_autocorrel_func(input_params, at_resolved, ph_resolved, print_data, comm, rank, size):
    # input_params -> input parameters data object
    # at_resolved -> atom resolved calculation
    # ph_resolved -> phonon resolved calculation
    #
    # first: extract unperturbed structure (ground state)
    GS = UnpertStruct(input_params.unpert_dir_gs)
    GS.read_poscar()
    # number of atoms in simulation
    nat = GS.nat
    # extract GS energy
    GS.read_free_energy()
    # second: extract unperturbed structure (excited state)
    EXC = UnpertStruct(input_params.unpert_dir_exc)
    EXC.read_poscar()
    # extract EXC energy
    EXC.read_free_energy()
    print("exciton energy= ", EXC.E - GS.E, "eV")
    #
    # compute GS config. forces
    #
    gradEGS = gradient_Eg(nat)
    # set GS forces
    gradEGS.set_forces(GS)
    # set force constants
    gradEGS.set_force_constants(GS, input_params.gs_fc_file)
    #
    # set EXC config. forces
    #
    gradEXC = gradient_Eg(nat)
    # set EXC forces
    gradEXC.set_forces(EXC)
    # set force constants
    #gradEXC.set_force_constants(EXC, input_params.exc_fc_file)
    # gradZPL -> gradEXC - gradEGS
    gradZPL = np.zeros(3*nat)
    gradZPL[:] = gradEXC.forces[:] - gradEGS.forces[:]
    # eV / Ang units
    #
    # define force constants -> HessZPL
    #
    hessZPL = np.zeros((3*nat,3*nat))
    hessZPL[:,:] = gradEXC.force_const[:,:] - gradEGS.force_const[:,:]
    # eV / Ang^2 units
    # print data
    #if rank == 0:
    #    print_zpl_fluct(gradZPL, hessZPL, input_params.out_dir)
    comm.Barrier()
    # set atoms dict
    atoms_dict = extract_atoms_coords(input_params, nat)
    # compute index maps
    index_to_ia_map = compute_index_to_ia_map(nat)
    # extract phonon data
    u, wu, nq, qpts, wq, mesh = extract_ph_data(input_params)
    if rank == 0:
        print("nq: ", nq)
        print("mesh: ", mesh)
    # initialize phonon amplitudes
    ph_ampl = PhononAmplitude(nat, input_params)
    #
    # prepare calculation over q pts.
    #
    qirr_list = range(nq)
    qirr_loc_list = np.array_split(qirr_list, size)
    qirr_list_loc_proc = qirr_loc_list[rank]
    #
    # compute energy fluct.
    #
    dE_oft = energy_level_fluctuations(at_resolved, ph_resolved, nat, input_params)
    # run over q pts.
    for iq in qirr_list_loc_proc:
        # w(q) data (THz)
        wuq = wu[iq]
        # eu(q) data
        euq = u[iq]
        # q vector
        qv = qpts[iq]
        # compute atoms displ. first
        ph_ampl.compute_ph_amplq(euq, wuq, nat, input_params, index_to_ia_map, atoms_dict)
        # compute deltaE(q,t)
        dEq_oft = energy_level_fluctuations_ofq(at_resolved, ph_resolved, nat, input_params)
        dEq_oft.compute_deltaEq_oft(at_resolved, ph_resolved, qv, wuq, nat, gradZPL, hessZPL, input_params, index_to_ia_map, atoms_dict, ph_ampl)
        # compute average deltaE(t)
        dE_oft.compute_deltaE_oft(at_resolved, ph_resolved, wq[iq], dEq_oft)
        print("iq: " + str(iq) + " -> completed")
    #
    comm.Barrier()
    # run over the temperature list
    for iT in range(input_params.ntmp):
        # collect data from different
        # processors
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
    #
    # set up auto-correlation functions
    #
    #Eg_acf = autocorrelation_function(at_resolved, ph_resolved, nat, input_params)
    #