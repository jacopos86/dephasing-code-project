import sys
from mpi4py import MPI
from pydephasing.create_displ_struct_files import gen_poscars, gen_2ndorder_poscar
from pydephasing.input_parameters import data_input
from pydephasing.compute_zfs_dephas import compute_zfs_autocorrel_func
from pydephasing.compute_exc_dephas import compute_hom_exc_autocorrel_func
from pydephasing.compute_hfi_dephas import compute_hfi_autocorrel_func
#
# set up parallelization
#
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#
if rank == 0:
    print("-------------------------------------------") 
    print("-------- START PYDEPHASING PROGRAM --------")
    print("-------------------------------------------")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#
calc_type = sys.argv[1]
if calc_type == "--energy":
    if rank == 0:
        print("-------------------------------------------")
        print("--- ENERGY LEVELS DEPHASING CALCULATION ---")
    # prepare energy dephasing calculation
    calc_type2 = sys.argv[2]
    if calc_type2 == "--homo":
        if rank == 0:
            print("-------------------------------------------")
            print("-------- HOMOGENEOUS CALCULATION ----------")
            print("-------------------------------------------")
        a = int(sys.argv[3])
        # action -> sys.argv[4]
        # 1 -> no ph / atoms resolved calculation
        # 2 -> atoms resolved calculation
        # 3 -> ph resolved only
        # 4 -> atoms + ph resolved calculation
        if a == 1:
            at_resolved = False
            ph_resolved = False
        elif a == 2:
            at_resolved = True
            ph_resolved = False
        elif a == 3:
            at_resolved = False
            ph_resolved = True
        elif a == 4:
            at_resolved = True
            ph_resolved = True
        else:
            if rank == 0:
                print("----------wrong action flag------------")
                print(-----------------"usage: ---------------")
                print("1 -> no ph / atoms resolved calculation")
                print("----2 -> atoms resolved calculation----")
                print("--------3 -> ph resolved only----------")
                print("--4 -> atoms + ph resolved calculation-")
            sys.exit(1)
        # read input file
        input_file = sys.argv[4]
        input_params = data_input()
        input_params.read_data(input_file)
        # compute auto correl. function first
        compute_hom_exc_autocorrel_func(input_params, at_resolved, ph_resolved)
elif calc_type == "--spin":
    if rank == 0:
        print("-------------------------------------------")
        print("------- SPIN DEPHASING CALCULATION --------")
    # prepare spin dephasing calculation
    calc_type2 = sys.argv[2]
    if calc_type2 == "--homo":
        if rank == 0:
            print("-------------------------------------------")
            print("-------- HOMOGENEOUS CALCULATION ----------")
            print("------------ COMPUTING T2* ----------------")
            print("-------------------------------------------")
        a = int(sys.argv[3])
        # action -> sys.argv[3]
        # 1 -> no ph / atoms resolved calculation
        # 2 -> atoms resolved calculation
        # 3 -> ph resolved only
        # 4 -> atoms + ph resolved calculation
        if a == 1:
            at_resolved = False
            ph_resolved = False
        elif a == 2:
            at_resolved = True
            ph_resolved = False
        elif a == 3:
            at_resolved = False
            ph_resolved = True
        elif a == 4:
            at_resolved = True
            ph_resolved = True
        else:
            if rank == 0:
                print("----------wrong action flag------------")
                print(-----------------"usage: ---------------")
                print("1 -> no ph / atoms resolved calculation")
                print("----2 -> atoms resolved calculation----")
                print("--------3 -> ph resolved only----------")
                print("--4 -> atoms + ph resolved calculation-")
            sys.exit(1)
        # read input file
        calc_type3 = sys.argv[4]
        input_file = sys.argv[5]
        input_params = data_input()
        input_params.read_data(input_file)
        # compute auto correl. function first
        if calc_type3 == "--debug":
            compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved, True, False, comm, rank, size)
        elif calc_type3 == "--plot":
            compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved, False, True, comm, rank, size)
        elif calc_type3 == "--noplot":
            compute_zfs_autocorrel_func(input_params, at_resolved, ph_resolved, False, False, comm, rank, size)
        else:
            if rank == 0:
                print("----------wrong plot/debug flag------------")
                print("-----------------usage: -------------------")
                print("--debug -> debug calc. -> check gradZFS----")
                print("--plot -> extract gradZFS data ------------")
                print("--noplot -> no data extraction-------------")
            sys.exit(1)
    elif calc_type2 == "--inhomo":
        if rank == 0:
            print("-------------------------------------------")
            print("------- INHOMOGENEOUS CALCULATION ---------")
            print("------------ COMPUTING T2* ----------------")
            print("-------------------------------------------")
elif calc_type == "--init":
    # read data file
    order = sys.argv[2]
    input_file = sys.argv[3]
    input_params = data_input()
    input_params.read_data_pre(input_file)
    if rank == 0:
        print("------- BUILD DISPLACED STRUCTURES --------")
    if int(order) == 1:
        gen_poscars(input_params)
    else:
        gen_2ndorder_poscar(input_params)
    if rank == 0:
        print("-------------------------------------------")
elif calc_type == "--post":
    # post process output data from VASP
    input_file = sys.argv[2]
else:
    if rank == 0:
        print("-------------------------------------------")
        print("------ CALC_TYPE FLAG NOT RECOGNIZED ------")
        print("---------- END OF THE PROGRAM -------------")
        print("-------------------------------------------")
    sys.exit(1)
if rank == 0:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")