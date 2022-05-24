import sys
from pydephasing.create_displ_struct_files import gen_poscars
from pydephasing.input_parameters import data_input
#
print("-------------------------------------------") 
print("-------- START PYDEPHASING PROGRAM --------")
print("-------------------------------------------")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#
calc_type = sys.argv[1]
if calc_type == "--energy":
    print("-------------------------------------------")
    print("--- ENERGY LEVELS DEPHASING CALCULATION ---")
    # prepare energy dephasing calculation
elif calc_type == "--spin":
    print("-------------------------------------------")
    print("------- SPIN DEPHASING CALCULATION --------")
    # prepare spin dephasing calculation
    calc_type2 = sys.argv[2]
    if calc_type2 == "--homo":
        print("-------------------------------------------")
        print("-------- HOMOGENEOUS CALCULATION ----------")
        print("------------ COMPUTING T2 -----------------")
        print("-------------------------------------------")
        calc_type3 = sys.argv[3]
        # case 1) -> ZFS
        # case 2) -> HFI
        if calc_type3 == "--zfs":
            print("ok -> ready to go")
    elif calc_type2 == "--inhomo":
        print("-------------------------------------------")
        print("------- INHOMOGENEOUS CALCULATION ---------")
        print("------------ COMPUTING T2* ----------------")
        print("-------------------------------------------")
elif calc_type == "--init":
    # read data file
    input_file = sys.argv[2]
    input_params = data_input()
    input_params.read_data(input_file)
    print("------- BUILD DISPLACED STRUCTURES --------")
    gen_poscars(input_params)
    print("-------------------------------------------")
else:
    print("-------------------------------------------")
    print("------ CALC_TYPE FLAG NOT RECOGNIZED ------")
    print("---------- END OF THE PROGRAM -------------")
    print("-------------------------------------------")
    sys.exit(1)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")