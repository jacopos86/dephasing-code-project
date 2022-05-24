import sys
from pydephasing.create_input_files import gen_input_files
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
        gen_input_files()
    elif calc_type2 == "--inhomo":
        print("-------------------------------------------")
        print("------- INHOMOGENEOUS CALCULATION ---------")
        print("------------ COMPUTING T2* ----------------")
        print("-------------------------------------------")
        
else:
    print("-------------------------------------------")
    print("------ CALC_TYPE FLAG NOT RECOGNIZED ------")
    print("---------- END OF THE PROGRAM -------------")
    print("-------------------------------------------")
    sys.exit(1)