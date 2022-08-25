# this subroutine creates the input files
# to be run in the VASP calculation
from shutil import copyfile
from pydephasing.set_structs import UnpertStruct, DisplacedStructs, DisplacedStructures2ndOrder
import os
#
def gen_poscars(input_params):
    print("-------------------------------------------")
    print("------- FIRST UNPERTURBED STRUCTURE -------")
    print("-------------------------------------------")
    struct0 = UnpertStruct(input_params.unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define all the displced atoms structures
    for i in range(len(input_params.pert_dirs)):
        displ_struct = DisplacedStructs(input_params.pert_dirs[i])
        # set atoms displacements
        displ_struct.atom_displ(input_params.atoms_displ[i])   # Ang
        # build displaced atomic structures
        displ_struct.build_atom_displ_structs(struct0)
        # write data on file
        displ_struct.write_structs_on_file()
        # delete structure object
        del displ_struct
    print("---------- POSCAR FILES WRITTEN -----------")
    # create calculation directory
    nat = struct0.nat
    for i in range(len(input_params.calc_dirs)):
        if not os.path.exists(input_params.calc_dirs[i]):
            os.mkdir(input_params.calc_dirs[i])
        # run over all calculations
        for ia in range(nat):
            for idx in range(3):      # x-y-z index
                for s in range(2):    # +/- index
                    namef = input_params.pert_dirs[i] + "/POSCAR-" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                    named = input_params.calc_dirs[i] + "/" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                    if not os.path.exists(named):
                        os.mkdir(named)
                    # copy files in new directory
                    files = os.listdir(input_params.copy_files_dir)
                    for fil in files:
                        copyfile(input_params.copy_files_dir + "/" + fil, named + "/" + fil)
                    copyfile(namef, named + "/POSCAR")
#
# generate 2nd order displ
# POSCAR
#
def gen_2ndorder_poscar(input_params):
    print("-------------------------------------------")
    print("------- FIRST UNPERTURBED STRUCTURE -------")
    print("-------------------------------------------")
    struct0 = UnpertStruct(input_params.unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define displ. atomic structures
    for i in range(len(input_params.pert_dirs)):
        # init structures
        displ_struct = DisplacedStructures2ndOrder(input_params.pert_dirs[i])
        # set atoms displacement
        displ_struct.atom_displ(input_params.atoms_displ[i])   # ang
        # build displ. atoms structures
        displ_struct.build_atom_displ_structs(struct0)
        # write data on file
        displ_struct.write_structs_on_file()
        print("---------- POSCAR FILES WRITTEN -----------")
        # check if directory exists
        if not os.path.exists(input_params.calc_dirs[i]):
            os.mkdir(input_params.calc_dirs[i])
        # read calc. files
        namef = input_params.pert_dirs[i] + "/summary"
        f = open(namef, 'r')
        dirlist = f.readlines()
        f.close()
        # run over all calculations
        for named in dirlist:
            # check named exists
            out_dir = input_params.calc_dirs[i]
            if not os.path.exists(out_dir + "/" + named[:-1]):
                os.mkdir(out_dir + "/" + named[:-1])
            # copy files in new directory
            files = os.listdir(input_params.copy_files_dir)
            for fil in files:
                copyfile(input_params.copy_files_dir + "/" + fil, out_dir + "/" + named[:-1] + "/" + fil)
            namef = input_params.pert_dirs[i] + "/POSCAR-" + named[:-1]
            copyfile(namef, out_dir + "/" + named[:-1] + "/POSCAR")