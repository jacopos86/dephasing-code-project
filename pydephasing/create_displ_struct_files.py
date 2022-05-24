# this subroutine creates the input files
# to be run in the VASP calculation
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
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