# This is the main subroutine
# for the homogeneous exciton dephasing calculation
# it computes the energy auto-correlation function
# and it returns it for further processing
import numpy as np
from pydephasing.set_structs import UnpertStruct, DisplacedStructs
from pydephasing.gradient_interactions import gradient_Eg
#
def compute_hom_exc_autocorrel_func(input_params, at_resolved, ph_resolved):
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
    # set GS displ. structures
    #
    GS_displ = DisplacedStructs(input_params.gs_pert_dirs, input_params.outcar_gs_dir)
    GS_displ.atom_displ(input_params.atoms_displ[0])
    # set gradient GS energy
    gradEGS = gradient_Eg(nat)
    # compute GS energy gradient
    gradEGS.set_gradE(GS_displ, nat)
    #
    # set EXC displ. structures
    #
    EXC_displ = DisplacedStructs(input_params.exc_pert_dirs, input_params.outcar_exc_dir)
    EXC_displ.atom_displ(input_params.atoms_displ[0])
    # set gradient exciton energy
    gradEXC = gradient_Eg(nat)
    # compute EXC energy gradient
    gradEXC.set_gradE(EXC_displ, nat)
    # compute grad Eg
    gradEg = np.zeros(3*nat)
    gradEg[:] = gradEXC.gradE[:] - gradEGS.gradE[:]
    # eV / Ang units
    #
    # set up auto-correlation functions
    #
    Eg_acf = autocorrelation_function(at_resolved, ph_resolved, nat, input_params)
    #