# This is the main subroutine
# for the homogeneous exciton dephasing calculation
# it computes the energy auto-correlation function
# and it returns it for further processing
import numpy as np
from pydephasing.set_structs import UnpertStruct
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
    # compute GS config. forces
    #
    gradEGS = gradient_Eg(nat)
    # set GS forces
    gradEGS.set_forces(GS)
    #
    # set EXC config. forces
    #
    gradEXC = gradient_Eg(nat)
    # set EXC forces
    gradEXC.set_forces(EXC)
    # gradZPL -> gradEXC - gradEGS
    gradZPL = np.zeros(3*nat)
    gradZPL[:] = gradEXC.forces[:] - gradEGS.forces[:]
    # eV / Ang units
    #
    # set up auto-correlation functions
    #
    #Eg_acf = autocorrelation_function(at_resolved, ph_resolved, nat, input_params)
    #