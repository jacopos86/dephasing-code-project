#
#  This module defines the atomic displacement
#  needed to compute the ACF
#
import numpy as np
import sys
import yaml
from pydephasing.utility_functions import bose_occup
from pydephasing.phys_constants import hbar, THz_to_eV, mp
#
class atomic_displ:
    # initialization
    def __init__(self, nph, nat, ntmp):
        # set atomic displ.
        self.u_ql_ja = np.zeros((nph,3*nat,ntmp))
    # extract atoms dictionary method
    def extract_atoms_coords(self, input_params, nat):
        # extract atom positions from yaml file
        with open(input_params.yaml_pos_file) as f:
            data = yaml.full_load(f)
            key = list(data.keys())[6]
            self.atoms_dict = list(data[key]['points'])
        #
        if len(self.atoms_dict) != nat:
            print("Value error exception: wrong number of atoms...")
            sys.exit(1)
    def compute_atoms_displq(self, euq, wuq, nat, nph, ntmp, input_params, index_to_ia_map):
        #
        # run over ph modes
        #
        for im in range(nph):
            for jax in range(3*nat):
                # atom index in list
                ia = index_to_ia_map[jax] - 1
                m_ia = self.atoms_dict[ia]['mass']
                m_ia = m_ia * mp
                #
                # eV ps^2 / ang^2 units
                #
                e_uq_ja = np.sqrt((euq[jax,im] * np.conjugate(euq[jax,im])).real)
                # T displ.
                for iT in range(ntmp):
                    T = input_params.temperatures[iT]
                    # K units
                    E = wuq[im] * THz_to_eV    # eV
                    nph = bose_occup(E, T)
                    #
                    # phonon amplitude
                    #
                    A = np.sqrt(hbar*(1+2*nph)/wuq[im]/nat)
                    self.u_ql_ja[im,jax,iT] = A * e_uq_ja / np.sqrt(m_ia)
                    # ang units