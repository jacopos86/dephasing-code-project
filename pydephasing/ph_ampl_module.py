#
#  This module defines the atomic displacement
#  needed to compute the ACF
#
import numpy as np
from pydephasing.utility_functions import bose_occup
from pydephasing.phys_constants import hbar, THz_to_ev, mp
#
class PhononAmplitude:
    # initialization
    def __init__(self, nat, input_params):
        # set atomic ph. amplitude
        self.nph = 3*nat
        self.u_ql_ja = np.zeros((self.nph,3*nat,input_params.ntmp))
    def compute_ph_amplq(self, euq, wuq, nat, input_params, index_to_ia_map, atoms_dict):
        #
        # run over ph modes
        #
        for im in range(self.nph):
            for jax in range(3*nat):
                # atom index in list
                ia = index_to_ia_map[jax] - 1
                m_ia = atoms_dict[ia]['mass']
                m_ia = m_ia * mp
                #
                # eV ps^2 / ang^2 units
                #
                e_uq_ja = np.sqrt((euq[jax,im] * np.conjugate(euq[jax,im])).real)
                # T displ.
                for iT in range(input_params.ntmp):
                    T = input_params.temperatures[iT]
                    # K units
                    E = wuq[im] * THz_to_ev    # eV
                    nph = bose_occup(E, T)
                    #
                    # phonon amplitude
                    #
                    A = np.sqrt(hbar*(1+2*nph)/(2.*np.pi*wuq[im]*nat))
                    self.u_ql_ja[im,jax,iT] = A * e_uq_ja / np.sqrt(m_ia)
                    # ang units