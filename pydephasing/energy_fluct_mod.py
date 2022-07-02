#
#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
from pydephasing.atom_displ_module import atomic_displ
#
class energy_fluctuations:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.ntmp = input_params.ntmp
        self.nph = 3*nat
        # if atom resolved
        if atom_res:
            self.deltaEq_atr = np.zeros((self.nt2, nat, self.ntmp))
        # if phonon resolved
        if ph_res:
            self.deltaEq_phm = np.zeros((self.nt2, self.nph, self.ntmp))
        # full auto correlation
        self.deltaEq_oft = np.zeros((self.nt, self.ntmp))
        # first initialize atomic displacements
        # object
        self.at_displ = atomic_displ(self.nph, nat, self.ntmp)
        # extract atom coordinates
        self.at_displ.extract_atoms_coords(input_params, nat)
    # compute energy fluctuations
    def compute_deltaEq_oft(self, atom_res, ph_res, qv, euq, wuq, nat, F_ax, input_params, index_to_ia_map):
        #
        # compute deltaEq_oft
        #
        # compute atoms displ. first
        self.at_displ.compute_atoms_displq(euq, wuq, nat, self.nph, self.ntmp, input_params, index_to_ia_map)
        # iterate over ph. modes
        for im in range(self.nph):
            for jax in range(3*nat):
                ia = index_to_ia_map[jax] - 1
                # direct atom coordinate
                R0 = self.at_displ.atoms_dict[ia]['coordinates']
                # temporal part
                # cos(wq t - q R0)
                if ph_res or atom_res:
                    cos_wt2 = np.zeros(input_params.nt2)
                    cos_wt2[:] = np.cos(2*np.pi*wuq[im]*input_params.time2[:]-2*np.pi*np.dot(qv,R0))
                cos_wt = np.zeros(input_params.nt)
                cos_wt[:] = np.cos(2*np.pi*wuq[im]*input_params.time[:]-2*np.pi*np.dot(qv,R0))
                # run over temperature list
                for iT in range(self.ntmp):
                    if ph_res:
                        self.deltaEq_phm[:,im,iT] = self.deltaEq_phm[:,im,iT] + self.at_displ.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                    if atom_res:
                        self.deltaEq_atr[:,ia,iT] = self.deltaEq_atr[:,ia,iT] + self.at_displ.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                    self.deltaEq_oft[:,iT] = self.deltaEq_oft[:,iT] + self.at_displ.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt[:]
    # compute final deltaE(t)