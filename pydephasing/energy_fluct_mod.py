#
#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
#
class spin_level_fluctuations_ofq:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph = 3*nat
        # if atom resolved
        if atom_res:
            self.deltaEq_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # if phonon resolved
        if ph_res:
            self.deltaEq_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        # full auto correlation
        self.deltaEq_oft = np.zeros((self.nt,input_params.ntmp))
    # compute energy fluctuations
    def compute_deltaEq_oft(self, atom_res, ph_res, qv, wuq, nat, F_ax, input_params, index_to_ia_map, atoms_dict, ph_ampl):
        #
        # compute deltaEq_oft
        #
        # iterate over ph. modes
        for im in range(self.nph):
            if wuq[im] > min_freq:
                for jax in range(3*nat):
                    ia = index_to_ia_map[jax] - 1
                    # direct atom coordinate
                    R0 = atoms_dict[ia]['coordinates']
                    # temporal part
                    # cos(wq t - q R0)
                    if ph_res or atom_res:
                        cos_wt2 = np.zeros(input_params.nt2)
                        cos_wt2[:] = np.cos(2*np.pi*wuq[im]*input_params.time2[:]-2*np.pi*np.dot(qv,R0))
                    cos_wt = np.zeros(input_params.nt)
                    cos_wt[:] = np.cos(2*np.pi*wuq[im]*input_params.time[:]-2*np.pi*np.dot(qv,R0))
                    # run over temperature list
                    for iT in range(input_params.ntmp):
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] = self.deltaEq_phm[:,im,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                        if atom_res:
                            self.deltaEq_atr[:,ia,iT] = self.deltaEq_atr[:,ia,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                        self.deltaEq_oft[:,iT] = self.deltaEq_oft[:,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt[:]
# compute final deltaE(t)
class spin_level_fluctuations:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        # atom resolved calc.
        if atom_res:
            self.deltaE_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # ph. resolved calc.
        if ph_res:
            self.deltaE_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        self.deltaE_oft = np.zeros((self.nt,input_params.ntmp))
    # compute energy fluctuations
    def compute_deltaE_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        #
        if atom_res:
            self.deltaE_atr[:,:,:] = self.deltaE_atr[:,:,:] + wq * deltaEq.deltaEq_atr[:,:,:]
        if ph_res:
            self.deltaE_phm[:,:,:] = self.deltaE_phm[:,:,:] + wq * deltaEq.deltaEq_phm[:,:,:]
        # energy fluctuations
        self.deltaE_oft[:,:] = self.deltaE_oft[:,:] + wq * deltaEq.deltaEq_oft[:,:]
# energy levels fluctuations
class energy_level_fluctuations_ofq:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2 = input_params.nt2
        self.nph = 3*nat
        # atom res. calc.
        if atom_res:
            self.deltaEq_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # ph. resolved
        if ph_res:
            self.deltaEq_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        self.deltaEq_oft = np.zeros((self.nt,input_params.ntmp))
    # compute energy fluctuations
    def compute_deltaEq_oft(self, atom_res, ph_res, qv, wuq, nat, Fax, Fc_axby, input_params, index_to_ia_map, atoms_dict, ph_ampl):
        #
        # compute deltaE(t)
        #
        # iterate over ph. modes
        for im in range(self.nph):
            if wuq[im] > min_freq:
                # temporal part
                if atom_res or ph_res:
                    cos_wt2 = np.zeros(self.nt2, dtype=np.complex128)
                    cos_wt2[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time2[:])
                cos_wt = np.zeros(self.nt, dtype=np.complex128)
                cos_wt[:] = np.exp(1j*2.*np.pi*wuq[im]*input_params.time[:])
                # run over atom index ia/ix
                for iax in range(3*nat):
                    # atom index ia
                    ia = index_to_ia_map[iax] - 1
                    # direct atom coordinates
                    R0 = atoms_dict[ia]['coordinates']
                    
