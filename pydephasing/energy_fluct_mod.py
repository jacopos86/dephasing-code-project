#
#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
from pydephasing.phys_constants import min_freq
from phys_constants import hbar
#
class spin_level_fluctuations_ofq:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params, homo=True, nconf=0):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph = 3*nat
        if homo:
            # if atom resolved
            if atom_res:
                self.deltaEq_atr = np.zeros((self.nt2,nat,input_params.ntmp))
            # if phonon resolved
            if ph_res:
                self.deltaEq_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
            # full auto correlation
            self.deltaEq_oft = np.zeros((self.nt,input_params.ntmp))
        else:
            # HFI calculation
            if atom_res:
                self.deltaEq_hfi_atr = np.zeros((self.nt2,nat))
            # phonon res.
            if ph_res:
                self.deltaEq_hfi_phm = np.zeros((self.nt2,self.nph))
            # full fluct.
            self.deltaEq_hfi_oft = np.zeros((self.nt,nconf))
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
                        cos_wt2 = np.zeros(self.nt2)
                        cos_wt2[:] = np.cos(2*np.pi*wuq[im]*input_params.time2[:]-2*np.pi*np.dot(qv,R0))
                    cos_wt = np.zeros(self.nt)
                    cos_wt[:] = np.cos(2*np.pi*wuq[im]*input_params.time[:]-2*np.pi*np.dot(qv,R0))
                    # run over temperature list
                    for iT in range(input_params.ntmp):
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] = self.deltaEq_phm[:,im,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                        if atom_res:
                            self.deltaEq_atr[:,ia,iT] = self.deltaEq_atr[:,ia,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt2[:]
                        self.deltaEq_oft[:,iT] = self.deltaEq_oft[:,iT] + ph_ampl.u_ql_ja[im,jax,iT] * F_ax[jax] * cos_wt[:]
    # compute HFI energy fluctuations
    def compute_deltaEq_hfi_oft(self, atom_res, ph_res, qv, wuq, nat, spin_config, input_params, index_to_ia_map, atoms_dict, ph_ampl, icl):
        #
        # compute deltaEq_hfi(t)
        #
        # applied force (THz/Ang)
        F_ax = spin_config.Fax
        # iterate over ph. modes
        for im in range(self.nph):
            if wuq[im] > min_freq:
                for jax in range(3*nat):
                    ia = index_to_ia_map[jax] - 1
                    # direct atom coord.
                    R0 = atoms_dict[ia]['coordinates']
                    # temporal part
                    # cos(wq t - q R0)
                    if ph_res or atom_res:
                        cos_wt2 = np.zeros(self.nt2)
                        cos_wt2[:] = np.cos(2.*np.pi*wuq[im]*input_params.time2[:]-2.*np.pi*np.dot(qv,R0))
                    cos_wt = np.zeros(self.nt)
                    cos_wt[:] = np.cos(2.*np.pi*wuq[im]*input_params.time[:]-2.*np.pi*np.dot(qv,R0))
                    # ph res / at res
                    if ph_res:
                        self.deltaEq_hfi_phm[:,im] = self.deltaEq_hfi_phm[:,im] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt2[:]
                    if atom_res:
                        self.deltaEq_hfi_atr[:,ia] = self.deltaEq_hfi_atr[:,ia] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt2[:]
                    self.deltaEq_hfi_oft[:,icl] = self.deltaEq_hfi_oft[:,icl] + ph_ampl.u_ql_ja[im,jax,0] * F_ax[jax] * cos_wt[:]
                    # THz units
# compute final deltaE(t)
class spin_level_fluctuations:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params, homo=True, nconf=0):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        if homo:
            # atom resolved calc.
            if atom_res:
                self.deltaE_atr = np.zeros((self.nt2,nat,input_params.ntmp))
            # ph. resolved calc.
            if ph_res:
                self.deltaE_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
            self.deltaE_oft = np.zeros((self.nt,input_params.ntmp))
        else:
            # HFI calculation
            if atom_res:
                self.deltaE_hfi_atr = np.zeros((self.nt2,nat))
            if ph_res:
                self.deltaE_hfi_phm = np.zeros((self.nt2,self.nph))
            # time fluct.
            self.deltaE_hfi_oft = np.zeros((self.nt,nconf))
    # compute energy fluctuations
    def compute_deltaE_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        #
        if atom_res:
            self.deltaE_atr[:,:,:] = self.deltaE_atr[:,:,:] + wq * deltaEq.deltaEq_atr[:,:,:] * hbar
        if ph_res:
            self.deltaE_phm[:,:,:] = self.deltaE_phm[:,:,:] + wq * deltaEq.deltaEq_phm[:,:,:] * hbar
        # energy fluctuations
        self.deltaE_oft[:,:] = self.deltaE_oft[:,:] + wq * deltaEq.deltaEq_oft[:,:] * hbar
        # eV units
    # HFI energy fluct.
    def compute_deltaE_hfi_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        if atom_res:
            self.deltaE_hfi_atr[:,:] = self.deltaE_hfi_atr[:,:] + wq * deltaEq.deltaEq_hfi_atr[:,:] * hbar
        if ph_res:
            self.deltaE_hfi_phm[:,:] = self.deltaE_hfi_phm[:,:] + wq * deltaEq.deltaEq_hfi_phm[:,:] * hbar
        # energy fluct.
        self.deltaE_hfi_oft[:,:] = self.deltaE_hfi_oft[:,:] + wq * deltaEq.deltaEq_hfi_oft[:,:] * hbar
        # eV units
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
                    # time array
                    ft = np.zeros(self.nt)
                    ft[:] = (cos_wt[:] * np.exp(-1j*2.*np.pi*np.dot(qv, R0))).real
                    if atom_res or ph_res:
                        ft2 = np.zeros(self.nt2)
                        ft2[:] = (cos_wt2[:] * np.exp(-1j*2.*np.pi*np.dot(qv, R0))).real
                    # run over the temperature list
                    for iT in range(input_params.ntmp):
                        # compute energy fluct.
                        if atom_res:
                            self.deltaEq_atr[:,ia,iT] = self.deltaEq_atr[:,ia,iT] + ph_ampl.u_ql_ja[im,iax,iT] * ft2[:] * Fax[iax]
                        if ph_res:
                            self.deltaEq_phm[:,im,iT] = self.deltaEq_phm[:,im,iT] + ph_ampl.u_ql_ja[im,iax,iT] * ft2[:] * Fax[iax]
                        self.deltaEq_oft[:,iT] = self.deltaEq_oft[:,iT] + ph_ampl.u_ql_ja[im,iax,iT] * ft[:] * Fax[iax]
                    # run over atom index ja/iy
                    for jax in range(3*nat):
                        # atom index ja
                        ja = index_to_ia_map[jax] - 1
                        # direct atom coordinate
                        R0p = atoms_dict[ja]['coordinates']
                        # time array
                        gt = np.zeros(self.nt)
                        gt[:] = (cos_wt[:] * np.exp(-1j*2.*np.pi*np.dot(qv, R0p))).real
                        if atom_res or ph_res:
                            gt2 = np.zeros(self.nt2)
                            gt2[:] = (cos_wt2[:] * np.exp(-1j*2.*np.pi*np.dot(qv, R0p))).real
                        # run over T list
                        for iT in range(input_params.ntmp):
                            # compute 2nd order fluct.
                            if atom_res:
                                self.deltaEq_atr[:,ia,iT] = self.deltaEq_atr[:,ia,iT] + 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * ft2[:] * Fc_axby[iax,jax] * gt2[:] * ph_ampl.u_ql_ja[im,jax,iT]
                            if ph_res:
                                self.deltaEq_phm[:,im,iT] = self.deltaEq_phm[:,im,iT] + 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * ft2[:] * Fc_axby[iax,jax] * gt2[:] * ph_ampl.u_ql_ja[im,jax,iT]
                            self.deltaEq_oft[:,iT] = self.deltaEq_oft[:,iT] + 0.5 * ph_ampl.u_ql_ja[im,iax,iT] * ft[:] * Fc_axby[iax,jax] * gt[:] * ph_ampl.u_ql_ja[im,jax,iT]
# total energy fluct. class
class energy_level_fluctuations:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        # atom res. calc.
        if atom_res:
            self.deltaE_atr = np.zeros((self.nt2,nat,input_params.ntmp))
        # ph. resolved
        if ph_res:
            self.deltaE_phm = np.zeros((self.nt2,self.nph,input_params.ntmp))
        self.deltaE_oft = np.zeros((self.nt,input_params.ntmp))
    # compute final energy fluctuations
    def compute_deltaE_oft(self, atom_res, ph_res, wq, deltaEq):
        #
        # compute deltaE(t)
        #
        if atom_res:
            self.deltaE_atr[:,:,:] = self.deltaE_atr[:,:,:] + wq * deltaEq.deltaEq_atr[:,:,:]
        if ph_res:
            self.deltaE_phm[:,:,:] = self.deltaE_phm[:,:,:] + wq * deltaEq.deltaEq_phm[:,:,:]
        # energy fluct.
        self.deltaE_oft[:,:] = self.deltaE_oft[:,:] + wq * deltaEq.deltaEq_oft[:,:]
# static spin fluctuations class
class spin_level_static_fluctuations:
    # initialization
    def __init__(self, spin_config):
        self.nt = len(spin_config.time)
        # deltaE(t)
        self.deltaE_oft = np.zeros(self.nt)
    # compute energy fluctuations
    def compute_deltaE_oft(self, spins_config):
        # run over nuclear spins
        for isp in range(spins_config.nsp):
            # spin fluctuations
            dIt = spins_config.nuclear_spins[isp]['dIt']
            # forces (THz)
            F = spins_config.nuclear_spins[isp]['F']
            # run over time steps
            for i in range(self.nt):
                self.deltaE_oft[i] = self.deltaE_oft[i] + np.dot(F[:], dIt[:,i])
        # eV units
        self.deltaE_oft[:] = self.deltaE_oft[:] * hbar