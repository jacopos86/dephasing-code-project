#
# This module sets up the methods
# needed to compute the energy fluctuations
# auto correlation function
#
import numpy as np
import statsmodels.api as sm
from pydephasing.phys_constants import hbar
#
class autocorrel_func:
    # initialization
    def __init__(self, nat, input_params):
        # array parameters
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph = 3*nat
        self.nlags = input_params.nlags
        # acf dictionaries
        self.acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        self.acf_phm_dict = {'time' : 0, 'acf' : {}, 'ft' : {}}
        self.acf_at_dict = {'time' : 0, 'acf' : {}, 'ft' : {}}
        # T2_inv - tau_c - D2 arrays
        self.T2i_ofT = np.zeros(2)
        self.T2i_phm_ofT = np.zeros((2,self.nph))
        self.T2i_atr_ofT = np.zeros((2,nat))
        #
        self.tauc_ofT = np.zeros(2)
        self.tauc_phm_ofT = np.zeros((2,self.nph))
        self.tauc_atr_ofT = np.zeros((2,nat))
        #
        self.D2_ofT = 0.
        self.D2_phm_ofT = np.zeros(self.nph)
        self.D2_atr_ofT = np.zeros(nat)
        # time arrays
        self.time = input_params.time
        self.time2 = input_params.time2
    # compute auto-correlation
    # function
    def compute_acf(self, deltaE_oft):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt    # eV^2
        # extract T2 time
        tau_c, T2_inv, ft = extract_T2(self.time[:nct], Ct, D2)
        # store data in array
        self.T2i_ofT[:] = 1./T2_inv[:] * 1.E-12     # sec units
        self.tauc_ofT[:] = tau_c[:]                 # ps units
        self.D2_ofT = D2                            # eV^2
        # dictionaries
        self.acf_dict['acf'] = Ct
        self.acf_dict['ft'] = ft
        self.acf_dict['time'] = self.time[:nct]
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_phm(self, deltaEphm_oft):
        # run over ph. modes
        for im in range(self.nph):
            Ct = sm.tsa.acf(deltaEphm_oft[:,im], nlags=self.nlags, fft=True)
            nct = len(Ct)
            #
            # compute C(t) = acf(t) * <DeltaE^2>_T
            #
            D2 = sum(deltaEphm_oft[:,im] * deltaEphm_oft[:,im]) / self.nt2    # eV^2
            # extract T2 time
            tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
            # store data in array
            if tau_c is not None and T2_inv is not None:
                self.T2i_phm_ofT[:,im] = 1./T2_inv[:] * 1.E-12     # sec units
                self.tauc_phm_ofT[:,im] = tau_c[:]                 # ps units
                self.D2_phm_ofT[im] = D2                           # eV^2
            # dictionaries
            if tau_c is not None and T2_inv is not None:
                self.acf_dict['acf'][im] = Ct
                self.acf_dict['ft'][im] = ft
        self.acf_dict['time'] = self.time2[:nct]
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_atr(self, deltaEatr_oft, nat):
        # run over atoms
        for ia in range(nat):
            Ct = sm.tsa.acf(deltaEatr_oft[:,ia], nlags=self.nlags, fft=True)
            nct = len(Ct)
            #
            # compute C(t) = acf(t) * <DeltaE^2>_T
            #
            D2 = sum(deltaEatr_oft[:,ia] * deltaEatr_oft[:,ia]) / self.nt2    # eV^2
            # extract T2 time
            tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
            # store data in array
            if tau_c is not None and T2_inv is not None:
                self.T2i_atr_ofT[:,ia] = 1./T2_inv[:] * 1.E-12     # sec units
                self.tauc_atr_ofT[:,ia] = tau_c[:]                 # ps units
                self.D2_atr_ofT[ia] = D2                           # eV^2
            # dictionaries
            if tau_c is not None and T2_inv is not None:
                self.acf_dict['acf'][ia] = Ct
                self.acf_dict['ft'][ia] = ft
        self.acf_dict['time'] = self.time2[:nct]
#
# define auto correlation function
# for energy excitations
#
class autocorrel_func_en_exc:
    # initialization
    def __init__(self, nat, input_params):
        # array parameters
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        self.nlags = input_params.nlags
        # acf dictionaries
        self.acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        self.acf_phm_dict = {'time' : 0, 'acf' : {}, 'ft' : {}}
        self.acf_atr_dict = {'time' : 0, 'acf' : {}, 'ft' : {}}
        # lw - tau_c - D2 arrays
        self.lw_ofT = np.zeros(2)
        self.lw_phm_ofT = np.zeros((2,self.nph))
        self.lw_atr_ofT = np.zeros((2,nat))
        #
        self.tauc_ofT = np.zeros(2)
        self.tauc_phm_ofT = np.zeros((2,self.nph))
        self.tauc_atr_ofT = np.zeros((2,nat))
        #
        self.D2_ofT = 0.
        self.D2_phm_ofT = np.zeros((2,self.nph))
        self.D2_atr_ofT = np.zeros((2,nat))
    # compute auto-correlation
    # function
    def compute_acf(self, deltaE_oft):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt    # eV^2
        # extract T2 time
        tau_c, T2_inv, ft = extract_T2(self.time[:nct], Ct, D2)
        # store data in array
        self.tauc_ofT[:] = tau_c[:]               # ps units
        self.D2_ofT = D2                          # eV^2
        self.lw_ofT[:] = 2.*np.pi*hbar*T2_inv[:]  # eV
        # dictionaries
        self.acf_dict['acf'] = Ct
        self.acf_dict['ft'] = ft
        self.acf_dict['time'] = self.time[:nct]
    # compute phonon resolved auto-correlation
    # function
    def compute_acf_phm(self, deltaEphm_oft):
        # compute auto correlation
        for im in range(self.nph):
            Ct = sm.tsa.acf(deltaEphm_oft[:,im], nlags=self.nlags, fft=True)
            nct = len(Ct)
            #
            # compute C(t) = acf(t) * <DeltaE^2>_T
            #
            D2 = sum(deltaEphm_oft[:,im] * deltaEphm_oft[:,im]) / self.nt2    # eV^2
            # extract T2 time
            tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
            # store data in array
            if tau_c is not None and T2_inv is not None:
                self.tauc_phm_ofT[:,im] = tau_c[:]               # ps units
                self.D2_phm_ofT[im] = D2                         # eV^2
                self.lw_phm_ofT[:,im] = 2.*np.pi*hbar*T2_inv[:]  # eV
            # dictionaries
            if tau_c is not None and T2_inv is not None:
                self.acf_phm_dict['acf'][im] = Ct
                self.acf_phm_dict['ft'][im] = ft
        self.acf_phm_dict['time'] = self.time2[:nct]
    # compute phonon resolved auto-correlation
    # function
    def compute_acf_atr(self, nat, deltaEatr_oft):
        # compute auto correlation
        for ia in range(nat):
            Ct = sm.tsa.acf(deltaEatr_oft[:,ia], nlags=self.nlags, fft=True)
            nct = len(Ct)
            #
            # compute C(t) = acf(t) * <DeltaE^2>_T
            #
            D2 = sum(deltaEatr_oft[:,ia] * deltaEatr_oft[:,ia]) / self.nt2    # eV^2
            # extract T2 time
            tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
            # store data in array
            if tau_c is not None and T2_inv is not None:
                self.tauc_atr_ofT[:,ia] = tau_c[:]               # ps units
                self.D2_atr_ofT[ia] = D2                         # eV^2
                self.lw_atr_ofT[:,ia] = 2.*np.pi*hbar*T2_inv[:]  # eV
            # dictionaries
            if tau_c is not None and T2_inv is not None:
                self.acf_atr_dict['acf'][ia] = Ct
                self.acf_atr_dict['ft'][ia] = ft
        self.acf_atr_dict['time'] = self.time2[:nct]