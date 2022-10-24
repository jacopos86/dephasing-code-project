#
# This module sets up the methods
# needed to compute the energy fluctuations
# auto correlation function
#
import numpy as np
import statsmodels.api as sm
import yaml
from pydephasing.phys_constants import hbar
from pydephasing.T2_calc import T2_eval
#
class autocorrel_func:
    # initialization
    def __init__(self, nat, input_params):
        # array parameters
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph = 3*nat
        self.nlags = input_params.nlags
        self.nlags2= input_params.nlags2
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
        # output dir
        self.write_dir = input_params.write_dir
    # compute auto-correlation
    # function
    def compute_acf(self, deltaE_oft, input_params):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_ofT[:] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_ofT[:] = tau_c[:]                 # ps units
            self.D2_ofT = D2                            # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data.yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_phm(self, deltaE_oft, input_params, im):
        # compute acf
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_phm_ofT[:,im] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_phm_ofT[:,im] = tau_c[:]                 # ps units
            self.D2_phm_ofT[im] = D2                           # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-phr" + str(im+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_atr(self, deltaE_oft, input_params, ia):
        # compute acf
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_atr_ofT[:,ia] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_atr_ofT[:,ia] = tau_c[:]                 # ps units
            self.D2_atr_ofT[ia] = D2                           # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-atr" + str(ia+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # print dictionary function
    def print_dict(self, Ct, ft, nct, namef):
        # acf dictionaries
        acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        acf_dict['acf'] = Ct
        acf_dict['ft'] = ft
        acf_dict['time'] = self.time[:nct]
        #
        # save dicts on file
        #
        with open(self.write_dir+namef, 'w') as out_file:
            yaml.dump(acf_dict, out_file)
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
        self.nlags2= input_params.nlags2
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
        # time arrays
        self.time = input_params.time
        self.time2 = input_params.time2
        # output dir
        self.write_dir = input_params.write_dir
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
        if tau_c is not None and T2_inv is not None:
            self.tauc_ofT[:] = tau_c[:]               # ps units
            self.D2_ofT = D2                          # eV^2
            self.lw_ofT[:] = 2.*np.pi*hbar*T2_inv[:]  # eV
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data.yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute phonon resolved auto-correlation
    # function
    def compute_acf_phm(self, deltaE_oft, im):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.tauc_phm_ofT[:,im] = tau_c[:]               # ps units
            self.D2_phm_ofT[im] = D2                         # eV^2
            self.lw_phm_ofT[:,im] = 2.*np.pi*hbar*T2_inv[:]  # eV
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-phr" + str(im+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute phonon resolved auto-correlation
    # function
    def compute_acf_atr(self, deltaE_oft, ia):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        tau_c, T2_inv, ft = extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.tauc_atr_ofT[:,ia] = tau_c[:]               # ps units
            self.D2_atr_ofT[ia] = D2                         # eV^2
            self.lw_atr_ofT[:,ia] = 2.*np.pi*hbar*T2_inv[:]  # eV
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-atr" + str(ia+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # print dictionary function
    def print_dict(self, Ct, ft, nct, namef):
        # acf dictionaries
        acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        acf_dict['acf'] = Ct
        acf_dict['ft'] = ft
        acf_dict['time'] = self.time[:nct]
        #
        # save dicts on file
        #
        with open(self.write_dir+namef, 'w') as out_file:
            yaml.dump(acf_dict, out_file)
#
# auto-correlation function class
# HF interaction calculations
#
class autocorrel_func_hfi_dyn:
    # initialization
    def __init__(self, nat, input_params):
        # array parameters
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.nph= 3*nat
        self.nlags = input_params.nlags
        self.nlags2= input_params.nlags2
        self.nconf = input_params.nconf
        # T2_inv - tau_c - D2 arrays
        self.T2i_ofT = np.zeros((2,self.nconf+1))
        self.T2i_phm_ofT = np.zeros((2,self.nph))
        self.T2i_atr_ofT = np.zeros((2,nat))
        #
        self.tauc_ofT = np.zeros((2,self.nconf+1))
        self.tauc_phm_ofT = np.zeros((2,self.nph))
        self.tauc_atr_ofT = np.zeros((2,nat))
        #
        self.D2_ofT = np.zeros(self.nconf+1)
        self.D2_phm_ofT = np.zeros(self.nph)
        self.D2_atr_ofT = np.zeros(nat)
        # time arrays
        self.time = input_params.time
        self.time2 = input_params.time2
        # output dir
        self.write_dir = input_params.write_dir
    # compute auto-correlation
    # function
    def compute_acf(self, deltaE_oft, input_params, ic):
        # compute auto correlation
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_ofT[:,ic] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_ofT[:,ic] = tau_c[:]                 # ps units
            self.D2_ofT[ic] = D2                           # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-c" + str(ic) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_phm(self, deltaE_oft, input_params, im):
        # im -> ph. mode
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_phm_ofT[:,im] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_phm_ofT[:,im] = tau_c[:]                 # ps units
            self.D2_phm_ofT[im] = D2                           # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-phr" + str(im+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # compute auto-correlation (ph. resolved)
    # function
    def compute_acf_atr(self, deltaE_oft, input_params, ia):
        # ia -> atom index
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags2, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt2    # eV^2
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2(self.time2[:nct], Ct, D2)
        # store data in array
        if tau_c is not None and T2_inv is not None:
            self.T2i_atr_ofT[:,ia] = 1./T2_inv[:] * 1.E-12     # sec units
            self.tauc_atr_ofT[:,ia] = tau_c[:]                 # ps units
            self.D2_atr_ofT[ia] = D2                           # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data-atr" + str(ia+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # print dictionary method
    def print_dict(self, Ct, ft, nct, namef):
        # acf dictionaries
        acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        acf_dict['acf'] = Ct
        acf_dict['ft'] = ft
        acf_dict['time'] = self.time[:nct]
        #
        # save dicts on file
        #
        with open(self.write_dir+namef, 'w') as out_file:
            yaml.dump(acf_dict, out_file)
#
# auto-correlation HFI static
# class
#
class autocorrel_func_hfi_stat:
    # initialization
    def __init__(self, input_params):
        # array variables
        self.nt = int(input_params.T_mus/input_params.dt_mus)
        self.nlags = input_params.nlags2
        # time (mu sec)
        self.time = input_params.time2
        # output dir
        self.write_dir = input_params.write_dir
        # arrays
        # T2i
        self.T2i_dat  = np.zeros(input_params.nconf+1)
        # tau_c
        self.tauc_dat = np.zeros(input_params.nconf+1)
        # D2
        self.D2_dat   = np.zeros(input_params.nconf+1)
    # compute auto correlation
    # function
    def compute_acf(self, deltaE_oft, input_params, ic):
        # acf
        Ct = sm.tsa.acf(deltaE_oft, nlags=self.nlags, fft=True)
        nct = len(Ct)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        # eV^2
        #
        D2 = sum(deltaE_oft[:] * deltaE_oft[:]) / self.nt
        # extract T2 time
        T2 = T2_eval(input_params)
        tau_c, T2_inv, ft = T2.extract_T2_star(self.time[:nct], Ct, D2)
        # tau_c (mu sec)
        # T2_inv (ps^-1)
        if tau_c is not None and T2_inv is not None:
            self.T2i_dat[ic+1] = 1./T2_inv * 1.E-12   # sec units
            self.tauc_dat[ic+1] = tau_c               # mu sec
        self.D2_dat[ic+1] = D2                        # eV^2
        # write data on file
        if tau_c is not None and T2_inv is not None:
            namef = "acf-data" + str(ic+1) + ".yml"
            self.print_dict(Ct, ft, nct, namef)
    # print dict. method
    def print_dict(self, Ct, ft, nct, namef):
        # acf dictionaries
        acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
        acf_dict['acf'] = Ct
        acf_dict['ft'] = ft
        acf_dict['time'] = self.time[:nct]
        #
        # save dict. on file
        #
        with open(self.write_dir+namef, 'w') as out_file:
            yaml.dump(acf_dict, out_file)