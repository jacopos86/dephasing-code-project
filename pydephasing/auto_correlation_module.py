#
# This module sets up the methods
# needed to compute the energy fluctuations
# auto correlation function
#
import numpy as np
#
class autocorrel_func:
    # initialization
    def __init__(self, nat, input_params):
        # array parameters
        self.nt = input_params.nt
        self.nt2= input_params.nt2
        self.ntmp = input_params.ntmp
        self.nph = 3*nat
        self.nlags = input_params.nlags
    # compute auto-correlation
    # function
    def compute_acf(self, nat, input_params, deltaE):
        # run over T list
        for iT in range(self.ntmp):
            #
            Ct = sm.tsa.acf(deltaE.deltaE_oft[:,iT], nlags=self.nlags, fft=True)
            nCt = len(Ct)
            #
            # compute C(t) = acf(t) * <DeltaE^2>_T
            #
            D2 = sum(deltaE.deltaE_oft[:,iT] * deltaE.deltaE_oft[:,iT]) / self.nt    # eV^2
            # extract T2 time