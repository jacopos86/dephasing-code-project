#
#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
import sys
import yaml
import h5py
import time
#
class autocorrelation_function:
    # initialization
    def __init__(self, atom_res, ph_res, nat, input_params):
        self.nt = input_params.nt
        self.ntmp = input_params.ntmp
        self.nph = 3*nat
        # if atom resolved
        if atom_res == True:
            self.acf = np.zeros((nat, self.nt))
        # if phonon resolved
        elif ph_res == True:
            self.acf = np.zeros((self.nph, self.nt))
        # full auto correlation
        else:
            self.acf = np.zeros((self.nt, self.ntmp))
        # extract atom coordinates
        self.extract_atoms_coords(input_params, nat)
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
    # compute auto correlation function
    def compute_autocorrel_func(self, input_params):
        # extract ph modes
        #
        # open HDF5 and read the
        # eigenvectors
        #
        start = time.time()
        with h5py.File(input_params.h5_eigen_file, 'r') as f:
            # list all groups
            print("Keys: %s" % f.keys())
            eig_key = list(f.keys())[0]
            # get the eigenvectors
            # Eigenvectors is a numpy array of three dimension.
            # The first index runs through q-points.
            # In the second and third indices, eigenvectors obtained
            # using numpy.linalg.eigh are stored.
            # The third index corresponds to the eigenvalue's index.
            # The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
            eigenv = list(f[eig_key])
            print("shape eigenv. matrix=", eigenv[0].shape)
            # extract phonon frequencies
            f_key = list(f.keys())[1]
            freq = list(f[f_key])
            nq = len(freq)
            print("n. inequiv. q pts= ", nq)