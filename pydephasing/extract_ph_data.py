# This routine extract the phonon data information
# and return it for processing
import numpy as np
import h5py
#
def extract_ph_data(input_params):
    # input_params -> input data structure
    # q pt. index
    # open file
    with h5py.File(input_params.h5_eigen_file, 'r') as f:
        # dict keys
        eig_key = list(f.keys())[0]
        # get the eigenvectors
        # Eigenvectors is a numpy array of three dimension.
        # The first index runs through q-points.
        # In the second and third indices, eigenvectors obtained
        # using numpy.linalg.eigh are stored.
        # The third index corresponds to the eigenvalue's index.
        # The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        eigenv = list(f[eig_key])
        # get frequencies
        f_key = list(f.keys())[1]
        freq = list(f[f_key])
        # n. q pts.
        nq = len(freq)
        # q mesh
        m_key = list(f.keys())[2]
        mesh = list(f[m_key])
        # q pts.
        qpts_key = list(f.keys())[3]
        qpts = list(f[qpts_key])
        # q pts. weight
        wq_key = list(f.keys())[4]
        wq = list(f[wq_key])
        wq = np.array(wq, dtype=float)
        r = sum(wq)
        wq[:] = wq[:] / r
    # return data
    return eigenv, freq, nq, qpts, wq, mesh