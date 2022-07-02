#
#   This module defines all the
#   parameters needed in input for the
#   calculations
#
import numpy as np
#
class data_input():
    # initialization
    def __init__(self):
        # working directory
        self.out_dir = ''
        # unperturbed directory calculation
        self.unpert_dir = ''
        # GS unperturbed directory
        self.unpert_dir_gs = ''
        # EXC unperturbed directory
        self.unpert_dir_exc = ''
        # perturbed calculations directory
        self.pert_dirs = []
        # perturbed calculations outcars directory
        self.outcar_dirs = []
        # perturbed GS calculation
        self.outcar_gs_dir = ''
        # perturbed GS files
        self.gs_pert_dirs = ''
        # perturbed EXC calculation
        self.outcar_exc_dir = ''
        # perturbed EXC files
        self.exc_pert_dirs = ''
        # grad info file
        self.grad_info = ''
        # atoms displacements
        self.atoms_displ = []
        # n. temperatures
        self.ntmp = 0
    # read data from file
    def read_data(self, input_file):
        dx = []
        dy = []
        dz = []
        # open input file
        f = open(input_file, 'r')
        lines = f.readlines()
        for line in lines:
            l = line.split()
            # directories
            if l[0] == "working_dir":
                self.out_dir = l[2] + '/'
            elif l[0] == "unpert_data_dir":
                self.unpert_dir = l[2]
            elif l[0] == "unpert_gs_dir":
                self.unpert_dir_gs = l[2]
            elif l[0] == "unpert_exc_dir":
                self.unpert_dir_exc = l[2]
            elif l[0] == "dir_first_order_displ":
                for i in range(2, len(l)):
                    self.pert_dirs.append(l[i])
            elif l[0] == "dir_displ_outcars":
                for i in range(2, len(l)):
                    self.outcar_dirs.append(l[i])
            elif l[0] == "dir_gs_outcars":
                self.outcar_gs_dir = l[2]
            elif l[0] == "dir_gs_displ":
                self.gs_pert_dirs = l[2]
            elif l[0] == "dir_exc_displ":
                self.exc_pert_dirs = l[2]
            elif l[0] == "dir_exc_outcars":
                self.outcar_exc_dir = l[2]
            elif l[0] == "grad_info_file":
                self.grad_info = l[2]
            elif l[0] == "yaml_pos_file":
                self.yaml_pos_file = l[2]
            elif l[0] == "hdf5_eigen_file":
                self.h5_eigen_file = l[2]
            # variables
            elif l[0] == "dx":
                for i in range(2, len(l)):
                    dx.append(float(l[i]))
            elif l[0] == "dy":
                for i in range(2, len(l)):
                    dy.append(float(l[i]))
            elif l[0] == "dz":
                for i in range(2, len(l)):
                    dz.append(float(l[i]))
            # time variables
            elif l[0] == 'T':
                self.T = float(l[2])
            elif l[0] == 'dt':
                self.dt = float(l[2])
            elif l[0] == 'T2':
                self.T2 = float(l[2])
            elif l[0] == 'dt2':
                self.dt2 = float(l[2])
            elif l[0] == 'nlags':
                self.nlags = int(l[2])
            # temperature
            elif l[0] == 'Tin':
                Tin = float(l[2])
            elif l[0] == 'Tfin':
                Tfin = float(l[2])
            elif l[0] == "dTmp":
                dTmp = float(l[2])
        f.close()
        # set atom displ. list
        for i in range(len(dx)):
            self.atoms_displ.append(np.array([dx[i], dy[i], dz[i]]))
        # set time array length
        self.nt = int(self.T / self.dt)
        self.nt2= int(self.T2 / self.dt2)
        # set time arrays
        self.time = np.linspace(0., self.T, self.nt)
        self.time2 = np.linspace(0., self.T2, self.nt2)
        # set temperature list
        self.ntmp = 1 + int((Tfin - Tin) / dTmp)
        self.temperatures = np.zeros(self.ntmp)
        self.temperatures[0] = Tin
        for it in range(1, self.ntmp):
            self.temperatures[it] = self.temperatures[it-1] + dTmp
    #