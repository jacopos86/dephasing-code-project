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
        # calculation directories
        self.calc_dirs = []
        # perturbed calculations outcars directory
        self.outcar_dirs = []
        # copy directory
        self.copy_files_dir = ''
        # grad info file
        self.grad_info = ''
        # atoms displacements
        self.atoms_displ = []
        # n. temperatures
        self.ntmp = 0
    # read data from file
    def read_data(self, input_file):
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
            elif l[0] == "grad_info_file":
                self.grad_info = l[2]
            elif l[0] == "yaml_pos_file":
                self.yaml_pos_file = l[2]
            elif l[0] == "hdf5_eigen_file":
                self.h5_eigen_file = l[2]
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
            elif l[0] == 'Tlist':
                Tlist = []
                for iT in range(2, len(l)):
                    Tlist.append(float(l[iT]))
            # read quantum states
            # to be normalized
            elif l[0] == "qs1":
                self.qs1 = np.array([complex(l[2]), complex(l[3]), complex(l[4])])
                nrm = np.sqrt(sum(self.qs1[:] * self.qs1[:].conjugate()))
                self.qs1[:] = self.qs1[:] / nrm
            elif l[0] == "qs2":
                self.qs2 = np.array([complex(l[2]), complex(l[3]), complex(l[4])])
                nrm = np.sqrt(sum(self.qs2[:] * self.qs2[:].conjugate()))
                self.qs2[:] = self.qs2[:] / nrm
        f.close()
        # set atom displ. list
        for i in range(len(self.pert_dirs)):
            file_name = self.pert_dirs[i] + "/displ"
            f = open(file_name, 'r')
            lines = f.readlines()
            l = lines[0].split()
            self.atoms_displ.append(np.array([float(l[0]), float(l[1]), float(l[2])]))
            f.close()
        # set time array length
        self.nt = int(self.T / self.dt)
        self.nt2= int(self.T2 / self.dt)
        # set time arrays
        self.time = np.linspace(0., self.T, self.nt)
        self.time2 = np.linspace(0., self.T2, self.nt2)
        # set temperature list
        self.ntmp = len(Tlist)
        self.temperatures = np.zeros(self.ntmp)
        for iT in range(self.ntmp):
            self.temperatures[iT] = Tlist[iT]
    #
    # prepare data_pre
    def read_data_pre(self, input_file):
        # open input file
        f = open(input_file, 'r')
        lines = f.readlines()
        dx = []
        dy = []
        dz = []
        for line in lines:
            l = line.split()
            if l[0] == "working_dir":
                self.out_dir = l[2] + '/'
            elif l[0] == "unpert_data_dir":
                self.unpert_dir = l[2]
            elif l[0] == "vasp_files_dir":
                self.copy_files_dir = l[2]
            elif l[0] == "dir_first_order_displ":
                for i in range(2, len(l)):
                    self.pert_dirs.append(l[i])
            elif l[0] == "dir_calc":
                for i in range(2, len(l)):
                    self.calc_dirs.append(l[i])
            elif l[0] == "dx":
                for i in range(2, len(l)):
                    dx.append(float(l[i]))
            elif l[0] == "dy":
                for i in range(2, len(l)):
                    dy.append(float(l[i]))
            elif l[0] == "dz":
                for i in range(2, len(l)):
                    dz.append(float(l[i]))
        # set atomic displacements
        for i in range(len(dx)):
            self.atoms_displ.append([dx[i], dy[i], dz[i]])

