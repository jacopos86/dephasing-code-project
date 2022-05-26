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
        # perturbed calculations directory
        self.pert_dirs = []
        # perturbed calculations outcars directory
        self.outcar_dirs = []
        # grad info file
        self.grad_info = ''
        # atoms displacements
        self.atoms_displ = []
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
            if l[0] == "working_dir":
                self.out_dir = l[2] + '/'
            if l[0] == "unpert_data_dir":
                self.unpert_dir = l[2]
            elif l[0] == "dir_first_order_displ":
                for i in range(2, len(l)):
                    self.pert_dirs.append(l[i])
            elif l[0] == "dir_displ_outcars":
                for i in range(2, len(l)):
                    self.outcar_dirs.append(l[i])
            elif l[0] == "grad_info_file":
                self.grad_info = l[2]
            elif l[0] == "dx":
                for i in range(2, len(l)):
                    dx.append(float(l[i]))
            elif l[0] == "dy":
                for i in range(2, len(l)):
                    dy.append(float(l[i]))
            elif l[0] == "dz":
                for i in range(2, len(l)):
                    dz.append(float(l[i]))
        # set atom displ. list
        for i in range(len(dx)):
            self.atoms_displ.append(np.array([dx[i], dy[i], dz[i]]))
        f.close()
    #