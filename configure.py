import os
import sys


# config for experiment
class Config():
    def __init__(self):
        # for model
        self.train = True
        self.test = True
        # self.predict = True
        # self.feature_write = True
        self.epoch = 30
        self.batch_size = 128

        # for descriptor
        self.sample_box_length = 16
        self.grid_resolution = 1

        # for files and data
        self.main_path = os.path.dirname(sys.modules['__main__'].__file__)
        self.raw_data_train = os.path.join(self.main_path, "data", "data_raw", "train")  # sc-pdb version 2018
        self.raw_data_valid = os.path.join(self.main_path, "data", "data_raw", "valid")
        self.raw_data_test = os.path.join(self.main_path, "data", "data_raw", "test")
        self.result_test_path = os.path.join(self.main_path, "results")
        #
        self.model_save_path = os.path.join(self.main_path, "model")
        self.feature_path = os.path.join(self.main_path, "data", "feature")
        self.feature_train_path = os.path.join(self.feature_path, "train")
        self.feature_valid_path = os.path.join(self.feature_path, "valid")
        self.feature_test_path = os.path.join(self.feature_path, "test")

        # for protein and atom
        self.buffer_size = 8
        self.probe_list_CH3 = ["C", "H", "H", "H"]
        self.probe_list_OH = ["O", "H"]
        self.probe_charge = 1.0
        self.cut_off_radius = 8
        self.pi = 3.141592654
        self.columb_sigma = 8.854187817


# return the vdw radius of the element
def elementVDWRadius(element_name, vdw_potential_flag=0):
    # if element_name=='H' or element_name=='h':
    #    return 1.100
    # elif element_name=='C' or element_name=='c':
    #    return 1.548
    # elif element_name=='N' or element_name=='n':
    #    return 1.400
    # elif element_name=='O' or element_name=='o':
    #    return 1.348
    # elif element_name=='P' or element_name=='p':
    #    return 1.880
    # elif element_name=='S' or element_name=='s':
    #    return 1.808
    # elif element_name=='Ca' or element_name=='ca':
    #    return 1.948
    # elif element_name=='Fe' or element_name=='fe':
    #    return 1.948
    # elif element_name=='Cd' or element_name=='cd':
    #    return 1.748
    # elif element_name=='I' or element_name=='i':
    #    return 1.748
    # else:
    #    return 1.500

    # if element_name == 'C.3' or element_name == 'C.2' or element_name == 'C.1' or element_name == 'C.ar' or element_name == 'C.cat':
    #     return 1.548
    # elif element_name == 'N.4' or element_name == 'N.3' or element_name == 'N.2' or element_name == 'N.1' or element_name == 'N.ar' or element_name == 'N.pl3' or element_name == 'N.am':
    #     return 1.400
    # elif element_name == 'O.3' or element_name == 'O.2' or element_name == 'O.co2' or element_name == 'O.spc' or element_name == 'O.t3p':
    #     return 1.348
    # elif element_name == 'S.3' or element_name == 'S.2' or element_name == 'S.o' or element_name == 'S.o2':
    #     return 1.808
    # elif element_name == 'P.3':
    #     return 1.880
    # elif element_name == 'H' or element_name == 'H.spc' or element_name == 'H.t3p':
    #     return 1.100
    # elif element_name == 'I':
    #     return 1.748
    # elif element_name == 'Ca':
    #     return 1.948
    # elif element_name == 'Fe':
    #     return 1.948
    # else:
    #     return 1.800
    element_autodock_H = ["H", "HD", "HS"]
    element_mol2_H = ["H", "H.spc", "H.t3p"]
    element_autodock_C = ["C", "A"]
    element_mol2_C = ["C.3", "C.2", "C.1", "C.ar", "C.cat"]
    element_autodock_N = ["N", "NA", "NS"]
    element_mol2_N = ["N.4", "N.3", "N.2", "N.1", "N.ar", "N.pl3", "N.am"]
    element_autodock_O = ["OA", "OS"]
    element_mol2_O = ["O.3", "O.2", "O.co2", "O.spc", "O.t3p"]
    element_autodock_S = ["SA", "S"]
    element_mol2_S = ["S.3", "S.2", "S.o2"]
    element_autodock_P = ["P"]
    element_mol2_P = ["P.3"]
    element_I = ["I"]
    element_Ca = ["Ca", "CA"]
    element_Fe = ["Fe", "FE"]
    element_F = ["F"]
    element_Mg = ["Mg", "MG"]
    element_Cl = ["Cl", "CL"]
    element_Mn = ["Mn", "MN"]
    element_Zn = ["Zn", "ZN"]
    element_Br = ["Br", "BR"]

    if (element_name in element_autodock_H) or (element_autodock_H in element_mol2_H):
        return float(2.0 / 2)
    elif (element_name in element_autodock_C) or (element_name in element_mol2_C):
        return float(4.0 / 2)
    elif (element_name in element_autodock_N) or (element_name in element_mol2_N):
        return float(3.50 / 2)
    elif (element_name in element_autodock_O) or (element_name in element_mol2_O):
        return float(3.20 / 2)
    elif (element_name in element_autodock_S) or (element_name in element_mol2_S):
        return float(4.0 / 2)
    elif (element_name in element_autodock_P) or (element_name in element_mol2_P):
        return float(4.20 / 2)
    elif element_name in element_I:
        return float(4.72 / 2)
    elif element_name in element_Ca:
        return float(1.98 / 2)
    elif element_name in element_Fe:
        return float(1.30 / 2)
    elif element_name in element_F:
        return float(3.09 / 2)
    elif element_name in element_Mg:
        return float(1.30 / 2)
    elif element_name in element_Cl:
        return float(4.09 / 2)
    elif element_name in element_Mn:
        return float(1.30 / 2)
    elif element_name in element_Zn:
        return float(1.48 / 2)
    elif element_name in element_Br:
        return float(4.33 / 2)
    elif vdw_potential_flag == 1:
        return 0
    else:
        return 1.800


# return vdw well depth http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
def elementVDWWellDepth(element_name):
    element_autodock_H = ["H", "HD", "HS"]
    element_mol2_H = ["H", "H.spc", "H.t3p"]
    element_autodock_C = ["C", "A"]
    element_mol2_C = ["C.3", "C.2", "C.1", "C.ar", "C.cat"]
    element_autodock_N = ["N", "NA", "NS"]
    element_mol2_N = ["N.4", "N.3", "N.2", "N.1", "N.ar", "N.pl3", "N.am"]
    element_autodock_O = ["OA", "OS"]
    element_mol2_O = ["O.3", "O.2", "O.co2", "O.spc", "O.t3p"]
    element_autodock_S = ["SA", "S"]
    element_mol2_S = ["S.3", "S.2", "S.o2"]
    element_autodock_P = ["P"]
    element_mol2_P = ["P.3"]
    element_I = ["I"]
    element_Ca = ["Ca", "CA"]
    element_Fe = ["Fe", "FE"]
    element_F = ["F"]
    element_Mg = ["Mg", "MG"]
    element_Cl = ["Cl", "CL"]
    element_Mn = ["Mn", "MN"]
    element_Zn = ["Zn", "ZN"]
    element_Br = ["Br", "BR"]

    if (element_name in element_autodock_H) or (element_autodock_H in element_mol2_H):
        return 0.020
    elif (element_name in element_autodock_C) or (element_name in element_mol2_C):
        return 0.150
    elif (element_name in element_autodock_N) or (element_name in element_mol2_N):
        return 0.160
    elif (element_name in element_autodock_O) or (element_name in element_mol2_O):
        return 0.200
    elif (element_name in element_autodock_S) or (element_name in element_mol2_S):
        return 0.200
    elif (element_name in element_autodock_P) or (element_name in element_mol2_P):
        return 0.200
    elif element_name in element_I:
        return 0.550
    elif element_name in element_Ca:
        return 0.550
    elif element_name in element_Fe:
        return 0.010
    elif element_name in element_F:
        return 0.080
    elif element_name in element_Mg:
        return 0.875
    elif element_name in element_Cl:
        return 0.276
    elif element_name in element_Mn:
        return 0.875
    elif element_name in element_Zn:
        return 0.550
    elif element_name in element_Br:
        return 0.389
    else:
        return 0


# return the hbond well depth of the donor and recptor
def elenmentHbondWellDepth(element_name):
    if element_name == "NA" or element_name == "NS" or element_name == "OA" or element_name == "OS" or element_name == "N":
        return 5
    elif element_name == "SA":
        return 1
    elif element_name == "HD" or element_name == "HS":
        return 1
    else:
        return 0


# return the hbond radius of the donor and recptor
def elenmentHbondRadius(element_name):
    if element_name == "NA" or element_name == "NS" or element_name == "OA" or element_name == "OS":
        return 1.9
    elif element_name == "SA":
        return 2.5
    elif element_name == "HD" or element_name == "HS":
        return 1
    else:
        return 0


if __name__ == '__main__':
    config = Config()
    print(config.main_path)
