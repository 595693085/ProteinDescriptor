import numpy as np
import os
import random
import sys
import keras

sys.path.append("./")
from components.descriporConstruction.gridConstruction import atomCoorToGridPosition, createProteinFillGrid, \
    createLigsiteGrid, createHBondGrid, createVDWGrid, createCoulombForceGrid
from components.descriporConstruction.fileOperation import readProtein


#######################################################################################
# to accelerate the program, this section need to be implemented in parallel in future
#######################################################################################

# compute site center for mol2 file
def computerMol2Center(site_file):
    read_flag = False
    atom_list = []
    for line in open(site_file, "r").readlines():
        if "@<TRIPOS>ATOM" in line:
            read_flag = True
            continue
        if "@<TRIPOS>" in line and read_flag == True:
            break
        if read_flag == True:
            arr = line.strip().split()
            atom_list.append([float(arr[2]), float(arr[3]), float(arr[4])])
    return np.mean(np.array(atom_list), axis=0)


# compute site center for pdb file
def computerPDBCenter(pdb_file):
    atom_list = []
    for line in open(pdb_file).readlines():
        if "ATOM" in line:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_list.append([x, y, z])
    return np.mean(np.array(atom_list), axis=0)


# get the protein all channel grids
def getProteinGrids(config, mol2_file_name, pdbqt_file_name, feature_path, pdb_name="protein", site_mol2_file_name=None,
                    buffer_size=8, resolution=1, train_flag=False,display_flag=False):
    try:
        site_center_np = np.array([0, 0, 0])
        if train_flag:
            site_center_np = computerMol2Center(site_mol2_file_name)
        protein = readProtein(config, mol2_file_name, pdbqt_file_name, pdb_name)
        site_center_grid = atomCoorToGridPosition(site_center_np, protein.Mol2MinCoorNp, buffer_size=buffer_size,resolution=resolution)

        protein_span = [int(protein.Mol2CoorSpanNp[0]), int(protein.Mol2CoorSpanNp[1]), int(protein.Mol2CoorSpanNp[2])]
        protein_atom_grid = createProteinFillGrid(protein.Mol2AtomsList, protein_span, protein.Mol2MinCoorNp,
                                                  buffer_size=buffer_size, resolution=resolution, fill_value=1)
        grid_channel_1 = createLigsiteGrid(protein_atom_grid, protein_span)
        grid_channel_2 = createHBondGrid(config, protein.PdbqtAtomsList, protein_span, protein.Mol2MinCoorNp,
                                         buffer_size=buffer_size, resolution=resolution,display_flag=display_flag)
        grid_channel_3 = createVDWGrid(config, protein.Mol2AtomsList, protein_span, protein.Mol2MinCoorNp, buffer_size=buffer_size,
                                       resolution=resolution,display_flag=display_flag)
        grid_channel_4 = createCoulombForceGrid(config, protein.PdbqtAtomsList, protein_span, protein.Mol2MinCoorNp,
                                                buffer_size=buffer_size, resolution=resolution,display_flag=display_flag)

        # print(grid_channel_1.shape)
        # print(grid_channel_2.shape)
        # print(grid_channel_3.shape)
        # print(grid_channel_4.shape)

        if train_flag:
            # save
            if not os.path.exists(os.path.join(feature_path, pdb_name)):
                os.makedirs(os.path.join(feature_path, pdb_name))
            np.save(os.path.join(feature_path, pdb_name, "ligsite.npy"), np.array(grid_channel_1))
            np.save(os.path.join(feature_path, pdb_name, "hbond.npy"), np.array(grid_channel_2))
            np.save(os.path.join(feature_path, pdb_name, "vdw.npy"), np.array(grid_channel_3))
            np.save(os.path.join(feature_path, pdb_name, "coulomb.npy"), np.array(grid_channel_4))
            np.save(os.path.join(feature_path, pdb_name, "center_position.npy"), np.array(site_center_grid))
        else:
            protein_grid = np.zeros((grid_channel_1.shape[0], grid_channel_1.shape[1], grid_channel_1.shape[2],4))
            protein_grid[:, :, :, 0] = grid_channel_1
            protein_grid[:, :, :, 1] = grid_channel_2
            protein_grid[:, :, :, 2] = grid_channel_3
            protein_grid[:, :, :, 3] = grid_channel_4
            return protein_grid
            # return np.array([grid_channel_1, grid_channel_2, grid_channel_3, grid_channel_4])
    except:
        import traceback
        traceback.print_exc()
        print(pdb_name, "read error.")


# def writePoteinFeatures(protein_list, protein_dataset_path, protein_np_path, sample_box_side_length, resolution,
#                         pdbbind_flag=0):
#     random.shuffle(protein_list)
#
#     for pdb in protein_list:
#         protein_np_save_file_1 = os.path.join(protein_np_path, pdb, "ligsite.npy")
#         protein_np_save_file_2 = os.path.join(protein_np_path, pdb, "hbond.npy")
#         protein_np_save_file_3 = os.path.join(protein_np_path, pdb, "vdw.npy")
#         protein_np_save_file_4 = os.path.join(protein_np_path, pdb, "coulomb.npy")
#         protein_center_save_file = os.path.join(protein_np_path, pdb, "center_position.npy")
#         if os.path.exists(protein_np_save_file_1) and os.path.exists(protein_np_save_file_2) and os.path.exists(
#                 protein_np_save_file_3) and os.path.exists(protein_np_save_file_4) and os.path.exists(
#             protein_center_save_file):
#             continue
#         getProteinGrids(protein_dataset_path, pdb, protein_np_path, sample_box_side_length, resolution, pdbbind_flag)
#         print(pdb, "save over.")


def loadAllSamles(feature_path):
    sample_x = []
    sample_y = []
    channel_1 = np.load(os.path.join(feature_path, "ligsite.npy"))
    channel_2 = np.load(os.path.join(feature_path, "hbond.npy"))
    channel_3 = np.load(os.path.join(feature_path, "vdw.npy"))
    channel_4 = np.load(os.path.join(feature_path, "coulomb.npy"))
    center = np.load(os.path.join(feature_path, "center_position.npy")).astype(int)
    protein_grid_channel = np.zeros((channel_1.shape[0], channel_1.shape[1], channel_1.shape[2], 4))
    protein_grid_channel[:, :, :, 0] = channel_1
    protein_grid_channel[:, :, :, 1] = channel_2
    protein_grid_channel[:, :, :, 2] = channel_3
    protein_grid_channel[:, :, :, 3] = channel_4

    # print(channel_1.shape,channel_2.shape,channel_3.shape,channel_4.shape)

    # a box center the site, for example, if site center is (center[0],center[1],center[2]) and the box is counted from top left
    # then the 16 sampling box positive area range from (-9,-6),produce 128 samples
    # -9 -8 -7 -6 -5 -4 -3 -2 -1 0  1  2  3  4  5  6  7  8
    # #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
    protein_range_max = [int(channel_1.shape[0]) - 1, int(channel_1.shape[1]) - 1, int(channel_1.shape[2]) - 1]
    positive_range_min = [int(max(0, center[0] - 9)), int(max(0, center[1] - 9)), int(max(0, center[2] - 9))]
    positive_range_max = [int(max(0, center[0] - 6)), int(max(0, center[1] - 6)), int(max(0, center[2] - 6))]

    # positive samples
    for i in range(positive_range_min[0], positive_range_max[0] + 1, 1):
        for j in range(positive_range_min[1], positive_range_max[1] + 1, 1):
            for k in range(positive_range_min[2], positive_range_max[2] + 1, 1):
                if (i + 15) > protein_range_max[0] or (j + 15) > protein_range_max[1] or (k + 15) > protein_range_max[
                    2]:
                    continue
                sample_x.append(protein_grid_channel[i:i + 16, j:j + 16, k:k + 16, :])
                sample_y.append(1.0)
    # print(np.array(sample_x).shape)
    # negative samples, random select
    while len(sample_x) < np.array(sample_x).shape[0]:
        i = random.randint(0, protein_range_max[0] - 1 - 15)
        j = random.randint(0, protein_range_max[1] - 1 - 15)
        k = random.randint(0, protein_range_max[2] - 1 - 15)
        # if in positive area
        if positive_range_min[0] <= i <= positive_range_max[0] and positive_range_min[1] <= j <= positive_range_max[
            1] and positive_range_min[2] <= k <= positive_range_max[2]:
            continue
        sample_x.append(protein_grid_channel[i:i + 16, j:j + 16, k:k + 16, :])
        sample_y.append(0.0)

    # for samples shuffle
    sample_index = [x for x in range(len(sample_x))]
    random.shuffle(sample_index)
    result_x = []
    result_y = []
    for i in sample_index:
        result_x.append(sample_x[i])
        result_y.append(sample_y[i])
    return np.array(result_x), np.array(result_y)


class dataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, feature_path, list_IDs, batch_size=128, dim=(16, 16, 16), n_channels=4,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.feature_path = feature_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        list_IDs_temp = self.list_IDs[index]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # print(X.shape, y.shape)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        X, y = loadAllSamles(os.path.join(self.feature_path, list_IDs_temp))
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return (2.0 * (np.arctan(X)) / np.pi), y
