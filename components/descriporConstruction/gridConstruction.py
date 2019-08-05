import math
import numpy as np
import sys

sys.path.append("./")

from tqdm import tqdm
from configure import elementVDWRadius, elementVDWWellDepth, elenmentHbondWellDepth, elenmentHbondRadius


#######################################################################################
# to accelerate the program, this section need to be implemented in parallel in future
#######################################################################################

# create an empty grid
def createGrid(x_length, y_length, z_length, buffer_size=8, resolution=1):
    x_length = int(math.ceil(x_length / resolution)) + 2 * buffer_size
    y_length = int(math.ceil(y_length / resolution)) + 2 * buffer_size
    z_length = int(math.ceil(z_length / resolution)) + 2 * buffer_size
    return np.zeros((x_length, y_length, z_length))


# map the coordinate to the grid position
def atomCoorToGridPosition(atom_coors, min_coors, buffer_size=8, resolution=1):
    return [int((atom_coors[0] - min_coors[0]) / resolution) + buffer_size,
            int((atom_coors[1] - min_coors[1]) / resolution) + buffer_size,
            int((atom_coors[2] - min_coors[2]) / resolution) + buffer_size]


# map the grid position to the coordinate
def atomGridPositionToCoor(atom_position, min_coors, buffer_size=8, resolution=1):
    return [(float(atom_position[0] - buffer_size)) * resolution + min_coors[0],
            (float(atom_position[1] - buffer_size)) * resolution + min_coors[1],
            (float(atom_position[2] - buffer_size)) * resolution + min_coors[2]]


# fill the protein grid according to the vdw adius of atom
def createProteinFillGrid(atom_list, grid_length, min_coors, buffer_size=8, resolution=1, fill_value=1):
    fill_grid = createGrid(grid_length[0], grid_length[1], grid_length[2], buffer_size=buffer_size,
                           resolution=resolution)

    # print("fill_grid",fill_grid.shape,grid_length)
    for atom in atom_list:
        if atom == None:
            continue
        radius = elementVDWRadius(atom.atom_type)
        atom_coor = [atom.x, atom.y, atom.z]
        radius_resolution = radius / resolution
        atom_position = atomCoorToGridPosition(atom_coor, min_coors, buffer_size=buffer_size, resolution=resolution)
        atom_min = [int(math.floor(atom_position[0] - radius_resolution)),
                    int(math.floor(atom_position[1] - radius_resolution)),
                    int(math.floor(atom_position[2] - radius_resolution))]
        atom_max = [int(math.ceil(atom_position[0] + radius_resolution)),
                    int(math.ceil(atom_position[1] + radius_resolution)),
                    int(math.ceil(atom_position[2] + radius_resolution))]

        # a sphere around the atom
        radius_square = radius_resolution * radius_resolution
        for i in range(atom_min[2], atom_max[2] + 1):
            z_counter = atom_position[2] - i
            z_square = z_counter * z_counter
            if (z_square > radius_square):
                continue
            sqr1 = math.sqrt(radius_square - z_square)
            y_min = int(math.floor(atom_position[1] - sqr1))
            y_max = int(math.ceil(atom_position[1] + sqr1))
            if y_min < atom_min[1]:
                y_min = atom_min[1]
            if y_max > atom_max[1]:
                y_max = atom_max[1]
            for j in range(y_min, y_max + 1):
                y_counter = j - atom_position[1]
                y_square = y_counter * y_counter
                if (y_square > (radius_resolution * radius_resolution - z_square)):
                    continue
                sqr2 = math.sqrt(radius_resolution * radius_resolution - z_square - y_square)
                x_min = int(math.floor(atom_position[0] - sqr2))
                x_max = int(math.ceil(atom_position[0] + sqr2))
                if x_min < atom_min[0]:
                    x_min = atom_min[0]
                if x_max > atom_max[0]:
                    x_max = atom_max[0]
                for k in range(x_min, x_max + 1):
                    # for test
                    # if protein_id=="2eg5":
                    # print(atom_min,atom_max,[k,j,i],np.array(fill_grid).shape,atom_position)
                    # print(grid_length)
                    fill_grid[k][j][i] = fill_value
    return fill_grid


# create ligste grid
def createLigsiteGrid(protein_grid, grid_length):
    # print("FillLigsiteGrid Start...")

    # print("createLigsiteGrid",protein_grid.shape)
    ligsite_grid = createGrid(grid_length[0], grid_length[1], grid_length[2])
    x_length = ligsite_grid.shape[0]
    y_length = ligsite_grid.shape[1]
    z_length = ligsite_grid.shape[2]
    grid_length = max(x_length, y_length, z_length)

    # print("axis")
    # three coordinate axis
    result_found1 = []
    for l in range(0, grid_length):
        result_found1.append(0)
    length = [x_length, y_length, z_length]
    # index = []
    # for l in range(0, 3):
    #     index.append(0)
    index = [0, 0, 0]
    for i in range(0, 3):
        Dimension = i
        Dimension1 = (Dimension + 1) % 3
        Dimension2 = (Dimension + 2) % 3
        for j in range(0, length[Dimension2]):
            # print("length[Dimension2]",j)
            index[Dimension2] = j
            for k in range(0, length[Dimension1]):
                # print("length[Dimension1]",k)
                index[Dimension1] = k
                found1 = 0
                for m in range(0, length[Dimension]):
                    index[Dimension] = m
                    # print("x:%d,y:%d,z:%d",index[0],index[1],index[2])
                    # print("LigsiteGrid test1")
                    flag = protein_grid[index[0]][index[1]][index[2]]
                    # print("LigsiteGrid test2")
                    if (flag > 0):
                        found1 = 1
                    else:
                        result_found1[index[Dimension]] = found1
                found2 = 0
                for n in range(length[Dimension] - 1, -1, -1):
                    # print("length[Dimension]",n)
                    index[Dimension] = n
                    flag = protein_grid[index[0]][index[1]][index[2]]
                    if (flag > 0):
                        found2 = 1
                    elif (found2 > 0 and result_found1[index[Dimension]] > 0):
                        # print("ligsite_grid test1")
                        ligsite_grid[index[0]][index[1]][index[2]] += 1

    # print("dignal")
    # four dignals
    for l in range(0, grid_length):
        result_found1.append(0)
    step = [0, 0, 1]
    index = [0, 0, 0]
    for i in range(-1, 2, 2):
        # print(i)
        step[0] = i
        for j in range(-1, 2, 2):
            step[1] = j
            for k in range(0, 3):
                Dimension = k
                Dimension1 = (Dimension + 1) % 3
                Dimension2 = (Dimension + 2) % 3
                start = 0
                stop1 = length[Dimension1] - 1
                stop2 = length[Dimension2] - 1
                if (Dimension == 2):
                    start = 1
                    stop1 = length[Dimension1] - 2
                    stop2 = length[Dimension2] - 2
                for i1 in range(start, stop1 + 1):
                    for i2 in range(start, stop2 + 1):
                        i0 = length[Dimension] - 1
                        if (step[Dimension] > 0):
                            i0 = 0
                        index[Dimension] = i0
                        index[Dimension1] = i1
                        index[Dimension2] = i2
                        found1 = 0
                        while (index[2] < length[2]):
                            if (index[0] < 0):
                                break
                            if (index[0] >= length[0]):
                                break
                            if (index[1] < 0):
                                break
                            if (index[1] >= length[1]):
                                break
                            flag = protein_grid[index[0]][index[1]][index[2]]
                            if (flag > 0):
                                found1 = 1
                            else:
                                result_found1[index[2]] = found1
                            index[0] += step[0]
                            index[1] += step[1]
                            index[2] += step[2]
                        found2 = 0
                        index[0] -= step[0]
                        index[1] -= step[1]
                        index[2] -= step[2]
                        while (index[2] >= 0):
                            if (index[0] < 0):
                                break
                            if (index[0] >= length[0]):
                                break
                            if (index[1] < 0):
                                break
                            if (index[1] >= length[1]):
                                break
                            flag = protein_grid[index[0]][index[1]][index[2]]
                            if (flag > 0):
                                found2 = 1
                            elif ((found2 > 0) and (result_found1[index[2]] > 0)):
                                ligsite_grid[index[0]][index[1]][index[2]] += 1
                                # print("test2")
                            index[0] -= step[0]
                            index[1] -= step[1]
                            index[2] -= step[2]

    # print("LigsiteGridSize:",np.array(ligsite_grid).shape)
    return ligsite_grid


# create the hydrogen bond grid
def createHBondGrid(config, atom_list, grid_length, min_coors, buffer_size=8, resolution=1, display_flag=False):
    # print("CreateHBondGrid")
    # print(len(atom_list))
    probe_list = config.probe_list_OH  # probe
    cut_off_radius = config.cut_off_radius

    hbond_grid = createGrid(grid_length[0], grid_length[1], grid_length[2], buffer_size=buffer_size,
                            resolution=resolution)
    x_length = hbond_grid.shape[0]
    y_length = hbond_grid.shape[1]
    z_length = hbond_grid.shape[2]

    display_message = "createHBondGrid"  # for progress display
    for n in tqdm(range(0, len(atom_list)), ascii=True, desc=display_message, disable=display_flag):
        # for n in range(len(atom_list)):
        a = atom_list[n]
        if a == None:
            continue
        # print(a.atom_type)
        if a.atom_type not in ["OA", "OS", "HS", "HD"]:  # the atom types that can format hbond
            continue
        a_position = atomCoorToGridPosition([a.x, a.y, a.z], min_coors, buffer_size=buffer_size, resolution=resolution)
        # by setting the truncation radius, only the contribution of the atom to the surrounding cut-off radius is calculated.
        x_bound_low = max(round(a_position[0] - cut_off_radius), 0)
        x_bound_high = min(round(a_position[0] + cut_off_radius), x_length - 1)
        y_bound_low = max(round(a_position[1] - cut_off_radius), 0)
        y_bound_high = min(round(a_position[1] + cut_off_radius), y_length - 1)
        z_bound_low = max(round(a_position[2] - cut_off_radius), 0)
        z_bound_high = min(round(a_position[2] + cut_off_radius), z_length - 1)

        hbond_epsilon_a = elenmentHbondWellDepth(a.atom_type)  # atom hbond well depth
        hbond_sigma_a = elenmentHbondRadius(a.atom_type)  # atom hbond radius
        for i in range(x_bound_low, x_bound_high + 1):
            for j in range(y_bound_low, y_bound_high):
                # a little mistake,should be "for j in range(y_bound_low, y_bound_high + 1)",and
                # in order to ensure repeatability, no modification is made for the time being.
                for k in range(z_bound_low, z_bound_high):
                    hbond_atom1_coor = np.array([i, j, k])  # grid coordinate
                    hbond_radius = np.linalg.norm(hbond_atom1_coor - a_position) + 1e-3  # the distance r
                    hbond_energy = 0  #
                    if a.atom_type in ["OA", "OS"]:  # as accptor "NA" ,"NS" also should be included in future
                        # hbond_epsilon_p = 1
                        hbond_sigma_p = 1
                        hbond_sigma = hbond_sigma_p + hbond_sigma_a
                        hbond_epsilon = hbond_epsilon_a
                        hbond_A = hbond_epsilon * 5 * (hbond_sigma ** 12)
                        hbond_B = hbond_epsilon * 6 * (hbond_sigma ** 10)
                        # 12-10 L-J potential
                        hbond_energy = hbond_A / (hbond_radius ** 12) - hbond_B / (hbond_radius ** 10)
                    elif a.atom_type in ["HD", "HS"]:  # as donator
                        for p_l in probe_list:  # only the atom "O" can format hbond with HD and HS
                            if p_l == "H":
                                continue
                            # for acceleration, not call from config class
                            hbond_epsilon_p = 5  # for OA and OS
                            hbond_sigma_p = 1.9
                            hbond_sigma = hbond_sigma_p + hbond_sigma_a
                            hbond_epsilon = hbond_epsilon_p
                            hbond_A = hbond_epsilon * 5 * (hbond_sigma ** 12)
                            hbond_B = hbond_epsilon * 6 * (hbond_sigma ** 10)
                            # 12-10 L-J potential
                            hbond_energy = hbond_A / (hbond_radius ** 12) - hbond_B / (hbond_radius ** 10)

                    if abs(hbond_energy) > abs(hbond_grid[i][j][k]):
                        hbond_grid[i][j][k] = hbond_energy

    # print("HBondGridSize:", np.array(hbond_grid).shape)
    return hbond_grid


# create the vdw grid
def createVDWGrid(config, atom_list, grid_length, min_coors, buffer_size=8, resolution=1, display_flag=False):
    # print("CreateVDWGrid")
    probe_list = config.probe_list_CH3
    cut_off_radius = config.cut_off_radius
    vdw_grid = createGrid(grid_length[0], grid_length[1], grid_length[2], buffer_size=buffer_size,
                          resolution=resolution)
    x_length = vdw_grid.shape[0]
    y_length = vdw_grid.shape[1]
    z_length = vdw_grid.shape[2]

    display_message = "createVDWGrid"
    for n in tqdm(range(0, len(atom_list)), ascii=True, desc=display_message, disable=display_flag):
        # for n in range(len(atom_list)):
        a = atom_list[n]
        if a == None:
            continue
        a_position = np.array(
            atomCoorToGridPosition([a.x, a.y, a.z], min_coors, buffer_size=buffer_size, resolution=resolution))

        # only calculate the atoms in the cut off radius
        x_bound_low = max(round(a_position[0] - cut_off_radius), 0)
        x_bound_high = min(round(a_position[0] + cut_off_radius), x_length - 1)
        y_bound_low = max(round(a_position[1] - cut_off_radius), 0)
        y_bound_high = min(round(a_position[1] + cut_off_radius), y_length - 1)
        z_bound_low = max(round(a_position[2] - cut_off_radius), 0)
        z_bound_high = min(round(a_position[2] + cut_off_radius), z_length - 1)

        # get the info ot the atom
        vdw_epsilon_a = a.vdw_welldepth
        vdw_sigma_a = a.vdw_radius
        # vdw_atom2_coor = [a.x, a.y, a.z]
        vdw_epsilon_p = 0
        vdw_sigma_p = 0
        # search the cut off radius
        for i in range(x_bound_low, x_bound_high + 1):
            for j in range(y_bound_low, y_bound_high + 1):
                for k in range(z_bound_low, z_bound_high + 1):
                    # vdw_atom1_coor = GetPreviousCoordinate([i, j, k], min_coors, resolution)
                    vdw_atom1_coor = np.array([i, j, k])
                    vdw_radius = np.linalg.norm(vdw_atom1_coor - a_position) + 1e-3
                    # vdw_radius = math.sqrt(
                    #     (vdw_atom1_coor[0] - vdw_atom2_coor[0]) * (vdw_atom1_coor[0] - vdw_atom2_coor[0]) + (
                    #             vdw_atom1_coor[1] - vdw_atom2_coor[1]) * (vdw_atom1_coor[1] - vdw_atom2_coor[1]) + (
                    #             vdw_atom1_coor[2] - vdw_atom2_coor[2]) * (vdw_atom1_coor[2] - vdw_atom2_coor[2]))
                    for p_l in probe_list:
                        # C and H is fixed, so not call from config for acceleration
                        if p_l == "C":
                            vdw_epsilon_p = 0.150
                            vdw_sigma_p = float(4.0 / 2)
                        elif p_l == "H":
                            vdw_epsilon_p = 0.020
                            vdw_sigma_p = float(2.0 / 2)

                        vdw_epsilon = math.sqrt(vdw_epsilon_p * vdw_epsilon_a)
                        vdw_sigma = vdw_sigma_a + vdw_sigma_p
                        vdw_A = vdw_epsilon * (vdw_sigma ** 12)
                        vdw_B = vdw_epsilon * 2 * (vdw_sigma ** 6)
                        # 12-6 L-J potential
                        vdw_energy = vdw_A / (vdw_radius ** 12) - vdw_B / (vdw_radius ** 6)
                        vdw_grid[i][j][k] += vdw_energy

    # print(VDWGridSize:", np.array(vdw_grid).shape)
    return vdw_grid


# create the coulomb force grid
def createCoulombForceGrid(config, atom_list, grid_length, min_coors, buffer_size=8, resolution=1, display_flag=False):
    # print("Create Coulomb Grid")
    probe_charge = config.probe_charge
    cut_off_radius = config.cut_off_radius
    pi = config.pi
    columb_grid = createGrid(grid_length[0], grid_length[1], grid_length[2], buffer_size=buffer_size,
                             resolution=resolution)
    x_length = columb_grid.shape[0]
    y_length = columb_grid.shape[1]
    z_length = columb_grid.shape[2]

    display_message = "createCoilombGrid"
    for n in tqdm(range(0, len(atom_list)), ascii=True, desc=display_message, disable=display_flag):
        # for n in range(len(atom_list)):
        a = atom_list[n]
        if a == None:
            continue
        a_position = np.array(
            atomCoorToGridPosition([a.x, a.y, a.z], min_coors, buffer_size=buffer_size, resolution=resolution))
        x_bound_low = max(round(a_position[0] - cut_off_radius), 0)
        x_bound_high = min(round(a_position[0] + cut_off_radius), x_length - 1)
        y_bound_low = max(round(a_position[1] - cut_off_radius), 0)
        y_bound_high = min(round(a_position[1] + cut_off_radius), y_length - 1)
        z_bound_low = max(round(a_position[2] - cut_off_radius), 0)
        z_bound_high = min(round(a_position[2] + cut_off_radius), z_length - 1)
        # columb_atom2_coor = np.array([a.x, a.y, a.z])
        # columb_sigma0 = 8.854187817 * (10 ** -12)
        columb_sigma0 = config.columb_sigma
        q_1 = probe_charge
        q_2 = a.partial_charge
        for i in range(x_bound_low, x_bound_high + 1):
            for j in range(y_bound_low, y_bound_high):
                for k in range(z_bound_low, z_bound_high):
                    if a.partial_charge == None:
                        continue
                    if a.partial_charge == 0:
                        continue
                    columb_atom1_coor = np.array([i, j, k])
                    columb_radius = np.linalg.norm(columb_atom1_coor - a_position) + 1e-3
                    # coulomb force calculation
                    columb_energy = (q_1 * q_2) / (4.0 * pi * columb_sigma0 * columb_radius * columb_radius)
                    # print("columb",columb_energy)
                    columb_grid[i][j][k] += columb_energy
    # print(ColumbGridSize:", np.array(columb_grid).shape)
    return columb_grid
