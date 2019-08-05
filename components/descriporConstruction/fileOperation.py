import os
import numpy as np
import sys

sys.path.append("../")
from components.protein.proteinClass import Protein
from components.protein.atomClass import Mol2Atom, PdbqtAtom


def readProtein(config,mol2_file_name,pdbqt_file_name, pdb):
    # mol2_file_name = os.path.join(protein_file_path, pdb, "protein.mol2")
    # pdbqt_file_name = os.path.join(protein_file_path, pdb, "protein.pdbqt")
    protein = Protein()
    # if os.path.exists(mol2_file_name) and os.path.exists(pdbqt_file_name):
    # print("test1")
    # print(mol2_file_name)
    # print(pdbqt_file_name)
    if os.path.exists(mol2_file_name) and os.path.exists(pdbqt_file_name):
        # read file to protein class
        # print("exists")
        atom_coor_list = []
        atom_flag = 0
        file_mol2_object = open(mol2_file_name)
        for line1 in file_mol2_object.readlines():
            if line1.strip() == "":
                continue
            if "@<TRIPOS>ATOM" in line1:
                atom_flag = 1
                continue
            if "@<TRIPOS>" in line1 and atom_flag == 1:
                break
            if atom_flag == 1:
                atom_array = line1.strip().split()
                # print(atom_array)
                atom_id = int(atom_array[0])
                atom_name = atom_array[1]
                atom_x = float(atom_array[2])
                atom_y = float(atom_array[3])
                atom_z = float(atom_array[4])
                atom_type = atom_array[5]
                atom = Mol2Atom(config,atom_id=atom_id, atom_name=atom_name, x=atom_x, y=atom_y, z=atom_z, atom_type=atom_type)
                # print(line1)
                # print(atom_id,atom_name,atom_x,atom_y,atom_z,atom_type)
                # print(atom)
                protein.AddMol2Atom(atom)
                atom_coor_list.append([atom_x, atom_y, atom_z])
        file_pdbqt_object = open(pdbqt_file_name)
        line_count = 0
        for line2 in file_pdbqt_object.readlines():
            if "ATOM" not in line2:
                continue
            # print(line2)
            line_count += 1
            atom_id = int(line2[6:11])
            atom_x = float(line2[30:38])
            atom_y = float(line2[38:46])
            atom_z = float(line2[46:54])
            partial_charge = float(line2[70:76])
            atom_type = line2[77:79]
            # print(line2)
            # print(atom_x, atom_y, atom_z, partial_charge, atom_type)
            atom2 = PdbqtAtom(atom_id=atom_id, x=atom_x, y=atom_y, z=atom_z, atom_type=atom_type,
                              partial_charge=partial_charge)
            protein.AddPdbqtAtom(atom2)
        file_mol2_object.close()
        file_pdbqt_object.close()

        max_coor = np.max(np.array(atom_coor_list), axis=0)
        min_coor = np.min(np.array(atom_coor_list), axis=0)
        span = max_coor - min_coor
        protein.Mol2MinCoorNp = min_coor
        protein.Mol2MaxCoorNp = max_coor
        protein.Mol2CoorSpanNp = span

        return protein
    else:
        print("read", pdb, "error.")
