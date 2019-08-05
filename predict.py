import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.append("../")

from components.descriporConstruction.dataProcess import getProteinGrids
from components.descriporConstruction.gridConstruction import atomGridPositionToCoor
from components.descriporConstruction.fileOperation import readProtein


# get protein grids and block sampling by a step of 16
def prepareDataForPredict(config, mol2_file_name, pdbqt_file_name, feature_path, pdb, load_flag=False,
                          display_flag=False):
    # protein grids construction
    # print(feature_path)
    if load_flag:
        protein_channel1 = np.load(os.path.join(feature_path, pdb, "ligsite.npy"))
        protein_channel2 = np.load(os.path.join(feature_path, pdb, "hbond.npy"))
        protein_channel3 = np.load(os.path.join(feature_path, pdb, "vdw.npy"))
        protein_channel4 = np.load(os.path.join(feature_path, pdb, "coulomb.npy"))
        protein_grid = np.zeros((protein_channel1.shape[0], protein_channel1.shape[1], protein_channel1.shape[2], 4))
        protein_grid[:, :, :, 0] = protein_channel1
        protein_grid[:, :, :, 1] = protein_channel2
        protein_grid[:, :, :, 2] = protein_channel3
        protein_grid[:, :, :, 3] = protein_channel4
    else:
        protein_grid = getProteinGrids(config=config, mol2_file_name=mol2_file_name, pdbqt_file_name=pdbqt_file_name,
                                       feature_path=None, pdb_name=pdb, buffer_size=8, resolution=1, train_flag=False,
                                       display_flag=display_flag)

    protein_grid = (2.0 * (np.arctan(protein_grid)) / np.pi)
    # print(mol2_file_name)
    # print(pdbqt_file_name)
    protein = readProtein(config, mol2_file_name, pdbqt_file_name, pdb)
    # for sampling
    step_para = 4
    temp_coor_list = []  # real cooridates for sampling blocks
    temp_x_list = []  # sampling blocks
    for i in range(0, protein_grid.shape[0] - 16, step_para):
        for j in range(0, protein_grid.shape[1] - 16, step_para):
            for k in range(0, protein_grid.shape[2] - 16, step_para):
                temp_sample_block = protein_grid[i:i + 16, j:j + 16, k:k + 16, :]
                temp_x_list.append(temp_sample_block)
                temp_coor_list.append(
                    atomGridPositionToCoor([i + 8, j + 8, k + 8], protein.Mol2MinCoorNp, buffer_size=8,
                                           resolution=1))
    return temp_x_list, temp_coor_list


# DBSCAN for coordinates
def DBSCANCluster(probs, coors):
    # print(probs)
    # print(coors)
    result_list_prob = []
    result_list_center = []
    step_para = 4.0

    # DBSCAN parameters
    eps, min_samples = (step_para + 1.0, 7)
    # clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto', leaf_size=30).fit(
        np.array(coors))
    # process with each clusters
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # no effective clusters, use the class with -1 label
    if n_clusters_ == 0:
        cluster_coors = np.array(coors)[labels == -1]
        cluster_probs = np.array(probs)[labels == -1]
        temp_centers = np.mean(cluster_coors, axis=0)
        # print(cluster_center)
        temp_prob = np.mean(cluster_probs)
        result_list_prob.append(temp_prob)
        result_list_center.append(temp_centers)

    # for each clusters
    for i in range(n_clusters_):
        cluster_coor = np.array(coors)[labels == i]
        cluster_prob = np.array(probs)[labels == i]
        temp_centers = np.mean(cluster_coor, axis=0)  # center
        # print(cluster_center)
        temp_probs = np.mean(cluster_prob)  # score
        result_list_prob.append(temp_probs)
        result_list_center.append(temp_centers)

    # rank the results according to scores
    result_list_top_index = np.argsort(np.array(result_list_prob))
    result_list_prob = np.array(result_list_prob)[result_list_top_index][::-1]
    result_list_center = np.array(result_list_center)[result_list_top_index][::-1]

    return result_list_prob, result_list_center


# predict
def predict(config, mol2_file_name, pdbqt_file_name, model_path, feature_path, result_save_file, pdb, load_flag=False,
            display_flag=False, top_pocket=3):
    # get blocks and their coors
    blocks_x, coors_x = prepareDataForPredict(config, mol2_file_name, pdbqt_file_name, feature_path, pdb,
                                              load_flag=load_flag, display_flag=display_flag)
    # model predict
    from keras.models import load_model
    model = load_model(model_path)
    label_predict_x = model.predict(np.array(blocks_x))

    # filter probs>=0.5
    positive_coor_list = []
    positive_prob_list = []
    for l in range(label_predict_x.shape[0]):
        if label_predict_x[l] >= 0.5:
            positive_coor_list.append(coors_x[l])
            positive_prob_list.append(label_predict_x[l])

    # DBSCAN clustering
    result_list_prob = []  # result for score
    result_list_center = []  # result for center
    # no blocks >0.5, choose the block with max pro. will not happen according to test.
    if label_predict_x.shape[0] == 0:
        result_list_top_index = np.argmax(label_predict_x)
        result_list_prob.append(label_predict_x[result_list_top_index])
        result_list_center.append(blocks_x[result_list_top_index])
    else:  # DBSCAN
        result_list_prob, result_list_center = DBSCANCluster(positive_prob_list, positive_coor_list)

    # for print and save
    result_str = "result for " + pdb + "\r\n"
    # top3 prediction output
    pocket_num = result_list_prob.shape[0]
    for i in range(min(top_pocket, pocket_num)):
        pro = result_list_prob[i]
        center = result_list_center[i]
        result_str += "score: " + str(pro) + " predicted center: " + str(center) + "\r\n"
    result_str += "number of all predicted pockets: " + str(pocket_num) + "\r\n"
    print(result_str)
    open(result_save_file, "w").writelines(result_str)


# top3 prediction
def predictTop3(config, mol2_file_name, pdbqt_file_name, model_path, result_save_file, pdb_name):
    predict(config, mol2_file_name, pdbqt_file_name, model_path, config.feature_test_path, result_save_file, pdb_name,
            load_flag=False,
            display_flag=False, top_pocket=3)


# top5 prediction
def predictTop5(config, mol2_file_name, pdbqt_file_name, model_path, result_save_file, pdb_name):
    predict(config, mol2_file_name, pdbqt_file_name, model_path, config.feature_test_path, result_save_file, pdb_name,
            load_flag=False,
            display_flag=False, top_pocket=5)


def printUsage():
    print("python predict [protein.mol2] [protein.pdbqt] [3/5] [save_file]")


def main(config, argv):
    try:
        # print(argv)
        if len(argv) != 4:
            printUsage()
            exit(0)
        protein_mol2_file = argv[0]
        protein_pdbqt_file = argv[1]
        top_pocket = int(argv[2])
        save_file = argv[3]
        # print(protein_mol2_file,protein_pdbqt_file,top_pocket,save_file)
        if ".mol2" not in protein_mol2_file and not os.path.exists(protein_mol2_file):
            print("no mol2 file found.")
            printUsage()
            exit(0)
        if ".pdbqt" not in protein_pdbqt_file and not os.path.exists(protein_pdbqt_file):
            print("no pdbqt file found.")
            printUsage()
            exit(0)
        if top_pocket != 3 and top_pocket != 5:
            printUsage()
            exit(0)
        model_file = os.path.join(config.model_save_path, "model.h5")
        if top_pocket == 3:
            predictTop3(config, protein_mol2_file, protein_pdbqt_file, model_file, save_file, "protein")
        else:
            predictTop5(config, protein_mol2_file, protein_pdbqt_file, model_file, save_file, "protein")
    except:
        import traceback
        traceback.print_exc()
        print("An unexpected error occur.")


if __name__ == '__main__':
    # gpu setting
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=gpu_config))
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    #
    from configure import Config

    config = Config()
    # # print(sys.argv)

    # should prepare mol and pdbqt files for the predicted protein (open babel or autodock script)
    # example: python predict.py example/1c6y_1/protein.mol2 example/1c6y_1/protein.pdbqt 3 ./results_example.txt
    main(config, sys.argv[1:])
