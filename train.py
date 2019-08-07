import os
import random
import sys
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

sys.path.append(".")

from components.descriporConstruction.dataProcess import dataGenerator, getProteinGrids
from components.descriporConstruction.modelConstruction import modelConstruct
from configure import Config
from predict import predict


# in order for acceleration, write all training proteins features
def writeFeatures(pdb_list, raw_data_path, feature_path, test_flag=False):
    # random.shuffle(pdb_list)
    # print("writeFeatures",pdb_list)
    for pdb in pdb_list:
        print("processing ", pdb)
        protein_np_save_file_1 = os.path.join(feature_path, pdb, "ligsite.npy")
        protein_np_save_file_2 = os.path.join(feature_path, pdb, "hbond.npy")
        protein_np_save_file_3 = os.path.join(feature_path, pdb, "vdw.npy")
        protein_np_save_file_4 = os.path.join(feature_path, pdb, "coulomb.npy")
        protein_center_save_file = os.path.join(feature_path, pdb, "center_position.npy")
        # print(protein_np_save_file_1)
        if os.path.exists(protein_np_save_file_1) and os.path.exists(protein_np_save_file_2) and os.path.exists(
                protein_np_save_file_3) and os.path.exists(protein_np_save_file_4):
            # for test, there is no need to save site position,
            # but for training and validation,site position is used for label determination
            if not test_flag and os.path.exists(protein_center_save_file) or test_flag:
                print(pdb, "already exists.")
                continue

        mol2_file_name = os.path.join(raw_data_path, pdb, "protein.mol2")
        pdbqt_file_name = os.path.join(raw_data_path, pdb, "protein.pdbqt")
        site_mol2_file_name = os.path.join(raw_data_path, pdb, "site.mol2")

        # for test, there is no need to parse site.mol2
        if test_flag:
            site_mol2_file_name = ""
        getProteinGrids(config, mol2_file_name, pdbqt_file_name, feature_path, pdb_name=pdb,
                        site_mol2_file_name=site_mol2_file_name,
                        buffer_size=8, resolution=1, train_flag=True, display_flag=True)
        print(pdb, "save over.")


# for training
def train(train_pdb_list, valid_pdb_list, train_feature_dir, valid_feature_dir, model_path, box_size=16):
    params = {'dim': (box_size, box_size, box_size),
              'batch_size': config.batch_size,  # no effect
              'n_classes': 2,  # no effecct
              'n_channels': 4,
              'shuffle': True}
    random.shuffle(train_pdb_list)
    random.shuffle(valid_pdb_list)
    print("training processing ...")
    print("train proteins:", len(train_pdb_list), "validation proteins:", len(valid_pdb_list))
    train_data_generator = dataGenerator(train_feature_dir, train_pdb_list, **params)
    valid_data_generator = dataGenerator(valid_feature_dir, valid_pdb_list, **params)
    model = modelConstruct(config.sample_box_length)
    if os.path.exists(os.path.join(model_path, "model.h5")):
        model = load_model(os.path.join(model_path, "model.h5"))
    epoch_models = os.path.join(model_path,
                                "saved-model-{epoch:02d}-{acc:.2f}-{loss:.2f}-{val_acc:.2f}-{val_loss:.2f}.h5")
    checkpoint = ModelCheckpoint(epoch_models, monitor='val_acc', verbose=1, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]
    model.fit_generator(train_data_generator, epochs=30, verbose=1, use_multiprocessing=True, workers=16,
                        callbacks=callbacks_list, validation_data=valid_data_generator)
    model.save(os.path.join(model_path, "model.h5"))
    print("training over.")


def test(config, test_pdb_list, test_feature_dir, model_path, result_save_dir):
    for pdb in test_pdb_list:
        mol2_file_name = os.path.join(config.raw_data_test, pdb, "protein.mol2")
        pdbqt_file_name = os.path.join(config.raw_data_test, pdb, "protein.pdbqt")
        model_file = os.path.join(model_path, "model.h5")
        result_save_file = os.path.join(result_save_dir, pdb + "_prediction.txt")
        predict(config, mol2_file_name, pdbqt_file_name, model_file, test_feature_dir, result_save_file, pdb,
                load_flag=True, display_flag=True, top_pocket=3)


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    from multiprocessing import Pool, cpu_count
    import multiprocessing

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=gpu_config))
    config = Config()

    # files
    train_data_dir = config.raw_data_train  # raw train data
    valid_data_dir = config.raw_data_valid  # raw validation data
    test_data_dir = config.raw_data_test  # raw test data
    train_feature_dir = config.feature_train_path  # train feature save path
    valid_feature_dir = config.feature_valid_path  # validation feature save path
    test_feature_dir = config.feature_test_path  # test feature save path
    model_path = config.model_save_path  # model save path
    test_result_save_dir = config.result_test_path  # test results save path

    # pdb list
    train_pdb_list = os.listdir(train_data_dir)
    valid_pdb_list = os.listdir(valid_data_dir)
    test_pdb_list = os.listdir(test_data_dir)
    random.shuffle(train_pdb_list)
    random.shuffle(valid_pdb_list)
    random.shuffle(test_pdb_list)

    # write features in parallel
    # print(len(train_pdb_list))
    # print(len(valid_pdb_list))
    # print(len(test_pdb_list))
    # pool = Pool(processes=cpu_count())
    # writeFeatures(train_pdb_list, train_data_dir, train_feature_dir)
    # writeFeatures(valid_pdb_list, valid_data_dir, valid_feature_dir)
    # writeFeatures(test_pdb_list, test_data_dir, test_feature_dir, True)
    with multiprocessing.Pool(processes=cpu_count()) as pool:
        for i in range(0, len(train_pdb_list)):
            temp_protein = [train_pdb_list[i]]
            pool.apply_async(writeFeatures, (temp_protein, train_data_dir, train_feature_dir))

        for i in range(0, len(valid_pdb_list)):
            temp_protein = [valid_pdb_list[i]]
            pool.apply_async(writeFeatures, (temp_protein, valid_data_dir, valid_feature_dir))

        for i in range(0, len(test_pdb_list)):
            temp_protein = [test_pdb_list[i]]
            pool.apply_async(writeFeatures, (temp_protein, test_data_dir, test_feature_dir, True))

        pool.close()
        pool.join()

    # for training
    if config.train:
        train(train_pdb_list, valid_pdb_list, train_feature_dir, valid_feature_dir, model_path, box_size=16)

    # for test
    if config.test:
        test(config, test_pdb_list, test_feature_dir, model_path, test_result_save_dir)
