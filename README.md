# ProteinDescriptor
a protein descriptor for site prediction

A four-channel grid protein descriptor is constructed based on ligsite, L-J potential, and Coulomb force, which can be used for protein binding site prediction.By using 16 X 16 X 16 sampling to classify and cluster the blocks, the binding sites of the proteins are finally determined. The detailed processing is shown in the figure blow.

<div align=center><img width="700" height="400" src="https://github.com/595693085/ProteinDescriptor/blob/master/docs/figure1.jpg"/></div>


## requirement
python == 3.6.x  
keras == 2.2.4  
tenforflow-gpu == 1.13.1  
numpy == 1.16.4  
tqdm == 4.13.1 
sklearn == 0.20.1


## training
### dataset （scPDB-2017）
To ensure training and testing, the data set should look like this.  
>ProteinDescriptor
>>data
>>>data_raw
>>>>train
>>>>>1a4i_1
>>>>>>protein.mol2, protein.pdbqt, site.mol2


>>data
>>>data_raw
>>>>valid
>>>>>1a4l_2
>>>>>>protein.mol2, protein.pdbqt, site.mol2


>>data  
>>>data_raw
>>>>test
>>>>>1aiq_2
>>>>>>protein.mol2, protein.pdbqt

The training set and validation set must contain site.mol2 file for every protein (for label determination). At the same time, in order to obtain protein feature, the mol2 and pdbqt files of each protein should be included in the dataset, and the pdbqt files can be obtained through openbabel or autodock script.  

Because the data of scPDB is too large, only a small part is provided for operation.

### usage 
```bash
python train.py
```
The features (grid descriptor) of the training set，validation set and testing set are stored in data/feature directory. In order to be more efficient, the features of all proteins are calculted and stored before model training and testing.

## prediction
A trained model is saved in the /model/model.h5. For a new protein in prediction, the mol2 and pdbqt files shuold be prepared for feature calculation. 
### usage
```bash
python predict.py protein.mol2 protein.pdbqt 3/5(pocket number) result_file
```
example:
```bash
python predict.py example/1c6y_1/protein.mol2 example/1c6y_1/protein.pdbqt 3 ./results_example/result.txt
```

