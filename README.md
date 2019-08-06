# ProteinDescriptor
a protein descriptor for site prediction

A four-channel grid protein descriptor is constructed based on ligsite, L-J potential, and Coulomb force, which can be used for protein binding site prediction.By using 16 X 16 X 16 sampling to classify and cluster the blocks, the binding sites of the proteins are finally determined. The detailed processing is shown in the figure blow.

<div align=center><img width="700" height="400" src="https://github.com/595693085/ProteinDescriptor/blob/master/docs/figure1.jpg"/></div>


## requirement:
python == 3.6.x  
keras == 2.2.4  
tenforflow-gpu == 1.13.1  
numpy == 1.16.4  
tqdm == 4.13.1  


## training:
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

Because the data of scPDB is too large, only a small part is provided for operation.

### usage 
```bash
python train.py
```
The features (grid descriptor) of the training set，validation set and testing set are stored in data/feature directory. In order to be more efficient， the features of all proteins are calculted before model training and testing.

