
# Installation
1. Download the code from https://github.com/pujols/Zero-shot-learning-journal and insert it into the folder /comp 

2. Download the 
 [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
and insert it into the folder /tool


# Data

1. For al-cone-learning: Download the resnet features and class splits from http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip and http://datasets.d2.mpi-inf.mpg.de/xian/standard_split.zip. Unzip and put the standard_split folder in the data folder. Run data_transfer.m to generate .mat files ending with "resnet.mat".

2. For pucc-based-SVM: Download the data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna

# Execution
For both approaches, the main-file can be executed.
The code is not optimized due to its prototypical nature.