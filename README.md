
# CaMeRe

Source code and datasets for CaMeRe.

## Requirements

Python == 3.6

torch == 1.7.0+cu110

scipy == 1.5.4

torch-scatter == 2.0.5

torch-sparse == 0.6.8

torch-cluster == 1.5.8

scikit-learn == 0.24.2

pandas == 1.1.5

torch-geometric == 1.6.1

rdkit == 2019.09.3.0

pytor-scatterchtools == 0.0.2


## Usage

### 0. Create a Virtual Environment

Recommended to use conda:

conda create -n CaMeRe python=3.6 cudatoolkit

conda activate CaMeRe

Install Dependencies:

conda install pytorch==1.7.0 cudatoolkit=11.0 -c pytorch

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy

pip install torch-scatter==2.0.5 torch-sparse==0.6.8 torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html

pip install torch-geometric==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install -c conda-forge rdkit==2019.09.3.0

pip install pytorchtools==0.0.2


### 1. Create Sample Feature Dictionaries

If you would like to create sample feature dictionaries xxx.p files, please run feature_generation.py and feature_direction.py.

### 2. Meta-training 

If you would like to learn the weights and biases of the CaMeRe, please run Main_Train_patient_w_Configure.py for Clinical datasets or Main_Train_PDX_w_Configure.py for PDXs datasets.

### 3. Meta-test

If you would like to test the power of CaMeRe, please run Main_Test_patient_w_Configure.py for Clinical datasets or Main_Test_PDX_w_Configure.py for PDXs datasets.

