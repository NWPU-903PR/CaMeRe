
# CaMeRe

Source code and dataset for CaMeRe.

## Requirements

Python == 3.6

torch == 1.7.0+cu110

torch-geometric == 1.6.1

torch-scatter == 2.0.5

torch-sparse == 0.6.8

torch-cluster == 1.5.8

rdkit == 2019.09.3.0

pytorchtools == 0.0.2


## Usage

### 1. Meta-training 

If you would like to learn the weights and biases of the CaMeRe, please run Main_Train_patient_w.py for Clinical datasets or Main_Train_PDX_w.py for PDXs datasets.

### 2. Meta-test

If you would like to test the power of CaMeRe, please run Main_Test_patient_w.py for Clinical datasets or Main_Test_PDX_w.py for PDXs datasets.

