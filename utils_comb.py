import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA
import torch

from sklearn.metrics import roc_curve
import csv
# 细胞系
class TestbedDataset_cell(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_cell, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):
        ce = self.df[idx, 0]
        #print("cell是：",ce)
        cell = self.codes['cells'].item2idx.get(int(ce))
        #print("np.shape(cell)",cell)
        d1 = self.df[idx, 1]
        #print("d1是：", d1)
        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        syn = self.df[idx, 5]
        if syn > 3.4:  # 3.4:#3.4:#0.0871#0.7313:#  #3.4 :#-0.7569
        # syn = self.df[idx, 4]#3
        # if syn > 0.0871:
        # syn = self.df[idx, 3]  # 3
        # if syn > 0.7313:
            syn= 1.0
        else:
            syn= 0.0
        GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_cell(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_cell, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx, 2]
        #print("d2是：",d2)
        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']
        #print(features2.size())
        #print(edge_index2.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

# PDX
class TestbedDataset_pdx(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_pdx, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):
        cell = self.df[idx, 0]

        cell = self.codes['PDXs'].item2idx.get(cell)

        d1 = self.df[idx, 1]

        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        syn = self.df[idx, 4]
        GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_pdx(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_pdx, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx, 2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']
        #print(features2.size())
        #print(edge_index2.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

#patient
class TestbedDataset_patient(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_patient, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):

        cell = self.df[idx, 0]

        cell = self.codes['Patients'].item2idx.get(str(cell))

        d1 = self.df[idx, 1]

        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        syn = self.df[idx, 4]
        GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_patient(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_patient, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx, 2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']
        #print(features2.size())
        #print(edge_index2.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

#unlabeled patient
class TestbedDataset_unpatient(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_unpatient, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes


    def len(self):
        return len(self.df)

    def get(self,idx):


        cell = self.df[idx, 0]
        d1 = self.df[idx, 1]

        cell = self.codes['Patients'].item2idx.get(str(cell))

        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        #syn = self.df[idx, 5]
        #GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_unpatient(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_unpatient, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes


    def  len(self):
        return len(self.df)

    def  get(self, idx):

        d2 = self.df[idx, 2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1
#unlabeled patient
class TestbedDataset_unpatient_T(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_unpatient_T, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes


    def len(self):
        return len(self.df)

    def get(self,idx):


        cell = self.df[idx][0]
        d1 = self.df[idx][1]

        cell = self.codes['Patients'].item2idx.get(str(cell))

        d1 = self.codes['drugs'].item2idx.get(int(d1))
        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        #syn = self.df[idx, 5]
        #GCNData.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_unpatient_T(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_unpatient_T, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes


    def  len(self):
        return len(self.df)

    def  get(self, idx):


        d2 = self.df[idx][2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

#patient——test
class TestbedDataset_patient_test(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_patient_test, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):

        cell = self.df[idx][0]

        cell = self.codes['Patients'].item2idx.get(str(cell))

        d1 = self.df[idx][1]

        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        syn = self.df[idx][4]
        GCNData.y = torch.tensor([int(syn)], dtype=torch.int16)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_patient_test(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_patient_test, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        cell = self.df[idx][0]

        d2 = self.df[idx][2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

class TestbedDataset_pdx_test(Dataset):
    def __init__(self, df,drug_features,cell_features,codes):
        super(TestbedDataset_pdx_test, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def len(self):
        return len(self.df)

    def get(self,idx):
        cell = self.df[idx][0]

        cell = self.codes['PDXs'].item2idx.get(str(cell))
        #print("np.shape(cell)",cell)
        d1 = self.df[idx][1]

        d1 = self.codes['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features.loc[d1, 'c_size']
        features1=self.drug_features.loc[d1, 'features']
        #print(features1.size())
        edge_index1=self.drug_features.loc[d1, 'edge_index']
        #print(edge_index1.size())
        #cell_feature
        target = self.cell_features.loc[cell, 'cg']
        #print(target.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)
        syn = self.df[idx][4]
        GCNData.y = torch.tensor([int(syn)], dtype=torch.int16)  # regress
        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_pdx_test(Dataset):
    def __init__(self, df, drug_features, cell_features, codes):
        super(TestbedDataset1_pdx_test, self).__init__()
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.codes = codes

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx][2]

        d2 = self.codes['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features.loc[d2, 'c_size']
        features2 = self.drug_features.loc[d2, 'features']
        edge_index2 = self.drug_features.loc[d2, 'edge_index']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1

#patient——test
class TestbedDataset_patient_T1(Dataset):
    def __init__(self, df,cell_features,codes,drug_features_P,codes_P):
        super(TestbedDataset_patient_T1, self).__init__()
        self.df = df

        self.cell_features = cell_features
        self.codes = codes
        self.drug_features_P = drug_features_P

        self.codes_P = codes_P

    def len(self):
        return len(self.df)

    def get(self,idx):

        cell = self.df[idx][0]

        cell = self.codes['Patients'].item2idx.get(str(cell))

        d1 = self.df[idx][1]

        d1 = self.codes_P['drugs'].item2idx.get(int(d1))

        #drug1_feature
        c_size1=self.drug_features_P.loc[d1, 'c_size']
        features1=self.drug_features_P.loc[d1, 'features']

        edge_index1=self.drug_features_P.loc[d1, 'edge_index']

        #cell_feature
        target = self.cell_features.loc[cell, 'cg']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=features1,edge_index=edge_index1.transpose(1, 0))
        GCNData.c = torch.unsqueeze(target,0)

        GCNData.__setitem__('c_size1', torch.tensor([c_size1],dtype=torch.long))
        return GCNData

class TestbedDataset1_patient_T1(Dataset):
    def __init__(self, df, cell_features,codes,drug_features_P,codes_P):
        super(TestbedDataset1_patient_T1, self).__init__()
        self.df = df
        self.cell_features = cell_features
        self.codes = codes
        self.drug_features_P = drug_features_P
        self.codes_P = codes_P

    def  len(self):
        return len(self.df)

    def  get(self, idx):
        d2 = self.df[idx][2]

        d2 = self.codes_P['drugs'].item2idx.get(int(d2))
        # drug_feature
        c_size2 = self.drug_features_P.loc[d2, 'c_size']
        features2 = self.drug_features_P.loc[d2, 'features']
        edge_index2 = self.drug_features_P.loc[d2, 'edge_index']

        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=features2,edge_index=edge_index2.transpose(1, 0))
        GCNData1.__setitem__('c_size2', torch.tensor([c_size2],dtype=torch.long))
        return GCNData1


def save_statistics(experiment_name, line_to_add): #save
    with open("{}.csv".format(experiment_name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line_to_add)

def load_statistics(experiment_name): #
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict

def BACC(label,pred):
    FPR,TPR,_ = roc_curve(label, pred)
    TNR=1-FPR[1]
    TPR=TPR[1]
    return (TNR+TPR)/2