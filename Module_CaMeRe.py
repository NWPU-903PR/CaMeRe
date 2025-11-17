#finnaly
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from utils_comb import *
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import random
import torch.autograd as autograd
from Loss import *

from torch.distributions.multivariate_normal import MultivariateNormal


class Drug_feature_embedding(nn.Module):
    def __init__(self, num_features_xd=78,output_dimd=128,dropout=0.2):
        super(Drug_feature_embedding, self).__init__()
        # Drugs
        self.conv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.conv3 = GCNConv(num_features_xd * 4, num_features_xd * 2)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd * 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(num_features_xd * 2)

    def forward(self, data_train, data0_train):
        # Drug A
        x1_train, edge_index1_train, batch1_train = data_train['x'], data_train['edge_index'], data_train['batch']
        x1 = self.conv1(x1_train, edge_index1_train)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1_train)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1_train)
        x1 = x1 + x11
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1_train)
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc_g2(x1)
        #x1 = self.relu(self.fc_g2(x1))

        # Drug B
        x2_train, edge_index2_train, batch2_train = data0_train['x'], data0_train['edge_index'], data0_train[ 'batch']
        x2 = self.conv1(x2_train, edge_index2_train)
        x22 = self.relu(x2)
        x22 = self.conv2(x22, edge_index2_train)
        x22 = self.relu(x22)
        x22 = self.conv3(x22, edge_index2_train)
        x2 = x2 + x22
        x2 = self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2_train)
        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2(x2)
        # x2 = self.relu(self.fc_g2(x2))
        return x1,x2

#cell_line feature embedding_sem
class Cell_feature_embedding_sem(nn.Module):
    def __init__(self,output_dimc=256):
        super(Cell_feature_embedding_sem, self).__init__()
        # Cell lines
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.fcc1 = nn.Linear(64 * 25, output_dimc * 2)
        self.relu = nn.ReLU()

    def forward(self,data_train):
        # Cell lines
        cell = data_train.c
        spc, h = cell.size()
        cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        cell = cell.unsqueeze(1)
        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        #print(xt)
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = self.relu(self.gconv3(xt))
        xt = self.relu(self.gconv4(xt))
        bs, kn, ce, ce1 = xt.size()  # batch size *(K+Q),channel, width, height
        xt = xt.view(-1, kn * ce * ce1)
        xt = self.fcc1(xt)
        # xt = self.relu(self.fcc1(xt))
        return xt


class task_embedding_sem_2(nn.Module):
    def __init__(self, output_dimd=128, output_dimc=256):
        super(task_embedding_sem_2, self).__init__()
        self.relu = nn.ReLU()
        self.normf = nn.BatchNorm1d(4 * output_dimd + output_dimc * 2)
        #self.normf = nn.LayerNorm(2 * output_dimd + output_dimc * 1)
        # self.normf = nn.InstanceNorm1d(2 * output_dimd + output_dimc * 1)
        self.fc = nn.Linear(output_dimd * 4 + output_dimc * 2, 512)#1024
        # self.fc1 = nn.Linear(1024, 512)
        self.normf1 = nn.BatchNorm1d(256)

    def forward(self,  xt,c_n):
        #print("xt_before:",xt.size())
        xc = self.normf(xt)
        xc = self.relu(xc)
        xc = self.relu(self.fc(xc))
        # xc = self.relu(self.fc1(xc))
        #print("xt_before_1:", xc.size())
        _,fn = xc.size()
        xc = xc.view(c_n,-1,fn)
        #print("xt_before_2:", xc.size())
        # variational
        xc = xc.mean(dim=1)
        #print("xt_after:", xc.size())

        embeddings, z_n, c_mu, c_log_var = self.variational_distribution(xc)
        embeddings = self.normf1(embeddings)
        kl_loss = F.kl_div(embeddings.softmax(dim=-1).log(), z_n.softmax(dim=-1), reduction='batchmean')

        return  embeddings,kl_loss, c_mu, c_log_var

    def variational_distribution(self, x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, :z_dim]
        c_log_var = x[:, z_dim:]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c, z_signal, c_mu, c_log_var

class task_embedding_sem_1(nn.Module):
    def __init__(self, output_dimd=128, output_dimc=256):
        super(task_embedding_sem_1, self).__init__()
        self.relu = nn.ReLU()
        self.normf = nn.BatchNorm1d(4 * output_dimd + output_dimc * 2)
        #self.normf = nn.LayerNorm(2 * output_dimd + output_dimc * 1)
        # self.normf = nn.InstanceNorm1d(2 * output_dimd + output_dimc * 1)
        self.fc = nn.Linear(output_dimd * 4 + output_dimc * 2, 512)#1024
        # self.fc1 = nn.Linear(1024, 512)
        self.normf1 = nn.BatchNorm1d(256)

    def forward(self, xt,c_n):
        #print("xt_before:",xt.size())
        xc = self.relu(xt)
        xc = self.normf(xc)
        xc = self.relu(self.fc(xc))
        # xc = self.relu(self.fc1(xc))
        #print("xt_before_1:", xc.size())
        _,fn = xc.size()
        xc = xc.view(c_n,-1,fn)
        #print("xt_before_2:", xc.size())
        # variational
        xc = xc.mean(dim=1)
        #print("xt_after:", xc.size())

        embeddings, z_n, c_mu, c_log_var = self.variational_distribution(xc)
        embeddings = self.normf1(embeddings)
        kl_loss = F.kl_div(embeddings.softmax(dim=-1).log(), z_n.softmax(dim=-1), reduction='batchmean')

        return  embeddings,kl_loss, c_mu, c_log_var

    def variational_distribution(self, x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, :z_dim]
        c_log_var = x[:, z_dim:]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c, z_signal, c_mu, c_log_var

#cell_line feature embedding_var
class Cell_feature_embedding_var(nn.Module):
    def __init__(self,output_dimc=256):
        super(Cell_feature_embedding_var, self).__init__()

        # Cell lines
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.fcc1 = nn.Linear(64 * 25, output_dimc * 4)
        self.relu = nn.ReLU()
        #self.normf = nn.BatchNorm1d(output_dimc * 2)
        #self.fcc2 = nn.Linear(output_dimc * 2, 512)
        #self.normf1 = nn.BatchNorm1d(512)

    def forward(self, data_train):
        # Cell lines
        cell = data_train.c
        spc, h = cell.size()
        cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        cell = cell.unsqueeze(1)
        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = self.relu(self.gconv3(xt))
        xt = self.relu(self.gconv4(xt))
        bs, kn, ce, ce1 = xt.size()  # batch size *(K+Q),channel, width, height
        xt = xt.view(-1, kn * ce * ce1)
        # xt = self.relu(self.fcc1(xt))
        xt = self.fcc1(xt)
        return xt

# backdoor adjust
class Backdoor_adjust_2(nn.Module):
    def __init__(self, output_dimd=128, output_dimc=256):
        super(Backdoor_adjust_2, self).__init__()
        self.relu = nn.ReLU()
        self.normf = nn.BatchNorm1d(4 * output_dimd + output_dimc * 6)
        self.fc = nn.Linear(output_dimd * 4 + output_dimc * 6, 512)
        self.normf1 = nn.BatchNorm1d(256)
        self.Synergy_Cell_1 = nn.Linear(256, 512)
        # self.Synergy_Cell_2 = nn.Linear(512, 512)
        self.Synergy_Cell_3 = nn.Linear(512, 2)

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, xs, xv):
        xc = torch.cat((x1, x2, xs, xv), 1)
        xc = self.normf(xc)
        xc = self.relu(xc)
        xc = self.relu(self.fc(xc))
        # variational
        embeddings_n,z_n ,c_mu,c_log_var= self.variational_distribution(xc)
        embeddings = self.normf1(embeddings_n)

        Synergy_Cell_1 = self.dropout_1(self.relu(self.Synergy_Cell_1(embeddings_n)))
        # Synergy_Cell_2 = self.dropout_2(self.relu(self.Synergy_Cell_2(Synergy_Cell_1)))
        # Synergy_Cell_3 = self.sigmoid(self.Synergy_Cell_3(Synergy_Cell_1))
        Synergy_Cell_3 = self.Synergy_Cell_3(Synergy_Cell_1)
        kl_loss =  F.kl_div(embeddings.softmax(dim=-1).log(), z_n.softmax(dim=-1),reduction='batchmean')
        #print("kl_loss")
        return Synergy_Cell_3,kl_loss,embeddings_n

    def variational_distribution(self, x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, z_dim:]
        c_log_var = x[:, :z_dim]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c,z_signal,c_mu,c_log_var

class Backdoor_adjust_1(nn.Module):
    def __init__(self, output_dimd=128, output_dimc=256):
        super(Backdoor_adjust_1, self).__init__()
        self.relu = nn.ReLU()
        self.normf = nn.BatchNorm1d(4 * output_dimd + output_dimc * 6)
        self.fc = nn.Linear(output_dimd * 4 + output_dimc * 6, 512)
        self.normf1 = nn.BatchNorm1d(256)
        self.Synergy_Cell_1 = nn.Linear(256, 512)
        # self.Synergy_Cell_2 = nn.Linear(512, 512)
        self.Synergy_Cell_3 = nn.Linear(512, 2)

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, xs, xv):
        xc = torch.cat((x1, x2, xs, xv), 1)

        xc = self.relu(xc)
        xc = self.normf(xc)
        xc = self.relu(self.fc(xc))
        # variational
        embeddings_n,z_n ,c_mu,c_log_var= self.variational_distribution(xc)
        embeddings = self.normf1(embeddings_n)

        Synergy_Cell_1 = self.dropout_1(self.relu(self.Synergy_Cell_1(embeddings_n)))
        # Synergy_Cell_2 = self.dropout_2(self.relu(self.Synergy_Cell_2(Synergy_Cell_1)))
        # Synergy_Cell_3 = self.sigmoid(self.Synergy_Cell_3(Synergy_Cell_1))
        Synergy_Cell_3 = self.Synergy_Cell_3(Synergy_Cell_1)
        kl_loss =  F.kl_div(embeddings.softmax(dim=-1).log(), z_n.softmax(dim=-1),reduction='batchmean')
        #print("kl_loss1")
        return Synergy_Cell_3,kl_loss,embeddings_n

    def variational_distribution(self, x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, z_dim:]
        c_log_var = x[:, :z_dim]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c,z_signal,c_mu,c_log_var

# label classifier
class Label_classifier(nn.Module):
    def __init__(self,output_dimd=128,output_dimc=256,drop_out=0.2):
        super(Label_classifier, self).__init__()
        self.normf = nn.BatchNorm1d(4 * output_dimd + output_dimc * 2)
        self.fc = nn.Linear(output_dimd * 4 + output_dimc * 2, 512)
        self.normf1 = nn.BatchNorm1d(256)
        self.Synergy_Cell_1 = nn.Linear(256, 512)
        # self.Synergy_Cell_2 = nn.Linear(1024, 512)
        self.Synergy_Cell_3 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(drop_out)
        self.dropout_2 = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, xt):
        xc = torch.cat((x1, x2, xt), 1)
        xc = self.relu(xc)
        xc = self.normf(xc)

        xc = self.relu(self.fc(xc))
        # variational
        embeddings_n, z_n = self.variational_distribution(xc)
        embeddings = self.normf1(embeddings_n)
        # embeddings_mean = embeddings_n.mean(dim=0).cuda()
        # embeddings_std = embeddings_n.std(dim=0).cuda()
        # embeddings = ((embeddings_n - embeddings_mean) / embeddings_std).cuda()
        kl_loss = F.kl_div(embeddings.softmax(dim=-1).log(), z_n.softmax(dim=-1), reduction='batchmean')

        Synergy_Cell_1 = self.dropout_1(self.relu(self.Synergy_Cell_1(embeddings_n)))
        # Synergy_Cell_2 = self.dropout_2(self.relu(self.Synergy_Cell_2(Synergy_Cell_1)))
        # Synergy_Cell_3 = self.sigmoid(self.Synergy_Cell_3(Synergy_Cell_1))
        Synergy_Cell_3 = self.Synergy_Cell_3(Synergy_Cell_1)

        return Synergy_Cell_3,kl_loss,embeddings_n
    def variational_distribution(self, x):
        c_dim = list(x.size())[-1]
        z_dim = c_dim // 2
        c_mu = x[:, z_dim:]
        c_log_var = x[:, :z_dim]
        z_signal = torch.randn(c_mu.size()).cuda()
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c,z_signal
