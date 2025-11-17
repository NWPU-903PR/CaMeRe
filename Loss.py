"""Copyright (c) Hyperconnect, Inc. and its affiliates.
All rights reserved.
"""

import functools
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# label distribution disentangling loss
class LADELoss(nn.Module):
    def __init__(self, num_classes=10,prior_txt=None,  remine_lambda=0.1):
        super().__init__()

        self.img_num_per_cls = calculate_prior(num_classes,  prior_txt).float().cuda()
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()


        self.balanced_prior = torch.tensor(1. / num_classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

def LADE_loss(num_classes,  prior_txt, remine_lambda=0.1):
    print("Loading LADELoss.")
    return LADELoss(
        num_classes=num_classes,

        prior_txt=prior_txt,
        remine_lambda=remine_lambda,
    )

#prior ce loss
class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes,  prior_txt):
        super().__init__()
        #self.img_num_per_cls = calculate_prior(num_classes, img_max, prior, prior_txt, return_num=True).float().cuda()
        self.img_num_per_cls = calculate_prior(num_classes, prior_txt).float().cuda()

        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        #print(self.prior)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        #print(x.size())
        #y = torch.nn.functional.one_hot(y)
        #print(y.size())
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y.long())

        return loss

def Prior_ce_loss(num_classes, prior_txt):
    #print('Loading PriorCELoss Loss.')
    return PriorCELoss(
        num_classes=num_classes,
        prior_txt=prior_txt
    )

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            assert False

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def Focal_loss():
    return FocalLoss(gamma = 2.0)

# weighted softmax loss
def weighted_softmax_loss ():
    print('Loading Weighted Softmax Loss.')
    # Imagenet_LT class distribution
    dist = [0 for _ in range(1000)]
    with open('./data/ImageNet_LT/ImageNet_LT_train.txt') as f:
        for line in f:
            dist[int(line.split()[1])] += 1 #calculate the number of samples in each class
    num = sum(dist)
    prob = [i/num for i in dist]
    prob = torch.FloatTensor(prob)
    # normalization
    max_prob = prob.max().item()
    prob = prob / max_prob
    # class reweight
    weight = - prob.log() + 1

    return nn.CrossEntropyLoss(weight=weight)


def calculate_prior(num_classes, prior_txt):
    #print(prior_txt.size())
    labels = []

    for i in range(prior_txt.size(0)):
        labels.append(prior_txt[i,0].numpy().tolist())
    occur_dict = dict(Counter(labels))
    #print(occur_dict.keys())
    if 0.0 in occur_dict.keys() and 1.0 in occur_dict.keys():
        #print("1")
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    if 0.0 in occur_dict.keys() and 1.0 not in occur_dict.keys():
        #print("2")
        img_num_per_cls = [0.0,0.0]
        img_num_per_cls[0] = occur_dict[0]
    if 0.0 not in occur_dict.keys() and 1.0 in occur_dict.keys():
        #print("3")
        img_num_per_cls = [0.0, 0.0]
        img_num_per_cls[1] = occur_dict[1]

    img_num_per_cls = torch.Tensor(img_num_per_cls)
    return img_num_per_cls


