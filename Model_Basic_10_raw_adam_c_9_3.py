#finnaly
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
from Module_CaMeRe import *
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss, auc, precision_recall_curve,matthews_corrcoef

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= -ctx.alpha
        return grad_input, None

class GradientReversalFunction(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalFunction, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

#loss define
# class FocalLoss(nn.Module):
#     def __init__(self,alpha=0.25,gamma= 2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#     def forward(self,inputs,targets):
#         ce_loss = F.binary_cross_entropy(inputs,targets,reduction='none')
#         pt= torch.exp(-ce_loss)
#         focal_loss = self.alpha*(1-pt)**self.gamma*ce_loss
#         return focal_loss.mean()
# class PostCompensatedSoftmax(nn.Module):
#     def __init__(self,num_classes,alpha = 0.1):
#         super(PostCompensatedSoftmax, self).__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#     def forward(self,inputs):
#         softmax_out = torch.softmax(inputs,dim=1)
#         class_weights = self.compute_class_weights(softmax_out)
#         postcomp_softmax_out = softmax_out*class_weights
#         return postcomp_softmax_out
#     def compute_class_weights(self,softmax_out):
#         class_counts = softmax_out.sum(dim=0)
#         class_weights = (1-self.alpha)/(1-self.alpha**class_counts)
#         return class_weights

class CaMeRe(nn.Module):
    def __init__(self,args):
        super(CaMeRe, self).__init__()
        self.drug_features = Drug_feature_embedding()
        self.cell_features_sem = Cell_feature_embedding_sem()
        self.cell_features_var = Cell_feature_embedding_var()
        self.label_classifier = Label_classifier()

        if args.batnorm == "true":
            self.task_features_sem = task_embedding_sem_2()
            self.back_door_adjust = Backdoor_adjust_2()
        else:
            self.back_door_adjust = Backdoor_adjust_1()
            self.task_features_sem = task_embedding_sem_1()

        # classifer
        model_param_group_classifer = []
        model_param_group_classifer.append({"params": self.label_classifier.parameters()})
        model_param_group_classifer.append({"params": self.back_door_adjust.parameters()})
        # label embedding
        model_param_group_label = []
        model_param_group_label.append({"params": self.drug_features.parameters()})
        model_param_group_label.append({"params": self.cell_features_sem.parameters()})
         # SGD
        if args.optimize == "SGD":
            self.optimizor_classifer = optim.SGD(params=model_param_group_classifer, lr=args.lr_c, momentum=args.mc,
                                                 weight_decay=args.wdc)
            self.optimizor_task = optim.SGD(params=self.task_features_sem.parameters(), lr=args.lr_c, momentum=args.mc,
                                            weight_decay=args.wdc)
            self.optimizor_label = optim.SGD(params=model_param_group_label, lr=args.lr_l, momentum=args.ml,
                                             weight_decay=args.wdl)
            self.optimizor_domain = optim.SGD(params=self.cell_features_var.parameters(), lr=args.lr_d,
                                              momentum=args.md, weight_decay=args.wdd)

        # ADAM
        if args.optimize == "ADAM":
            self.optimizor_classifer = optim.Adam(params=model_param_group_classifer, lr=args.lr_c_a,
                                                  weight_decay=args.wdc)
            self.optimizor_task = optim.Adam(params=self.task_features_sem.parameters(), lr=args.lr_t_a,
                                             weight_decay=args.wdc)
            self.optimizor_label = optim.Adam(params=model_param_group_label, lr=args.lr_l_a, weight_decay=args.wdl)
            self.optimizor_domain = optim.Adam(params=self.cell_features_var.parameters(), lr=args.lr_d_a,
                                               weight_decay=args.wdd)

        # Rmsprop
        if args.optimize == "RMSprop":
            self.optimizor_classifer = optim.RMSprop(params=model_param_group_classifer, lr=args.lr_c_a,
                                                     weight_decay=args.wdc)
            self.optimizor_task = optim.RMSprop(params=self.task_features_sem.parameters(), lr=args.lr_t_a,
                                                weight_decay=args.wdc)
            self.optimizor_label = optim.RMSprop(params=model_param_group_label, lr=args.lr_l_a, weight_decay=args.wdl)
            self.optimizor_domain = optim.RMSprop(params=self.cell_features_var.parameters(), lr=args.lr_d_a,
                                                  weight_decay=args.wdd)

        self.scheduler_c_a = lr_scheduler.MultiStepLR(optimizer=self.optimizor_classifer,
                                                    milestones=args.milestones, gamma=args.lr_c_s)  #
        self.scheduler_l_a = lr_scheduler.MultiStepLR(optimizer=self.optimizor_label,
                                                    milestones=args.milestones, gamma=args.lr_l_s)  #
        self.scheduler_d_a = lr_scheduler.MultiStepLR(optimizer=self.optimizor_domain,
                                                    milestones=args.milestones, gamma=args.lr_d_s)  #
        self.scheduler_t_a = lr_scheduler.MultiStepLR(optimizer=self.optimizor_task,
                                                    milestones=args.milestones, gamma=args.lr_t_s)

    def forward(self, mini_test, drug_features_cell, cell_features, codes_cell, number_train_task,
                drug_features_unpatient, unpatient_features, codes_unpatient,
                drug_features_patient, patient_features, codes_patient, step0, device,args):

        # loss init
        loss_query_pa_id_s_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_query_pa_indis_dom_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_query_pa_invar_id_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_query_pa_dom_v_all = torch.tensor(0, dtype=torch.float64).to(device)

        loss_sem_out_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_var_out_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_classifer_out_all = torch.tensor(0, dtype=torch.float64).to(device)
        loss_task_out_all = torch.tensor(0, dtype=torch.float64).to(device)

        num_ta = 0
        self.drug_features.train()
        self.cell_features_sem.train()
        self.cell_features_var.train()
        self.task_features_sem.train()
        self.label_classifier.train()
        self.back_door_adjust.train()

        old_params_de = parameters_to_vector(self.drug_features.parameters())
        old_params_cs = parameters_to_vector(self.cell_features_sem.parameters())
        old_params_ts = parameters_to_vector(self.task_features_sem.parameters())
        old_params_cv = parameters_to_vector(self.cell_features_var.parameters())
        old_params_lc = parameters_to_vector(self.label_classifier.parameters())
        old_params_bd = parameters_to_vector(self.back_door_adjust.parameters())

        #cell lines and unpatient, patient
        if args.object == 'cell':
            x_support_set, x_support_unlabel_set0, x_target0_q,x_target0 = mini_test.get_train_batch(obj='pdx',augment=False)
        else:
            x_support_set, x_support_unlabel_set0, x_target0_q, x_target0 = mini_test.get_train_batch(obj=args.object,
                                                                                                  augment=False)
        batch_n, cells_n, samples_n, features_n = np.shape(x_support_set)
        samples_n_c = int(samples_n / number_train_task)

        batch_n, samples_un, features_un = np.shape(x_support_unlabel_set0)
        batch_n, samples_pn, features_pn = np.shape(x_target0)
        batch_n, cells_n_q, samples_n_q, features_n = np.shape(x_target0_q)

        total_preds_train = torch.Tensor()
        total_labels_train = torch.Tensor()

        for batch in range(batch_n):
            #print("batch:",batch)
            # cell line
            x_support_set_cell = x_support_set[batch]
            x_support_set_cell = x_support_set_cell.reshape(cells_n , number_train_task, samples_n_c, features_n)
            x_support_set_cell = x_support_set_cell.transpose(1,0,2,3)
            x_support_set_cell = x_support_set_cell.reshape(number_train_task * cells_n * samples_n_c, features_n)
            train_data = TestbedDataset_cell(x_support_set_cell, drug_features_cell, cell_features, codes_cell)
            train_data1 = TestbedDataset1_cell(x_support_set_cell, drug_features_cell, cell_features, codes_cell)
            data_train_cell = DataLoader(train_data, batch_size=cells_n * samples_n_c, shuffle=False)
            data1_train_cell = DataLoader(train_data1, batch_size=cells_n * samples_n_c, shuffle=False)

            # data_train_cell_p = DataLoader(train_data, batch_size=number_train_task * cells_n * samples_n_c, shuffle=False)
            # data1_train_cell_p = DataLoader(train_data1, batch_size=number_train_task * cells_n * samples_n_c, shuffle=False)

            # unlabeled patient
            x_support_unlabel_set = x_support_unlabel_set0[batch]
            x_support_unlabel_set = np.expand_dims(x_support_unlabel_set,0).repeat(number_train_task,axis=0)
            x_support_unlabel_set = x_support_unlabel_set.reshape( number_train_task*samples_un, features_un)
            train_data = TestbedDataset_unpatient(x_support_unlabel_set, drug_features_unpatient, unpatient_features,
                                                  codes_unpatient)
            train_data1 = TestbedDataset1_unpatient(x_support_unlabel_set, drug_features_unpatient, unpatient_features,
                                                    codes_unpatient)
            data_train_unpatient = DataLoader(train_data, batch_size=samples_un, shuffle=False)
            data1_train_unpatient = DataLoader(train_data1, batch_size=samples_un, shuffle=False)

            # train
            for batch_idx, data in enumerate(zip(data_train_cell, data1_train_cell, data_train_unpatient,data1_train_unpatient)):
                x_support = data[0].to(device)
                x_support1 = data[1].to(device)
                x_support_unlabel = data[2].to(device)
                x_support1_unlabel = data[3].to(device)

                self.inner_optimize(cells_n, samples_n_c, x_support, x_support1,x_support_unlabel,
                                    x_support1_unlabel, args.num_inner_updates, device,args)

            # patient
            x_target = x_target0[batch]
            if args.object =='patient':
                train_data = TestbedDataset_patient(x_target, drug_features_patient, patient_features, codes_patient)
                train_data1 = TestbedDataset1_patient(x_target, drug_features_patient, patient_features, codes_patient)
            else :
                train_data = TestbedDataset_pdx(x_target, drug_features_patient, patient_features, codes_patient)
                train_data1 = TestbedDataset1_pdx(x_target, drug_features_patient, patient_features, codes_patient)

            data_train_patient = DataLoader(train_data, batch_size=samples_pn, shuffle=False)
            data1_train_patient = DataLoader(train_data1, batch_size=samples_pn, shuffle=False)

            x_target_q = x_target0_q[batch]
            x_target_q = x_target_q.reshape(cells_n_q * samples_n_q, features_n)
            train_data = TestbedDataset_cell(x_target_q, drug_features_cell, cell_features, codes_cell)
            train_data1 = TestbedDataset1_cell(x_target_q, drug_features_cell, cell_features, codes_cell)

            data_train_patient_q = DataLoader(train_data, batch_size=cells_n_q*samples_n_q, shuffle=False)
            data1_train_patient_q = DataLoader(train_data1, batch_size=cells_n_q*samples_n_q, shuffle=False)

            for batch_idx, data in enumerate(zip(data_train_patient_q, data1_train_patient_q,data_train_unpatient,data1_train_unpatient,data_train_patient, data1_train_patient)):

                x_query_q = data[0].to(device)
                x_query1_q = data[1].to(device)
                x_support_unlabel  = data[2].to(device)
                x_support1_unlabel = data[3].to(device)
                x_query = data[4].to(device)
                x_query1 = data[5].to(device)

                # patient prediction
                drug1_features_patient, drug2_features_patient = self.drug_features(x_query, x_query1)
                patient_features_sem = self.cell_features_sem(x_query)


                patient_samples_label, kl_s,_ = self.label_classifier(drug1_features_patient, drug2_features_patient,
                                                      patient_features_sem)


                self.loss = Prior_ce_loss(2, prior_txt=x_query['y'].view(-1, 1).cpu())
                loss_query_pa_id_s_1 = self.loss(patient_samples_label,x_query['y'].to(torch.int64).to(device)) + args.lamda_kl * kl_s

                drug1_features, drug2_features = self.drug_features(x_query_q, x_query1_q)

                cell_features_sem = self.cell_features_sem(x_query_q)
                # print(patient_features_sem)


                cell_samples_label, kl_s_c,_ = self.label_classifier(drug1_features, drug2_features,cell_features_sem)
                # print(patient_samples_label)
                #print("kl_s_c:", kl_s_c)
                self.loss = Prior_ce_loss(2, prior_txt=x_query_q['y'].view(-1, 1).cpu())
                loss_query_cell_id_s = self.loss(cell_samples_label,x_query_q['y'].to(torch.int64).to(device)) + args.lamda_kl * kl_s_c
                loss_query_pa_id_s =(args.lamda_task*loss_query_pa_id_s_1 + loss_query_cell_id_s)
                # print("kl_s:", kl_s)
                # cell line
                cell_features_sem_sig = cell_features_sem.reshape(cells_n_q, samples_n_q, -1)
                cell_features_sem_sig = cell_features_sem_sig[:, 0, :]

                # unpatient
                drug1_features_unpa, drug2_features_unpa = self.drug_features(x_support_unlabel, x_support1_unlabel)
                unpatient_features_sem = self.cell_features_sem(x_support_unlabel)

                # VAR
                cell_var = self.cell_features_var(x_query_q)

                cell_var_sig = cell_var.reshape(cells_n_q, samples_n_q, -1)
                cell_var_sig = cell_var_sig[:, 0, :]

                unpatient_var = self.cell_features_var(x_support_unlabel)
                patient_var = self.cell_features_var(x_query)

                # drug pair
                drug1_s = torch.cat((drug1_features, drug1_features_unpa), dim=0)
                drug1_s = torch.cat((drug1_s, drug1_features_patient), dim=0)
                drug2_s = torch.cat((drug2_features, drug2_features_unpa), dim=0)
                drug2_s = torch.cat((drug2_s, drug2_features_patient), dim=0)
                # domain loss
                features_sem_c_p = torch.cat((cell_features_sem_sig, unpatient_features_sem), dim=0)
                features_sem_c_p = torch.cat((features_sem_c_p, patient_features_sem), dim=0)

                drug1_s_c = drug1_s.repeat_interleave(features_sem_c_p.size()[0], dim=0)
                drug2_s_c = drug2_s.repeat_interleave(features_sem_c_p.size()[0], dim=0)
                features_sem = torch.cat([features_sem_c_p] * drug1_s.size()[0], dim=0)

                embedding_var = torch.cat((cell_var_sig, unpatient_var), dim=0)
                embedding_var = torch.cat((embedding_var, patient_var), dim=0)


                reversed_embedding_var = embedding_var
                features_var, kl_var, features_var_mu, features_var_sigma = self.task_features_sem(
                    reversed_embedding_var, reversed_embedding_var.size()[0])

                patient_var_label = self.wasserstein_distance(features_var_mu, features_var_sigma, device)
                #print("patient_var_label:",patient_var_label)
                loss_query_pa_dom_v = patient_var_label + args.lamda_kl_s * kl_var
                #print("kl_var:",kl_var)
                c_up_patient_sem = torch.cat((drug1_s_c, drug2_s_c, features_sem), 1)
                c_up_patient_sem = GradientReversalLayer.apply(c_up_patient_sem, 1)
                _, kl_se, features_sem_mu, features_sem_sigma = self.task_features_sem(c_up_patient_sem,
                                                                                       c_up_patient_sem.size()[0])
                reversed_feature_mu = features_sem_mu.reshape(drug1_s.size()[0], features_sem_c_p.size()[0], -1)
                reversed_feature_sigma = features_sem_sigma.reshape(drug1_s.size()[0], features_sem_c_p.size()[0],-1)
                domain_label_sem = self.wasserstein_distance(reversed_feature_mu, reversed_feature_sigma, device)
                #print("domain_label_sem:", domain_label_sem)
                c_sem_task = torch.cat((drug1_features, drug2_features, cell_features_sem), 1)
                c_sem_task = GradientReversalLayer.apply(c_sem_task, 1)
                _, kl_tc, task_cell_sem_mu, task_cell_sem_sigma = self.task_features_sem(c_sem_task, args.cntq)
                up_sem_task = torch.cat((drug1_features_unpa, drug2_features_unpa, unpatient_features_sem), 1)
                up_sem_task = GradientReversalLayer.apply(up_sem_task, 1)
                _, kl_tup, task_unpatient_sem_mu, task_unpatient_sem_sigma = self.task_features_sem(up_sem_task,args.usn)
                patient_sem_task = torch.cat((drug1_features_patient, drug2_features_patient, patient_features_sem), 1)
                patient_sem_task = GradientReversalLayer.apply(patient_sem_task, 1)
                _, kl_tp, task_patient_sem_mu, task_patient_sem_sigma = self.task_features_sem(patient_sem_task,args.qsn)

                task_feature_mu = torch.cat((task_cell_sem_mu, task_unpatient_sem_mu), dim=0)
                task_feature_mu = torch.cat((task_feature_mu, task_patient_sem_mu), dim=0)

                task_feature_sigma = torch.cat((task_cell_sem_sigma, task_unpatient_sem_sigma), dim=0)
                task_feature_sigma = torch.cat((task_feature_sigma, task_patient_sem_sigma), dim=0)


                task_label_sem = self.wasserstein_distance(task_feature_mu, task_feature_sigma, device)

                loss_query_pa_indis_dom = (domain_label_sem + task_label_sem )+ args.lamda_kl_s * (kl_se) +args.lamda_kl_s1 * (kl_tc +kl_tup + kl_tp).mean()

                # backdoor adjustment

                features_var_random = torch.Tensor().to(device)
                for _ in range(patient_features_sem.size(0)):
                    index = torch.randperm(embedding_var.size()[0])[:args.mean_backdoor]
                    features_var_random = torch.cat((features_var_random, embedding_var[index]), dim=0)

                drug1_features_b = drug1_features_patient.repeat_interleave(args.mean_backdoor, dim=0)
                drug2_features_b = drug2_features_patient.repeat_interleave(args.mean_backdoor, dim=0)
                patient_features_sem_b = patient_features_sem.repeat_interleave(args.mean_backdoor, dim=0)

                label =x_query['y'].view(-1, 1)
                # label = torch.cat([x_query['y'].view(-1, 1)] * args.mean_backdoor, dim=0)
                backdoor_samples_label, kl,_ = self.back_door_adjust(drug1_features_b, drug2_features_b,
                                                                     patient_features_sem_b, features_var_random)

                backdoor_samples_label = backdoor_samples_label.view(-1,args.mean_backdoor,2)
                # print(backdoor_samples_label[0])
                backdoor_samples_label = torch.mean(backdoor_samples_label , dim=1)
                # print("1",backdoor_samples_label[0])
                self.loss = Prior_ce_loss(2, prior_txt=label.cpu())
                loss_query_pa_invar_id_1 = self.loss(backdoor_samples_label,label.view(-1).to(torch.int64).to(device))+args.lamda_kl*kl
                #print("kl:",kl)

                features_var_random = torch.Tensor().to(device)
                for _ in range(cell_features_sem.size(0)):
                    index = torch.randperm(embedding_var.size()[0])[:args.mean_backdoor]
                    features_var_random = torch.cat((features_var_random, embedding_var[index]), dim=0)

                drug1_features_b = drug1_features.repeat_interleave(args.mean_backdoor, dim=0)
                drug2_features_b = drug2_features.repeat_interleave(args.mean_backdoor, dim=0)
                cell_features_sem_b = cell_features_sem.repeat_interleave(args.mean_backdoor, dim=0)

                label = x_query_q['y'].view(-1, 1)
                # label = torch.cat([x_query['y'].view(-1, 1)] * args.mean_backdoor, dim=0)
                backdoor_samples_label, kl,_ = self.back_door_adjust(drug1_features_b, drug2_features_b,
                                                                   cell_features_sem_b, features_var_random)
                #print("kl:", kl)
                backdoor_samples_label = backdoor_samples_label.view(-1, args.mean_backdoor, 2)
                backdoor_samples_label = torch.mean(backdoor_samples_label, dim=1)
                self.loss = Prior_ce_loss(2, prior_txt=label.cpu())
                loss_query_cell_invar_id = self.loss(backdoor_samples_label,label.view(-1).to(torch.int64).to(device)) + args.lamda_kl * kl
                # print(((cells_n_q*samples_n_q)/samples_n))
                loss_query_pa_invar_id = (args.lamda_task*loss_query_pa_invar_id_1 + loss_query_cell_invar_id)

                loss_sem_out = torch.sum(
                    torch.stack([args.lamda_pa[0] * loss_query_pa_id_s, args.lamda_pa[1] * loss_query_pa_indis_dom,
                                 args.lamda_pa[2] * loss_query_pa_invar_id]))
                loss_var_out = torch.sum(
                    torch.stack([args.lamda_pa[3] * loss_query_pa_dom_v, args.lamda_pa[4] * loss_query_pa_invar_id]))
                loss_classifer_out = torch.sum(
                    torch.stack([args.lamda_pa[5] * loss_query_pa_id_s, args.lamda_pa[6] * loss_query_pa_invar_id]))

                loss_task_out = torch.sum(
                    torch.stack([args.lamda_pa[1] * loss_query_pa_indis_dom, args.lamda_pa[3] * loss_query_pa_dom_v]))

                #  all loss

                loss_query_pa_id_s_all = loss_query_pa_id_s_all + loss_query_pa_id_s
                # """
                loss_query_pa_indis_dom_all = loss_query_pa_indis_dom_all + loss_query_pa_indis_dom
                loss_query_pa_invar_id_all = loss_query_pa_invar_id_all + loss_query_pa_invar_id
                loss_query_pa_dom_v_all = loss_query_pa_dom_v_all + loss_query_pa_dom_v

                loss_sem_out_all = loss_sem_out_all + loss_sem_out
                loss_var_out_all = loss_var_out_all + loss_var_out
                loss_classifer_out_all = loss_classifer_out_all + loss_classifer_out
                loss_task_out_all = loss_task_out_all + loss_task_out
                # """
                num_ta = num_ta + 1

                total_labels_train = torch.cat((total_labels_train, x_query['y'].view(-1, 1).float().cpu()), 0)
                total_preds_train = torch.cat((total_preds_train, F.softmax(patient_samples_label, dim=1)[:, 1].cpu()),0)

            vector_to_parameters(old_params_de, self.drug_features.parameters())
            vector_to_parameters(old_params_cs, self.cell_features_sem.parameters())
            vector_to_parameters(old_params_ts, self.task_features_sem.parameters())
            vector_to_parameters(old_params_cv, self.cell_features_var.parameters())
            vector_to_parameters(old_params_lc, self.label_classifier.parameters())
            vector_to_parameters(old_params_bd, self.back_door_adjust.parameters())

        loss_query_pa_id_s = loss_query_pa_id_s_all / num_ta
        loss_query_pa_indis_dom = loss_query_pa_indis_dom_all / num_ta
        loss_query_pa_invar_id = loss_query_pa_invar_id_all / num_ta
        loss_query_pa_dom_v = loss_query_pa_dom_v_all / num_ta

        loss_query_pa_sem = loss_sem_out_all / num_ta
        loss_query_pa_var = loss_var_out_all / num_ta
        loss_query_pa_classifer = loss_classifer_out_all / num_ta
        loss_query_pa_task = loss_task_out_all / num_ta


        if step0 % args.inter == 0:
            print("current step/total step: {}/{}".format(step0, args.epochs))
            print("loss_query_pa_indis_dom: %.4f" % loss_query_pa_indis_dom.item(), "loss_qry_pa:%.4f" % loss_query_pa_id_s.item(),
                  "loss_query_pa_invar_id:%.4f" % loss_query_pa_invar_id.item(),
                  "loss_query_pa_dom_v :%.4f" % loss_query_pa_dom_v.item())


        self.optimizor_label.zero_grad()
        self.optimizor_domain.zero_grad()
        self.optimizor_classifer.zero_grad()
        self.optimizor_task.zero_grad()

        loss_query_pa_sem.backward(retain_graph=True)
        loss_query_pa_var.backward(retain_graph=True)
        loss_query_pa_classifer.backward(retain_graph=True)
        loss_query_pa_task.backward(retain_graph=True)

        self.optimizor_label.step()
        self.optimizor_domain.step()
        self.optimizor_classifer.step()
        self.optimizor_task.step()

        if self.scheduler_c_a.get_last_lr()[0] > 1e-6:
            self.scheduler_c_a.step()
        if self.scheduler_l_a.get_last_lr()[0] > 1e-6:
            self.scheduler_l_a.step()
        if self.scheduler_d_a.get_last_lr()[0] > 1e-6:
            self.scheduler_d_a.step()
        if self.scheduler_t_a.get_last_lr()[0] > 1e-6:
            self.scheduler_t_a.step()
            #print("self.scheduler_t_a:",self.scheduler_t_a.get_last_lr())

        return loss_query_pa_indis_dom.item(), loss_query_pa_id_s.item(), loss_query_pa_invar_id.item(),\
               loss_query_pa_dom_v.item()

    def update_params(self, loss, update_lr, parameters):
        grads = torch.autograd.grad(loss, parameters)
        return parameters_to_vector(grads), parameters_to_vector(parameters) - parameters_to_vector(grads) * update_lr

    def inner_optimize(self, cells_n, samples_n_c, x_support, x_support1, x_support_unlabel, x_support1_unlabel,
                       update_step, device,args):

        self.update_lr = args.inner_lr

        for k in range(update_step):
            # cell line
            drug1_features, drug2_features = self.drug_features(x_support, x_support1)
            cell_features_sem = self.cell_features_sem(x_support)
            # feature_sem,kl,_,_ = self.features_sem(drug1_features, drug2_features, cell_features_sem)
            samples_label,kl,_= self.label_classifier(drug1_features, drug2_features, cell_features_sem)
            self.loss = Prior_ce_loss(2, prior_txt=x_support['y'].view(-1, 1).cpu())
            loss_id_s_in = self.loss(samples_label, x_support['y'].to(torch.int64).to(device))+args.lamda_kl*kl
            # print("loss_id_s_in:", loss_id_s_in)

            cell_features_sem_sig = cell_features_sem.reshape(cells_n, samples_n_c, -1)
            cell_features_sem_sig = cell_features_sem_sig[:, 0, :]
            # unpatient
            unpatient_features_sem = self.cell_features_sem(x_support_unlabel)
            features_sem_c_up = torch.cat((cell_features_sem_sig, unpatient_features_sem), dim=0)

            drug1_features_unpa, drug2_features_unpa = self.drug_features(x_support_unlabel, x_support1_unlabel)
            # drug pair
            drug1_s = torch.cat((drug1_features, drug1_features_unpa), dim=0)
            drug2_s = torch.cat((drug2_features, drug2_features_unpa), dim=0)
            drug1_s_c = drug1_s.repeat_interleave(features_sem_c_up.size()[0], dim=0)
            drug2_s_c = drug2_s.repeat_interleave(features_sem_c_up.size()[0], dim=0)

            features_sem_in = torch.cat([features_sem_c_up] * drug1_s.size()[0], dim=0)

            cell_sem = torch.cat((drug1_s_c, drug2_s_c, features_sem_in), 1)
            reversed_cell_sem = GradientReversalLayer.apply(cell_sem, 1)
            _, kl_s, features_sem_mu_in, features_sem_sigma_in = self.task_features_sem(reversed_cell_sem, cell_sem.size()[0])
            # print(features_sem_sigma_in)
            #
            reversed_feature_mu_in = features_sem_mu_in.reshape(drug1_s.size()[0], features_sem_c_up.size()[0], -1)
            reversed_feature_sigma_in = features_sem_sigma_in.reshape(drug1_s.size()[0], features_sem_c_up.size()[0],
                                                                      -1)
            domain_label_sem_in = self.wasserstein_distance(reversed_feature_mu_in, reversed_feature_sigma_in, device)
            # print(domain_label_sem_in)
            cell_sem_task = torch.cat((drug1_features, drug2_features, cell_features_sem), 1)
            reversed_cell_sem_task = GradientReversalLayer.apply(cell_sem_task, 1)
            _, kl_tc_in, task_cell_sem_mu_in, task_cell_sem_sigma_in = self.task_features_sem(reversed_cell_sem_task, args.cnt)
            unpatient_sem_task = torch.cat((drug1_features_unpa, drug2_features_unpa, unpatient_features_sem,), 1)
            reversed_unpatient_sem_task = GradientReversalLayer.apply(unpatient_sem_task, 1)
            _, kl_tup_in, task_unpatient_sem_mu_in, task_unpatient_sem_sigma_in = self.task_features_sem(
                reversed_unpatient_sem_task, args.usn)

            task_feature_mu_in = torch.cat((task_cell_sem_mu_in, task_unpatient_sem_mu_in), dim=0)
            # task_feature_mu = torch.cat((task_feature_mu, task_patient_sem_mu), dim=0)

            task_feature_sigma_in = torch.cat((task_cell_sem_sigma_in, task_unpatient_sem_sigma_in), dim=0)
            # task_feature_sigma = torch.cat((task_feature_sigma, task_patient_sem_sigma), dim=0)
            # print(task_feature_sigma_in)

            task_label_sem_in = self.wasserstein_distance(task_feature_mu_in, task_feature_sigma_in, device)

            # loss_indis_dom_in = torch.mean(domain_label_sem_in) + args.lamda_kl * kl_s

            loss_indis_dom_in = (domain_label_sem_in + task_label_sem_in + args.lamda_kl_s * (kl_s) + args.lamda_kl_s1 *(kl_tc_in + kl_tup_in).mean())
            # print("loss_indis_dom_in:", loss_indis_dom_in)
            # VAR
            unpatient_var = self.cell_features_var(x_support_unlabel)

            cell_var = self.cell_features_var(x_support)
            cell_var_sig = cell_var.reshape(cells_n, samples_n_c, -1)
            cell_var_sig = cell_var_sig[:, 0, :]
            cell_unpa_var = torch.cat((cell_var_sig, unpatient_var), dim=0)

            reversed_embedding_var = cell_unpa_var
            features_var, kl_var, features_var_mu, features_var_sigma = self.task_features_sem(reversed_embedding_var,
                                                                                               reversed_embedding_var.size()[
                                                                                                   0])
            # print(features_var_mu)
            domain_label_var_in_all = self.wasserstein_distance(features_var_mu, features_var_sigma, device)
            # domain_label_var_in_all = self.matrix_distance(features_var, device)
            # print("0_1:", torch.cuda.memory_allocated())
            # loss_query_pa_dom_v = - torch.mean(patient_var_label)+ args.lamda_kl * (kl_v_c+kl_v_unp+kl_v_p)
            loss_dom_v_in = domain_label_var_in_all + args.lamda_kl_s * kl_var
            # print("loss_dom_v_in:", loss_dom_v_in)
            # backdoor
            # cell_var_sig = cell_var.reshape(cells_n, samples_n_c, -1)
            # cell_var_sig = cell_var_sig[:, 0, :]
            # cell_unpa_var = torch.cat((cell_var_sig, unpatient_var), dim=0)

            cell_features_var_random = torch.Tensor().to(device)
            for _ in range(cell_features_sem.size(0)):
                index = torch.randperm(cell_unpa_var.size()[0])[:args.mean_backdoor]
                cell_features_var_random = torch.cat((cell_features_var_random,cell_unpa_var[index]),dim=0)

            drug1_features_b = drug1_features.repeat_interleave(args.mean_backdoor, dim=0)
            drug2_features_b = drug2_features.repeat_interleave(args.mean_backdoor, dim=0)
            cell_features_sem_b = cell_features_sem.repeat_interleave(args.mean_backdoor, dim=0)
            label = x_support['y'].view(-1, 1)
            # label = torch.cat([x_support['y'].view(-1, 1)]* args.mean_backdoor,dim=0)
            backdoor_samples_label,kl,_ = self.back_door_adjust(drug1_features_b, drug2_features_b,cell_features_sem_b,cell_features_var_random)
            backdoor_samples_label = backdoor_samples_label.view(-1, args.mean_backdoor, 2)
            backdoor_samples_label = torch.mean(backdoor_samples_label, dim=1)

            self.loss = Prior_ce_loss(2, prior_txt=label.cpu())
            #print(backdoor_samples_label.size())
            loss_invar_id_in = self.loss(backdoor_samples_label,label.view(-1).to(torch.int64).to(device))+args.lamda_kl*kl
            #print("loss_invar_id_in:", loss_invar_id_in)
            loss_sem_in = torch.sum(torch.stack([args.lamda_pa[7] * loss_id_s_in, args.lamda_pa[8] * loss_indis_dom_in,
                                                 args.lamda_pa[9] * loss_invar_id_in]))

            loss_var_in = torch.sum(
                torch.stack([args.lamda_pa[10] * loss_dom_v_in, args.lamda_pa[11] * loss_invar_id_in]))

            loss_classifer_in = torch.sum(
                torch.stack([args.lamda_pa[12] * loss_id_s_in, args.lamda_pa[13] * loss_invar_id_in]))

            loss_task_in = torch.sum(
                torch.stack([args.lamda_pa[8] * loss_indis_dom_in, args.lamda_pa[10] * loss_dom_v_in]))

            # loss_sem_in
            # print(loss_sem_in)
            grads = torch.autograd.grad(loss_sem_in, self.drug_features.parameters(), retain_graph=True)
            # print("self.drug_features.parameters()", self.drug_features.parameters())
            # print("grads:drug_features",grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.drug_features.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.drug_features.parameters())

            grads = torch.autograd.grad(loss_sem_in, self.cell_features_sem.parameters(), retain_graph=True)
            # print("grads:cell_features_sem", grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.cell_features_sem.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.cell_features_sem.parameters())

            # loss_var_in
            # print(loss_var_in)

            grads = torch.autograd.grad(loss_var_in, self.cell_features_var.parameters(), retain_graph=True)
            # grads = torch.autograd.grad(loss_var_in, parameters_to_vector(self.cell_features_var.parameters()),retain_graph=True)
            # print("grads:cell_features_var",grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.cell_features_var.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.cell_features_var.parameters())

            grads = torch.autograd.grad(loss_task_in, self.task_features_sem.parameters(), retain_graph=True)
            # print("grads:task_features_sem", grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.task_features_sem.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.task_features_sem.parameters())

            # loss_classifer_in
            grads = torch.autograd.grad(loss_classifer_in, self.label_classifier.parameters(), retain_graph=True)
            # print("grads:label_classifier", grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.label_classifier.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.label_classifier.parameters())

            grads = torch.autograd.grad(loss_classifer_in, self.back_door_adjust.parameters(), retain_graph=True)
            # print("grads:label_classifier", grads)
            new_grad, new_params = parameters_to_vector(grads), parameters_to_vector(
                self.back_door_adjust.parameters()) - parameters_to_vector(
                grads) * self.update_lr
            vector_to_parameters(new_params, self.back_door_adjust.parameters())


    def wasserstein_distance(self,mu_list,sigma_list,device):
        #print(mu_list.dim())
        if mu_list.dim()==2:
            # print("true")
            mu_list = mu_list.unsqueeze(0)
            sigma_list = sigma_list.unsqueeze(0)
            # print(mu_list.size())
            # print(sigma_list.size())


        # mean_distance = self.distance_matrix(mu_list.sqrt())
        # var_distance = self.distance_matrix(sigma_list.sqrt())
        mean_distance = self.distance_matrix(mu_list)
        var_distance = self.distance_matrix(sigma_list)

        wassertein_distance = mean_distance +var_distance + 1e-6

        return wassertein_distance

    def distance_matrix(self,z,p=2):
        x = z.unsqueeze(2)
        y = z.unsqueeze(1)
        #print(x.size())
        distance = torch.sum((torch.abs(x-y))**p,dim=3)
        #print(distance.size())

        upper_mean_distance = torch.triu(distance[:,:,:],diagonal=1)
        upper_mean_distance = torch.mean(upper_mean_distance,dim=(1,2))
        upper_mean_distance = upper_mean_distance.mean()
        # print("upper_mean_distance:",upper_mean_distance)

        upper_mean_distance = torch.exp(- 0.1* upper_mean_distance)

        return upper_mean_distance
    #
    # def matrix_distance(self,X,device):
    #     #X = X.unsqueeze(0)
    #     #print(X)
    #     x = X.unsqueeze(1).expand(len(X),len(X),X.shape[1])
    #     y = X.unsqueeze(0).expand(len(X),len(X),X.shape[1])
    #     distance_matrix = ((x-y).pow(2).sum(2))#.pow(0.5)
    #     upper_mean_distance = torch.triu(distance_matrix[:, :], diagonal=1)
    #     # upper_mean_distance =  - torch.mean(upper_mean_distance)
    #     #upper_mean_distance = torch.exp(- upper_mean_distance)
    #     upper_mean_distance = torch.mean(upper_mean_distance)
    #     # upper_mean_distance = torch.log(upper_mean_distance + 1)
    #     upper_mean_distance = torch.exp(-0.1* upper_mean_distance)
    #     # upper_mean_distance = - upper_mean_distance
    #     # print(upper_mean_distance)
    #     return upper_mean_distance



    def meta_test(self, test_loader_cell, test_loader1_cell, test_loader_unpatient, test_loader1_unpatient,
                                    test_loader_target, test_loader1_target, cells_n, samples_n_c, total_preds_t, total_labels_t, device,args):

        old_params_de = parameters_to_vector(self.drug_features.parameters())
        old_params_cs = parameters_to_vector(self.cell_features_sem.parameters())
        old_params_ts = parameters_to_vector(self.task_features_sem.parameters())
        old_params_cv = parameters_to_vector(self.cell_features_var.parameters())
        old_params_lc = parameters_to_vector(self.label_classifier.parameters())
        old_params_bd = parameters_to_vector(self.back_door_adjust.parameters())

        for batch_idx, data_test in enumerate(zip(test_loader_cell, test_loader1_cell, test_loader_unpatient, test_loader1_unpatient)):
            data_cell = data_test[0].to(device)
            data_cell1 = data_test[1].to(device)
            data_unpa = data_test[2].to(device)
            data_unpa1 = data_test[3].to(device)
            self.inner_optimize(cells_n, samples_n_c, data_cell, data_cell1, data_unpa,data_unpa1,
                                args.num_inner_updates, device,args)


        loss_test_qry_all = []
        for batch_idx, data_test in enumerate(zip( test_loader_target, test_loader1_target)):
            data_target = data_test[0].to(device)
            data_target1 = data_test[1].to(device)
            with torch.no_grad():
                drug1_features, drug2_features = self.drug_features(data_target, data_target1)
                patient_features_sem = self.cell_features_sem(data_target)
                # embedding_de,kl,_,_ = self.features_sem(drug1_features, drug2_features,patient_features_sem)
                y_pre_pa,_ ,_= self.label_classifier(drug1_features, drug2_features,patient_features_sem)
                loss_test_qry = F.cross_entropy(y_pre_pa, data_target['y'].to(torch.int64).to(device))

                total_labels_t = torch.cat((total_labels_t, data_target['y'].view(-1, 1).float().cpu()), 0)
                total_preds_t = torch.cat((total_preds_t, F.softmax(y_pre_pa,dim=1)[:,1].cpu()), 0)
                loss_test_qry_all.append(loss_test_qry.item())

        vector_to_parameters(old_params_de, self.drug_features.parameters())
        vector_to_parameters(old_params_cs, self.cell_features_sem.parameters())

        vector_to_parameters(old_params_ts, self.task_features_sem.parameters())
        vector_to_parameters(old_params_cv, self.cell_features_var.parameters())

        vector_to_parameters(old_params_lc, self.label_classifier.parameters())

        vector_to_parameters(old_params_bd, self.back_door_adjust.parameters())
        #print(type(loss_test_qry_all))

        return sum(loss_test_qry_all)/len(loss_test_qry_all), total_preds_t, total_labels_t