import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

from Model_Basic_10_raw_adam_c_9_3_test import *
from Data_Load_new_P import *
from utils_comb import *
from utilities import *
import pickle
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score, \
    matthews_corrcoef
torch.backends.cudnn.enabled=False
# def sgd(parameters, lr, weight_decay=0.00005, momentum =0.9):
#     opt = optim.SGD(params= parameters,lr= lr, momentum= momentum, weight_decay= weight_decay)
#     return opt
#
def test_mddan(meta,synergydata,args):

    # print("--------------------begin test--------------------")

    auroc_patients = []
    aupr_patients = []
    bacc_patients = []
    mcc_patients = []

    for num in range(args.num_rep):

        cell_feature_pre_save = torch.Tensor().to(device)
        cell_back_feature_pre_save = torch.Tensor().to(device)
        patient_feature_pre_save = torch.Tensor().to(device)
        patient_back_feature_pre_save = torch.Tensor().to(device)
        drug_comb_obj_sem= torch.Tensor()
        obj_var = torch.Tensor()
        total_label = torch.Tensor()
        total_pred = torch.Tensor()

        save_list = [cell_feature_pre_save,cell_back_feature_pre_save, patient_feature_pre_save,  total_label, total_pred,drug_comb_obj_sem,obj_var,patient_back_feature_pre_save]

        x_support_set, x_support_unlabel_set, x_target = synergydata.get_test_batch(obj=args.object, augment=False)  # 产生样本
        # print(x_support_set)
        # cell line
        tissue_n, cell_n, samples_n, features_n = np.shape(x_support_set)
        samples_n_c = int(samples_n / args.num_task)
        # print(tissue_n, cell_n,samples_n_c,features_n)
        for i in range(tissue_n):

            x_support_set_ti_1 = x_support_set[i]
            cell_n, samples_n, features_n = np.shape(x_support_set_ti_1)
            # print("x_support_set_ti.size()",np.shape(x_support_set_ti))
            x_support_set_ti_1 = x_support_set_ti_1.reshape(cell_n, args.num_task, samples_n_c,
                                                        features_n)
            x_support_set_ti = x_support_set_ti_1.transpose(1, 0, 2, 3)
            x_support_set_ti = x_support_set_ti.reshape(args.num_task * cell_n * samples_n_c, features_n)
            test_data = TestbedDataset_cell(x_support_set_ti, drug_features_cell, cell_features, codes_cell)
            test_data1 = TestbedDataset1_cell(x_support_set_ti, drug_features_cell, cell_features, codes_cell)
            test_loader_cell = DataLoader(test_data, batch_size=cell_n * samples_n_c, shuffle=False)
            test_loader1_cell = DataLoader(test_data1, batch_size=cell_n * samples_n_c, shuffle=False)
            test_loader_cell_p = DataLoader(test_data, batch_size=args.num_task * cell_n * samples_n_c,shuffle=False)
            test_loader1_cell_p = DataLoader(test_data1, batch_size=args.num_task * cell_n * samples_n_c, shuffle=False)
            #
            # unlabel patient
            x_support_unlabel_set_ti_1 = x_support_unlabel_set[i]
            samples_un, features_un = np.shape(x_support_unlabel_set_ti_1)
            x_support_unlabel_set_ti = np.expand_dims(x_support_unlabel_set_ti_1, 0).repeat(args.num_task, axis=0)
            x_support_unlabel_set_ti = x_support_unlabel_set_ti.reshape(args.num_task * samples_un, features_un)

            test_data = TestbedDataset_unpatient_T(x_support_unlabel_set_ti, drug_features_unpatient,unpatient_features, codes_unpatient)
            test_data1 = TestbedDataset1_unpatient_T(x_support_unlabel_set_ti, drug_features_unpatient,unpatient_features, codes_unpatient)
            test_loader_unpatient = DataLoader(test_data, batch_size=samples_un, shuffle=False)
            test_loader1_unpatient = DataLoader(test_data1, batch_size=samples_un, shuffle=False)

            # test set
            if args.object == 'patient':
                x_target_ti = x_target[i]
                samples_n, features_n = np.shape(x_target_ti)
                test_data = TestbedDataset_patient_test(x_target_ti, drug_features_patient,
                                                        patient_features, codes_patient)
                test_data1 = TestbedDataset1_patient_test(x_target_ti, drug_features_patient,
                                                          patient_features, codes_patient)
                test_loader_target = DataLoader(test_data, batch_size=samples_n, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=samples_n, shuffle=False)

            elif args.object == 'pdx':
                x_target_ti = x_target[i]
                samples_n, features_n = np.shape(x_target_ti)
                test_data = TestbedDataset_pdx(x_target_ti, drug_features_pdx,pdx_features, codes_pdx)
                test_data1 = TestbedDataset1_pdx(x_target_ti, drug_features_pdx,pdx_features, codes_pdx)
                test_loader_target = DataLoader(test_data, batch_size=samples_n, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=samples_n, shuffle=False)

            elif args.object == 'cell':
                x_target_ti = x_target[i]
                #samples_n_c, features_n = np.shape(x_target_ti)
                #x_target_ti = x_target_ti[i].reshape(cell_n * samples_n_c, features_n)
                test_data = TestbedDataset_cell(x_target_ti, drug_features_cell,
                                                cell_features, codes_cell)
                test_data1 = TestbedDataset1_cell(x_target_ti, drug_features_cell,
                                                  cell_features, codes_cell)
                test_loader_target = DataLoader(test_data, batch_size=args.qst, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=args.qst, shuffle=False)

                #
            save_list = meta.meta_test(x_support_set_ti_1,x_target_ti,x_support_unlabel_set_ti_1,test_loader_cell, test_loader1_cell, test_loader_unpatient, test_loader1_unpatient,
                    test_loader_target, test_loader1_target,test_loader_cell_p,test_loader1_cell_p, cell_n,samples_n_c, save_list, device,args)

            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue)+"_"+args.object  + "_10_cell_feature_label.txt",
                       save_list[0].cpu().detach().numpy(),delimiter=",")
            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue) + "_" + args.object + "_10_cell_backdoor_feature_label.txt",
                       save_list[1].cpu().detach().numpy(), delimiter=",")
            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue) +"_"+args.object+ "_10_patient_feature_label.txt",
                       save_list[2],delimiter=",", fmt="%s")
            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue) + "_" + args.object + "_10_patient_backdoor_feature_label.txt",
                save_list[7], delimiter=",", fmt="%s")

            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue)+"_"+args.object + "_10_drug_combination_object_sem_embedding.txt",
                       save_list[5],delimiter=",", fmt="%s")
            np.savetxt(args.save_results +str(args.seed)+ "_"+ str(num) + "_" + str(tissue) + "_" + args.object + "_10_object_var_embedding.txt",
                       save_list[6], delimiter=",", fmt="%s")

            G, P = save_list[3].numpy().flatten(), save_list[4].numpy().flatten()

            np.savetxt(args.save_results + "label_camere_10_" +str(args.seed)+ "_" + args.object + "_" + str(num) + "_" + str(tissue) + ".txt", G,
                       delimiter=",")
            np.savetxt(args.save_results + "predict_camere_10_" +str(args.seed)+ "_" + args.object + "_" + str(num) + "_" + str(tissue) + ".txt", P,
                       delimiter=",")

            auroc_patient = roc_auc_score(y_true=G, y_score=P)
            aupr_patient = average_precision_score(y_true=G, y_score=P)
            bacc_patient = BACC(label=G, pred=(P > 0.5).astype('int'))
            mcc_patient = matthews_corrcoef(G, (P > 0.5).astype('int'))

            auroc_patients.append(auroc_patient)
            aupr_patients.append(aupr_patient)
            bacc_patients.append(bacc_patient)
            mcc_patients.append(mcc_patient)

            print("test_auroc:%.4f" % auroc_patient, "test_aupr:%.4f" % aupr_patient,
                  "test_bacc:%.4f" % bacc_patient, "test_mcc:%.4f" % mcc_patient)
            save_statistics(experiment_nameTE, [num, auroc_patient, aupr_patient, bacc_patient, mcc_patient])

        auroc_patient = np.array(auroc_patients).mean(axis=0).astype(np.float16)
        aupr_patient = np.array(aupr_patients).mean(axis=0).astype(np.float16)
        bacc_patient = np.array(bacc_patients).mean(axis=0).astype(np.float16)
        mcc_patient = np.array(mcc_patients).mean(axis=0).astype(np.float16)

        save_statistics(experiment_nameTE,
                        ["AVE", auroc_patient, aupr_patient, bacc_patient, mcc_patient])

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_logs", type=str, default='logs_pdx_w/', help='path of folder to write log')
    parser.add_argument("--save_models", type=str, default='saved_models_pdx_w/', help='folder for saving model')
    parser.add_argument("--save_results", type=str, default='results_pdx/', help='folder for saving result')
    parser.add_argument('--fpath', type=str, default='data/feature/', help='Feature folder')

    parser.add_argument('--meta_batch_size', type=int, default=20,
                        help='Meta-learning batch size, i.e. how many different tasks need to be sampled')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
    # parser.add_argument('--step_size', type=int, default=1500 * 100, help='Number of learning rate decay')

    parser.add_argument('--num_inner_updates', type=int, default=1, help='the number of inner optimize')
    # parser.add_argument("--wd", type=float, default=0.005, help='weight decay')
    parser.add_argument("--inter", type=int, default=5, help='number of ites showing trainging results')
    parser.add_argument("--inter_val", type=int, default=5, help='number of ites showing valing results')

    parser.add_argument('--lr_c_a', type=float, default=0.001,
                        help='Learning rate of classifer for meta-learning update')
    parser.add_argument('--lr_l_a', type=float, default=0.001,
                        help='Learning rate of feature_sem for meta-learning update')
    parser.add_argument('--lr_d_a', type=float, default=0.001,
                        help='Learning rate of features_var for meta-learning update')
    parser.add_argument('--lr_t_a', type=float, default=0.0005,
                        help='Learning rate of task_features_sem for meta-learning update')

    parser.add_argument('--lr_c_s', type=float, default=1,
                        help='Learning rate of classifer for meta-learning update')
    parser.add_argument('--lr_l_s', type=float, default=1, help='Learning rate of feature_sem for meta-learning update')
    parser.add_argument('--lr_d_s', type=float, default=1,
                        help='Learning rate of features_var for meta-learning update')
    parser.add_argument('--lr_t_s', type=float, default=1,
                        help='Learning rate of task_features_sem for meta-learning update')

    # parser.add_argument('--lr_c', type=float, default=0.001, help='Learning rate of classifer for meta-learning update')
    # parser.add_argument('--mc', type=float, default=0.95, help='momentum of classifer for meta-learning update')
    parser.add_argument('--wdc', type=float, default=0.00005, help='weight decay of classifer for meta-learning update')
    # parser.add_argument('--lr_l', type=float, default=0.001, help='Learning rate of feature_sem for meta-learning update')
    # parser.add_argument('--ml', type=float, default=0.95, help='momentum of feature_sem for meta-learning update')
    parser.add_argument('--wdl', type=float, default=0.00005,
                        help='weight decay of feature_sem for meta-learning update')
    # parser.add_argument('--lr_d', type=float, default=0.001, help='Learning rate of features_var for meta-learning update')
    # parser.add_argument('--md', type=float, default=0.95, help='momentum of features_var for meta-learning update')
    parser.add_argument('--wdd', type=float, default=0.00005,
                        help='weight decay of features_var for meta-learning update')

    parser.add_argument('--inner_lr', type=float, default=0.000078, help='Learning rate for inner optimization')
    parser.add_argument('--milestones', type=list, default=[100, 600, 1200],
                        help='milestones of learning rate for meta-learning update')
    parser.add_argument('--lamda_pa', type=list,
                        default=[1, 0.5, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 0.5, 1, 1, 1],
                        help='weight coefficient of different loss in meta-learning update')
    parser.add_argument('--lamda_task', type=float, default=0.5, help='weight coefficient of auxiliary task')

    parser.add_argument('--lamda_dis', type=float, default=0.1, help='weight coefficient in Wassertein distance')
    parser.add_argument('--lamda_kl', type=float, default=0.001, help='weight coefficient in KL')
    parser.add_argument('--lamda_kl_s', type=float, default=0.0001, help='weight coefficient in KL')
    parser.add_argument('--lamda_kl_s1', type=float, default=0.0001, help='weight coefficient in KL')
    parser.add_argument('--batnorm', choices=["true", "false"], help='RELU')

    parser.add_argument('--mean_backdoor', type=int, default=8,
                        help='the number of samples in backdoor sampling')
    parser.add_argument('--gpu', type=int, default=1, help='set using GPU ')
    parser.add_argument("--tnt", type=int, default=1, help='the number of tissue in val/test set')

    parser.add_argument("--cnt", type=int, default=3, help='the number of cell lines in support set')
    parser.add_argument("--ssn", type=int, default=100, help='the number of support samples in each cell lines task')
    parser.add_argument("--num_task", type=int, default=5, help='the number of group in each cell lines')

    parser.add_argument("--cntq", type=int, default=2, help='the number of cell lines in query set')
    parser.add_argument("--sqn", type=int, default=50, help='the number of query samples in each cell lines task')

    parser.add_argument("--usn", type=int, default=6, help='the number of unlabeled patients samples in each task')
    parser.add_argument("--qsn", type=int, default=10, help='the number of query samples in each task')
    parser.add_argument("--qst", type=int, default=1000, help='the number of samples in each val cell lines')
    parser.add_argument("--num_rep", type=int, default=10, help='the number for repeat test')
    parser.add_argument("--num_rep_val", type=int, default=1, help='the number for repeat val')

    parser.add_argument("--object", type=str, default='pdx', help='the object of test')
    parser.add_argument("--object_val", type=str, default='cell', help='the object of test')
    parser.add_argument('--seed', type=int, default=24, help='Random seed.')
    parser.add_argument('--optimize', type=str, default="ADAM", help='optimizer')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    paths = [args.save_logs, args.save_models, args.save_results]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path + '/args.txt', 'a') as f:
            f.write('------------*******--------------\n')
            f.write(str(args))
    #read feature
    # cell lines
    codes_cell = pickle.load(open(args.fpath + 'codes_cell_comb.p', 'rb'))
    drug_features_cell = pickle.load(open(args.fpath + 'drug_feature_cell_comb.p', 'rb'))
    cell_features = pickle.load(open(args.fpath + 'cell_feature_900_comb.p', 'rb'))
    # pdx
    codes_pdx = pickle.load(open(args.fpath + 'codes_PDX.p', 'rb'))
    drug_features_pdx = pickle.load(open(args.fpath + 'drug_feature_PDX.p', 'rb'))
    pdx_features = pickle.load(open(args.fpath + 'PDX_feature_900.p', 'rb'))
    # patient
    codes_patient = pickle.load(open(args.fpath + 'codes_patient.p', 'rb'))
    drug_features_patient = pickle.load(open(args.fpath + 'drug_feature_patient.p', 'rb'))
    patient_features = pickle.load(open(args.fpath + 'patient_feature_900.p', 'rb'))
    # patient_unlabel
    codes_unpatient = pickle.load(open(args.fpath + 'codes_patient_label_unlabel.p', 'rb'))
    drug_features_unpatient = pickle.load(open(args.fpath + 'drug_feature_patient_label_unlabel.p', 'rb'))
    unpatient_features = pickle.load(open(args.fpath + 'patient_feature_900_label_unlabel.p', 'rb'))

    #model train
    for tissue in ["SKI","BRE","LUN","COL"]:
        # read dataset
        synergydata = MiniSynergyDataSet(args.meta_batch_size, args.tnt, args.cnt, args.ssn,
                                         args.usn, args.qsn, args.cntq, args.sqn,tissue_t=tissue,seed = args.seed)
        _, _, x_target = synergydata.get_test_batch(obj=args.object,augment=False)  # test sample
        print("test_shape:",np.shape(np.array(x_target)))

        # test
        if tissue == "COL":
            step_best, mcc_best = 1160, 0.2486572265625

        elif tissue == "SKI":
            step_best, mcc_best = 1425, 0.2498779296875

        elif tissue == "BRE":
            step_best, mcc_best = 1425, 0.267333984375

        elif tissue == "LUN":
            step_best, mcc_best = 1480, 0.250244140625

        # model
        logs_path = 'log_outputs/'
        experiment_nameTE = f'CaMeRe_comb_TEST_10_pdx'
        logs = "Tissue:" + str(tissue) + str(args)
        save_statistics(experiment_nameTE, ["Experimental details: {}".format(logs)])
        save_statistics(experiment_nameTE, ["num", "AUROC", "AUPR", "BACC", "MCC"])

        model = CaMeRe(args, drop_d=0, drop_l=0).to(device)

        save_path = args.save_models + "{}_{}_{}_{}_{}_w10.pth".format(args.object, step_best, mcc_best, tissue,args.seed)
        model.load_state_dict(torch.load(save_path))
        test_mddan(model, synergydata, args)

