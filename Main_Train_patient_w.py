import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from Model_Basic_10_raw_adam_c_9_3 import *

from Data_Load_new import *
from utils_comb import *
from utilities import *
import pickle
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score,  \
    matthews_corrcoef
torch.backends.cudnn.enabled=False

def train_mddan(meta,synergydata,args):

    mcc_best,step_best = 0, 0

    print("-------------------------begin train------------------------")
    for step in range(args.epochs+1):
        if args.object == 'pdx' or args.object == 'cell':
            loss_indis_dom, loss_id_s, loss_invar_id, loss_dom_v = meta(synergydata, drug_features_cell,cell_features, codes_cell,args.num_task,
                                                drug_features_unpatient,unpatient_features,codes_unpatient,drug_features_pdx, pdx_features,
                                                                                     codes_pdx, step, device, args)
        else:
            loss_indis_dom, loss_id_s, loss_invar_id, loss_dom_v = meta(synergydata, drug_features_cell,cell_features, codes_cell,args.num_task,
                                               drug_features_unpatient,unpatient_features,codes_unpatient,drug_features_patient, patient_features,
                                                                                     codes_patient, step, device, args)

        if step % args.inter == 0:
            save_statistics(experiment_nameT, [step, loss_indis_dom, loss_id_s, loss_invar_id,loss_dom_v])
            #val
            mcc_best,step_best =val_mddan(meta,synergydata,mcc_best,step_best,step,args)

def val_mddan(meta,synergydata,best_mcc,best_step,step,args):
    # print("--------------------begin val--------------------")
    accs = []
    auroc_patients = []
    aupr_patients = []
    bacc_patients = []
    mcc_patients = []

    for _ in range(args.num_rep_val // 1):

        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        # db_train.next('test')
        # print(i)

        x_support_set, x_support_unlabel_set, x_target = synergydata.get_val_batch(obj=args.object_val, augment=False)
        # print(x_support_set)
        # cell lines
        tissue_n, cell_n, samples_n, features_n = np.shape(x_support_set)
        samples_n_c = int(samples_n / args.num_task)
        # print(tissue_n, cell_n,samples_n_c,features_n)
        for i in range(tissue_n):
            x_support_set_ti = x_support_set[i]
            cell_n, samples_n, features_n = np.shape(x_support_set_ti)
            # print("x_support_set_ti.size()",np.shape(x_support_set_ti))
            x_support_set_ti = x_support_set_ti.reshape(cell_n, args.num_task, samples_n_c,
                                                        features_n)
            x_support_set_ti = x_support_set_ti.transpose(1, 0, 2, 3)
            x_support_set_ti = x_support_set_ti.reshape(args.num_task * cell_n * samples_n_c, features_n)
            test_data = TestbedDataset_cell(x_support_set_ti, drug_features_cell, cell_features, codes_cell)
            test_data1 = TestbedDataset1_cell(x_support_set_ti, drug_features_cell, cell_features, codes_cell)
            test_loader_cell = DataLoader(test_data, batch_size=cell_n * samples_n_c, shuffle=False)
            test_loader1_cell = DataLoader(test_data1, batch_size=cell_n * samples_n_c, shuffle=False)

            # unlabeled patient
            x_support_unlabel_set_ti = x_support_unlabel_set[i]
            samples_un, features_un = np.shape(x_support_unlabel_set_ti)
            x_support_unlabel_set_ti = np.expand_dims(x_support_unlabel_set_ti, 0).repeat(args.num_task, axis=0)
            x_support_unlabel_set_ti = x_support_unlabel_set_ti.reshape(args.num_task * samples_un, features_un)

            test_data = TestbedDataset_unpatient_T(x_support_unlabel_set_ti, drug_features_unpatient,unpatient_features, codes_unpatient)
            test_data1 = TestbedDataset1_unpatient_T(x_support_unlabel_set_ti, drug_features_unpatient,unpatient_features, codes_unpatient)
            test_loader_unpatient = DataLoader(test_data, batch_size=samples_un, shuffle=False)
            test_loader1_unpatient = DataLoader(test_data1, batch_size=samples_un, shuffle=False)

            #  val set
            if args.object_val == 'patient':
                x_target_ti = x_target[i]
                samples_n, features_n = np.shape(x_target_ti)
                test_data = TestbedDataset_patient_test(x_target_ti, drug_features_patient,
                                                        patient_features, codes_patient)
                test_data1 = TestbedDataset1_patient_test(x_target_ti, drug_features_patient,
                                                          patient_features, codes_patient)
                test_loader_target = DataLoader(test_data, batch_size=samples_n, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=samples_n, shuffle=False)

            elif args.object_val == 'pdx':
                x_target_ti = x_target[i]
                samples_n, features_n = np.shape(x_target_ti)
                test_data = TestbedDataset_pdx(x_target_ti, drug_features_pdx,
                                               pdx_features, codes_pdx)
                test_data1 = TestbedDataset1_pdx(x_target_ti, drug_features_pdx,
                                                 pdx_features, codes_pdx)
                test_loader_target = DataLoader(test_data, batch_size=samples_n, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=samples_n, shuffle=False)

            elif args.object_val == 'cell':
                x_target_ti = x_target[i]
                #samples_n_c, features_n = np.shape(x_target_ti)
                #x_target_ti = x_target_ti[i].reshape(cell_n * samples_n_c, features_n)
                test_data = TestbedDataset_cell(x_target_ti, drug_features_cell,
                                                cell_features, codes_cell)
                test_data1 = TestbedDataset1_cell(x_target_ti, drug_features_cell,
                                                  cell_features, codes_cell)
                test_loader_target = DataLoader(test_data, batch_size=args.qst, shuffle=False)
                test_loader1_target = DataLoader(test_data1, batch_size=args.qst, shuffle=False)

            loss_t, total_preds, total_labels = meta.meta_test(test_loader_cell, test_loader1_cell,
                                                               test_loader_unpatient, test_loader1_unpatient,
                                                               test_loader_target, test_loader1_target, cell_n,
                                                               samples_n_c, total_preds,
                                                               total_labels, device,args)
            accs.append(loss_t)

        G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
        auroc_patient = roc_auc_score(y_true=G, y_score=P)
        aupr_patient = average_precision_score(y_true=G, y_score=P)
        bacc_patient = BACC(label=G, pred=(P > 0.5).astype('int'))
        mcc_patient = matthews_corrcoef(G, (P > 0.5).astype('int'))

        auroc_patients.append(auroc_patient)
        aupr_patients.append(aupr_patient)
        bacc_patients.append(bacc_patient)
        mcc_patients.append(mcc_patient)

    accs = np.array(accs).mean(axis=0).astype(np.float16)

    auroc_patient = np.array(auroc_patients).mean(axis=0).astype(np.float16)
    aupr_patient = np.array(aupr_patients).mean(axis=0).astype(np.float16)
    bacc_patient = np.array(bacc_patients).mean(axis=0).astype(np.float16)
    mcc_patient = np.array(mcc_patients).mean(axis=0).astype(np.float16)

    print("val_cross entropy:%.4f" % accs, "val_auroc:%.4f" % auroc_patient, "val_aupr:%.4f" % aupr_patient,
          "val_bacc:%.4f" % bacc_patient, "val_mcc:%.4f" % mcc_patient)
    save_statistics(experiment_nameval, [step, accs, auroc_patient, aupr_patient, bacc_patient, mcc_patient])

    if mcc_patient > best_mcc:
        best_mcc = mcc_patient
        best_step = step
        if not os.path.exists(args.save_models):
            os.makedirs(args.save_models, exist_ok=False)
        save_path = args.save_models + "{}_{}_{}_{}_{}_w10.pth".format(args.object, best_step, best_mcc, tissue,
                                                                       args.seed)
        torch.save(meta.state_dict(), save_path)

    if (step == args.epochs):
        if not os.path.exists(args.save_models):
            os.makedirs(args.save_models, exist_ok=False)
        save_path = args.save_models + "{}_{}_{}_{}_{}_w10.pth".format(args.object, step, mcc_patient, tissue,
                                                                       args.seed)
        torch.save(meta.state_dict(), save_path)

    return best_mcc,best_step

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_logs", type=str, default='logs_patient_w/', help='path of folder to write log')
    parser.add_argument("--save_models", type=str, default='saved_models_patient_w/', help='folder for saving model')
    parser.add_argument("--save_results", type=str, default='results_patient/', help='folder for saving results')
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
    parser.add_argument('--lr_d_a', type=float, default=0.0005,
                        help='Learning rate of features_var for meta-learning update')
    parser.add_argument('--lr_t_a', type=float, default=0.0005,
                        help='Learning rate of task_features_sem for meta-learning update')

    parser.add_argument('--lr_c_s', type=float, default=0.9,
                        help='Learning rate of classifer for meta-learning update')
    parser.add_argument('--lr_l_s', type=float, default=0.5, help='Learning rate of feature_sem for meta-learning update')
    parser.add_argument('--lr_d_s', type=float, default=0.5,
                        help='Learning rate of features_var for meta-learning update')
    parser.add_argument('--lr_t_s', type=float, default=0.5,
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
    parser.add_argument('--gpu', type=int, default=2, help='set using GPU ')
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

    parser.add_argument("--object", type=str, default='patient', help='the object of test')
    parser.add_argument("--object_val", type=str, default='cell', help='the object of val')
    parser.add_argument('--seed', type=int, default=24, help='Random seed.')  #
    parser.add_argument('--optimize', type=str, default="ADAM", help='optimizor.')

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
    # read features
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
    #train model
    for tissue in ["OVA","BRE","LUN","BRA","COL"]:
        # read dataset
        synergydata = MiniSynergyDataSet(args.meta_batch_size, args.tnt, args.cnt, args.ssn,
                                         args.usn, args.qsn, args.cntq, args.sqn,tissue_t=tissue, seed = args.seed)
        _, _, x_target = synergydata.get_test_batch(obj=args.object,augment=False)  # test samples
        print("test_shape:",np.shape(np.array(x_target)))

        logs_path = 'log_outputs/'
        experiment_nameT = f'CaMeRe_comb_Train_patient'
        experiment_nameval = f'CaMeRe_comb_VAL_patient'

        logs = "Tissue:" + str(tissue) + str(args)

        save_statistics(experiment_nameT, ["Experimental details: {}".format(logs)])

        save_statistics(experiment_nameT,
                        ["epoch", "loss_indis_dom", "loss_id_s", "loss_invar_id", "loss_dom_v"])
        save_statistics(experiment_nameval, ["Experimental details: {}".format(logs)])
        save_statistics(experiment_nameval, ["epoch", "LOSS", "auroc", "aupr", "bacc", "mcc"])

        #train
        model = CaMeRe(args).to(device)
        train_mddan(model,synergydata,args)
