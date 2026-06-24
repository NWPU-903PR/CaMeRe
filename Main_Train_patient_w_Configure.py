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
    parser.add_argument("--config", type=str, default="config_patient.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    cli_args = parser.parse_args()
    args, raw_cfg = load_config(cli_args.config)
    if cli_args.gpu is not None:
        args.gpu = cli_args.gpu

    if cli_args.seed is not None:
        args.seed = cli_args.seed

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
    for tissue in raw_cfg["tissues"]["default"]:
        # read dataset
        synergydata = MiniSynergyDataSet(args.meta_batch_size, args.tnt, args.cnt, args.ssn,
                                         args.usn, args.qsn, args.cntq, args.sqn,tissue_t=tissue, seed = args.seed)
        _, _, x_target = synergydata.get_test_batch(obj=args.object,augment=False)  # test samples

        print("tissue:", tissue)
        print("test_shape:", np.shape(np.array(x_target)))

        if tissue in raw_cfg.get("tissue_specific", {}):
            tissue_cfg = raw_cfg["tissue_specific"][tissue]

            for key, value in tissue_cfg.items():
                setattr(args, key, value)

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
