import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from Model_Basic_10_raw_adam_c_9_3_test import *


from Data_Load_new import *
from utils_comb import *
from utilities import *
import pickle
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score,  \
    matthews_corrcoef
torch.backends.cudnn.enabled=False



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

        x_support_set, x_support_unlabel_set, x_target = synergydata.get_test_batch(obj=args.object, augment=False)  #
        # print(x_support_set)
        # cell lines
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


            # unlabeled patient
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
    #
            G, P = save_list[3].numpy().flatten(), save_list[4].numpy().flatten()
    #

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

            print( "test_auroc:%.4f" % auroc_patient, "test_aupr:%.4f" % aupr_patient,
                  "test_bacc:%.4f" % bacc_patient, "test_mcc:%.4f" % mcc_patient)
            save_statistics(experiment_nameTE, [num,  auroc_patient, aupr_patient, bacc_patient, mcc_patient])

    auroc_patient = np.array(auroc_patients).mean(axis=0).astype(np.float16)
    aupr_patient = np.array(aupr_patients).mean(axis=0).astype(np.float16)
    bacc_patient = np.array(bacc_patients).mean(axis=0).astype(np.float16)
    mcc_patient = np.array(mcc_patients).mean(axis=0).astype(np.float16)


    save_statistics(experiment_nameTE,
                    ["AVE",  auroc_patient, aupr_patient, bacc_patient, mcc_patient])


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

    #
    for tissue in raw_cfg["tissues"]["default"]:
        #
        # read date
        synergydata = MiniSynergyDataSet(args.meta_batch_size, args.tnt, args.cnt, args.ssn,
                                         args.usn, args.qsn, args.cntq, args.sqn,tissue_t=tissue,seed = args.seed)
        _, _, x_target = synergydata.get_test_batch(obj=args.object,augment=False)  # test samples

        print("tissue:", tissue)
        print("test_shape:", np.shape(np.array(x_target)))

        #test
        if tissue in raw_cfg.get("tissue_specific", {}):
            tissue_cfg = raw_cfg["tissue_specific"][tissue]

            for key, value in tissue_cfg.items():
                setattr(args, key, value)

        if tissue == "OVA":
            step_best, mcc_best = 1495, 0.334228515625

        logs_path = 'log_outputs/'
        experiment_nameTE = f'CaMeRe_comb_TEST_10_patient'
        logs = "Tissue:" + str(tissue) + str(args)
        save_statistics(experiment_nameTE, ["Experimental details: {}".format(logs)])
        save_statistics(experiment_nameTE, ["num",  "AUROC", "AUPR", "BACC","MCC"])

        save_path = args.save_models + "{}_{}_{}_{}_{}_w10.pth".format(args.object, step_best, mcc_best, tissue,args.seed)

        model = CaMeRe(args, drop_d=0, drop_l=0).to(device)
        model.load_state_dict(torch.load(save_path))
        test_mddan(model, synergydata, args)

