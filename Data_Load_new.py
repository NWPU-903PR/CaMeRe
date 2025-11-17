import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

def Load_Synergydata(tissue_t,seed):
    np.random.seed(seed)
    print("Loading Samples...")
    txt_files_dir='./data/sample_patient'
    sample_files_n = os.listdir(txt_files_dir)
    files_ids = np.arange(len(sample_files_n))
    np.random.shuffle(files_ids)
    sample_files = []
    for x in files_ids:
        sample_files.append(sample_files_n[x])
    test_files = [tissue_t]
    print("Test_tissue:",test_files)
    train_files = filter(lambda x: x not in test_files,sample_files)
    def data_loader(files):

        data_list_all =[]
        for file in files:
            objs_path = os.path.join(txt_files_dir,file)
            objs = os.listdir(objs_path)
            data_list_tissue= {'cell':[],'patient':[],'pdx':[],'un_patient':[]}
            for obj in objs:
                all_files = glob.glob(os.path.join(os.path.join(objs_path,obj),"*txt"))
                data_list = []
                for f in tqdm(all_files):
                    data = pd.read_table(f, header=0,sep='\t')
                    data_list.append(data)
                if obj == 'cell_comb':
                    data_list_tissue['cell'].extend(data_list)
                elif obj == 'patient':
                    data_list_tissue['patient'].extend(data_list)
                elif obj == 'pdx':
                    data_list_tissue['pdx'].extend(data_list)
                elif obj == 'un_patient':
                    data_list_tissue['un_patient'].extend(data_list)
            data_list_all.append(data_list_tissue)
        return data_list_all
    train = data_loader(train_files) #
    test = data_loader(test_files)
    print(len(train),len(test))

    return train, test

class MiniSynergyDataSet():
    def __init__(self, batch_size, tissue_class, cell_number, samples_support_label, samples_support_unlabel,
                 samples_query,cell_number_query, samples_query_label,tissue_t, seed=2000):

        self.seed = seed
        self.tissue_t = tissue_t

        self.train, self.test = Load_Synergydata(self.tissue_t, self.seed)
        self.batch_size = batch_size
        self.tissue_class = tissue_class
        self.cell_number = cell_number
        self.samples_support_label = samples_support_label
        self.samples_support_unlabel = samples_support_unlabel

        self.samples_query = samples_query

        self.cell_number_query = cell_number_query
        self.samples_query_label = samples_query_label

        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.train, "test": self.test}

    def __len__(self):
        return len(self.train)

    def preprocess_batch(self, x_batch):  #

        x_batch = x_batch
        return x_batch

    def sample_new_batch_train(self, data_pack,obj):
        """
        Collect batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return:  A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_samples_labeled = np.zeros((self.batch_size, self.cell_number, self.samples_support_label, 6),
                                           dtype=np.float64)
        support_samples_unlabeled = []
        query_samples = []
        query_samples_labeled = np.zeros((self.batch_size, self.cell_number_query, self.samples_query_label, 6),
                                           dtype=np.float64)

        for i in range(self.batch_size):
            # Each idx in batch contains a task
            tissue_class = np.arange(len(data_pack))
            choose_tissue = np.random.choice(tissue_class, size=self.tissue_class, replace=False)

            support_label = []
            support_unlabel = []
            query_label = []
            query_label_cell = []
            for x in choose_tissue:
                # unlabeled patient
                patient_unlabeled = data_pack[x].get('un_patient')
                sample_patient_unlabel_index = np.arange(len(patient_unlabeled))
                choose_patient = np.random.choice(sample_patient_unlabel_index, size=self.samples_support_unlabel,
                                                  replace=False)
                patient_unlabeled_samples = []
                for patient_unlabel in choose_patient:
                    samples_idx = np.arange(np.array(patient_unlabeled[patient_unlabel]).shape[0])
                    choose_samples = np.random.choice(samples_idx, size=1, replace=False)
                    patient_unlabeled_samples.extend(np.array(patient_unlabeled[patient_unlabel])[choose_samples])
                support_unlabel.append(patient_unlabeled_samples)

                #cell lines
                cell_lines = data_pack[x].get('cell')
                sample_cell_index = np.arange(len(cell_lines))
                choose_cell = np.random.choice(sample_cell_index, size=self.cell_number, replace=False)
                cell_samples = []
                for cell in choose_cell:
                    samples_idx = np.arange(np.array(cell_lines[cell]).shape[0])
                    choose_samples = np.random.choice(samples_idx, size=self.samples_support_label, replace=False)
                    # print(choose_samples)
                    cell_samples.append(np.array(cell_lines[cell])[choose_samples])
                support_label.append(cell_samples)

                cell_lines_query = data_pack[x].get('cell')

                choose_patient = np.random.choice(list(set(sample_cell_index).difference(set(choose_cell))),
                                                  size=self.cell_number_query, replace=False)
                cell_labeled_samples = []
                for patient_label in choose_patient:
                    samples_idx = np.arange(np.array(cell_lines_query[patient_label]).shape[0])
                    choose_samples = np.random.choice(samples_idx, size=self.samples_query_label, replace=False)  # 1
                    # print(choose_samples)
                    cell_labeled_samples.append(np.array(cell_lines_query[patient_label])[choose_samples])

                query_label_cell.append(cell_labeled_samples)

                patient_labeled = data_pack[x].get(str(obj))#'patient','pdx'
                sample_patient_label_index = np.arange(len(patient_labeled))
                choose_patient = np.random.choice(sample_patient_label_index, size=self.samples_query, replace=False)

                #print(choose_patient)
                patient_labeled_samples = []
                for patient_label in choose_patient:
                    samples_idx = np.arange(np.array(patient_labeled[patient_label]).shape[0])
                    choose_samples = np.random.choice(samples_idx, size=1, replace=False)#1
                    #print(choose_samples)
                    patient_labeled_samples.extend(np.array(patient_labeled[patient_label])[choose_samples])
                query_label.append(patient_labeled_samples)

            support_samples_labeled[i] = np.array(support_label)
            support_samples_unlabeled.append(np.array(support_unlabel).squeeze())
            query_samples.append(np.array(query_label).squeeze())
            query_samples_labeled[i] = np.array(query_label_cell)

        return np.array(support_samples_labeled), np.array(support_samples_unlabeled),np.array(query_samples_labeled), np.array(query_samples)

    def sample_new_batch_test(self, data_pack, obj):
        #np.random.seed(self.seed)
        tissue_class = len(data_pack)
        support_samples_labeled = np.zeros((tissue_class, self.cell_number, self.samples_support_label, 6),
                                           dtype=np.float64)
        support_samples_unlabeled = []
        query_samples = []

        for i in range(tissue_class):

            #un_patient
            patient_unlabeled = data_pack[i].get('un_patient')
            sample_patient_unlabel_index = np.arange(len(patient_unlabeled))
            choose_patient = np.random.choice(sample_patient_unlabel_index, size=self.samples_support_unlabel,
                                              replace=False)
            patient_unlabeled_samples = []
            for patient_unlabel in choose_patient:
                samples_idx = np.arange(np.array(patient_unlabeled[patient_unlabel]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=1, replace=False)
                patient_unlabeled_samples.extend(np.array(patient_unlabeled[patient_unlabel])[choose_samples])
            support_samples_unlabeled.append(patient_unlabeled_samples)
            #cell
            cell_lines = data_pack[i].get('cell')
            sample_cell_index = np.arange(len(cell_lines))
            choose_cell = np.random.choice(sample_cell_index, size=self.cell_number, replace=False)
            cell_samples = []
            for cell in choose_cell:
                samples_idx = np.arange(np.array(cell_lines[cell]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=self.samples_support_label, replace=False)
                cell_samples.append(np.array(cell_lines[cell])[choose_samples])
            support_samples_labeled[i] = np.array(cell_samples)

            # obj
            if obj=='cell':
                patient_labeled = data_pack[i].get(str(obj))
                choose_patient = list(set(sample_cell_index).difference(set(choose_cell)))
            else:
                patient_labeled = data_pack[i].get(str(obj))
                sample_patient_label_index = np.arange(len(patient_labeled))
                choose_patient = np.random.choice(sample_patient_label_index, size=len(patient_labeled), replace=False)

            patient_labeled_samples = []
            for patient_label in choose_patient:
                samples_idx = np.arange(np.array(patient_labeled[patient_label]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=len(samples_idx), replace=False)
                patient_labeled_samples.extend(np.array(patient_labeled[patient_label])[choose_samples])
            query_samples.append(patient_labeled_samples)

        return np.array(support_samples_labeled), np.array(support_samples_unlabeled), np.array(query_samples)

    def sample_new_batch_val(self, data_pack, obj):
        #np.random.seed(self.seed)
        #print(data_pack)
        tissue_class = len(data_pack)

        support_samples_labeled = np.zeros((tissue_class, self.cell_number, self.samples_support_label, 6),dtype=np.float64)
        support_samples_unlabeled = []
        query_samples = []

        #for _ in range(1):
        for i in range(tissue_class):

            #samples_idx = np.arange(tissue_class)
            #print(samples_idx)
            #i = np.random.choice(samples_idx, size=1, replace=False)[0]
            #print(i)
            #un_patient
            patient_unlabeled = data_pack[i].get('un_patient')
            sample_patient_unlabel_index = np.arange(len(patient_unlabeled))
            choose_patient = np.random.choice(sample_patient_unlabel_index, size=self.samples_support_unlabel,
                                              replace=False)
            patient_unlabeled_samples = []
            for patient_unlabel in choose_patient:
                samples_idx = np.arange(np.array(patient_unlabeled[patient_unlabel]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=1, replace=False)
                patient_unlabeled_samples.extend(np.array(patient_unlabeled[patient_unlabel])[choose_samples])
            support_samples_unlabeled.append(patient_unlabeled_samples)
            #cell
            cell_lines = data_pack[i].get('cell')
            sample_cell_index = np.arange(len(cell_lines))
            choose_cell = np.random.choice(sample_cell_index, size=self.cell_number, replace=False)
            cell_samples = []
            for cell in choose_cell:
                samples_idx = np.arange(np.array(cell_lines[cell]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=self.samples_support_label, replace=False)
                cell_samples.append(np.array(cell_lines[cell])[choose_samples])
            support_samples_labeled[i] = np.array(cell_samples)

            # obj
            if obj=='cell':
                patient_labeled = data_pack[i].get(str(obj))
                choose_patient = list(set(sample_cell_index).difference(set(choose_cell)))
            else:
                patient_labeled = data_pack[i].get(str(obj))
                sample_patient_label_index = np.arange(len(patient_labeled))
                choose_patient = np.random.choice(sample_patient_label_index, size=len(patient_labeled), replace=False)

            patient_labeled_samples = []
            for patient_label in choose_patient:
                samples_idx = np.arange(np.array(patient_labeled[patient_label]).shape[0])
                choose_samples = np.random.choice(samples_idx, size=len(samples_idx), replace=False)
                patient_labeled_samples.extend(np.array(patient_labeled[patient_label])[choose_samples])
            query_samples.append(patient_labeled_samples)

        return np.array(support_samples_labeled), np.array(support_samples_unlabeled), np.array(query_samples)

    def get_batch(self, dataset_name, obj, augment=False):  #
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        # x_support_set, x_support_unlabel_set,x_target_c, x_target = [], [],[], []
        if dataset_name == 'train':
            x_support_set, x_support_unlabel_set,x_target_c, x_target = self.sample_new_batch_train(self.datasets[dataset_name],obj)
            return x_support_set, x_support_unlabel_set,x_target_c, x_target
        elif dataset_name == 'test':
            x_support_set, x_support_unlabel_set, x_target = self.sample_new_batch_test(self.datasets[dataset_name], obj)
            return x_support_set, x_support_unlabel_set, x_target
        elif dataset_name == 'val':
            x_support_set, x_support_unlabel_set, x_target = self.sample_new_batch_val(self.datasets['test'],obj)
            return x_support_set, x_support_unlabel_set, x_target


    def get_train_batch(self,obj, augment=False):  #

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train",obj, augment)

    def get_test_batch(self, obj, augment=False):  #

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", obj, augment)
    def get_val_batch(self, obj, augment=False):  #

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("val", obj, augment)