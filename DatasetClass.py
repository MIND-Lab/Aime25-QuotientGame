import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import pickle


class Dataset:
    def __init__(self, spectra, x_axis, raw, user, name, category, labels,
                 assumed_drugs=[], label_dictionary={'cov': 0, 'covNeg': 1, 'ctrl': 2}):
        self.spectra = spectra
        self.spectra_to_numpy()
        self.x_axis = x_axis
        self.x_axis_to_numpy()
        self._raw = raw
        self.user = user
        self._name = name
        self.category = category
        self.assumed_drugs = np.array(assumed_drugs)
        self._n_elements = len(spectra)
        if labels.size != 0:
            self.labels = labels
        else:
            self.create_label(label_dictionary)
        self.n_dims = self.spectra.shape[1]

    def __getitem__(self, items):
        try:
            return self.spectra[items], self.x_axis[items], self._raw[items], self.user[items], self._name[items], \
                   self.category[items], self.labels[items], self.assumed_drugs[items]
        except IndexError:
            return self.spectra[items], self.x_axis[items], self._raw[items], self.user[items], self._name[items], \
                   self.category[items], self.labels[items]

    def __len__(self):
        return self._n_elements

    def extend(self, dataset):
        self.spectra = np.append(self.spectra, dataset.spectra, axis=0)
        self.x_axis = np.append(self.x_axis, dataset.x_axis, axis=0)
        self._raw = np.append(self._raw, dataset._raw, axis=0)
        self.user = np.append(self.user, dataset.user, axis=0)
        self._name = np.append(self._name, dataset._name, axis=0)
        self.category = np.append(self.category, dataset.category, axis=0)
        self.labels = np.append(self.labels, dataset.labels, axis=0)
        self._n_elements = self._n_elements + len(dataset)

    @classmethod
    def load_file(ds, file_type, file_name, parquet_engine='fastparquet',
                  label_dictionary={'cov': 0, 'covNeg': 1, 'ctrl': 2}):
        if file_type == 'pkl':
            df = pd.read_pickle(file_name)
        elif file_type == 'csv':
            df = pd.read_csv(file_name)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_name, engine=parquet_engine)
        else:
            print("Error file type not supported")
        spectra = df['spectra'].to_numpy()
        x_axis = df['x-axis'].to_numpy()
        raw = df['raw'].to_numpy()
        user = df['user'].to_numpy()
        name = df['name'].to_numpy()
        category = df['category'].to_numpy()
        if 'labels' in df.columns:
            labels = df['label'].to_numpy()
        else:
            labels = np.empty(shape=(0, 0))
        return ds(spectra, x_axis, raw, user, name, category, labels, [], label_dictionary)

    def create_label(self, label_dictionary):
        labels = []
        for el in self.category:
            labels.append(label_dictionary[el])
        self.labels = np.array(labels)

    def to_df(self, save_label=True):
        if save_label:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category', 'label'])
        else:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category'])
        df['spectra'] = self.spectra.tolist()
        df['x-axis'] = self.x_axis.tolist()
        df['raw'] = self._raw.tolist()
        df['user'] = self.user.tolist()
        df['name'] = self._name.tolist()
        df['category'] = self.category.tolist()
        if save_label:
            df['labels'] = self.labels.tolist()
        return df

    def save_pickle(self, filename, save_label=True):
        df = self.to_df(save_label)
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

    def k_fold(self, k):
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(GroupKFold(n_splits=k).split(x_train, y_train, groups=groups))
            return folds

    def leave_one_patient_cv(self):
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(LeaveOneGroupOut().split(x_train, y_train, groups=groups))
            return folds

    def spectra_to_numpy(self):
        spectra_list = []
        for el in self.spectra:
            spectra_list.append(np.array(el))
        spectra_list = np.array(spectra_list)
        self.spectra = spectra_list

    def x_axis_to_numpy(self):
        x_axis_list = []
        for el in self.x_axis:
            x_axis_list.append(np.array(el))
        x_axis_list = np.array(x_axis_list)
        self.x_axis = x_axis_list
