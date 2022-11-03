import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA

class Processor:
    def __init__(self, X, Y, kaggle_test_data=None):
        self.columns = X.columns
        self.X = pd.DataFrame(X)
        self.Y = pd.Series(Y)
        self.train_indices = None
        self.dev_indices = None
        self.test_indices = None
        self.pca_transformed = False
        self.pca = PCA(whiten=True, n_components=40)
        self.kaggle_test_data = None
        if kaggle_test_data is not None:
            self.kaggle_test_data = kaggle_test_data

    def pca_transform(self):
        if not self.pca_transformed:
            self.X = pd.DataFrame(self.pca.fit_transform(self.X))
            self.pca_transformed = True
            if self.kaggle_test_data is not None:
                self.kaggle_test_data = pd.DataFrame(self.pca.transform(self.kaggle_test_data))

    def reduce_dimensions(self, trained_coefficients):
        self.X = self.X[self.columns[trained_coefficients == 0]]
        print(f"Reduced from {len(self.columns)} B's to {self.X.shape[1]} B's]\nNow re-calculate train, dev, and test sets")

    # Expects dataframe w/o Y
    def process_data(self, numeric=[], dummify=True):
        self.X = self.standardize(numeric)
        if dummify:
            self.X = pd.get_dummies(self.X, drop_first=True)
        self.X = self.pad_B0(ret_numpy=False)
        self.columns = self.X.columns
        if self.kaggle_test_data is not None:
            self.kaggle_test_data = self.standardize(numeric)
            if dummify:
                self.kaggle_test_data = pd.get_dummies(self.kaggle_test_data, drop_first=True)
            self.kaggle_test_data = self.pad_B0(ret_numpy=False)

    def pad_B0(self, ret_numpy=True):
        self.X['B0'] = 1
        self.X = self.X[['B0'] + list(self.X.drop(columns='B0').columns)]
        if ret_numpy:
            return self.X.to_numpy() 
        return self.X

    # Can pass in a list of the numeric variables to standardize or rely on type inference
    def standardize(self, numeric=[]):
        if numeric == []:
            numeric = [column for column in self.X.columns if is_numeric_dtype(self.X[column].dtype)]
        for col in numeric:
            self.X[col] = (self.X[col]-np.mean(self.X[col]))/np.std(self.X[col])
        return self.X

    # Returns previously calculated train, dev, test, sets in format: train_X, train_Y, dev_X, dev_Y, test_X, test_Y
    def get_train_dev_test_sets(self, ret_numpy=True):
        if self.train_indices is None:
            raise ValueError('Must run train_dev_test_split. This returns previously calculated sets.')
        return self.format_train_dev_test_sets_(ret_numpy)

    # Calculates the train, dev, and test set indices given a split type and proportions
    # If train, dev, test sets already exist, use the get_ function (this will recompute them)
    # For set_class_prop_undersample, class_prop_1_0 is the ratio of # 1s / # 0s
    def calculate_train_dev_test_split(self, split_type='random', train_prop=.8, dev_prop=.1, 
                                       class_prop_1_0=1, silence=False, random_state=9):
        if split_type not in ['random', 'stratified_class', 'set_class_prop_undersample', 'non-random']:
            raise ValueError('Bad type is provided')

        if not silence and self.train_indices is not None:
            print("Re-calculating train, dev, and test sets")

        if split_type == 'random':
            train_indices = self.X.sample(frac=train_prop, random_state=random_state).index
            left_over_prop = 1 - train_prop
            dev_indices = self.X[~(self.X.index.isin(train_indices))].sample(frac=dev_prop / left_over_prop, random_state=random_state).index
            test_indices = self.X[~(self.X.index.isin(train_indices)) & ~(self.X.index.isin(dev_indices))].index
        elif split_type == 'stratified_class':
            zero_class = self.X[self.Y == 0]
            pos_class = self.X[self.Y == 1]

            train_zero_indices = zero_class.sample(frac=train_prop, random_state=random_state).index
            train_pos_indices = pos_class.sample(frac=train_prop, random_state=random_state).index
            train_indices = train_zero_indices.union(train_pos_indices)

            left_over_prop = 1 - train_prop
            dev_set_prop = dev_prop / left_over_prop
            dev_zero_indices = zero_class[~(zero_class.index.isin(train_zero_indices))].sample(frac=dev_set_prop, random_state=random_state).index
            dev_pos_indices = pos_class[~(pos_class.index.isin(train_pos_indices))].sample(frac=dev_set_prop, random_state=random_state).index
            dev_indices = dev_zero_indices.union(dev_pos_indices)

            test_zero_indices = zero_class[~(zero_class.index.isin(train_zero_indices.union(dev_zero_indices)))].index
            test_pos_indices = pos_class[~(pos_class.index.isin(train_pos_indices.union(dev_pos_indices)))].index
            test_indices = test_zero_indices.union(test_pos_indices)
        elif split_type == 'set_class_prop_undersample':
            # ASSUMES DEV PROP = TEST PROP and undersamples for the training set accordingly
            zero_class = self.X[self.Y == 0]
            pos_class = self.X[self.Y == 1]

            dev_zero_indices = zero_class.sample(frac=dev_prop, random_state=random_state).index
            dev_pos_indices = pos_class.sample(frac=dev_prop, random_state=random_state).index
            dev_indices = dev_zero_indices.union(dev_pos_indices)

            test_zero_indices = zero_class[~(zero_class.index.isin(dev_zero_indices))].sample(n=len(dev_zero_indices), random_state=random_state).index
            test_pos_indices = pos_class[~(pos_class.index.isin(dev_pos_indices))].sample(n=len(dev_pos_indices), random_state=random_state).index
            test_indices = test_zero_indices.union(test_pos_indices)
            # At least one pos class
            max_pos_n = len(pos_class) - len(dev_pos_indices) - len(test_pos_indices)
            max_zero_n = len(zero_class) - len(dev_zero_indices) - len(test_zero_indices)
            # frac * #0's = #1's
            calculated_pos_n = class_prop_1_0 * max_zero_n 
            # Clip to 1 - max_pos_n range
            train_pos_n = int(max(min(max_pos_n, calculated_pos_n), 1))
            train_zeros_n = int(min(train_pos_n / class_prop_1_0, max_zero_n)) if class_prop_1_0 != 0 else max_zero_n
            # max is the size of the leftovers
            train_pos_indices = pos_class[~(pos_class.index.isin(dev_pos_indices.union(test_pos_indices)))
                                          ].sample(n=train_pos_n, random_state=random_state).index
            train_zero_indices = zero_class[~(zero_class.index.isin(dev_zero_indices.union(test_zero_indices)))
                                            ].sample(n=train_zeros_n, random_state=random_state).index
            train_indices = train_pos_indices.union(train_zero_indices)
        elif split_type == 'non-random':
            if self.kaggle_test_data is None:
                raise ValueError('Must provide a link to the kaggle data')

        self.train_indices = train_indices
        self.dev_indices = dev_indices
        self.test_indices = test_indices

    # Do not directly call this function
    def format_train_dev_test_sets_(self, ret_numpy):
        train_X = self.X.loc[self.train_indices]
        train_Y = self.Y.loc[self.train_indices]
        dev_X = self.X.loc[self.dev_indices]
        dev_Y = self.Y.loc[self.dev_indices]
        test_X = self.X.loc[self.test_indices]
        test_Y = self.Y.loc[self.test_indices]
        if ret_numpy:
            train_X = train_X.to_numpy()
            train_Y = train_Y.to_numpy()
            dev_X = dev_X.to_numpy()
            dev_Y = dev_Y.to_numpy()
            test_X = test_X.to_numpy()
            test_Y = test_Y.to_numpy()
        return train_X, train_Y, dev_X, dev_Y, test_X, test_Y