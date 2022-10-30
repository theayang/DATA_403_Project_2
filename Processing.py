import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class Processor:
    def __init__(self, X, Y):
        self.X = pd.DataFrame(X)
        self.Y = pd.Series(Y)
        self.train_indices = None
        self.dev_indices = None
        self.test_indices = None

    # Expects dataframe w/o Y
    def process_data(self, numeric=[], dummify=True):
        self.X = self.standardize(numeric)
        if dummify:
            self.X = pd.get_dummies(self.X, sparse=True)
        self.X = self.pad_B0(ret_numpy=False)

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
    # TODO: add stratified and smart future split code
    def calculate_train_dev_test_split(self, split_type='random', train_prop=.8, dev_prop=.1):
        if split_type not in ['random', 'stratified', 'non-random']:
            raise ValueError('Bad type is provided')

        if self.train_indices is not None:
            print("Re-calculating train, dev, and test sets")

        if split_type == 'random':
            train_indices = self.X.sample(frac=train_prop).index
            left_over_prop = 1 - train_prop
            dev_indices = self.X[~(self.X.index.isin(train_indices))].sample(frac=dev_prop / left_over_prop).index
            test_indices = self.X[~(self.X.index.isin(train_indices)) & ~(self.X.index.isin(dev_indices))].index
        #elif split_type == 'stratified':
        
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