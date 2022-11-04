from re import S
import pandas as pd
import numpy as np
from Processing import Processor
import Metrics
from Models import Model


class ModelGridBuilder:

    def __init__(self, model_type, parameters=[]):
        if parameters == []:
            self.models = [Model(model_type)]
        else:
            self.models = [Model(model_type, param) for param in parameters]

    def get_models(self):
        return self.models


class AnalysisPipeline:

    def __init__(self, X, Y, models, score_func):
        if type(models) != list:
            models = [models]

        self.score_func = score_func

        # Process data and build splits
        self.processor = Processor(X, Y)

        self.models = models
        self.best_model = None

        self.dev_set_analysis = None
        self.cv_dev_set_analysis = None

    def process_data(self, split_type='random', train_prop=.8, dev_prop=.1, class_prop_1_0=1, numeric=[], pca=False):
        self.processor.process_data(numeric)
        if pca:
            self.processor.pca_transform()
        self.processor.calculate_train_dev_test_split(
            split_type, train_prop, dev_prop, class_prop_1_0)

    def cv(self, k, split_type='random', class_prop_1_0=1, adaptive_descent=False, initial_B=None, max_iterations=None,
           etas=None, tol=None, err=None, show_iter=False, train_X=None, train_Y=None):

        trains, devs = self.processor.get_cv_splits(
            k, split_type=split_type, class_prop_1_0=class_prop_1_0)
        model_tracking = []
        best_score = 0
        best_model = None
        best_model_specs = None
        best_predictions = None
        for i in range(k):
            train_X = self.processor.X.loc[trains[i]].to_numpy()
            train_Y = self.processor.Y.loc[trains[i]].to_numpy()
            dev_X = self.processor.X.loc[devs[i]].to_numpy()
            dev_Y = self.processor.Y.loc[devs[i]].to_numpy()

            self.fit_models(train_X=train_X, train_Y=train_Y, adaptive_descent=adaptive_descent, initial_B=initial_B, max_iterations=max_iterations,
                            etas=etas, tol=tol, err=err, show_iter=show_iter)

            for i in range(len(self.models)):
                model = self.models[i]
                predictions = model.predict(dev_X)
                score = self.score_func(predictions, dev_Y)
                specs = model.get_model_specs()
                    
                if len(model_tracking) != len(self.models):
                    model_tracking.append([model, predictions, specs, score/k])
                else:
                    model_tracking[i][-1] += score/k


        for model in model_tracking:
            if model[-1] > best_score:
                best_model = model[0]
                best_predictions = model[1]
                best_model_specs = model[2]
                best_score = model[3]
        
        self.best_model = best_model
        self.cv_dev_set_analysis = model_tracking
        return best_model_specs

    def fit_models(self, adaptive_descent=False, initial_B=None, max_iterations=None,
                   etas=None, tol=None, err=None, show_iter=False, train_X=None, train_Y=None):
        if train_X is None and train_Y is None:
            train_X, train_Y, _, _, _, _ = self.processor.get_train_dev_test_sets()
            print(train_X.shape, train_Y.shape)
        for model in self.models:
            model.fit(train_X, train_Y, adaptive_descent=adaptive_descent, initial_B=initial_B, max_iterations=max_iterations,
                      etas=etas, tol=tol, err=err, show_iter=show_iter)

    # Tests all models on the dev set and returns their scores and the best spec
    def test_models(self):
        _, _, dev_X, dev_Y, _, _ = self.processor.get_train_dev_test_sets()
        scores = []
        best_score = 0
        best_model = None
        best_model_specs = None
        best_predictions = None
        for model in self.models:
            predictions = model.predict(dev_X)
            score = self.score_func(predictions, dev_Y)
            specs = model.get_model_specs()
            if score > best_score:
                best_score = score
                best_model = model
                best_model_specs = specs
                best_predictions = predictions
            scores.append((specs, score))
        self.best_model = best_model
        self.dev_set_analysis = scores
        best_model_conf_matrix = Metrics.confusion_matrix(best_predictions, dev_Y)
        return best_model_specs, best_model_conf_matrix

    def testscore_best_model(self):
        if self.best_model is None:
            raise ValueError('Must run test_models')
        _, _, _, _, test_X, test_Y = self.processor.get_train_dev_test_sets()
        predictions = self.best_model.predict(test_X)
        return self.score_func(predictions, test_Y)
