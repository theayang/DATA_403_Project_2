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

    def process_data(self, split_type='random', train_prop=.8, dev_prop=.1, numeric=[]):
        self.processor.process_data(numeric)
        self.processor.calculate_train_dev_test_split(split_type, train_prop, dev_prop)

    def fit_models(self, adaptive_descent=False, initial_B=None, max_iterations=None, 
                   etas=None, tol=None, err=None, show_iter=False):
        train_X, train_Y, _, _, _, _ = self.processor.get_train_dev_test_sets()
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