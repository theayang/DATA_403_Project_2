# %%
# Our code imports
import json
from Metrics import accuracy, fmeasure, roc_auc, net_revenue, confusion_matrix
from Pipelines import ModelGridBuilder, AnalysisPipeline
# Standard lib imports
import pandas as pd
import numpy as np
import altair as alt

credit = pd.read_csv('cleaned_training_data.csv')
credit_X = credit.drop(columns='TARGET')
credit_Y = credit['TARGET']

def get_metrics(analysis):
    ratio_best_model = analysis.best_model
    roc = analysis.testscore_best_model()
    _, _, dev_X, dev_Y, test_X, test_Y = analysis.processor.get_train_dev_test_sets()

    def get_metrics_for_set(X, Y, istest=True):
        predictions = ratio_best_model.predict(X)
        acc = accuracy(predictions, Y)
        f_4 = fmeasure(predictions, Y, B=4)
        f_1 = fmeasure(predictions, Y, B=1)
        f_p5 = fmeasure(predictions, Y, B=.5)
        if istest:
            annuity = credit_X['AMT_ANNUITY'].loc[analysis.processor.test_indices]
        else:
            annuity = credit_X['AMT_ANNUITY'].loc[analysis.processor.dev_indices]
        tpaa = net_revenue(predictions, Y, annuity)
        return {'ROC-AUC': f'{roc:.4f}', 'Accuracy': f'{acc:.4f}', 'F-4': f'{f_4:.4f}', 'F-1': f'{f_1:.4f}', 'F-.5': f'{f_p5:.4f}', 'TPAA': f'{tpaa:.6f}'}

    return {'DEV': get_metrics_for_set(dev_X, dev_Y, False), 'TEST': get_metrics_for_set(test_X, test_Y)}


def net_revenue(p, Y, ann):
    if len(p) != len(ann) or len(Y) != len(ann):
        print('Wrong lengths of inputs')
        return None
    tot = 0
    realized = 0
    Y = list(Y)
    ann = list(ann)
    p = list(p)
    for i in range(len(Y)):
        if (p[i] > .5) != Y[i] and p[i] < .5:
            tot -= ann[i]
        elif (p[i] < .5) == Y[i] and p[i] < .5:
            tot += ann[i]
        if Y[i] == 0:
            realized += ann[i]
    return tot / realized


ratios = [i * 0.05 for i in range(1, 21)]
d = {}
for r in ratios:
    print(r)
    d[r] = {}
    
    print("LDA")
    ldaSearchBuilder = ModelGridBuilder('LDA')
    models = ldaSearchBuilder.get_models()

    ratio_lda_analysis = AnalysisPipeline(credit_X, credit_Y, models, roc_auc)
    ratio_lda_analysis.process_data(
        pca=True, split_type='set_class_prop_undersample', dev_prop=.05, class_prop_1_0=r)
    ratio_lda_analysis.fit_models()
    d[r]['lda'] = {}
    d[r]['lda']['best'] = {'model': ratio_lda_analysis.test_models()[0]}
    d[r]['lda']['best']['metrics'] = get_metrics(ratio_lda_analysis)
    d[r]['lda']['all'] = ratio_lda_analysis.dev_set_analysis

    print("Logistic")
    lambdas = np.linspace(0, 50, num=7)

    logisticSearchBuilder = ModelGridBuilder(
        'Logistic Lasso', parameters=lambdas)
    models = logisticSearchBuilder.get_models()

    ratio_logistic_analysis = AnalysisPipeline(
        credit_X, credit_Y, models, roc_auc)
    ratio_logistic_analysis.process_data(
        pca=True, split_type='set_class_prop_undersample', dev_prop=.05, class_prop_1_0=r)
    ratio_logistic_analysis.fit_models(max_iterations=8000)
    d[r]['logistic'] = {}
    d[r]['logistic']['best'] = {'model': ratio_logistic_analysis.test_models()[0]}
    d[r]['logistic']['best']['metrics'] = get_metrics(ratio_logistic_analysis)
    d[r]['logistic']['all'] = ratio_logistic_analysis.dev_set_analysis

    print("SVC")
    lambdas = np.linspace(0, 10, num=7)
    svcSearchBuilder = ModelGridBuilder('SVC', parameters=lambdas)
    models = svcSearchBuilder.get_models()

    ratio_svc_analysis = AnalysisPipeline(credit_X, credit_Y, models, roc_auc)
    ratio_svc_analysis.process_data(
        pca=True, split_type='set_class_prop_undersample', dev_prop=.05, class_prop_1_0=r)
    ratio_svc_analysis.fit_models(max_iterations=8000)
    d[r]['svc'] = {}
    d[r]['svc']['best'] = {'model': ratio_svc_analysis.test_models()[0]}
    d[r]['svc']['best']['metrics'] = get_metrics(ratio_svc_analysis)
    d[r]['svc']['all'] = ratio_svc_analysis.dev_set_analysis
    print(d)

    with open("model_results.json", "w") as outfile:
        json.dump(d, outfile)