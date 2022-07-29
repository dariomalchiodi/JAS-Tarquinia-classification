#!/usr/bin/env python
# coding: utf-8

import copy
import logging
import numpy as np
import os
import pandas as pd
import random
import sys

from tarquinia.experiments import get_results
from tarquinia.model_selection import MeasureStratifiedKFold, \
                                      FragmentStratifiedKFold, kfold_factory
from tarquinia.classifiers import MLP


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed) 
    random.seed(seed) 
    np.random.seed(seed) 

def experiment(experiment_name, dataset, col_features, col_label,
               names, algs, grids,
               cv_class=FragmentStratifiedKFold, outer_splits=4, inner_splits=3,
               num_samples=2, logger=None):

        cv_object = kfold_factory(cv_class, outer_splits, num_samples)
        suite_name = cv_object.suite_name()

        result, predictions = get_results(experiment_name, dataset,
                                          col_features, col_label, names,
                                          algs, grids, cv_class=cv_class,
                                          outer_splits=outer_splits,
                                          inner_splits=inner_splits,
                                          num_samples=num_samples,
                                          logger=logger)
        for name in names:
                first_col = f'{suite_name}-{name}-0'
                last_col = f'{suite_name}-{name}-{outer_splits-1}'

                majority_vote = (predictions.loc[:, first_col:last_col]
                                            .sum(axis=1)
                                            .apply(lambda x: 0 \
                                                   if x <= outer_splits/2 \
                                                   else 1))
                confidence = (predictions.loc[:, first_col:last_col]
                                         .sum(axis=1)
                                         .apply(lambda x: x/outer_splits)
                                         .apply(lambda x: x if x >= 0.5 \
                                                            else 1-x))
                predictions[f'{suite_name}-{name}-majority'] = majority_vote
                predictions[f'{suite_name}-{name}-confidence'] = confidence

                for i in range(outer_splits):
                        del predictions[f'{suite_name}-{name}-{i}']

        result.to_csv(f'models/{experiment_name}/{suite_name}/'
                      f'global_results.csv')
        predictions.to_csv(f'models/{experiment_name}/{suite_name}/'
                           f'global_predictions.csv')

        return result, predictions


def main():
    experiment_name = 'JAS'

    logging.basicConfig(
        level=logging.INFO,
        format='[{%(asctime)s %(filename)s:%(lineno)d} %(levelname)s - '
            '%(message)s',
        handlers=[
            logging.FileHandler(filename=f'logs/{experiment_name}/tarquinia.log'),
            logging.StreamHandler(sys.stderr)
        ]
    )
    logger = logging.getLogger('tarquinia')

    np.random.seed(20220225)
    random.seed(20220225)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    dataset = 'data/data-tarquinia-latest.csv'
    tarquinia = pd.read_csv(dataset, sep='\t', decimal=',', index_col='id')

    col_features = tarquinia.columns[4:13]
    print(col_features)
    col_label = 'PROVENIENZA'

    names = ['LDA', 'MLP', 'SVM-lin', 'SVM-rbf', 'SVM-poly',
             'DT', 'RF', 'KNN', 'LR', 'NB']

    algs = [LinearDiscriminantAnalysis, MLP,
            SVC, SVC, SVC, DecisionTreeClassifier,
            RandomForestClassifier, KNeighborsClassifier,
            LogisticRegression, GaussianNB]

    lp_lda = {'solver': ['svd', 'lsqr']}

    lp_mlp = {'hidden_layer_sizes': [[2], [3], [2, 2]],
              'activation': ['logistic', 'relu'],
              'alpha': [1E-4, 1E-3],
              'learning_rate': ['constant', 'adaptive'],
              'learning_rate_init': [1E-4, 1E-3, 1E-2],
              'shuffle': [True, False],
              'momentum': [0.8, 0.9]
            }

    c_values = np.logspace(-4, 3, 10)
    gamma_values = ['auto', 'scale'] + list(np.logspace(-4, 3, 10))
    lp_svc_lin = {'C': c_values, 'kernel': ['linear']}
    lp_svc_rbf = {'C': c_values, 'kernel': ['rbf'], 'gamma': gamma_values}
    lp_svc_poly = {'C': c_values, 'kernel': ['poly'], 'degree': [2, 3, 5, 9]}

    lp_dt = {'criterion': ['gini', 'entropy'],
             #'max_leaf_nodes': [2],
             'max_features': [None, 'sqrt'],
             'max_depth': [None] + list(range(2, 10)),
             'min_samples_split': list(range(2, 6)),
             'min_samples_leaf': list(range(2, 6)),
             'ccp_alpha': [0, 0.5, 1, 1.5]}

    lp_rf = copy.deepcopy(lp_dt)
    lp_rf['n_estimators'] = [3, 5, 7, 9]

    lp_knn = {'n_neighbors': np.arange(1, 8),
              'metric': ['minkowski'],
              'p': list(range(2, 4))}

    lp_lr = {'penalty': ['l1', 'l2'],
             'C': c_values,
             'solver': ['liblinear'],
             'max_iter': [5000]}

    lp_nb = {}
    
    grids = [lp_lda, lp_mlp, lp_svc_lin, lp_svc_rbf, lp_svc_poly,
             lp_dt, lp_rf, lp_knn, lp_lr, lp_nb]


    cv_class = MeasureStratifiedKFold
    outer_splits = 4
    inner_splits = 3
    num_samples = None

    experiment(experiment_name, tarquinia, col_features, col_label,
               names, algs, grids, cv_class=cv_class,
               outer_splits=outer_splits, inner_splits=inner_splits,
               num_samples=num_samples, logger=logger)


    cv_class = FragmentStratifiedKFold

    experiment(experiment_name, tarquinia, col_features, col_label,
               names, algs, grids, cv_class=cv_class,
               outer_splits=outer_splits, inner_splits=inner_splits,
               num_samples=num_samples, logger=logger)

    num_samples = 2

    experiment(experiment_name, tarquinia, col_features, col_label,
               names, algs, grids, cv_class=cv_class,
               outer_splits=outer_splits, inner_splits=inner_splits,
               num_samples=num_samples, logger=logger)


if __name__ == '__main__':
    main()