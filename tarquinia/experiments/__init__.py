import itertools as it
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, \
                                  MinMaxScaler, QuantileTransformer

from tarquinia.model_selection import FragmentStratifiedKFold, kfold_factory


def create_if_not_exists(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def make_experiment(experiment_name, model_name, learning_alg, params,
                    dataset, col_features, col_label,
                    cv_class=FragmentStratifiedKFold,
                    outer_splits=4, inner_splits=3, num_samples=2,
                    logger=None):

    '''Perform a classification experiment, choosing among three different
    settings, each characterized by a different way to consider the rows
    of the dataset:

    - an 'all_data' setting, using all rows as different cases (triggered
      when all_data=True),
    - an 'obj_sample' setting, keeping all cases referring to a same objec
      either in the train or in the test set, sampling num_samples fragments
      for each object so as to guarantee that the cross-validation folds have
      approximately the same length (triggered when all_data=False and
      num_sample is not None),
    - an 'obj_nosample' setting, for a behavior analogous to that of
      'object_sample', though using all fragments of the selected object
      (triggered when all_data=False and num_sample is set to None).

    Parameters are as follows:

    - model_name: symbolic name of the machine learning model to be used;
    - learning_alg: sklearn object corresponding to the learning algorithm
      to be used;
    - learning_params: grid with the hyper-parameters values to be considered
      in the model selection phase;
    - dataset: a pandas dataframe containing the dataset to be processed;
    - all_data: boolean value triggering the 'all_data' setting;
    - num_samples: number of samples used in case of 'obj_nosample' setting;
    - outer_folds: number of folds for the external cross-validation
      procedure used in order to estimate generalization ability;
    - inner_folds: number of folds for the internal cross-validation
      procedure used in the model selection phase;
    - logger: logger to be used to vehiculate information generated during
      the learning process.

    The function returns a dictionary with the following keys:

    - 'accuracy': test accuracy for each of the outer cross-validation fold,
    - 'specificity': test specificities for each of the outer cross-validation
       fold,
    - 'sensibility': test sensibilities for each of the outer cross-validation
       fold,
    - 'f1_score': test F1 scores for each of the outer cross-validation fold,
    - 'best_models': best_model for each of the outer cross-validation fold,
    - 'best_params': best_params for each of the outer cross-validation fold.
    '''

    if logger:
        logger.info(f'start experiment with model {model_name}')

    scalers = [None, StandardScaler(), RobustScaler(),
               MinMaxScaler(), QuantileTransformer(n_quantiles=20)]

    n_components_PCA = np.arange(2, len(col_features)+1)
    n_components_SVD = np.arange(2, len(col_features))

    dim_reductions = [PCA(n) for n in n_components_PCA] + \
                    [TruncatedSVD(n) for n in n_components_SVD] + \
                    [None]

    accuracies = []
    specificities = []
    sensibilities  = []
    f1_scores = []
    best_models = []
    best_params = []

    outer_cv = kfold_factory(cv_class, outer_splits, num_samples)
    inner_cv = kfold_factory(cv_class, inner_splits, num_samples)
    suite_name = outer_cv.suite_name()
    assert(suite_name == inner_cv.suite_name())
    
    for i, (trainval_idx, test_idx) in enumerate(outer_cv.split(dataset)):
        if logger:
            logger.info(f'Outer fold {i}')

        data_trainval = dataset.iloc[trainval_idx]
        data_test = dataset.iloc[test_idx]
        X_trainval = data_trainval[col_features]
        X_test = data_test[col_features]
        y_trainval = data_trainval[col_label]
        y_test = data_test[col_label]
        
        
        for (train_idx, test_idx) in inner_cv.split(data_trainval):
            data_train = data_trainval.iloc[train_idx]
            data_val = data_trainval.iloc[test_idx]
            X_train = data_train[col_features]
            X_val = data_val[col_features]
            y_train = data_train[col_label]
            y_val = data_val[col_label]

            best_accuracy = -1
            best_params_ = None

            for scaler in scalers:
                for dim_reduction in dim_reductions:
                    param_names = (params.keys())
                    param_values = params.values()

                    for conf in it.product(*param_values):
                        args = {name: value
                                for name, value in zip(param_names, conf)}

                        alg = learning_alg(**args)
                        pipe = Pipeline([('scaler', scaler),
                                ('dim_reduction', dim_reduction),
                                ('learning_algorithm', alg)])

                        pipe.fit(X_train, y_train)

                        accuracy = pipe.score(X_val, y_val)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params_ = {'scaler': scaler,
                                           'dim_reduction': dim_reduction,
                                           'args': args}


        best_pipe = Pipeline([('scaler', best_params_['scaler']),
                              ('dim_reduction', best_params_['dim_reduction']),
                              ('learning_algorithm', \
                                   learning_alg(**best_params_['args']))])
        best_pipe.fit(X_trainval, y_trainval)

        y_hat = best_pipe.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)
        accuracies.append(accuracy)

        specificity = recall_score(y_test, y_hat, pos_label=0)
        specificities.append(specificity)
        sensibility = recall_score(y_test, y_hat)
        sensibilities.append(sensibility)
        f1 = f1_score(y_test, y_hat)
        f1_scores.append(f1)

        if logger:
            logger.info(f"best scaler: {best_params_['scaler']}")
            logger.info(f"best dim. reduction: {best_params_['dim_reduction']}")
            logger.info(f"best model args.: {best_params_['args']}")
            logger.info(f'accuracy: {accuracy}')
            logger.info(f'specificity: {specificity}')
            logger.info(f'sensibility: {sensibility}')
            logger.info(f'F1 score: {f1}')

        best_models.append(best_pipe)
        best_params.append(best_params_)

        dir_name = f'./models/{experiment_name}/{suite_name}'
        create_if_not_exists(dir_name)

        file_name = f'{dir_name}/{model_name}-{i}'

        if logger:
            logger.info(f'writing model in file "{file_name}"')

        with open(file_name, 'wb') as f:
            pickle.dump(best_pipe, f)

    return {'accuracy': accuracies,
            'specificity': specificities,
            'sensibility': sensibilities,
            'f1_score': f1_scores,
            'best_models': best_models,
            'best_params': best_params}


def pretty_dict(d):
    return '\n'.join([f'{k}: {v}' for k, v in d.items()])


def generate_table(experiment_name,
                   dataset, col_features, col_label, names, algs, grids,
                   cv_class=FragmentStratifiedKFold,
                   outer_splits=4, inner_splits=3,
                   num_samples=2, logger=None):

    cv_object = kfold_factory(cv_class, outer_splits, num_samples)
    suite_name = cv_object.suite_name()

    result = []
    predictions = {}
    logger.info(f'start experiment suite {suite_name}')

    for name, alg, grid in zip(names, algs, grids):
        dir_name = f'./models/{experiment_name}/{suite_name}'
        create_if_not_exists(dir_name)
        file_name = f'{dir_name}/{name}-performance.csv'
        if os.path.isfile(file_name):
            logger.info(f'experiment with {name} has already been done,'
                        f'retrieving results and skipping experiment')

            best_fold = -1
        else:
            logger.info(f'start experiment with {name}')
            res = make_experiment(experiment_name, name, alg, grid, dataset,
                                  col_features, col_label,
                                  cv_class=cv_class, outer_splits=outer_splits,
                                  inner_splits=inner_splits,
                                  num_samples=num_samples,
                                  logger=logger)
            logger.info(f'end experiment with {name}')

            accuracy_mean = np.mean(res['accuracy'])
            accuracy_std = np.std(res['accuracy'])

            specificity_mean = np.mean(res['specificity'])
            specificity_std = np.std(res['specificity'])

            sensibility_mean = np.mean(res['sensibility'])
            sensibility_std = np.std(res['sensibility'])

            f1_mean = np.mean(res['f1_score'])
            f1_std = np.std(res['f1_score'])

            logger.info(f'accuracy:{accuracy_mean:.3f}±{accuracy_std:.3f}')
            logger.info(f'sensibility:'
                        f'{sensibility_mean:.3f}±{sensibility_std:.3f}')
            logger.info(f'specificity:'
                        f'{specificity_mean:.3f}±{specificity_std:.3f}')
            logger.info(f'F1:{f1_mean:.3f}±{f1_std:.3f}')

            for i, bp in enumerate(res['best_params']):
                logger.info(f'round {i}, best parameters:')
                for k in bp:
                    logger.info(f'- {k}: {bp[k]}')
            best_fold = np.argmax(res['accuracy'])
            best_of_bests = pretty_dict(res['best_params'][best_fold])
            logger.info(f'Best fold is {best_fold}')

            col_names = ('', 'Method', 'Acc mean', 'Acc std', 'Sens mean',
                         'Sens std', 'Spec mean', 'Spec std', 'F1 mean',
                         'F1 std', 'Best fold', 'Best params')

            data_values = pd.Series([0, name, accuracy_mean, accuracy_std,
                                     sensibility_mean, sensibility_std,
                                     specificity_mean, specificity_std,
                                     f1_mean, f1_std, best_fold, best_of_bests],
                                    index=col_names)

            data_dict = {k: v for k, v in zip(col_names, data_values)}

            logger.info(f'writing partial results to {file_name}')

            tmp = pd.DataFrame([data_dict], columns=col_names)
            tmp.to_csv(file_name)
            logger.info(f'end experiment with {name}')

        retrieved_results = pd.read_csv(file_name)
        data_values = retrieved_results.iloc[0,1:]
        data_dict = data_values.to_dict()
        result.append(data_values)

        if best_fold != -1:
            assert(best_fold == data_dict['Best fold'])
        best_fold = data_dict['Best fold']

        file_name = f'{dir_name}/{name}-{best_fold}'
        logger.info(f'reading model from file "{file_name}"')
        with open(file_name, 'rb') as f:
            model = pickle.load(f)

        predictions[f'{suite_name}-{name}-best-{best_fold}'] = \
            pd.Series(model.predict(dataset[col_features]),
                      index=dataset.index)
        
        for i in range(outer_splits):
            file_name = f'{dir_name}/{name}-{i}'
            with open(file_name, 'rb') as f:
                model = pickle.load(f)
            predictions[f'{suite_name}-{name}-{i}'] = \
            pd.Series(model.predict(dataset[col_features]),
                      index=dataset.index)

    logger.info(f'ended experiment suite {suite_name}')

    return result, predictions


def get_results(experiment_name, dataset, col_features, col_label,
                names, algs, grids,
                cv_class=FragmentStratifiedKFold,
                outer_splits=4, inner_splits=3, num_samples=2,
                logger=None):

    t, p = generate_table(experiment_name, dataset, col_features, col_label,
                          names, algs, grids,
                          cv_class=cv_class, outer_splits=outer_splits,
                          inner_splits=inner_splits, num_samples=num_samples,
                          logger=logger)
    col_names = ['Method', 'Acc mean', 'Acc std', 'Sens mean', 'Sens std',
                 'Spec mean', 'Spec std', 'F1 mean', 'F1 std', 'Best fold',
                 'Best params']
    result = pd.DataFrame(t, columns=col_names)
    result_sorted = result.set_index('Method').sort_values(by='Acc mean',
                                                           ascending=False)
    predictions = pd.DataFrame(p)

    return result_sorted, predictions

