import numpy as np
import pandas as pd
from sklearn import model_selection, base, linear_model, ensemble

from models_optimization import bayesian_optimization, hyperparameters_spaces
from utility_functions import logging
from . import models_stacking


def first_level_predictions(classification_task, datasets_models, y_train_df, folds_number, random_state):
    models_no = len(datasets_models)
    instances_no = datasets_models[0][1].shape[0]
    predictions_dict = {}

    for i in range(models_no):
        predictions_dict[datasets_models[i][0]] = instances_no * [-1]

    predictions_df = pd.DataFrame(predictions_dict)
    if not classification_task:
        for column_name in predictions_df.columns:
            predictions_df[column_name] = predictions_df[column_name].astype('float64')

    if classification_task:
        k_fold = model_selection.StratifiedKFold(n_splits=folds_number, random_state=random_state, shuffle=True)
    else:
        k_fold = model_selection.KFold(n_splits=folds_number, random_state=random_state, shuffle=True)

    for train_indices, test_indices in k_fold.split(datasets_models[0][1], y_train_df):
        fold_train_y_df = y_train_df[y_train_df.index.isin(train_indices)]

        for (model_name, dataset, model) in datasets_models:
            fold_train_x_df = dataset[dataset.index.isin(train_indices)]
            fold_test_x_df = dataset[dataset.index.isin(test_indices)]
            model.fit(fold_train_x_df, fold_train_y_df)
            predicted_labels = model.predict(fold_test_x_df)
            if classification_task:
                predictions_df[model_name][predictions_df.index.isin(test_indices)] = np.array(predicted_labels). \
                    astype(int).flatten()
            else:
                predictions_df[model_name][predictions_df.index.isin(test_indices)] = np.array(
                    predicted_labels).flatten()

    return predictions_df


def stack_models(classification_task, datasets_models, meta_model, y_train_df, metric, random_state, params_space=None,
                 iterations=None, folds_number_meta=None):
    predictions_df = first_level_predictions(classification_task, datasets_models, y_train_df, folds_number_meta,
                                             random_state)

    if params_space is not None and iterations is not None:
        if classification_task:
            y_train_df[y_train_df.columns[-1]] = y_train_df[y_train_df.columns[-1]].astype(int)

        try:
            optimization_results = bayesian_optimization.bayesian_optimization('meta_model', meta_model, params_space,
                                                                               iterations, folds_number_meta, metric,
                                                                               predictions_df, y_train_df,
                                                                               random_state)
            trained_meta_model = optimization_results[2]
        except ValueError:
            logging.display('Bayesian Optimization failed, trying without it.', p=1)
            meta_model.fit(predictions_df, y_train_df)
            trained_meta_model = meta_model
    else:
        meta_model.fit(predictions_df, y_train_df)
        trained_meta_model = meta_model

    if classification_task:
        k_fold = model_selection.StratifiedKFold(n_splits=folds_number_meta, random_state=random_state, shuffle=True)
    else:
        k_fold = model_selection.KFold(n_splits=folds_number_meta, random_state=random_state, shuffle=True)

    val_score = model_selection.cross_val_score(base.clone(trained_meta_model), predictions_df, y_train_df, cv=k_fold,
                                                scoring=metric, n_jobs=-1).mean()

    if metric == 'neg_root_mean_squared_error':
        true_values = list(y_train_df[y_train_df.columns[-1]].values)
        val_score_ = -val_score / (max(true_values) - min(true_values))
    else:
        val_score_ = val_score

    if metric == 'neg_root_mean_squared_error':
        info = 'nrmse: {}'.format(val_score_)
    else:
        info = '{}: {}'.format(metric, val_score_)

    print(info)

    return trained_meta_model, val_score_, info


def stacked_prediction(datasets_models, meta_model, y_test_df=None, metric=None):
    models_no = len(datasets_models)
    instances_no = datasets_models[0][1].shape[0]
    predictions_dict = {}

    for i in range(models_no):
        predictions_dict[datasets_models[i][0]] = instances_no * [-1]

    predictions_df = pd.DataFrame(predictions_dict)

    for (model_name, dataset, model) in datasets_models:
        predicted_labels = model.predict(dataset)
        predictions_df[model_name] = predicted_labels

    final_predictions = meta_model.predict(predictions_df)

    score = None

    if y_test_df is not None and metric is not None:
        true_y_test = list(y_test_df[y_test_df.columns[-1]].values)
        score = models_stacking.get_score(true_y_test, final_predictions, metric)

    return final_predictions, score


def configure_standard_stacking(classification_task, meta_model_type, random_state):
    meta_model = None

    if classification_task:
        classifiers_names = ['log_reg', 'rfc', 'gbc', 'mlpc']
        params_spaces = hyperparameters_spaces.load_params_spaces(classifiers_names, classifiers=True)

        if meta_model_type == 0:
            meta_model = linear_model.LogisticRegression(n_jobs=-1, class_weight='balanced', random_state=random_state)
        elif meta_model_type == 1:
            meta_model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=random_state)

    else:
        regressors_names = ['ridge_reg', 'rfr', 'gbr']
        params_spaces = hyperparameters_spaces.load_params_spaces(regressors_names, classifiers=False)

        if meta_model_type == 0:
            meta_model = linear_model.Ridge(random_state=random_state)
        elif meta_model_type == 1:
            meta_model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=random_state)

    meta_param_space = list(params_spaces.values())[meta_model_type]

    return meta_model, meta_param_space
