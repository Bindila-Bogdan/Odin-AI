import sys

import pandas as pd
from sklearn import model_selection, base, linear_model, ensemble

from models_optimization import bayesian_optimization, hyperparameters_spaces
from . import models_stacking


def first_level_prob_predictions(classification_task, datasets_models, y_train_df, folds_number, random_state):
    models_no = len(datasets_models)
    instances_no = datasets_models[0][1].shape[0]
    classes_no = len(set(y_train_df[y_train_df.columns[-1]].values))
    predictions_dict = {}

    for i in range(models_no):
        for j in range(classes_no):
            predictions_dict[datasets_models[i][0] + '_' + str(j)] = instances_no * [-1]

    predictions_df = pd.DataFrame(predictions_dict)
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
            predicted_prob = model.predict_proba(fold_test_x_df)
            index = 0
            for column_name in list(predictions_df.columns):
                if model_name in column_name:
                    predictions_df[column_name][predictions_df.index.isin(test_indices)] = predicted_prob[:, index]
                    index += 1

    return predictions_df, classes_no


def get_highest_prob_classes(meta_predictions_list, test_instances_no, classes_no):
    final_meta_predictions = []

    for j in range(test_instances_no):
        max_prob = -sys.maxsize
        pred_class = -sys.maxsize

        for i in range(classes_no):
            if max_prob < meta_predictions_list[i][j]:
                max_prob = meta_predictions_list[i][j]
                pred_class = i

        final_meta_predictions.append(pred_class)

    return final_meta_predictions


def cross_val_prob_stacking(x, y, meta_models, classes_no, folds_number, metric, random_state):
    sum_score = 0.0

    k_fold = model_selection.StratifiedKFold(n_splits=folds_number, random_state=random_state, shuffle=True)

    for train_index, val_index in k_fold.split(x, y):
        x_train, x_val = x.iloc[train_index, :], x.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        meta_models_clone = []
        meta_predictions_list = []

        for i in range(classes_no):
            meta_models_clone.append(base.clone(meta_models[i]))
            current_columns = []

            for column_name in x_train.columns:
                if str(i) not in column_name:
                    current_columns.append(column_name)

            current_predictions_df = x_train.drop(current_columns, axis=1)
            true_class_df = y_train[y_train.columns[-1]].map(lambda value: 1 if value == i else 0)

            meta_models_clone[i].fit(current_predictions_df, true_class_df)

        for i in range(classes_no):
            current_columns = []

            for column_name in x_val.columns:
                if str(i) not in column_name:
                    current_columns.append(column_name)

            current_predictions_df = x_val.drop(current_columns, axis=1)
            meta_predictions = meta_models_clone[i].predict(current_predictions_df)
            meta_predictions_list.append(meta_predictions)

        test_classes = list(y_val[y_val.columns[-1]].values)
        final_meta_predictions = get_highest_prob_classes(meta_predictions_list, len(test_classes), classes_no)

        score = models_stacking.get_score(test_classes, final_meta_predictions, metric)
        sum_score += score

    return sum_score / folds_number


def stack_models_prob(classification_task, datasets_models, meta_model, y_train_df, metric_meta, metric, random_state,
                      params_space=None, iterations=None, folds_number_meta=None):
    predictions_df, classes_no = first_level_prob_predictions(classification_task, datasets_models, y_train_df,
                                                              folds_number_meta, random_state)

    meta_models = []

    for i in range(classes_no):
        meta_models.append(base.clone(meta_model))
        current_columns = []

        for column_name in predictions_df.columns:
            if str(i) not in column_name:
                current_columns.append(column_name)

        current_predictions_df = predictions_df.drop(current_columns, axis=1)
        true_class_df = y_train_df[y_train_df.columns[-1]].map(lambda x: 1 if x == i else 0)

        if params_space is not None and iterations is not None:
            optimization_results = bayesian_optimization.bayesian_optimization('meta_model', meta_models[i],
                                                                               params_space.copy(), iterations,
                                                                               folds_number_meta, metric_meta,
                                                                               current_predictions_df, true_class_df,
                                                                               random_state)
            meta_models[i] = optimization_results[2]
        else:
            meta_models[i].fit(current_predictions_df, true_class_df)

    val_score = cross_val_prob_stacking(predictions_df, y_train_df, meta_models, classes_no, folds_number_meta, metric,
                                        random_state)

    if metric == 'neg_root_mean_squared_error':
        info = 'nrmse: {}'.format(val_score)
    else:
        info = '{}: {}'.format(metric, val_score)
    print(info)

    return meta_models, val_score, info


def first_level_prob_predictions_test(datasets_models, meta_models):
    models_no = len(datasets_models)
    instances_no = datasets_models[0][1].shape[0]
    classes_no = len(meta_models)
    predictions_dict = {}

    for i in range(models_no):
        for j in range(classes_no):
            predictions_dict[datasets_models[i][0] + '_' + str(j)] = instances_no * [-1]

    predictions_df = pd.DataFrame(predictions_dict)

    for column_name in predictions_df.columns:
        predictions_df[column_name] = predictions_df[column_name].astype('float64')

    for (model_name, dataset, model) in datasets_models:
        predicted_probs = model.predict_proba(dataset)
        index = 0
        for column_name in list(predictions_df.columns):
            if model_name in column_name:
                predictions_df[column_name] = predicted_probs[:, index]
                index += 1

    return predictions_df, instances_no, classes_no


def stacked_prediction_prob(datasets_models, meta_models, y_test_df=None, metric=None):
    predictions_df, instances_no, classes_no = first_level_prob_predictions_test(datasets_models, meta_models)
    meta_predictions_list = []

    for i in range(classes_no):
        current_columns = []

        for column_name in predictions_df.columns:
            if str(i) not in column_name:
                current_columns.append(column_name)

        current_predictions_df = predictions_df.drop(current_columns, axis=1)
        meta_predictions = meta_models[i].predict(current_predictions_df)
        meta_predictions_list.append(meta_predictions)

    final_meta_predictions = get_highest_prob_classes(meta_predictions_list, instances_no, classes_no)

    score = None

    if y_test_df is not None and metric is not None:
        true_y_test = list(y_test_df[y_test_df.columns[-1]].values)
        score = models_stacking.get_score(true_y_test, final_meta_predictions, metric)

    return final_meta_predictions, score


def configure_prob_based_stacking(meta_model_type, random_state):
    meta_model = None
    regressors_names = ['ridge_reg', 'rfr', 'gbr']
    params_spaces = hyperparameters_spaces.load_params_spaces(regressors_names, classifiers=False)

    if meta_model_type == 0:
        meta_model = linear_model.Ridge(random_state=random_state)
    elif meta_model_type == 1:
        meta_model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=random_state)

    meta_param_space = list(params_spaces.values())[meta_model_type]

    return meta_model, meta_param_space
