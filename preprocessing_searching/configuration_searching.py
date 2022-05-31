import copy
import os
import sys
import time
from os import path

import pandas as pd
from sklearn import linear_model, ensemble, neural_network, model_selection

from utility_functions import files_storing, logging, reporting
from . import preprocessing_training, preprocessing_spaces


def get_next_config(config_number, configs, index_non_tree_based, index_tree_based):
    [best_config_non_tree_based, best_config_tree_based, non_tree_based_configs, tree_based_configs] = configs
    params = None
    tree_based = None

    if config_number == 0:
        tree_based = False
        params = best_config_non_tree_based

    elif config_number == 1:
        tree_based = True
        params = best_config_tree_based

    elif config_number % 2 == 0:
        try:
            tree_based = False
            params = non_tree_based_configs[index_non_tree_based]
            index_non_tree_based += 1
        except IndexError:
            tree_based = True
            params = tree_based_configs[index_tree_based]
            index_tree_based += 1

    elif config_number % 2 == 1:

        try:
            tree_based = True
            params = tree_based_configs[index_tree_based]
            index_tree_based += 1
        except IndexError:
            tree_based = False
            params = non_tree_based_configs[index_non_tree_based]
            index_non_tree_based += 1

    return [tree_based, params, index_non_tree_based, index_tree_based]


def constructing_training_dataset(train_continuous_df, train_categorical_df, y_train_df):
    x_train_df = None

    if train_continuous_df is None or train_categorical_df is None:
        if train_continuous_df is None:
            x_train_df = train_categorical_df
        if train_categorical_df is None:
            x_train_df = train_continuous_df
    else:
        x_train_df = pd.concat([train_continuous_df, train_categorical_df], axis=1)

    return [x_train_df, y_train_df]


def get_models(classification_task, include_gbc, random_state):
    if classification_task:
        log_reg = linear_model.LogisticRegression(n_jobs=-1, class_weight='balanced', random_state=random_state)
        rfc = ensemble.RandomForestClassifier(n_jobs=-1, max_features='sqrt', class_weight='balanced',
                                              random_state=random_state)
        mlpc = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(64, 32, 64), random_state=random_state)

        if include_gbc:
            gbc = ensemble.GradientBoostingClassifier(n_estimators=128, n_iter_no_change=20, max_depth=100,
                                                      max_features='sqrt', random_state=random_state)
            scores = {'log_reg': [None, None], 'rfc': [None, None], 'gbc': [None, None], 'mlpc': [None, None]}
            models = {'log_reg': log_reg, 'rfc': rfc, 'gbc': gbc, 'mlpc': mlpc}
        else:
            scores = {'log_reg': [None, None], 'rfc': [None, None], 'mlpc': [None, None]}
            models = {'log_reg': log_reg, 'rfc': rfc, 'mlpc': mlpc}
    else:
        ridge_reg = linear_model.Ridge(random_state=random_state)
        rfr = ensemble.RandomForestRegressor(n_jobs=-1, max_features='sqrt', random_state=random_state)
        gbr = ensemble.GradientBoostingRegressor(n_estimators=128, n_iter_no_change=20, max_depth=100,
                                                 max_features='sqrt', random_state=random_state)

        scores = {'ridge_reg': [None, None], 'rfr': [None, None], 'gbr': [None, None]}
        models = {'ridge_reg': ridge_reg, 'rfr': rfr, 'gbr': gbr}

    return models, scores


def get_config_score(classification_task, x_train_df, y_train_df, models, scores, metric, configs_no, index,
                     config_number, random_state, tree_based):
    scores_copy = copy.deepcopy(scores)
    models_names = []

    if classification_task:
        if configs_no == 1 and not tree_based:
            models_names = ['log_reg', 'mlpc']
        elif (configs_no == 1 and tree_based) or (configs_no == 2 and index == 0):
            models_names = ['rfc']
        elif configs_no == 2 and index == 1:
            models_names = ['gbc']
    else:
        if configs_no == 1:
            models_names = ['ridge_reg']
        elif index == 0:
            models_names = ['rfr']
        elif index == 1:
            models_names = ['gbr']

    if x_train_df.shape[0] <= 1000:
        folds_number = 10
    else:
        folds_number = 5

    for model_name in models_names:
        model = models[model_name]

        if classification_task:
            k_fold = model_selection.StratifiedKFold(n_splits=folds_number, random_state=random_state, shuffle=True)
        else:
            k_fold = model_selection.KFold(n_splits=folds_number, random_state=random_state, shuffle=True)

        score = model_selection.cross_val_score(model, x_train_df, y_train_df, cv=k_fold, scoring=metric,
                                                n_jobs=-1).mean()

        if metric == 'neg_root_mean_squared_error':
            score = -score / (y_train_df.values.max() - y_train_df.values.min())

        config_name = str(config_number) + '_' + str(index)

        if scores_copy[model_name][0] is None:
            scores_copy[model_name][0] = score
            scores_copy[model_name][1] = config_name
        elif metric == 'neg_root_mean_squared_error' and scores_copy[model_name][0] > score:
            scores_copy[model_name][0] = score
            scores_copy[model_name][1] = config_name
        elif metric != 'neg_root_mean_squared_error' and scores_copy[model_name][0] < score:
            scores_copy[model_name][0] = score
            scores_copy[model_name][1] = config_name

    return scores_copy, folds_number


def clean_store_files(dataset_name, target_column_name, config_number, index, config_dict, models_dict, x_train_df,
                      y_train_df, scores, new_scores, store_y):
    new_config_added = False

    for model_name in scores.keys():
        cond1 = (scores[model_name][0] is None and new_scores[model_name][0] is not None)
        try:
            cond2 = (scores[model_name][0] != new_scores[model_name][0])
        except TypeError:
            cond2 = False

        if cond1 or cond2:
            new_config_added = True
            files_storing.file_writer(dataset_name, target_column_name, str(config_number), str(index), config_dict)
            files_storing.serialize_prep_models(dataset_name, target_column_name, str(config_number), str(index),
                                                models_dict)
            files_storing.save_preprocessed_dataset(dataset_name, target_column_name, str(config_number), str(index),
                                                    x_train_df, y_train_df, store_y)
            store_y = False

    if new_config_added:
        config_files_path = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + \
                            target_column_name + '/'
        datasets_path = '/odinstorage/automl_data/training_results/preprocess_data/' + dataset_name + '/' + \
                        target_column_name + '/'
        config_files_names = [f for f in os.listdir(config_files_path) if path.isfile(path.join(config_files_path, f))]
        stored_datasets_names = [f for f in os.listdir(datasets_path) if path.isfile(path.join(datasets_path, f))]

        used_config_names = [conf_name for _, [_, conf_name] in new_scores.items()]

        for i, files_names in enumerate([config_files_names, stored_datasets_names]):
            for file_name in files_names:
                used = False

                for used_config_name in used_config_names:
                    if used_config_name is not None and used_config_name in file_name:
                        used = True

                if not used:
                    if i == 0:
                        os.remove(config_files_path + file_name)
                    elif i == 1 and file_name != 'y_train.csv':
                        os.remove(datasets_path + file_name)

        files_storing.file_writer(dataset_name, target_column_name, 'model_config_mapping', None, new_scores)

    return new_scores, store_y


def search_preprocessing_steps(dataset_name, predicted_column, classification_task, metric, max_search_time,
                               configurations_no, include_gbc):
    print('*Searching preprocessing configuration*')
    print('model      config  score')

    if max_search_time == sys.maxsize:
        time_limited_searching = False
    else:
        time_limited_searching = True

    store_y = True
    start_search_time = time.time()
    [params, config_number, folds_number, models, scores] = [None, None, None, None, None]

    files_storing.gen_config_folders(dataset_name, predicted_column, True)
    [best_config_non_tree_based, best_config_tree_based, non_tree_based_configs,
     tree_based_configs] = preprocessing_spaces.load_configs(configurations_no, time_limited_searching, 42)

    index_non_tree_based = 0
    index_tree_based = 0

    for config_number in range(configurations_no):
        print('Config number: {}'.format(config_number))
        logging.display('clean')

        configs = [best_config_non_tree_based, best_config_tree_based, non_tree_based_configs, tree_based_configs]
        [tree_based, params, index_non_tree_based, index_tree_based] = get_next_config(config_number, configs,
                                                                                       index_non_tree_based,
                                                                                       index_tree_based)
        results = preprocessing_training.data_preprocessing_training(dataset_name, predicted_column, params,
                                                                     classification_task, include_gbc)

        if config_number == 0:
            models, scores = get_models(classification_task, include_gbc, params[1])

        for index, result in enumerate(results):
            files_storing.write_log_file(dataset_name, predicted_column, str(config_number), str(index))

        for index, result in enumerate(results):
            [config_dict, models_dict, train_continuous_df, train_categorical_df, y_train_df] = result

            result = constructing_training_dataset(train_continuous_df, train_categorical_df, y_train_df)
            [x_train_df, y_train_df] = result

            new_scores, folds_number = get_config_score(classification_task, x_train_df, y_train_df, models, scores,
                                                        metric, len(results), index, config_number, params[1],
                                                        tree_based)
            scores, store_y = clean_store_files(dataset_name, predicted_column, config_number, index, config_dict,
                                                models_dict, x_train_df, y_train_df, scores, new_scores, store_y)

        for model_name, data in scores.items():
            print(model_name.ljust(11), end='')
            print(str(data[1]).ljust(7), str(data[0]).ljust(12))

        passed_time = time.time() - start_search_time

        if passed_time >= max_search_time and config_number > 1:
            search_info = reporting.get_searching_info(passed_time, config_number, folds_number, metric, scores)
            print(53 * '*' + '\nTime required by searching: {} s\n'.format(round(passed_time, 4)))
            return params[1], search_info

    passed_time = time.time() - start_search_time
    print(53 * '*' + '\nTime required by searching: {} s\n'.format(round(passed_time, 4)))
    search_info = reporting.get_searching_info(passed_time, config_number, folds_number, metric, scores)
    return params[1], search_info
