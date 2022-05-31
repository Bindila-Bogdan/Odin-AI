import time

import numpy as np
from sklearn import linear_model, ensemble

import config
from . import bayesian_optimization, hyperparameters_spaces


def optimize_classifiers(classifiers_names, iterations, folds_number, metric, model_config_mapping, config_data,
                         random_state, include_gbc=False, verbose=True):
    params_spaces = hyperparameters_spaces.load_params_spaces(classifiers_names, classifiers=True)

    log_reg = linear_model.LogisticRegression(n_jobs=-1, class_weight='balanced', random_state=random_state)
    rfc = ensemble.RandomForestClassifier(n_jobs=-1, random_state=random_state)
    gbc = ensemble.GradientBoostingClassifier(n_estimators=128, n_iter_no_change=20, max_depth=100,
                                              random_state=random_state)

    if include_gbc:
        initial_classifiers = {'log_reg': log_reg, 'rfc': rfc, 'gbc': gbc, 'mlpc': None}
        optimized_classifiers = {'log_reg': None, 'rfc': None, 'gbc': None, 'mlpc': None}

    else:
        initial_classifiers = {'log_reg': log_reg, 'rfc': rfc, 'mlpc': None}
        optimized_classifiers = {'log_reg': None, 'rfc': None, 'mlpc': None}

    models_no = len(list(optimized_classifiers.keys()))
    config.max_model_optimization_time = config.max_optimization_time / models_no

    for classifier_name, classifier in initial_classifiers.items():
        if classifier_name == 'log_reg':
            config.linear_model_ = True
        else:
            config.linear_model_ = False

        time_before_opt = time.time()
        params_space = params_spaces[classifier_name]

        x_train_df = config_data[model_config_mapping[classifier_name]][0]
        y_train_df = config_data['y_train']

        if classifier_name == 'mlpc':
            if iterations < 2:
                mlpc_iterations = 1
            else:
                mlpc_iterations = iterations // 2

            optimization_results = bayesian_optimization.bayesian_optimization_mlpc(params_space, mlpc_iterations,
                                                                                    folds_number, metric,
                                                                                    x_train_df, y_train_df,
                                                                                    random_state, verbose)
        else:
            optimization_results = bayesian_optimization.bayesian_optimization(classifier_name, classifier,
                                                                               params_space, iterations,
                                                                               folds_number, metric, x_train_df,
                                                                               y_train_df, random_state,
                                                                               verbose)

        optimized_classifiers[classifier_name] = optimization_results

        models_no -= 1
        config.max_optimization_time -= (time.time() - time_before_opt)

        if models_no > 0:
            config.max_model_optimization_time = config.max_optimization_time / models_no

        if config.max_optimization_time_mlpc < optimization_results[-1]:
            config.max_optimization_time_mlpc = optimization_results[-1]

    return optimized_classifiers, params_spaces


def optimize_regressors(regressors_names, iterations, folds_number, metric, model_config_mapping, config_data,
                        random_state, verbose=True):
    params_spaces = hyperparameters_spaces.load_params_spaces(regressors_names, classifiers=False)

    ridge_reg = linear_model.Ridge(random_state=random_state)
    rfr = ensemble.RandomForestRegressor(n_jobs=-1, random_state=random_state)
    gbr = ensemble.GradientBoostingRegressor(n_estimators=128, n_iter_no_change=20, max_depth=100,
                                             random_state=random_state)

    initial_regressors = {'ridge_reg': ridge_reg, 'rfr': rfr, 'gbr': gbr}
    optimized_regressors = {'ridge_reg': None, 'rfr': None, 'gbr': None}

    models_no = len(list(optimized_regressors.keys()))
    config.max_model_optimization_time = config.max_optimization_time / models_no

    for regressor_name, regressor in initial_regressors.items():
        if regressor_name == 'ridge_reg':
            config.linear_model_ = True
        else:
            config.linear_model_ = False

        time_before_opt = time.time()
        params_space = params_spaces[regressor_name]

        x_train_df = config_data[model_config_mapping[regressor_name]][0]
        y_train_df = config_data['y_train']

        optimization_results = bayesian_optimization.bayesian_optimization(regressor_name, regressor, params_space,
                                                                           iterations, folds_number, metric, x_train_df,
                                                                           y_train_df, random_state, verbose)
        optimized_regressors[regressor_name] = optimization_results

        models_no -= 1
        config.max_optimization_time -= (time.time() - time_before_opt)

        if models_no > 0:
            config.max_model_optimization_time = config.max_optimization_time / models_no

    return optimized_regressors, params_spaces


def extract_optimization_info(optimized_models):
    optimization_times = []
    obtained_metric_values = []

    for model_name, optimization_data in optimized_models.items():
        optimization_times.append(optimization_data[-1])
        obtained_metric_values.append(optimization_data[0])

    total_time_required = round(np.array(optimization_times).sum(), 4)
    print(53 * '*' + '\nTime required by optimization: {} s\n'.format(round(total_time_required, 2)))

    return optimization_times, obtained_metric_values, total_time_required


def optimize_models(model_config_mapping, config_data, classification_task, metric, iterations_bo, folds_number_bo,
                    random_state, include_gbc):
    print('*Optimizing models*')
    if include_gbc:
        classifiers_names = ['log_reg', 'rfc', 'gbc', 'mlpc']
    else:
        classifiers_names = ['log_reg', 'rfc', 'mlpc']

    regressors_names = ['ridge_reg', 'rfr', 'gbr']

    if classification_task:
        optimized_models, models_params_spaces = optimize_classifiers(classifiers_names, iterations_bo, folds_number_bo,
                                                                      metric, model_config_mapping, config_data,
                                                                      random_state, include_gbc)
    else:
        optimized_models, models_params_spaces = optimize_regressors(regressors_names, iterations_bo, folds_number_bo,
                                                                     metric, model_config_mapping, config_data,
                                                                     random_state)

    optimization_times, obtained_metric_values, total_time_required = extract_optimization_info(optimized_models)

    return [optimized_models, models_params_spaces, optimization_times, obtained_metric_values, total_time_required]
