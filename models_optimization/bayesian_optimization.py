import time

import skopt
from sklearn import model_selection, neural_network

import config
from utility_functions import logging
from . import hyperparameters_spaces


def optimization_stopper(_):
    config.min_models_opt_iter += 1
    config.best_scores_optimization.append(config.bayesian_opt_.best_score_)

    if config.TIME_LIMITED_OPTIMIZATION is False:
        return False
    else:
        current_opt_time = time.time()
        time_elapsed = current_opt_time - config.initial_opt_time

        cond = False
        if len(config.best_scores_optimization) > config.EARLY_STOPPING_BO and config.linear_model_:
            if config.best_scores_optimization[-1] == config.best_scores_optimization[-config.EARLY_STOPPING_BO - 1] \
                    and config.EARLY_STOPPING_BO > 0:
                cond = True

        if (time_elapsed > config.max_model_optimization_time or cond) and config.min_models_opt_iter >= 2:
            return True
        else:
            return False


def bayesian_optimization(model_name, model, params_space, iterations, folds_number, metric, x_train_df, y_train_df,
                          random_state, verbose=False):
    if verbose:
        print(53 * '*' + '\nModel: {}'.format(model_name))

    optimized = False
    required_time = None

    if metric in ['balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
        k_fold = model_selection.StratifiedKFold(n_splits=folds_number, random_state=random_state, shuffle=True)
    else:
        k_fold = model_selection.KFold(n_splits=folds_number, random_state=random_state, shuffle=True)

    while not optimized:
        try:
            config.initial_opt_time = time.time()
            config.min_models_opt_iter = 0
            config.best_scores_optimization = []

            config.bayesian_opt_ = skopt.BayesSearchCV(model, params_space, n_iter=iterations,
                                                       cv=k_fold, scoring=metric, n_jobs=-1,
                                                       random_state=random_state,
                                                       optimizer_kwargs={'random_state': random_state,
                                                                         'n_jobs': -1,
                                                                         'n_initial_points': 100,
                                                                         'initial_point_generator': 'hammersly'})

            time_start = time.time()
            config.bayesian_opt_.fit(x_train_df, y_train_df, callback=optimization_stopper)
            required_time = time.time() - time_start
            optimized = True
        except ValueError:
            prev_iterations = iterations
            iterations -= 1

            logging.display('ValueError for optimizing using iterations = {}. '
                            'Trying with {} iterations.'.format(prev_iterations, iterations), p=1)

    if 'mean_squared_error' in metric:
        raw_score = -config.bayesian_opt_.best_score_
        best_score = raw_score / (max(y_train_df.values)[0] - min(y_train_df.values)[0])
    else:
        best_score = config.bayesian_opt_.best_score_

    best_params = config.bayesian_opt_.best_params_
    best_model = config.bayesian_opt_.best_estimator_

    if verbose:
        if metric == 'neg_root_mean_squared_error':
            print('nrmse: {}'.format(best_score))
        else:
            print('{}: {}'.format(metric, best_score))

        print('Optimization time: {}'.format(required_time))

    return [best_score, best_params, best_model, required_time]


def mlpc_optimization_stopper(_):
    config.min_models_opt_iter += 1

    current_opt_time = time.time()
    time_elapsed = current_opt_time - config.initial_opt_time

    if config.TIME_LIMITED_OPTIMIZATION is False:
        if time_elapsed > config.max_optimization_time_mlpc / 2:
            return True
        else:
            return False

    else:
        if time_elapsed > config.max_model_optimization_time / 2 and config.min_models_opt_iter >= 2:
            return True
        else:
            return False


def bayesian_optimization_mlpc(params_space, iterations, folds_number, metric, x_train_df, y_train_df, random_state,
                               verbose=False):
    if verbose:
        print(53 * '*' + '\nModel: mlpc')
    hidden_layers_architectures = hyperparameters_spaces.get_hidden_layers_architectures(rand_state=random_state)

    best_scores = []
    best_params = []
    best_models = []

    k_fold = model_selection.StratifiedKFold(n_splits=folds_number, random_state=random_state, shuffle=True)
    time_start = time.time()

    for hidden_layer_architecture in hidden_layers_architectures:
        mlpc = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden_layer_architecture,
                                            random_state=random_state)

        optimized = False

        while not optimized:
            try:
                config.initial_opt_time = time.time()
                config.min_models_opt_iter = 0

                config.bayesian_opt_ = skopt.BayesSearchCV(mlpc, params_space, n_iter=iterations,
                                                           cv=k_fold, scoring=metric, n_jobs=-1,
                                                           random_state=random_state,
                                                           optimizer_kwargs={'random_state': random_state,
                                                                             'n_jobs': -1,
                                                                             'n_initial_points': 100,
                                                                             'initial_point_generator': 'hammersly'})

                config.bayesian_opt_.fit(x_train_df, y_train_df, callback=mlpc_optimization_stopper)
                optimized = True
            except ValueError:
                prev_iterations = iterations
                iterations -= 1

                logging.display('ValueError for optimizing using iterations = {}. Trying with {} iterations.'.format(
                    prev_iterations, iterations), p=1)

        if 'mean_squared_error' in metric:
            raw_score = -config.bayesian_opt_.best_score_
            score = raw_score / (max(y_train_df.values)[0] - min(y_train_df.values)[0])
        else:
            score = config.bayesian_opt_.best_score_

        best_scores.append(score)
        best_params.append(config.bayesian_opt_.best_params_)
        best_models.append(config.bayesian_opt_.best_estimator_)

    required_time = time.time() - time_start
    best_index = best_scores.index(max(best_scores))

    if verbose:
        if metric == 'neg_root_mean_squared_error':
            print('nrmse: {}'.format(best_scores[best_index]))
        else:
            print('{}: {}'.format(metric, best_scores[best_index]))

        print('Optimization time: {}'.format(required_time))

    return [best_scores[best_index], best_params[best_index], best_models[best_index], required_time]
