import os

import numpy as np
from skopt.space import space

from utility_functions import files_storing, files_loading


def store_log_reg_param_space(folder_name):
    log_reg_param_space = {'C': space.Real(1e-6, 1e+1)}

    files_storing.serialize_param_space(folder_name, 'log_reg', log_reg_param_space)


def store_rfc_param_space(folder_name):
    n_estimators = list(range(64, 129, 4))
    max_features = ['sqrt', 'log2', None]
    max_samples = list(np.linspace(0.1, 0.9, 9)) + [None]
    min_samples_leaf = list(range(1, 21))
    class_weight = ['balanced', 'balanced_subsample']

    rfc_param_space = {'n_estimators': space.Categorical(n_estimators),
                       'max_features': space.Categorical(max_features),
                       'max_samples': space.Categorical(max_samples),
                       'min_samples_leaf': space.Categorical(min_samples_leaf),
                       'class_weight': space.Categorical(class_weight)}

    files_storing.serialize_param_space(folder_name, 'rfc', rfc_param_space)


def store_gbc_param_space(folder_name):
    loss = ['deviance']
    learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    subsample = list(np.linspace(0.1, 1.0, 10))
    criterion = ['friedman_mse', 'mse']
    max_features = ['sqrt', 'log2', None]
    min_samples_leaf = list(range(1, 21))

    gbc_param_space = {'loss': space.Categorical(loss),
                       'learning_rate': space.Categorical(learning_rate),
                       'subsample': space.Categorical(subsample),
                       'criterion': space.Categorical(criterion),
                       'max_features': space.Categorical(max_features),
                       'min_samples_leaf': space.Categorical(min_samples_leaf)}

    files_storing.serialize_param_space(folder_name, 'gbc', gbc_param_space)


def store_mlpc_param_space(folder_name):
    activation = ['logistic', 'tanh', 'relu']
    alpha = [0.00001, 0.00005, 0.0001, 0.0005,
             0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    mlpc_param_space = {'activation': space.Categorical(activation),
                        'alpha': space.Categorical(alpha)}

    files_storing.serialize_param_space(folder_name, 'mlpc', mlpc_param_space)


def store_ridge_reg_param_space(folder_name):
    solver = ['cholesky', 'sparse_cg', 'lsqr']

    ridge_reg_param_space = {'alpha': space.Real(0.0005, 5000),
                             'solver': space.Categorical(solver)}

    files_storing.serialize_param_space(folder_name, 'ridge_reg', ridge_reg_param_space)


def store_rfr_param_space(folder_name):
    n_estimators = list(range(64, 129, 4))
    max_features = ['sqrt', 'log2', None]
    max_samples = list(np.linspace(0.1, 0.9, 9)) + [None]
    min_samples_leaf = list(range(1, 21))

    rfr_param_space = {'n_estimators': space.Categorical(n_estimators),
                       'max_features': space.Categorical(max_features),
                       'max_samples': space.Categorical(max_samples),
                       'min_samples_leaf': space.Categorical(min_samples_leaf)}

    files_storing.serialize_param_space(folder_name, 'rfr', rfr_param_space)


def store_gbr_param_space(folder_name):
    loss = ['ls', 'lad']
    learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    subsample = list(np.linspace(0.1, 1.0, 10))
    criterion = ['friedman_mse', 'mse']
    max_features = ['sqrt', 'log2', None]
    min_samples_leaf = list(range(1, 21))

    gbr_param_space = {'loss': space.Categorical(loss),
                       'learning_rate': space.Categorical(learning_rate),
                       'subsample': space.Categorical(subsample),
                       'criterion': space.Categorical(criterion),
                       'max_features': space.Categorical(max_features),
                       'min_samples_leaf': space.Categorical(min_samples_leaf)}

    files_storing.serialize_param_space(folder_name, 'gbr', gbr_param_space)


def write_param_spaces(classifiers_spaces=True):
    if classifiers_spaces:
        subfolder_name = '/odinstorage/automl_data/parameters_spaces/classifiers/'
    else:
        subfolder_name = '/odinstorage/automl_data/parameters_spaces/regressors/'

    try:
        os.makedirs(subfolder_name)

        if classifiers_spaces:
            store_log_reg_param_space(subfolder_name)
            store_rfc_param_space(subfolder_name)
            store_gbc_param_space(subfolder_name)
            store_mlpc_param_space(subfolder_name)
        else:
            store_ridge_reg_param_space(subfolder_name)
            store_rfr_param_space(subfolder_name)
            store_gbr_param_space(subfolder_name)

    except FileExistsError:
        pass


def load_params_spaces(models_names, classifiers=True):
    spaces = {}

    if classifiers:
        folder_name = '/odinstorage/automl_data/parameters_spaces/classifiers/'
    else:
        folder_name = '/odinstorage/automl_data/parameters_spaces/regressors/'

    for model_name in models_names:
        spaces[model_name] = files_loading.deserialize_param_space(folder_name, model_name)

    return spaces


def display_param_spaces(param_spaces):
    for model_name, params in param_spaces.items():
        print('\n*' + model_name + '*')

        for param_name, domain in params.items():
            print(param_name.ljust(20) + ': ' + str(domain))


def generate_mlp_architectures(layers_sizes):
    one_hidden_layer = []
    two_hidden_layers = []

    for first_layer_size in layers_sizes:
        one_hidden_layer.append((first_layer_size,))
        for second_layer_size in layers_sizes:
            two_hidden_layers.append((first_layer_size, second_layer_size))

    neural_space = one_hidden_layer + two_hidden_layers

    return neural_space


def get_hidden_layers_architectures(rand_state, best_architecture=None, no_randomly_picked=1):
    if best_architecture is None:
        best_architecture = [(64, 32, 64)]

    np.random.seed(rand_state)
    architectures = generate_mlp_architectures([32, 64])

    randomly_picked_architectures = []
    indices = np.random.randint(len(architectures), size=no_randomly_picked)

    for index in indices:
        randomly_picked_architectures.append(architectures[index])

    selected_architectures = best_architecture + randomly_picked_architectures

    return selected_architectures
