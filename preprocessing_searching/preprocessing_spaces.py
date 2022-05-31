import itertools
import os
import pickle
import random

from utility_functions import logging


def get_params_reduced(model_type, construct_features, det_cont_outliers, detect_cat_outliers):
    # 2.
    test_size = [0.0]
    random_state = [42]

    # 3.
    normalize_text = [True]

    # 5.a.
    if model_type != -1:
        if det_cont_outliers:
            method_type = ['three_std']
        else:
            method_type = [None]
    else:
        method_type = ['three_std']

    # 5.b
    cat_outliers_det = [detect_cat_outliers]

    # 6.a.
    technique = ['kNN', 'mean']
    neighbors_no_cont = [5]

    # 7.d
    transform_skewed = [True, False]

    # 6.b.
    use_knn = [True, False]
    neighbors_no_cat = [1]

    if model_type != -1:
        if det_cont_outliers:
            use_continuous_data = [True]
        else:
            use_continuous_data = [False]
    else:
        use_continuous_data = [True]

        # 8.
    if model_type != -1:
        num_enc_type = [2, 0, -1]
        short_text_enc_type = [0]
    else:
        num_enc_type = [2, 0, 1, -1]
        short_text_enc_type = [0, 1]

    long_text_enc_type = [1]
    min_freq = [0.05]
    max_freq = [0.99]
    max_features = [100]
    reduce_dim = [True, False]

    if model_type != -1:
        # 9.a.
        scale_norm_type_cont = [2]

        # 9.b.
        scale_norm_type_cat = [2]
    else:
        # 9.a.
        scale_norm_type_cont = [2, 1]

        # 9.b.
        scale_norm_type_cat = [2, 1]

    scale_entire_df = [True]

    # 7.f.
    if model_type != -1:
        min_importance = [0.0]
    else:
        min_importance = [0.01]
    tree_based_model = [model_type]

    # 7.g
    related_to_all = [False]

    if construct_features:
        select_max = [3]

        add_is_missing = [True]

        # 7.c
        add_text_features = [True]
    else:
        select_max = [0]

        add_is_missing = [False]

        # 7.c
        add_text_features = [False]

    params_space = [test_size, random_state, normalize_text, method_type, cat_outliers_det, technique,
                    neighbors_no_cont, transform_skewed, use_knn, neighbors_no_cat, use_continuous_data, num_enc_type,
                    long_text_enc_type, short_text_enc_type, min_freq, max_freq, max_features, reduce_dim,
                    scale_norm_type_cont, scale_norm_type_cat, scale_entire_df, min_importance, tree_based_model,
                    related_to_all, select_max, add_is_missing, add_text_features]

    list_of_params = list(itertools.product(*params_space))

    return list_of_params


def get_params_extended(model_type, normalize_txt, use_tf_idf, construct_features, detect_cont_outliers,
                        detect_cat_outliers):
    # 2.
    test_size = [0.0]
    random_state = [42]

    # 3.
    normalize_text = [normalize_txt]

    # 5.a.
    if model_type != -1:
        if detect_cont_outliers:
            method_type = ['three_std', 'iqr']
        else:
            method_type = [None]
    else:
        method_type = ['three_std', 'iqr']

    # 5.b
    cat_outliers_det = [detect_cat_outliers]

    # 6.a.
    technique = ['kNN', 'mean', 'median']

    neighbors_no_cont = [5]

    # 7.d
    transform_skewed = [True, False]

    # 6.b.
    use_knn = [True, False]
    neighbors_no_cat = [1]

    if model_type != -1:
        if detect_cont_outliers:
            use_continuous_data = [True]
        else:
            use_continuous_data = [False]
    else:
        use_continuous_data = [True]

    # 8.
    if model_type != -1:
        num_enc_type = [2, 0, -1]
        short_text_enc_type = [0]
    else:
        num_enc_type = [2, 0, 1, -1]
        short_text_enc_type = [0, 1]

    if normalize_txt:
        if use_tf_idf:
            long_text_enc_type = [1]
        else:
            long_text_enc_type = [0]
    else:
        long_text_enc_type = [1]

    min_freq = [0.05, 0.01]
    max_freq = [0.99]
    max_features = [100]

    if use_tf_idf:
        reduce_dim = [True]
    else:
        reduce_dim = [False]

    if model_type != -1:
        # 9.a.
        scale_norm_type_cont = [2]

        # 9.b.
        scale_norm_type_cat = [2]
    else:
        # 9.a.
        if detect_cont_outliers:
            scale_norm_type_cont = [2, 0, 1]
        else:
            scale_norm_type_cont = [2, 0, 1, 3]

        # 9.b.
        scale_norm_type_cat = [2, 0, 1]

    if model_type != -1:
        scale_entire_df = [True, False]
    else:
        scale_entire_df = [True]

    # 7.f.
    if model_type != -1:
        min_importance = [0.0, -1]
    else:
        min_importance = [0.01, 0.0, -1]
    tree_based_model = [model_type]

    # 7.g
    if construct_features:
        related_to_all = [True, False]
        select_max = [3]
    else:
        related_to_all = [True]
        select_max = [0]

    add_is_missing = [True, False]

    # 7.c
    add_text_features = [True, False]

    params_space = [test_size, random_state, normalize_text, method_type, cat_outliers_det, technique,
                    neighbors_no_cont, transform_skewed, use_knn, neighbors_no_cat, use_continuous_data, num_enc_type,
                    long_text_enc_type, short_text_enc_type, min_freq, max_freq, max_features, reduce_dim,
                    scale_norm_type_cont, scale_norm_type_cat, scale_entire_df, min_importance, tree_based_model,
                    related_to_all, select_max, add_is_missing, add_text_features]

    list_of_params = list(itertools.product(*params_space))

    return list_of_params


def generate_store_params_reduced():
    model_types = [-1, 0]
    construct_features_options = [True, False]
    detect_cont_outliers_options = [True, False]
    detect_cat_outliers_options = [False, True]

    params_reduced = {-1: [], 0: []}

    for model_type in model_types:
        for construct_features in construct_features_options:
            for detect_cont_outliers in detect_cont_outliers_options:
                for detect_cat_outliers in detect_cat_outliers_options:
                    params = get_params_reduced(model_type, construct_features, detect_cont_outliers,
                                                detect_cat_outliers)
                    params_reduced[model_type] += params

    model_name = None

    for model_type in params_reduced.keys():
        if model_type == -1:
            model_name = 'non_tree_based'
        elif model_type == 0:
            model_name = 'tree_based'

        try:
            os.makedirs('/odinstorage/automl_data/preprocessing_params/reduced/')
        except FileExistsError:
            logging.display('automl_data/preprocessing_params/reduced subfolder already exists', p=4)

        file_name = '/odinstorage/automl_data/preprocessing_params/reduced/' + model_name + '.pickle'

        with open(file_name, 'wb') as pickle_file:
            pickle.dump(params_reduced[model_type], pickle_file)


def generate_store_params_extended(params_reduced, rand_state):
    random.seed(rand_state)

    model_types = [-1, 0]
    normalize_txt_options = [True, False]
    use_tf_idf_options = [True, False]
    construct_features_options = [True, False]
    detect_cont_outliers_options = [True, False]
    detect_cat_outliers_options = [False, True]

    params_extended = {-1: [], 0: []}

    for model_type in model_types:
        for normalize_txt in normalize_txt_options:
            for use_tf_idf in use_tf_idf_options:
                for construct_features in construct_features_options:
                    for detect_cont_outliers in detect_cont_outliers_options:
                        for detect_cat_outliers in detect_cat_outliers_options:
                            params_ = get_params_extended(model_type, normalize_txt, use_tf_idf, construct_features,
                                                          detect_cont_outliers, detect_cat_outliers)
                            params = sorted(list(set(params_) - set(params_reduced[model_type])))
                            params_extended[model_type] += random.sample(params, len(params) // 10)

    model_name = None

    for model_type in params_extended.keys():
        if model_type == -1:
            model_name = 'non_tree_based'
        elif model_type == 0:
            model_name = 'tree_based'

        try:
            os.makedirs('/odinstorage/automl_data/preprocessing_params/extended/')
        except FileExistsError:
            logging.display('automl_data/preprocessing_params/extended subfolder already exists', p=4)

        file_name = '/odinstorage/automl_data/preprocessing_params/extended/' + model_name + '.pickle'

        with open(file_name, 'wb') as pickle_file:
            pickle.dump(params_extended[model_type], pickle_file)

    return params_extended


def select_diverse_configs(configs, reduced_configs, tree_based_configs, rand_state):
    random.seed(rand_state)
    random.shuffle(configs)

    if reduced_configs:
        if tree_based_configs:
            evidence = [[], [], [True], ['three_std', None], [True, False], ['kNN', 'mean'], [5], [True, False],
                        [True, False], [1], [True, False], [2, 0, -1], [1], [0], [0.05], [0.99], [], [True, False],
                        [2], [2], [True, False], [0.0], [], [False], [3, 0], [True, False], [True, False]]
        else:
            evidence = [[], [], [True], ['three_std'], [True, False], ['kNN', 'mean'], [5], [True, False],
                        [True, False], [1], [True], [2, 0, 1, -1], [1], [0, 1], [0.05], [0.99], [], [True, False],
                        [2, 1], [2, 1], [True], [0.01], [], [False], [3, 0], [True, False], [True, False]]
    else:
        if tree_based_configs:
            evidence = [[], [], [True, False], ['three_std', 'iqr', None], [True, False], ['kNN', 'mean', 'median'],
                        [5], [True, False], [True, False], [1], [True, False], [2, 0, -1], [0, 1], [0], [0.05, 0.01],
                        [0.99], [], [True, False], [2], [2], [True, False], [0.0, -1], [], [True, False], [0, 3],
                        [True, False], [True, False]]
        else:
            evidence = [[], [], [True, False], ['three_std', 'iqr'], [True, False], ['kNN', 'mean', 'median'], [5],
                        [True, False], [True, False], [1], [True], [2, 0, 1, -1], [0, 1], [0, 1], [0.05, 0.01], [0.99],
                        [], [True, False], [2, 0, 1, 3], [2, 0, 1], [True], [0.01, 0.0, -1], [], [True, False], [0, 3],
                        [True, False], [True, False]]

    diverse_configs = []

    for config in configs:
        add_config = False

        for index, param in enumerate(config):
            if param in evidence[index]:
                add_config = True
                evidence[index].remove(param)

        if add_config:
            diverse_configs.append(config)

    return diverse_configs


def load_params_preprocessing(type_of_params, configs_no, rand_state):
    random.seed(rand_state)

    configs_no_non_tree_based = configs_no // 2
    configs_no_tree_based = configs_no // 2 + (configs_no - configs_no // 2 * 2)

    model_name = None
    params = {-1: [], 0: []}

    for model_type in params.keys():
        if model_type == -1:
            model_name = 'non_tree_based'
        elif model_type == 0:
            model_name = 'tree_based'

        with open('/odinstorage/automl_data/preprocessing_params/' + type_of_params + '/' + model_name + '.pickle', 'rb') as f:
            params[model_type] = pickle.load(f)

    if configs_no == -1:
        return params

    selected_configs = []

    if type_of_params == 'reduced':
        selected_configs.append(params[-1][0])
        selected_configs.append(params[0][0])

        diverse_non_tree_based = select_diverse_configs(params[-1][1:], True, False,
                                                        rand_state)[:configs_no_non_tree_based]
        diverse_tree_based = select_diverse_configs(params[0][1:], True, True, rand_state)[:configs_no_tree_based]

        remained_configs_non_tree_based = [param for param in params[-1][1:] if param not in diverse_non_tree_based]
        remained_configs_tree_based = [param for param in params[0][1:] if param not in diverse_tree_based]

    else:

        diverse_non_tree_based = select_diverse_configs(params[-1], False, False,
                                                        rand_state)[:configs_no_non_tree_based]
        diverse_tree_based = select_diverse_configs(params[0], False, True, rand_state)[:configs_no_tree_based]

        remained_configs_non_tree_based = [param for param in params[-1] if param not in diverse_non_tree_based]
        remained_configs_tree_based = [param for param in params[0] if param not in diverse_tree_based]

    remained_no_non_tree_based = configs_no_non_tree_based - len(diverse_non_tree_based)
    remained_no_tree_based = configs_no_tree_based - len(diverse_tree_based)

    if remained_no_non_tree_based > 0:
        selected_configs.append(
            diverse_non_tree_based + random.sample(remained_configs_non_tree_based, remained_no_non_tree_based))
    else:
        selected_configs.append(diverse_non_tree_based)

    if remained_no_tree_based > 0:
        selected_configs.append(
            diverse_tree_based + random.sample(remained_configs_tree_based, remained_no_tree_based))
    else:
        selected_configs.append(diverse_tree_based)

    return selected_configs


def load_configs(configurations_no, time_limited_searching, rand_state):
    try:
        os.makedirs('/odinstorage/automl_data/preprocessing_params/reduced/')
        os.makedirs('/odinstorage/automl_data/preprocessing_params/extended/')
        generate_store_params_reduced()
        generate_store_params_extended(load_params_preprocessing('reduced', -1, rand_state), rand_state)
    except FileExistsError:
        pass

    reduced_configs_no = (configurations_no - 2) // 3 * 2 + ((configurations_no - 2) - (configurations_no - 2) // 3 * 3)

    if reduced_configs_no > 766:
        reduced_configs_no = 766
        extended_configs_no = configurations_no - 768
    else:
        extended_configs_no = (configurations_no - 2) // 3

    selected_reduced_configs = load_params_preprocessing('reduced', reduced_configs_no, rand_state)

    best_config_non_tree_based = selected_reduced_configs[0]
    best_config_tree_based = selected_reduced_configs[1]
    del selected_reduced_configs[0]
    del selected_reduced_configs[0]

    selected_extended_configs = load_params_preprocessing('extended', extended_configs_no, rand_state)

    if time_limited_searching:
        i = 0
        non_tree_based_configs = []
        ntb_reduced_len = len(selected_reduced_configs[0])
        ntb_general_len = len(selected_extended_configs[0])

        for i in range(min(ntb_reduced_len, ntb_general_len)):
            non_tree_based_configs.append(selected_reduced_configs[0][i])
            non_tree_based_configs.append(selected_extended_configs[0][i])

        if ntb_reduced_len < ntb_general_len:
            non_tree_based_configs += selected_extended_configs[0][i + 1:]
        elif ntb_reduced_len > ntb_general_len:
            non_tree_based_configs += selected_reduced_configs[0][i + 1:]

        tree_based_configs = []
        tb_reduced_len = len(selected_reduced_configs[1])
        tb_general_len = len(selected_extended_configs[1])

        for i in range(min(tb_reduced_len, tb_general_len)):
            tree_based_configs.append(selected_reduced_configs[1][i])
            tree_based_configs.append(selected_extended_configs[1][i])

        if tb_reduced_len < tb_general_len:
            tree_based_configs += selected_extended_configs[1][i + 1:]
        elif tb_reduced_len > tb_general_len:
            tree_based_configs += selected_reduced_configs[1][i + 1:]

    else:
        non_tree_based_configs = selected_reduced_configs[0] + selected_extended_configs[0]
        tree_based_configs = selected_reduced_configs[1] + selected_extended_configs[1]

    return [best_config_non_tree_based, best_config_tree_based, non_tree_based_configs, tree_based_configs]
