import copy
import time

import pandas as pd

from data_cleaning import *
from feature_engineering import *
from utility_functions import logging, data_manipulation


def common_preprocessing(classification_task, train_continuous_df, train_categorical_df, y_train_df, config_dict,
                         models_dict, params, model_type):
    train_continuous_copy_df = train_continuous_df.copy(deep=True)
    train_categorical_copy_df = train_categorical_df.copy(deep=True)
    y_train_copy_df = y_train_df.copy(deep=True)
    config_copy_dict = copy.deepcopy(config_dict)
    models_copy_dict = copy.deepcopy(models_dict)

    if model_type == 0:
        first_log = True
    elif model_type == 1:
        first_log = False
    else:
        first_log = None

    # 7.f.
    if params[21] != -1:
        result = feature_selection.select_features(train_continuous_copy_df, train_categorical_copy_df, y_train_copy_df,
                                                   classification_task, params[21], model_type, params[1],
                                                   first_log=first_log)
        [train_continuous_copy_df, train_categorical_copy_df, discarded_columns, columns_importance] = result
        config_copy_dict['7f'] = discarded_columns

    else:
        config_copy_dict['7f'] = []
        columns_importance = None

    # 7.g
    if params[11] == 1 or params[13] == 1:
        related_to_all = False
    else:
        related_to_all = params[23]

    result = feature_construction.construct_features(train_continuous_copy_df, train_categorical_copy_df,
                                                     y_train_copy_df, columns_importance, classification_task,
                                                     params[1], related_to_all=related_to_all, select_max=params[24],
                                                     first_log=first_log)
    [selected_new_features, selected_new_features_df] = result
    config_copy_dict['7g'] = selected_new_features

    if selected_new_features is not None and selected_new_features_df is not None:
        result = scaling.scale_df(selected_new_features_df, scale_norm_type=params[18], continuous=True,
                                  first_log=first_log)
        [selected_new_features_df, scaler_new_features, _, _] = result
        models_copy_dict['9a_nf'] = scaler_new_features

        train_continuous_copy_df = pd.concat([train_continuous_copy_df, selected_new_features_df], axis=1)
    else:
        models_copy_dict['9a_nf'] = None

    cat_shape = train_categorical_copy_df.shape
    cont_shape = train_continuous_copy_df.shape

    if cat_shape[1] != 0 and cont_shape[1] != 0:
        if cat_shape[0] != cont_shape[0]:
            logging.display('Categorical and continuous dataframes have a different'
                            ' number of rows: {}, {}'.format(cat_shape[0], cont_shape[0]), p=0, first_log=first_log)
            return 7 * [None]

    if cat_shape[1] == 0:
        train_categorical_copy_df = None
    if cont_shape[1] == 0:
        train_continuous_copy_df = None

    logging.display('\nFinal shape of the training dataset: {}'.format((cat_shape[0], cat_shape[1] + cont_shape[1])),
                    end_line='', p=4, first_log=first_log)

    return [config_copy_dict, models_copy_dict, train_continuous_copy_df, train_categorical_copy_df, y_train_copy_df]


def data_preprocessing_training(dataset_name, predicted_column, params, classification_task, include_gbc):
    start_time = time.time()
    config_dict = {}
    models_dict = {}

    config_dict['add_is_missing'] = params[25]
    config_dict['add_text_features'] = params[26]

    # 0.
    input_df = data_manipulation.read_dataset('/odinstorage/automl_data/datasets/' + dataset_name + '/' + dataset_name + '.csv')
    logging.display('Initial shape of the dataset: {}'.format(input_df.shape), p=4)

    # 1.a. 1.b.
    result = useless_data_removal.remove_useless_data(input_df)
    [input_df, removed_columns_1a] = result
    config_dict['1a'] = removed_columns_1a

    # 2.
    result = data_manipulation.split_train_test(input_df, predicted_column, classification_task, test_size=params[0],
                                                random_state=params[1])
    [x_train_df, _, y_train_df, _, ordinal_encoding_target_dict, _, _] = result
    config_dict['2'] = {'predicted_column': predicted_column, 'classification_task': classification_task}
    models_dict['2'] = ordinal_encoding_target_dict

    # 1.c.
    current_rows_no = x_train_df.shape[0]
    result = useless_data_removal.remove_missing_data_columns(x_train_df, current_rows_no, add_is_missing=params[25])
    [x_train_df, removed_columns_1c] = result
    config_dict['1c'] = removed_columns_1c

    # 3. 7.c.
    x_train_copy_df = x_train_df.copy()
    x_train_df, added_columns = datatypes_conversion.convert_data_types_text_feat(x_train_df, normalize_text=params[2],
                                                                                  add_text_features=params[26])
    converted_columns_3 = datatypes_conversion.datatype_modification_checker(x_train_copy_df, x_train_df,
                                                                             normalize_text=params[2])
    config_dict['3'] = converted_columns_3
    if params[26]:
        config_dict['7c'] = added_columns
    else:
        config_dict['7c'] = params[26]

    # 4.
    result = columns_separation.split_dataset(x_train_df)
    [train_categorical_df, train_continuous_df, categorical_columns_4] = result
    config_dict['4'] = categorical_columns_4

    # 5.a.
    logging.display('5. Outliers detection', p=2)
    continuous_outliers_info_5a = None

    if params[3] is not None:
        result = outliers_detection.detect_outliers_continuous(train_continuous_df, method_type=params[3],
                                                               add_is_missing=params[25])
        [train_continuous_df, continuous_outliers_info_5a] = result
        config_dict['5a'] = continuous_outliers_info_5a
    else:
        logging.display('5.a. Without outliers detection in continuous columns', p=3)
        config_dict['5a'] = None

    # 5.b.
    if params[4]:
        result = outliers_detection.detect_outliers_categorical(train_categorical_df, add_is_missing=params[25])
        [train_categorical_df, categorical_outliers_info_5b] = result
        config_dict['5b'] = categorical_outliers_info_5b
    else:
        logging.display('5.b. Without outliers detection in categorical columns', p=3)
        config_dict['5b'] = None

    if params[3] is not None and params[25]:
        result = feature_construction.add_missing_columns(train_continuous_df, train_categorical_df,
                                                          continuous_outliers_info_5a)
        [train_categorical_df, train_continuous_df] = result

    # 6.a.
    result = missing_data_imputation.impute_missing_continuous_values(train_continuous_df, technique=params[5],
                                                                      neighbors_no=params[6])
    [train_continuous_df, imputation_continuous_info_6a, fitted_imputer_continuous] = result
    config_dict['6a'] = imputation_continuous_info_6a
    models_dict['6a'] = fitted_imputer_continuous

    # 7.d.
    if params[7]:
        result = feature_construction.transform_skewed_features(train_continuous_df, y_train_df)
        [train_continuous_df, skew_dict_7d, remained_columns] = result
        config_dict['7d'] = [skew_dict_7d, remained_columns]
    else:
        config_dict['7d'] = [None, None]

    # 6.b.
    result = missing_data_imputation.impute_missing_categorical_values(train_continuous_df, train_categorical_df,
                                                                       use_knn=params[8], neighbors_no=params[9],
                                                                       use_continuous_data=params[10])
    [train_categorical_df, imputation_categorical_info_6b, encoder_dict, fitted_imputer_categorical] = result
    config_dict['6b'] = imputation_categorical_info_6b
    models_dict['6b'] = {'encoder_dict': encoder_dict, 'fitted_imputer_categorical': fitted_imputer_categorical}

    # 7.a.
    result = feature_construction.process_dates(train_continuous_df, train_categorical_df)
    [train_continuous_df, train_categorical_df, date_transform_info_7a] = result
    config_dict['7a'] = date_transform_info_7a

    # 7.b. 8.
    result = encoding.encode_categorical_values(train_categorical_df, rand_state=params[1], reduce_dim=params[17],
                                                num_enc_type=params[11], long_text_enc_type=params[12],
                                                short_text_enc_type=params[13], min_freq=params[14],
                                                max_freq=params[15])
    [train_categorical_df, scalable_columns_names_8, encoders_dict, svd_dict] = result
    config_dict['8'] = [params[17], params[1]]
    models_dict['8'] = {'encoders_dict': encoders_dict, 'svd_dict': svd_dict}

    results = columns_separation.move_to_continuous(train_categorical_df, train_continuous_df, scalable_columns_names_8)
    [train_categorical_df, train_continuous_df, scalable_columns_names_8] = results
    config_dict['8'].append(scalable_columns_names_8)

    result = useless_data_removal.remove_zero_variance_columns(train_continuous_df, train_categorical_df)
    [train_continuous_df, train_categorical_df, removed_columns_wo_var] = result
    config_dict['removed_columns_wo_var'] = removed_columns_wo_var

    # 9.a.
    result = scaling.scale_df(train_continuous_df, scale_norm_type=params[18], continuous=True)
    [train_continuous_df, scaler_continuous, scaling_continuous_9a, _] = result
    config_dict['9a'] = scaling_continuous_9a
    models_dict['9a'] = scaler_continuous

    # 9.b.
    result = scaling.scale_categorical_columns(train_categorical_df, scalable_columns_names=scalable_columns_names_8,
                                               scale_norm_type=params[19], scale_entire_df=params[20])
    [train_categorical_df, scaler_categorical, scaling_categorical_9b] = result
    config_dict['9b'] = scaling_categorical_9b
    models_dict['9b'] = scaler_categorical

    if params[22] == -1:
        result = [
            common_preprocessing(classification_task, train_continuous_df, train_categorical_df, y_train_df,
                                 config_dict, models_dict, params, params[22])]
        logging.display('\nTime required for preparing the training'
                        ' dataset: {} s'.format(round(time.time() - start_time, 6)), p=4)
    else:
        common_time = time.time() - start_time
        start_time = time.time()
        result = [
            common_preprocessing(classification_task, train_continuous_df, train_categorical_df, y_train_df,
                                 config_dict, models_dict, params, 0)]
        logging.display('\nTime required for preparing the training dataset:'
                        ' {} s'.format(round(time.time() - start_time + common_time, 6)), p=4, first_log=True)

        if (classification_task and include_gbc) or (not classification_task):
            start_time = time.time()
            result += [
                common_preprocessing(classification_task, train_continuous_df, train_categorical_df, y_train_df,
                                     config_dict, models_dict, params, 1)]
            logging.display('\nTime required for preparing the training dataset: {} s'.format(
                round(time.time() - start_time + common_time, 6)), p=4, first_log=False)

    return result
