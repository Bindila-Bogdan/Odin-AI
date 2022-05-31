import time

from data_cleaning import *
from feature_engineering import encoding, feature_construction, scaling, feature_selection
from utility_functions import logging, files_storing
from . import test_data_prediction


def data_preprocessing_testing(x_test_df, y_test_df, config_dict, models_dict, handle_unknown_classes):
    start_time = time.time()

    logging.display('\nInitial shape of the testing dataset: {}'.format(x_test_df.shape), p=4, end_line='')

    add_is_missing = config_dict['add_is_missing']

    # 2.
    if y_test_df is not None:
        y_test_df = encoding.target_ordinal_encoding(y_test_df, config_dict, models_dict,
                                                     handle_unknown_classes=handle_unknown_classes)

    # 1.a
    x_test_df.drop(config_dict['1a'], axis=1, inplace=True)

    # 1.c.
    x_test_df = useless_data_removal.remove_missing_data_columns_test(x_test_df, config_dict, add_is_missing)

    # 3. 7.c.
    x_test_df = datatypes_conversion.convert_data_types_text_feat_test(x_test_df, config_dict)
    datatypes_conversion.failed_data_conversion(x_test_df, config_dict)

    # 4.
    result = columns_separation.split_dataset_test(x_test_df, config_dict)
    [test_categorical_df, test_continuous_df] = result

    # 5.a.
    if config_dict['5a'] is not None:
        test_continuous_df = outliers_detection.detect_outliers_continuous_test(test_continuous_df, config_dict,
                                                                                add_is_missing)

    # 5.b.
    if config_dict['5b'] is not None:
        test_categorical_df = outliers_detection.detect_outliers_categorical_test(test_categorical_df, config_dict,
                                                                                  add_is_missing)

    if config_dict['5a'] is not None and add_is_missing:
        result = feature_construction.add_missing_columns_test(test_continuous_df, test_categorical_df)
        [test_continuous_df, test_categorical_df] = result

    # 6.a.
    test_continuous_df = missing_data_imputation.impute_missing_continuous_values_test(test_continuous_df, config_dict,
                                                                                       models_dict)

    # 7.d.
    test_continuous_df = feature_construction.transform_skewed_features_test(test_continuous_df, config_dict)

    # 6.b.
    test_categorical_df = missing_data_imputation.impute_missing_categorical_values_test(test_categorical_df,
                                                                                         test_continuous_df,
                                                                                         config_dict, models_dict)

    # 7.a.
    result = feature_construction.process_dates_test(test_categorical_df, test_continuous_df, config_dict)
    [test_categorical_df, test_continuous_df] = result

    # 8.
    test_categorical_df = encoding.encode_categorical_values_test(test_categorical_df, config_dict, models_dict)
    categorical_scalable_columns = config_dict['8'][2]
    result = columns_separation.move_to_continuous_test(test_categorical_df, test_continuous_df)
    [test_categorical_df, test_continuous_df] = result

    result = useless_data_removal.remove_zero_variance_columns_test(test_continuous_df, test_categorical_df,
                                                                    config_dict)
    [test_continuous_df, test_categorical_df] = result

    # 9.a.
    test_continuous_df = scaling.scale_continuous_columns_test(test_continuous_df, config_dict, models_dict)
    # 9.b.
    test_categorical_df = scaling.scale_categorical_columns_test(test_categorical_df, config_dict, models_dict,
                                                                 categorical_scalable_columns)

    # 7.f.
    result = feature_selection.remove_features(test_continuous_df, test_categorical_df, config_dict['7f'])
    [test_continuous_df, test_categorical_df] = result

    cat_shape = test_categorical_df.shape
    cont_shape = test_continuous_df.shape

    # 7.g.
    test_continuous_df = feature_construction.construct_features_test(test_continuous_df, config_dict, models_dict)

    if cat_shape[1] != 0 and cont_shape[1] != 0:
        if cat_shape[0] != cont_shape[0]:
            logging.display('Categorical and continuous dataframes have a '
                            'different number of rows: {}, {}'.format(cat_shape[0], cont_shape[0]), p=0)
            return None, None

    if cat_shape[1] == 0:
        test_categorical_df = None
    if cont_shape[1] == 0:
        test_continuous_df = None

    logging.display('\nFinal shape of the testing dataset: {}'.format((cat_shape[0], cat_shape[1] + cont_shape[1])),
                    end_line='', p=4)
    logging.display('\nTime required for testing: {} s'.format(round(time.time() - start_time, 6)), p=4)

    return [test_continuous_df, test_categorical_df, y_test_df]


def preprocess_test_datasets(x_test_initial_df, y_test_initial_df, models_configs, config_dataset, dataset_name,
                             predicted_column, new_subfolder_name, handle_unknown_classes):
    models_data = {}
    encoded_y_test_df = None
    models_dict = None

    for config_name, [config_dict, models_dict] in models_configs.items():
        x_test_copy_df = x_test_initial_df.copy(deep=True)

        if y_test_initial_df is not None:
            y_test_copy_df = y_test_initial_df.copy(deep=True)
        else:
            y_test_copy_df = None

        result = data_preprocessing_testing(x_test_copy_df, y_test_copy_df, config_dict, models_dict,
                                            handle_unknown_classes)
        [test_continuous_df, test_categorical_df, y_test_df] = result

        x_train_df = config_dataset[config_name][0]
        x_test_df = test_data_prediction.check_construct(x_train_df, test_continuous_df, test_categorical_df)
        encoded_y_test_df = y_test_df

        if x_test_df is None:
            return None, None, None

        models_data[config_name] = x_test_df
        files_storing.save_preprocessed_dataset(dataset_name, predicted_column, config_name, new_subfolder_name,
                                                x_test_df, None, None, False)

    return models_data, encoded_y_test_df, models_dict
