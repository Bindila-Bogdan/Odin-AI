from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import impute, preprocessing

from utility_functions import logging, files_storing
from . import datatypes_conversion


def knn_imputation(input_df, neighbors_no):
    columns_names = input_df.columns

    knn_imp = impute.KNNImputer(n_neighbors=neighbors_no)
    fitted_imp = knn_imp.fit(input_df)
    imputed = fitted_imp.transform(input_df)
    imputed_df = pd.DataFrame(imputed, columns=columns_names)

    return imputed_df, fitted_imp


def impute_missing_continuous_values(continuous_df, technique='kNN', neighbors_no=5):
    imputation_continuous_info = {}

    if continuous_df.shape[1] == 0:
        imputation_continuous_info['method'] = 'w/o_cont_columns'
        return [continuous_df, imputation_continuous_info, None]

    logging.display('6. Imputation of missing values', p=2)
    logging.display('6.a. Imputation for continuous data', p=3)

    fitted_imp = None

    if technique == 'kNN':
        imputation_continuous_info['method'] = technique + '_' + str(neighbors_no)
        logging.display('kNN with {} neighbors is used for imputation'.format(neighbors_no), p=4)
    else:
        imputation_continuous_info['method'] = technique
        logging.display('{} is used for imputation'.format(technique), p=4)

    if technique == 'mean':
        mean = continuous_df.mean()
        imputed_df = continuous_df.fillna(mean)
        imputed_means = dict(zip(list(mean.index), list(mean.values)))
        imputation_continuous_info['imputed_values'] = imputed_means
        logging.display('Imputed values:', p=4)
        logging.display_json(imputed_means)
    elif technique == 'median':
        median = continuous_df.median()
        imputed_df = continuous_df.fillna(median)
        imputed_medians = dict(zip(list(median.index), list(median.values)))
        imputation_continuous_info['imputed_values'] = imputed_medians
        logging.display('Imputed values:', p=4)
        logging.display_json(imputed_medians)
    else:
        imputed_df, fitted_imp = knn_imputation(continuous_df, neighbors_no)
        imputation_continuous_info['imputed_values'] = {}

    return [imputed_df, imputation_continuous_info, fitted_imp]


def encode_categorical_columns(categorical_df):
    column_type_dict = {}

    for column in categorical_df.columns:
        data_type = categorical_df[column].dtype
        if data_type == 'int64':
            column_type_dict[column] = 'int64'
        elif data_type == 'float64':
            column_type_dict[column] = 'float64'
        elif data_type == 'O':
            column_type_dict[column] = 'O'
        elif data_type == 'datetime64[ns]':
            column_type_dict[column] = 'datetime64[ns]'

    encoder_dict = defaultdict(preprocessing.LabelEncoder)
    encoded_categorical_df = categorical_df.apply(
        lambda col: encoder_dict[col.name].fit_transform(col.astype(str)))
    encoded_final_categorical_df = encoded_categorical_df.where(~categorical_df.isna(), categorical_df)
    encoded_final_categorical_df = encoded_final_categorical_df.replace(pd.NaT, np.nan)

    return encoded_final_categorical_df, encoder_dict, column_type_dict


def decode_categorical_columns(encoded_categorical_df, encoder_dict, column_type_dict):
    decoded_categorical_df = encoded_categorical_df.apply(
        lambda column: encoder_dict[column.name].inverse_transform(column.astype(int)))

    for (column_name, column_type) in column_type_dict.items():
        if column_type == 'int64' or column_type == 'float64':
            decoded_categorical_df[column_name], _ = datatypes_conversion.object_to_numeric(
                decoded_categorical_df[column_name])
        elif column_type == 'datetime64[ns]':
            decoded_categorical_df[column_name], _ = datatypes_conversion.object_to_date(
                decoded_categorical_df[column_name], thresh=1)

    return decoded_categorical_df


def knn_impute_categorical(continuous_df, categorical_df, neighbors_no, use_continuous_data):
    encoded_categorical_df, encoder_dict, column_type_dict = encode_categorical_columns(categorical_df)

    if use_continuous_data:
        concatenated_df = pd.concat([encoded_categorical_df, continuous_df], axis=1)
        imputed_concatenated_df, fitted_imp = knn_imputation(concatenated_df, neighbors_no)
        imputed_categorical_df = imputed_concatenated_df[list(column_type_dict.keys())]
    else:
        imputed_categorical_df, fitted_imp = knn_imputation(encoded_categorical_df, neighbors_no)

    imputed_final_categorical_df = decode_categorical_columns(imputed_categorical_df, encoder_dict, column_type_dict)

    return imputed_final_categorical_df, encoder_dict, column_type_dict, fitted_imp


def mode_impute_categorical(categorical_df):
    categorical_copy_df = categorical_df.copy()
    modes_dict = {}

    for column in categorical_copy_df.columns:
        mode_value = categorical_copy_df[column].value_counts().index[0]
        if isinstance(mode_value, str) and len(mode_value) == 0:
            mode_value = categorical_copy_df[column].value_counts().index[1]
        categorical_copy_df[column].fillna(mode_value, inplace=True)
        modes_dict[column] = files_storing.type_conv(mode_value)

    return categorical_copy_df, modes_dict


def impute_missing_categorical_values(continuous_df, categorical_df, use_knn=True, neighbors_no=1,
                                      use_continuous_data=True):
    imputation_categorical_info = {}

    if categorical_df.shape[1] == 0:
        imputation_categorical_info['method'] = 'w/o_cat_columns'
        return [categorical_df, imputation_categorical_info, None, None]

    logging.display('6.b. Imputation for categorical data', p=3)
    logging.display('continuous data was used: {}'.format(use_continuous_data), p=4)

    if use_knn:
        imputation_categorical_info['method'] = 'knn_' + str(neighbors_no)
        logging.display('kNN with {} neighbors is used for imputation'.format(neighbors_no), p=4)
    else:
        imputation_categorical_info['method'] = 'mode'
        logging.display('mode is used for imputation', p=4)
    imputation_categorical_info['use_continuous_data'] = use_continuous_data

    if use_knn:
        imputed_categorical_df, encoder_dict, column_type_dict, fitted_imp = knn_impute_categorical(continuous_df,
                                                                                                    categorical_df,
                                                                                                    neighbors_no,
                                                                                                    use_continuous_data)
        imputation_categorical_info['column_type_dict'] = column_type_dict
        return [imputed_categorical_df, imputation_categorical_info, encoder_dict, fitted_imp]
    else:
        imputed_categorical_df, modes_dict = mode_impute_categorical(categorical_df)
        imputation_categorical_info['modes_dict'] = modes_dict
        logging.display('Modes:', p=4)
        logging.display_json(modes_dict)
        return [imputed_categorical_df, imputation_categorical_info, None, None]


def impute_missing_continuous_values_test(test_continuous_df, config_dict, models_dict):
    continuous_features_imputation = config_dict['6a']
    continuous_imputation_method = continuous_features_imputation['method']

    if continuous_imputation_method == 'w/o_cont_columns':
        return test_continuous_df

    if continuous_imputation_method in ['mean', 'median']:
        values_to_impute = continuous_features_imputation['imputed_values']

        for column, value in values_to_impute.items():
            test_continuous_df[column].fillna(value, inplace=True)
    elif continuous_imputation_method.find('kNN') == 0:
        fitted_knn_imp = models_dict['6a']
        columns_names = test_continuous_df.columns
        imputed_continuous = fitted_knn_imp.transform(test_continuous_df)
        test_continuous_df = pd.DataFrame(imputed_continuous, columns=columns_names)

    return test_continuous_df


def mode_impute_categorical_test(test_categorical_df, categorical_features_imputation):
    values_to_impute = categorical_features_imputation['modes_dict']
    date_formats = ['%Y%d%m', '%Y%m%d', '%d%Y%m', '%d%m%Y', '%m%Y%d', '%m%d%Y', None]

    for column, value in values_to_impute.items():
        if isinstance(value, str) and len(value) > 10:
            for date_format in date_formats:
                value_aux = pd.to_datetime(value, format=date_format, errors='coerce')

                if not pd.isnull(value_aux):
                    value = value_aux
                    break

        test_categorical_df[column].fillna(value, inplace=True)

    return test_categorical_df


def encode_categorical_columns_test(test_categorical_df, encoder_dict):
    for column, column_encoder in encoder_dict.items():
        encoded_classes = list(column_encoder.classes_)

        if test_categorical_df[column].dtype == 'datetime64[ns]':
            test_categorical_df[column] = pd.DataFrame({column: test_categorical_df[column].dt.date.astype(str).values})

        unseen_values = set(test_categorical_df[column].astype(str).values) - set(encoded_classes)

        test_categorical_df[column] = test_categorical_df[column].map(
            lambda x: np.nan if (str(x) in unseen_values or str(x) in ['NaN', 'NaT', '']) else x)

        i = 0
        while encoded_classes[i] in ['', 'NaN', 'NaT', 'nan', 'np.nan']:
            i += 1

        enc_class = encoded_classes[i]
        indices_unknown = test_categorical_df[column].isna()
        test_categorical_df[column][indices_unknown] = enc_class

        try:
            encoded_values = column_encoder.transform(test_categorical_df[column].astype(str).values)
        except ValueError:
            try:
                logging.display('Test encoding exception type 1. Converting to int.', p=1)
                encoded_values = column_encoder.transform(test_categorical_df[column].astype(int).astype(str).values)
            except ValueError:
                logging.display('Test encoding exception type 2. Converting to float.', p=1)
                encoded_values = column_encoder.transform(test_categorical_df[column].astype(float).astype(str).values)

        test_categorical_df[column] = pd.DataFrame({column: encoded_values})
        test_categorical_df[column][indices_unknown] = np.nan

    return test_categorical_df


def knn_impute_categorical_test(test_categorical_df, test_continuous_df, use_continuous_data, fitted_knn_imp):
    categorical_columns_names = list(test_categorical_df.columns)

    if use_continuous_data:
        concatenated_df = pd.concat([test_categorical_df, test_continuous_df], axis=1)
        columns_names = concatenated_df.columns
        imputed_categorical = fitted_knn_imp.transform(concatenated_df)
        test_categorical_df = pd.DataFrame(imputed_categorical, columns=columns_names)
        test_categorical_df = test_categorical_df[categorical_columns_names]
    else:
        imputed_categorical = fitted_knn_imp.transform(test_categorical_df)
        test_categorical_df = pd.DataFrame(imputed_categorical, columns=categorical_columns_names)

    return test_categorical_df


def decode_categorical_columns_test(test_categorical_df, encoder_dict, column_type_dict):
    for column, column_encoder in encoder_dict.items():
        decoded_values = column_encoder.inverse_transform(test_categorical_df[column].astype(int).values)
        test_categorical_df[column] = pd.DataFrame({column: decoded_values})
        col_type = column_type_dict[column]

        if col_type == 'int64' or col_type == 'float64':
            test_categorical_df[column], _ = datatypes_conversion.object_to_numeric(test_categorical_df[column],
                                                                                    thresh=1)
        elif col_type == 'datetime64[ns]':
            test_categorical_df[column], _ = datatypes_conversion.object_to_date(test_categorical_df[column],
                                                                                 thresh=1)

    return test_categorical_df


def impute_missing_categorical_values_test(test_categorical_df, test_continuous_df, config_dict, models_dict):
    categorical_features_imputation = config_dict['6b']
    categorical_imputation_method = categorical_features_imputation['method']

    if categorical_imputation_method == 'w/o_cat_columns':
        return test_categorical_df

    if categorical_imputation_method == 'mode':
        test_categorical_df = mode_impute_categorical_test(test_categorical_df, categorical_features_imputation)
    elif categorical_imputation_method.find('knn') == 0:
        test_categorical_copy_df = test_categorical_df.copy()

        use_continuous_data = categorical_features_imputation['use_continuous_data']
        column_type_dict = categorical_features_imputation['column_type_dict']
        fitted_knn_imp = models_dict['6b']['fitted_imputer_categorical']
        encoder_dict = models_dict['6b']['encoder_dict']

        test_categorical_df = encode_categorical_columns_test(test_categorical_df, encoder_dict)
        test_categorical_df = knn_impute_categorical_test(test_categorical_df, test_continuous_df, use_continuous_data,
                                                          fitted_knn_imp)
        test_categorical_df = decode_categorical_columns_test(test_categorical_df, encoder_dict, column_type_dict)

        test_categorical_df = test_categorical_df.where(test_categorical_copy_df.isna(), test_categorical_copy_df)

    return test_categorical_df
