import re
import string

import pandas as pd

import config
from feature_engineering import feature_construction
from utility_functions import logging


def int_to_date(column, thresh=0.05):
    date_formats = ['%Y%d%m', '%Y%m%d', '%d%Y%m', '%d%m%Y', '%m%Y%d', '%m%d%Y']
    not_null_rows_no = column.notnull().sum()
    null_rows_no = column.isnull().sum()

    for i in range(len(date_formats)):
        converted_column = pd.to_datetime(column, format=date_formats[i], errors='coerce')
        if (converted_column.isnull().sum() - null_rows_no) / not_null_rows_no < thresh:
            return converted_column, True

    return column, False


def object_to_date(column, thresh=0.05):
    not_null_rows_no = column.notnull().sum()
    null_rows_no = column.isnull().sum()

    converted_column = pd.to_datetime(column, errors='coerce')
    if (converted_column.isnull().sum() - null_rows_no) / not_null_rows_no < thresh:
        return converted_column, True

    try:
        int_column = column.astype('int')
        return int_to_date(int_column, thresh)
    except (ValueError, TypeError):
        return column, False


def object_to_numeric(column, thresh=0.05):
    not_null_rows_no = column.notnull().sum()
    null_rows_no = column.isnull().sum()

    converted_column = pd.to_numeric(column, errors='coerce')
    if (converted_column.isnull().sum() - null_rows_no) / not_null_rows_no < thresh:
        return converted_column, True
    else:
        return column, False


def text_normalization(text):
    if pd.isnull(text):
        return text
    elif bool(re.search('^[0-9,\-+]+$', text)):
        text_wo_commas = text.replace(',', '.')
        return text_wo_commas

    encoded_string = text.encode("ascii", "ignore")
    text_wo_non_ascii = encoded_string.decode()
    text_lowered = text_wo_non_ascii.lower()

    rep = ["'m", "'re", "'s", "n't", "'ve", "'d", "'ll"]

    text_wo_cf = text_lowered
    for index in range(len(rep)):
        text_wo_cf = text_wo_cf.replace(rep[index], '')

    punctuation = string.punctuation
    text_wo_punctuation = text_wo_cf.translate(str.maketrans(punctuation, ' ' * len(punctuation)))
    text_wo_sap = re.sub(' +', ' ', text_wo_punctuation)

    split_text = text_wo_sap.split(' ')

    if len(split_text) == 1:
        return split_text[0]

    split_text_wo_sw = [word for word in split_text if ((word not in config.STOP_WORDS) and len(word) > 0)]
    normalized_text = ' '.join(split_text_wo_sw)

    return normalized_text


def convert_data_types_text_feat(input_df, normalize_text, add_text_features):
    logging.display('3. Data types conversion for each column', p=2)
    logging.display('Code meanings:\n\t\t\t' +
                    '3a: bool -> int\n\t\t\t'
                    '3b: int -> datetime\n\t\t\t' +
                    '3c: object -> datetime\n\t\t\t' +
                    '3d: object -> int/float\n\t\t\t' +
                    '3e: text normalization\n\t\t\t' +
                    '3f: without text normalization', p=4)

    initial_columns = input_df.columns
    display_text_feat = True
    added_columns = []

    for column in initial_columns:
        if input_df[column].dtype == 'bool':
            input_df[column] = input_df[column].map(lambda x: 1 if x is True else 0)

        elif input_df[column].dtype == 'int64':
            input_df[column], _ = int_to_date(input_df[column])

        elif input_df[column].dtype == 'object':
            input_df[column], converted_to_date = object_to_date(input_df[column])

            if converted_to_date is False:
                input_df[column], converted_to_numeric = object_to_numeric(input_df[column])
                if converted_to_numeric is False:
                    if normalize_text:
                        input_df[column] = input_df[column].map(text_normalization)

                    if add_text_features:
                        new_features_df = feature_construction.create_text_features(input_df[column], column,
                                                                                    display_message=display_text_feat)

                        display_text_feat = False
                        if new_features_df is not None:
                            input_df = pd.concat([input_df, new_features_df], axis=1)
                            added_columns += list(new_features_df.columns)

                    if normalize_text:
                        input_df[column], _ = object_to_numeric(input_df[column])

    return input_df, added_columns


def datatype_modification_checker(input_df, input_converted_df, normalize_text=True):
    converted_columns = {}

    for column in input_df.columns:
        initial_type = input_df[column].dtype
        converted_type = input_converted_df[column].dtype

        if initial_type == 'bool' and converted_type == 'int64':
            converted_columns[column] = '3a'
        elif initial_type == 'int64' and converted_type == 'datetime64[ns]':
            converted_columns[column] = '3b'
        elif initial_type == 'O' and converted_type == 'datetime64[ns]':
            converted_columns[column] = '3c'
        elif initial_type == 'O' and (converted_type == 'int64' or converted_type == 'float64'):
            converted_columns[column] = '3d'
        elif initial_type == 'O' and converted_type == 'O':
            if normalize_text:
                converted_columns[column] = '3e'
            else:
                converted_columns[column] = '3f'

    logging.display_json(converted_columns)

    return converted_columns


def convert_data_types_single(test_df, datatype_conv_dict, empty_cols):
    for column, conv_type in datatype_conv_dict.items():
        if column in empty_cols:
            if conv_type == '3a':
                test_df[column] = test_df[column].astype('bool')
            elif conv_type == '3b' or conv_type == '3c':
                test_df[column] = test_df[column].astype('datetime64[ns]')
            elif conv_type == '3d':
                test_df[column] = test_df[column].astype('float64')
            elif conv_type in ['3e', '3f']:
                test_df[column] = test_df[column].astype('O')

    return test_df


def convert_data_types_text_feat_test(test_df, config_dict):
    datatype_conv_dict = config_dict['3']
    added_columns = config_dict['7c']

    if added_columns is False:
        add_text_columns = added_columns
    else:
        add_text_columns = True

    display_text_features = False
    rows = test_df.shape[0]
    empty_cols = []

    for column, conv_type in datatype_conv_dict.items():
        if conv_type == '3a':
            test_df[column] = test_df[column].map(lambda x: 1 if x is True else 0)
        elif conv_type == '3b':
            test_df[column], _ = int_to_date(test_df[column], thresh=1)
        elif conv_type == '3c':
            test_df[column], _ = object_to_date(test_df[column], thresh=1)
        elif conv_type == '3d':
            test_df[column], converted_to_numeric = object_to_numeric(test_df[column], thresh=1)
            if converted_to_numeric is False:
                test_df[column] = test_df[column].map(text_normalization)

                if add_text_columns:
                    new_features_df = feature_construction.create_text_features(test_df[column], column, added_columns,
                                                                                display_text_features)
                    if new_features_df is not None:
                        test_df = pd.concat([test_df, new_features_df], axis=1)

                test_df[column], _ = object_to_numeric(test_df[column], thresh=1)
        elif conv_type == '3e':
            test_df[column] = test_df[column].map(text_normalization)

            if add_text_columns:
                new_features_df = feature_construction.create_text_features(test_df[column], column, added_columns,
                                                                            display_text_features)
                if new_features_df is not None:
                    test_df = pd.concat([test_df, new_features_df], axis=1)

        elif conv_type == '3f':
            if add_text_columns:
                new_features_df = feature_construction.create_text_features(test_df[column], column, added_columns,
                                                                            display_text_features)
                if new_features_df is not None:
                    test_df = pd.concat([test_df, new_features_df], axis=1)

        if test_df[column].isna().sum() == rows:
            empty_cols.append(column)

    if len(empty_cols) > 0:
        test_df = convert_data_types_single(test_df, datatype_conv_dict, empty_cols)

    return test_df


def failed_data_conversion(test_df, config_dict):
    datatype_conv_dict = config_dict['3']

    for k, v in datatype_conv_dict.items():
        if v == '3b' or v == '3c':
            datatype_conv_dict[k] = ['datetime64[ns]']
        elif v == '3a' or v == '3d':
            datatype_conv_dict[k] = ['int64', 'float64']
        elif v == '3e' or v == '3f':
            datatype_conv_dict[k] = ['O']

    for column, datatype in datatype_conv_dict.items():
        if test_df[column].dtype not in datatype:
            logging.display('Datatype conversion for test dataset failed', p=0)


def find_target_column_type(input_df, target_column):
    categorical_column = False
    target_type = input_df[target_column].dtype

    if target_type == 'O':
        input_df[target_column], converted_to_numeric = object_to_numeric(input_df[target_column])
    
    if target_type in ['int64', 'float64'] or converted_to_numeric:
        unique_ratio = input_df[target_column].nunique() / input_df.shape[0]
        unique_values_no = input_df[target_column].nunique()

        if unique_ratio < config.CONTINUOUS_THRESHOLD or unique_values_no < config.MIN_UNIQUE_CONTINUOUS_VALUES:
            categorical_column = True

    elif target_type in ['bool', 'datetime64[ns]', 'O']:
        categorical_column = True

    return categorical_column
