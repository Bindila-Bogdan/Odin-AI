import pandas as pd

import config
from utility_functions import logging


def detect_categorical_columns(input_df):
    categorical_data = {}
    logging.display('Categorical columns:' + 12 * ' ' + 'Datatype:' + 12 * ' ' + 'Ratio unique values:', p=4)
    categorical_columns = []
    rows_no = input_df.shape[0]

    for column in input_df.columns:
        unique_ratio = input_df[column].nunique() / rows_no
        unique_values_no = input_df[column].nunique()
        categorical = False

        if input_df[column].dtype == 'O' or input_df[column].dtype == 'datetime64[ns]':
            categorical = True
        elif input_df[column].dtype == 'int64' or input_df[column].dtype == 'float64':
            if unique_ratio < config.CONTINUOUS_THRESHOLD or unique_values_no < config.MIN_UNIQUE_CONTINUOUS_VALUES:
                categorical = True

        if categorical:
            categorical_columns.append(column)
            categorical_data[column] = input_df[column].values
            logging.display(column.ljust(32) + str(input_df[column].dtype).ljust(20) + ' ' + str(unique_ratio), p=4)

    categorical_df = pd.DataFrame(categorical_data)

    return categorical_df, categorical_columns


def split_dataset(input_df):
    logging.display('4. Dataset splitting based on categorical columns', p=2)
    logging.display('Used threshold = {}'.format(config.CONTINUOUS_THRESHOLD), p=4)

    input_copy_df = input_df.copy()
    categorical_df, categorical_columns = detect_categorical_columns(input_copy_df)
    input_copy_df.drop(categorical_df.columns, axis=1, inplace=True)

    logging.display(' ', p=4)
    logging.display('Continuous columns:' + 13 * ' ' + 'Datatype:' + 12 * ' ' + 'Ratio unique values:', p=4)
    rows_no = input_copy_df.shape[0]

    for column in input_copy_df.columns:
        unique_ratio = input_copy_df[column].nunique() / rows_no
        logging.display(column.ljust(32) + str(input_copy_df[column].dtype).ljust(20) + ' ' + str(unique_ratio), p=4)

    return [categorical_df, input_copy_df, categorical_columns]


def move_to_continuous(train_categorical_df, train_continuous_df, scalable_columns_names):
    printed = False
    train_categorical_copy_df = train_categorical_df.copy()

    for column_name in train_categorical_copy_df.columns:
        if '_long_text_' in column_name or '_svd_' in column_name:

            if not printed:
                logging.display('Moving column {} from categorical to continuous'.format(column_name), p=5)
                printed = True

            scalable_columns_names.remove(column_name)
            train_continuous_df = pd.concat([train_continuous_df, train_categorical_df[column_name]], axis=1)
            train_categorical_df.drop(column_name, axis=1, inplace=True)

    return [train_categorical_df, train_continuous_df, scalable_columns_names]


def split_dataset_test(test_df, config_dict):
    categorical_columns = config_dict['4']
    test_categorical_df = test_df[categorical_columns]
    test_continuous_df = test_df.drop(columns=test_categorical_df)

    return [test_categorical_df, test_continuous_df]


def move_to_continuous_test(test_categorical_df, test_continuous_df):
    test_categorical_copy_df = test_categorical_df.copy()

    for column_name in test_categorical_copy_df:
        if '_long_text_' in column_name or '_svd_' in column_name:
            test_continuous_df = pd.concat([test_continuous_df, test_categorical_df[column_name]], axis=1)
            test_categorical_df.drop(column_name, axis=1, inplace=True)

    return [test_categorical_df, test_continuous_df]
