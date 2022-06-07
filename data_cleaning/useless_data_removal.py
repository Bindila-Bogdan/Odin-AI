import numpy as np

import config
from utility_functions import logging


def useless_columns_removal(input_df):
    logging.display('1.a. Remove useless columns', p=3)
    rows_no = input_df.shape[0]
    removed_columns = []
    deleted = False

    for column in input_df.columns:
        unique_values = input_df[column].nunique()
        if unique_values <= 1:
            logging.display('Column {} has zero variance.'.format(column), p=4)
            input_df.drop(column, axis=1, inplace=True)
            removed_columns.append(column)
            deleted = True
        elif (rows_no == input_df[column].nunique()) and (input_df[column].dtype == 'int64') and \
                (input_df[column].isnull().sum() == 0):
            logging.display('Column {} has all values unique.'.format(column), p=4)
            input_df.drop(column, axis=1, inplace=True)
            removed_columns.append(column)
            deleted = True

    if not deleted:
        logging.display('There are not useless columns.', p=4)

    return input_df, removed_columns


def duplicated_rows_removal(input_df):
    logging.display('1.b. Remove duplicated rows', p=3)
    initial_rows_no = input_df.shape[0]
    input_df.drop_duplicates(inplace=True)
    input_df.index = list(range(input_df.shape[0]))
    current_rows_no = input_df.shape[0]

    if initial_rows_no != current_rows_no:
        logging.display('Number of duplicated rows: {}.'.format(initial_rows_no - current_rows_no), p=4)
    else:
        logging.display('The dataset does not contain duplicated rows.', p=4)

    return input_df, current_rows_no


def remove_missing_data_columns(input_df, current_rows_no, add_is_missing):
    logging.display('1.c. Remove columns with missing data (after splitting)', p=3)
    removed_columns = []
    deleted = False

    for column in input_df.columns:
        missing_data_percentage = input_df[column].isnull().sum() * 100 / current_rows_no
        if missing_data_percentage > config.MAX_MISSING_PERCENTAGE:
            position = list(input_df.columns).index(column)
            to_insert = list(np.array(list(input_df[column].isna().values)).astype(int))
            if add_is_missing:
                input_df.insert(position, column + '_is_missing', to_insert)
                logging.display('Column {} was inserted.'.format(column + '_is_missing'), p=4)

            input_df.drop(column, axis=1, inplace=True)
            logging.display('Column \'{}\' has {}% of missing data.'.format(column, missing_data_percentage), p=4)
            removed_columns.append(column)
            deleted = True

    if not deleted:
        logging.display('Every column has < {}% missing data.'.format(config.MAX_MISSING_PERCENTAGE), p=4)

    return [input_df, removed_columns]


def remove_zero_variance_columns(train_continuous_df, train_categorical_df):
    logging.display('\nRemove columns without variation', p=3)
    removed_columns_wo_var = []

    for df in [train_continuous_df, train_categorical_df]:
        for column in df.columns:
            unique_values_no = df[column].nunique()
            if unique_values_no == 1:
                logging.display('Column {} has zero variance.'.format(column), p=4)
                df.drop(column, axis=1, inplace=True)
                removed_columns_wo_var.append(column)

    return [train_continuous_df, train_categorical_df, removed_columns_wo_var]


def remove_useless_data(input_df):
    logging.display('1. Removing duplicated rows and useless columns', p=2)

    input_df, removed_columns = useless_columns_removal(input_df)
    input_df, current_rows_no = duplicated_rows_removal(input_df)

    return [input_df, removed_columns]


def remove_missing_data_columns_test(test_df, config_dict, add_is_missing):
    columns_to_drop = config_dict['1c']

    for column in columns_to_drop:
        position = list(test_df.columns).index(column)
        to_insert = list(np.array(list(test_df[column].isna().values)).astype(int))
        if add_is_missing:
            test_df.insert(position, column + '_is_missing', to_insert)
        test_df.drop(column, axis=1, inplace=True)

    return test_df


def remove_zero_variance_columns_test(test_continuous_df, test_categorical_df, config_dict):
    columns_to_remove = config_dict['removed_columns_wo_var']

    for df in [test_continuous_df, test_categorical_df]:
        for column in df.columns:
            if column in columns_to_remove:
                df.drop(column, axis=1, inplace=True)
                logging.display('Removing column {} because it has zero variance.'.format(column), p=4)

    return [test_continuous_df, test_categorical_df]
