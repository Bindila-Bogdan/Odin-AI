import sys

import numpy as np
import pandas as pd

import config
from utility_functions import files_storing
from utility_functions import logging


def robust_z_score(values):
    med_values = np.median(values)
    med_abs_dev = np.median(np.abs(values - med_values))
    mod_z_score = 0.6475 * ((values - med_values) / med_abs_dev)

    return mod_z_score


def outliers_three_std(input_column):
    mean = np.array(input_column.values).mean()
    std = np.array(input_column.values).std()
    lower_thresh = mean - 3 * std
    upper_thresh = mean + 3 * std

    return lower_thresh, upper_thresh


def outliers_iqr(input_column):
    q1 = input_column.quantile(0.25)
    q3 = input_column.quantile(0.75)
    iqr = q3 - q1
    lower_thresh = q1 - 1.5 * iqr
    upper_thresh = q3 + 1.5 * iqr

    return lower_thresh, upper_thresh


def outliers_detection_column_continuous(input_column, method_type='three_std'):
    if method_type == 'three_std':
        lower_thresh, upper_thresh = outliers_three_std(input_column)
    else:
        lower_thresh, upper_thresh = outliers_iqr(input_column)

    lower_outliers = input_column[(input_column < lower_thresh)]
    upper_outliers = input_column[(input_column > upper_thresh)]

    outliers = lower_outliers.append(upper_outliers)
    percentage_unique_outliers = (outliers.nunique() / input_column.nunique() * 100)

    if percentage_unique_outliers < config.MAX_PERCENTAGE_UNIQUE_OUTLIERS:
        pass
    else:
        outliers = None
        lower_outliers = pd.Series()
        upper_outliers = pd.Series()

    return outliers, lower_outliers, upper_outliers


def detect_outliers_continuous(input_df, method_type, add_is_missing):
    continuous_df = input_df.copy()
    rows_no = continuous_df.shape[0]
    continuous_outliers_info = {'method': method_type}
    removed_columns = []
    outliers_boundaries = {}

    logging.display('5.a. Outliers detection in continuous columns', p=3)
    logging.display('{} is the used method for outliers detection'.format(method_type), p=4)
    logging.display('Column name:'.ljust(32) + 'outliers'.ljust(12) + 'missing values:'.ljust(10), p=4)

    for column in continuous_df.columns:
        process_column = True

        for wo_detection in config.WO_OUTLIERS_DET:
            if wo_detection in column:
                process_column = False
                break

        if not process_column:
            continue

        column_wo_na = continuous_df[column].dropna()
        outliers, lower_outliers, upper_outliers = outliers_detection_column_continuous(column_wo_na, method_type)
        if len(lower_outliers.values) == 0:
            lower_boundary = -int(sys.maxsize)
        else:
            lower_boundary = max(lower_outliers.values)

        if len(upper_outliers.values) == 0:
            upper_boundary = int(sys.maxsize)
        else:
            upper_boundary = min(upper_outliers.values)

        outliers_no = 0

        try:
            outliers_no = outliers.size
        except AttributeError:
            pass

        missing_values_no = continuous_df[column].isnull().sum()

        if (outliers_no + missing_values_no) / rows_no * 100 > config.MAX_MISSING_PERCENTAGE:
            position = list(continuous_df.columns).index(column)
            to_insert = list(np.array(list(continuous_df[column].isna().values)).astype(int))

            if add_is_missing:
                continuous_df.insert(position, column + '_is_missing', to_insert)
                logging.display('Column {} was inserted.'.format(column + '_is_missing'), p=4)

            continuous_df.drop(column, axis=1, inplace=True)
            logging.display("Column {} was dropped.".format(column), p=4)
            removed_columns.append(column)
        else:
            continuous_df[column][
                (continuous_df[column] <= lower_boundary) | (continuous_df[column] >= upper_boundary)] = np.nan

            lower_boundary = files_storing.type_conv(lower_boundary)
            upper_boundary = files_storing.type_conv(upper_boundary)
            outliers_boundaries[column] = [lower_boundary, upper_boundary]

        logging.display('{}{}  {}'.format(column.ljust(32), str(outliers_no).ljust(10), missing_values_no), p=4)

    continuous_outliers_info['removed_columns'] = removed_columns
    continuous_outliers_info['outliers_boundaries'] = outliers_boundaries

    logging.display('Lower and upper boundaries for outliers:', p=4)
    logging.display_json(outliers_boundaries)

    return [continuous_df, continuous_outliers_info]


def outliers_detection_column_categorical(input_column):
    column_wo_na = input_column.dropna()
    freq_list = list(column_wo_na.value_counts().values)

    if len(column_wo_na.value_counts()) < 3:
        lower_freq_thresh = 0
    else:
        lower_freq_thresh = np.floor(np.percentile(freq_list, 1))

    outliers_list = column_wo_na.value_counts()[column_wo_na.value_counts().values < lower_freq_thresh]

    outliers_percentage = np.sum(np.array(outliers_list.values)) / np.sum(np.array(column_wo_na.value_counts().values))

    return outliers_percentage, np.sum(np.array(outliers_list.values)), list(outliers_list.index), lower_freq_thresh


def detect_outliers_categorical(input_df, add_is_missing):
    categorical_copy_df = input_df.copy()
    rows_no = categorical_copy_df.shape[0]
    categorical_outliers_info = {}
    removed_columns = []
    outliers_boundaries = {}

    logging.display('5.b. Outliers detection in categorical columns', p=3)
    logging.display('Threshold for detecting categorical that will be verified for outliers: {}'.format(
        config.CATEGORICAL_OUTLIERS_DETECTION_THRESHOLD) +
                    '\n\t\twinsorization for 1st percentile is the method used for outliers detection', p=4)
    logging.display('Column name:'.ljust(32) + 'outliers'.ljust(12) + 'missing values:'.ljust(10), p=4)

    for column in categorical_copy_df.columns:
        process_column = True

        for wo_detection in config.WO_OUTLIERS_DET:
            if wo_detection in column:
                process_column = False
                break

        if not process_column:
            continue

        nulls_no = categorical_copy_df[column].isnull().sum()
        unique_ratio = categorical_copy_df[column].nunique() / rows_no
        column_datatype = categorical_copy_df[column].dtype

        if (unique_ratio < config.CATEGORICAL_OUTLIERS_DETECTION_THRESHOLD and column_datatype == 'O') or \
                column_datatype != 'O':
            outliers_percentage, outliers_no, outliers_list, lower_freq_thresh = outliers_detection_column_categorical(
                categorical_copy_df[column])
            missing_percentage = outliers_percentage + nulls_no / rows_no

            if missing_percentage > config.MAX_MISSING_PERCENTAGE:
                position = list(categorical_copy_df.columns).index(column)
                to_insert = list(np.array(list(categorical_copy_df[column].isna().values)).astype(int))
                if add_is_missing:
                    categorical_copy_df.insert(position, column + '_is_missing', to_insert)
                    logging.display('Column {} was inserted.'.format(column + '_is_missing'), p=4)

                categorical_copy_df.drop(column, axis=1, inplace=True)
                logging.display("Column {} was dropped.".format(column), p=4)
                removed_columns.append(column)
            else:
                categorical_copy_df[column] = categorical_copy_df[column].map(
                    lambda x: np.nan if x in outliers_list else x)

                lower_freq_thresh = files_storing.type_conv(lower_freq_thresh)
                outliers_boundaries[column] = lower_freq_thresh

            logging.display('{}{}  {}'.format(column.ljust(32), str(outliers_no).ljust(10), nulls_no), p=4)

    categorical_outliers_info['removed_columns'] = removed_columns
    categorical_outliers_info['outliers_boundaries'] = outliers_boundaries

    logging.display('Lower boundary for outliers:', p=4)
    logging.display_json(outliers_boundaries)

    return [categorical_copy_df, categorical_outliers_info]


def detect_outliers_continuous_test(test_continuous_df, config_dict, add_is_missing):
    continuous_outliers_detection = config_dict['5a']
    continuous_columns_to_drop = continuous_outliers_detection['removed_columns']
    continuous_outliers_boundaries = continuous_outliers_detection['outliers_boundaries']

    for column in continuous_columns_to_drop:
        position = list(test_continuous_df.columns).index(column)
        to_insert = list(np.array(list(test_continuous_df[column].isna().values)).astype(int))
        if add_is_missing:
            test_continuous_df.insert(position, column + '_is_missing', to_insert)
        test_continuous_df.drop(column, axis=1, inplace=True)

    for column, boundaries in continuous_outliers_boundaries.items():
        lower_boundary = boundaries[0]
        upper_boundary = boundaries[1]

        test_continuous_df[column][
            (test_continuous_df[column] <= lower_boundary) | (test_continuous_df[column] >= upper_boundary)] = np.nan

    return test_continuous_df


def detect_outliers_categorical_test(test_categorical_df, config_dict, add_is_missing):
    categorical_outliers_detection = config_dict['5b']
    categorical_columns_to_drop = categorical_outliers_detection['removed_columns']
    categorical_outliers_boundaries = categorical_outliers_detection['outliers_boundaries']

    for column in categorical_columns_to_drop:
        position = list(test_categorical_df.columns).index(column)
        to_insert = list(np.array(list(test_categorical_df[column].isna().values)).astype(int))
        if add_is_missing:
            test_categorical_df.insert(position, column + '_is_missing', to_insert)
        test_categorical_df.drop(column, axis=1, inplace=True)

    for column, lower_boundary in categorical_outliers_boundaries.items():
        column_wo_na = test_categorical_df[column].dropna()

        outliers_list = column_wo_na.value_counts()[column_wo_na.value_counts().values <= lower_boundary]

        outliers_to_remove = list(outliers_list.index)
        test_categorical_df[column] = test_categorical_df[column]. \
            map(lambda x: np.nan if x in outliers_to_remove else x)

    return test_categorical_df
