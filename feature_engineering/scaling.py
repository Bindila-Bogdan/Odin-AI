import pandas as pd
from sklearn import preprocessing

from utility_functions import logging


def scale_df(input_df, scale_norm_type=1, continuous=False, first_log=None):
    if input_df.shape[1] == 0:
        return [input_df, None, None, None]

    if continuous:
        logging.display('9. Scaling features', p=2, first_log=first_log)
        logging.display('9.a. Scaling continuous features', p=3, first_log=first_log)

    scaling_continuous = {}

    if scale_norm_type == 0:
        scaler = preprocessing.MaxAbsScaler().fit(input_df)
        scaling_continuous['scaler'] = 'MaxAbsScaler'
    elif scale_norm_type == 1:
        scaler = preprocessing.MinMaxScaler().fit(input_df)
        scaling_continuous['scaler'] = 'MinMaxScaler'
    elif scale_norm_type == 2:
        scaler = preprocessing.StandardScaler().fit(input_df)
        scaling_continuous['scaler'] = 'StandardScaler'
    else:
        scaler = preprocessing.RobustScaler().fit(input_df)
        scaling_continuous['scaler'] = 'RobustScaler'

    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=list(input_df.columns))

    if continuous:
        logging.display('Scaler type is {}'.format(scaling_continuous['scaler']), p=4, first_log=first_log)

    return [scaled_df, scaler, scaling_continuous, scaling_continuous['scaler']]


def scale_categorical_columns(categorical_df, scalable_columns_names=None, scale_norm_type=1, scale_entire_df=False):
    logging.display('9.b. Scaling categorical features', p=3)
    scaling_categorical = {'scale_entire': scale_entire_df}

    if scale_entire_df and len(list(categorical_df.columns)) != 0:
        [scaled_df, scaler, _, type_of_scaler] = scale_df(categorical_df, scale_norm_type)
        scaling_categorical['scaler'] = type_of_scaler
        final_scaled_df = scaled_df
    elif scalable_columns_names is not None and len(scalable_columns_names) != 0:
        scalable_columns_names_ = [column_name for column_name in scalable_columns_names if
                                   column_name in categorical_df.columns]
        scalable_df = categorical_df[scalable_columns_names_]
        non_scalable_df = categorical_df.drop(columns=scalable_columns_names_)
        [scaled_df, scaler, _, type_of_scaler] = scale_df(scalable_df, scale_norm_type)
        scaling_categorical['scaler'] = type_of_scaler
        final_scaled_df = pd.concat([scaled_df, non_scalable_df], axis=1)
    else:
        scaler = None
        scaling_categorical['scaler'] = None
        final_scaled_df = categorical_df

    logging.display('Number of scalable columns: {}'.format(len(scalable_columns_names)), p=4)
    logging.display('Scale entire categorical dataframe: {}'.format(scale_entire_df), p=4)
    logging.display('Scaler type is {}'.format(scaling_categorical['scaler']), p=4)

    return [final_scaled_df, scaler, scaling_categorical]


def scale_continuous_columns_test(test_continuous_df, config_dict, models_dict):
    continuous_scaler = models_dict['9a']
    continuous_scaling_dict = config_dict['9a']

    if continuous_scaling_dict is None:
        return test_continuous_df

    columns_names = list(test_continuous_df.columns)

    scaled_continuous_features = continuous_scaler.transform(test_continuous_df)
    test_continuous_df = pd.DataFrame(scaled_continuous_features, columns=columns_names)

    return test_continuous_df


def scale_categorical_columns_test(test_categorical_df, config_dict, models_dict, categorical_scalable_columns):
    categorical_scaler = models_dict['9b']
    categorical_scaling_dict = config_dict['9b']
    scale_entire = categorical_scaling_dict['scale_entire']

    if scale_entire and len(list(test_categorical_df.columns)) != 0:
        scaled_categorical_features = categorical_scaler.transform(test_categorical_df)
        test_categorical_df = pd.DataFrame(scaled_categorical_features, columns=list(test_categorical_df.columns))
    elif len(categorical_scalable_columns) != 0:
        categorical_scalable_columns_ = [column_name for column_name in categorical_scalable_columns if
                                         column_name in test_categorical_df.columns]
        scalable_df = test_categorical_df[categorical_scalable_columns_]
        non_scalable_df = test_categorical_df.drop(columns=categorical_scalable_columns_)
        columns_names = list(scalable_df.columns)
        scaled_categorical_features = categorical_scaler.transform(scalable_df)
        scaled_df = pd.DataFrame(scaled_categorical_features, columns=columns_names)
        test_categorical_df = pd.concat([scaled_df, non_scalable_df], axis=1)

    return test_categorical_df
