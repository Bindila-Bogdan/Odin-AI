import category_encoders
import numpy as np
import pandas as pd
from sklearn import preprocessing

from utility_functions import logging
from . import feature_extraction


def binary_encoding(column, column_name, encoder=None):
    if column.dtype != 'O':
        column = column.astype(str)

    if encoder is None:
        binary_encoder = category_encoders.BinaryEncoder()
        binary_encoder.fit(column.values.reshape(-1, 1))
    else:
        binary_encoder = encoder

    binary_encoded_df = binary_encoder.transform(column.values.reshape(-1, 1))
    binary_encoded_df.columns = [column_name + '_enc_' + str(x) for x in range(binary_encoded_df.shape[1])]

    return binary_encoded_df, binary_encoder


def one_hot_encoding(column, column_name, encoder=None):
    if encoder is None:
        one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit(column.values.reshape(-1, 1))
    else:
        one_hot_encoder = encoder

    one_hot_encoded = one_hot_encoder.transform(column.values.reshape(-1, 1))
    column_names = [column_name + '_enc_' + str(x) for x in range(one_hot_encoded.toarray().shape[1])]
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=column_names)

    return one_hot_encoded_df, one_hot_encoder


def ordinal_encoding(column, column_name, encoder=None, add_mode=True, target=''):
    if encoder is None:
        values = column.value_counts().index
        mode = values[0]
        classes = list(set(values))
        sorted_classes = sorted(classes)

        encoding = 0
        ordinal_encoding_dict = {}
        if add_mode:
            ordinal_encoding_dict['mode'] = mode
        else:
            ordinal_encoding_dict['mode'] = None

        for value in sorted_classes:
            ordinal_encoding_dict[value] = encoding
            encoding += 1

    else:
        mode = encoder['mode']
        del encoder['mode']
        column[~column.isin(list(encoder.keys()))] = mode
        ordinal_encoding_dict = encoder

    encoded_column = column.map(ordinal_encoding_dict)
    encoded_column_df = pd.DataFrame(encoded_column.values, columns=[column_name + '_enc' + target])

    return encoded_column_df, ordinal_encoding_dict


def numerical_encoding(column, column_name, enc_type, encoder=None):
    if enc_type == 0:
        return binary_encoding(column, column_name, encoder=encoder)
    elif enc_type == 1:
        return one_hot_encoding(column, column_name, encoder=encoder)
    else:
        return ordinal_encoding(column, column_name, encoder=encoder)


def long_text_encoding(column, column_name, enc_type, min_freq, max_freq, svd, reduce_dim, rand_state,
                       encoder=None):
    if enc_type == 0:
        return binary_encoding(column, column_name, encoder=encoder)
    else:
        return feature_extraction.extract_text_features(column, column_name, min_freq, max_freq, input_tfidf=encoder,
                                                        svd=svd, reduce_dim=reduce_dim, rand_state=rand_state)


def short_text_encoding(column, column_name, enc_type, encoder=None):
    if enc_type == 0:
        return binary_encoding(column, column_name, encoder=encoder)
    elif enc_type == 1:
        return one_hot_encoding(column, column_name, encoder=encoder)


def encode_categorical_values(categorical_df, rand_state, reduce_dim, num_enc_type=0, long_text_enc_type=0,
                              short_text_enc_type=0, min_freq=0.001, max_freq=0.999):
    if long_text_enc_type == 1:
        logging.display('7.b. Text feature engineering applied', p=3)

    if reduce_dim:
        logging.display('7.e. Reducing dimensionality of tf_idf result', p=4)

    logging.display('8. Encoding categorical features', p=2)
    logging.display('Column type'.ljust(32) + 'column name'.ljust(32) + 'encoding type', p=4)

    encoder = None
    encoded_df = None
    svd_dict = {}
    encoders_dict = {}
    scalable_columns_names = []
    categorical_copy_df = categorical_df.copy()

    for column in categorical_df.columns:
        datatype = categorical_df[column].dtype

        if datatype == 'int64' or datatype == 'float64':
            if num_enc_type != -1:
                encoded_df, encoder = numerical_encoding(categorical_copy_df[column], column, num_enc_type)
                encoders_dict['num_' + column + '_' + str(num_enc_type)] = encoder
                if num_enc_type == 2:
                    scalable_columns_names += list(encoded_df.columns)
            else:
                encoders_dict['num_' + column + '_' + str(num_enc_type)] = None
                encoded_df = categorical_df[column]
                scalable_columns_names.append(column)

        elif datatype == 'O':
            rows_no = categorical_copy_df.shape[0]
            unique_ratio = categorical_copy_df[column].nunique() / rows_no

            if unique_ratio > 0.1:
                svd = None

                if long_text_enc_type == 1:
                    encoded_df, encoder, svd = long_text_encoding(categorical_copy_df[column], column,
                                                                  long_text_enc_type, min_freq, max_freq,
                                                                  svd=None, reduce_dim=reduce_dim,
                                                                  rand_state=rand_state)
                    scalable_columns_names += list(encoded_df.columns)
                elif long_text_enc_type == 0:
                    encoded_df, encoder = long_text_encoding(categorical_copy_df[column], column, long_text_enc_type,
                                                             min_freq, max_freq, svd=None,
                                                             reduce_dim=reduce_dim, rand_state=rand_state)

                encoders_dict['longtext_' + column + '_' + str(long_text_enc_type)] = encoder
                svd_dict['svd_' + column] = svd
            else:
                encoded_df, encoder = short_text_encoding(categorical_copy_df[column], column, short_text_enc_type)
                encoders_dict['shorttext_' + column + '_' + str(short_text_enc_type)] = encoder

        categorical_copy_df.drop([column], axis=1, inplace=True)
        categorical_copy_df = pd.concat([categorical_copy_df, encoded_df], axis=1)

    logging.display_dict(encoders_dict)
    logging.display_dict(svd_dict)

    return [categorical_copy_df, scalable_columns_names, encoders_dict, svd_dict]


def target_ordinal_encoding(y_test_df, config_dict, models_dict, handle_unknown_classes=False):
    classification_task = config_dict['2']['classification_task']

    if not classification_task:
        return y_test_df

    ordinal_encoding_dict = models_dict['2']
    column_name = config_dict['2']['predicted_column']

    y_test_df[column_name][~y_test_df[column_name].isin(list(ordinal_encoding_dict.keys()))] = np.nan
    unknown_targets_no = y_test_df[column_name].isna().sum()

    if unknown_targets_no != 0:
        if handle_unknown_classes:
            indices_unknown_classes = list(y_test_df[column_name][y_test_df[column_name].isnull()].index)
            logging.display('Indices of unknown classes: {}.'.format(indices_unknown_classes), p=1)

            y_test_df.dropna(inplace=True)
            logging.display('No of target examples removed: {}'.format(len(indices_unknown_classes)), p=1)
        else:
            logging.display('Target column contains unknown classes.', p=0)

    encoded_column = y_test_df[column_name].map(ordinal_encoding_dict)
    y_test_df = pd.DataFrame(encoded_column.values, columns=[column_name + '_enc_target'])

    return y_test_df


def encode_categorical_values_test(test_categorical_df, config_dict, models_dict):
    encoders_dict = models_dict['8']['encoders_dict']

    encoded_df = None
    svd_index = 0
    svds = list(models_dict['8']['svd_dict'].values())

    reduce_dim = config_dict['8'][0]
    rand_state = config_dict['8'][1]

    for info, encoder in encoders_dict.items():
        col_type = info[:info.find('_')]
        col_name = info[info.find('_') + 1:info.rfind('_')]
        enc_code = int(info[info.rfind('_') + 1:])

        if col_type == 'num':
            if enc_code != -1:
                encoded_df, _ = numerical_encoding(test_categorical_df[col_name], col_name, enc_code, encoder=encoder)
            else:
                encoded_df = test_categorical_df[col_name]

        elif col_type == 'shorttext':
            encoded_df, _ = short_text_encoding(test_categorical_df[col_name], col_name, enc_code, encoder=encoder)
        elif col_type == 'longtext':
            if reduce_dim is False:
                reduce_dim_now = False
            elif svds[svd_index] is None:
                reduce_dim_now = False
            else:
                reduce_dim_now = True

            if enc_code == 0:
                encoded_df, _ = long_text_encoding(test_categorical_df[col_name], col_name, enc_code, None, None,
                                                   svd=svds[svd_index], reduce_dim=reduce_dim_now,
                                                   rand_state=rand_state, encoder=encoder)
            elif enc_code == 1:
                encoded_df, _, _ = long_text_encoding(test_categorical_df[col_name], col_name, enc_code, None,
                                                      None, svd=svds[svd_index], reduce_dim=reduce_dim_now,
                                                      rand_state=rand_state, encoder=encoder)
            svd_index += 1

        test_categorical_df.drop([col_name], axis=1, inplace=True)
        test_categorical_df = pd.concat([test_categorical_df, encoded_df], axis=1)

    return test_categorical_df
