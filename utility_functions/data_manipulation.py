import pandas as pd
from pandas.io import parsers
from sklearn import model_selection

import config
from . import logging


def read_dataset(csv_name, disable_logging=False):
    if not disable_logging:
        logging.display('0. Dataset reading', p=2)

    delimiters = [',', ', ', ';', '; ', '\t']

    for i in range(len(delimiters)):
        try:
            input_df = pd.read_csv(csv_name, sep=delimiters[i], na_values=config.MISSING_VALUES, skip_blank_lines=True)
            input_df.dropna(how="all", inplace=True)

            if input_df.shape[1] != 1:
                if not disable_logging:
                    logging.display('Used delimiter: {}'.format(delimiters[i]), p=4)
                
                return input_df
        except parsers.ParserError:
            pass

    return None


def ord_encoding(column, column_name, target=''):
    values = column.value_counts().index
    classes = list(set(values))
    sorted_classes = sorted(classes)

    encoding = 0
    ordinal_encoding_dict = {'mode': None}

    for value in sorted_classes:
        ordinal_encoding_dict[value] = encoding
        encoding += 1

    encoded_column = column.map(ordinal_encoding_dict)
    encoded_column_df = pd.DataFrame(encoded_column.values, columns=[column_name + '_enc' + target])

    return encoded_column_df, ordinal_encoding_dict


def split_train_test(dataframe, output_column_name, classification_task, test_size=0.2, random_state=42):
    logging.display('2. Split dataset in training and testing', p=2)
    logging.display('Test percentage = {}\n\t\tRandom state = {}\n\t\t'.format(test_size, random_state) +
                    'Output column = {}'.format(output_column_name), p=4)

    y = dataframe[output_column_name]
    ordinal_encoding_target_dict = None

    if y.isnull().sum() != 0:
        logging.display('Predicted column should not contain missing values.', p=0)
        return 7 * [None]
    elif classification_task is False and (y.dtype not in ['int64', 'float64']):
        logging.display('Predicted column should have a numeric type.', p=0)

    x = dataframe.drop([output_column_name], axis=1)

    if test_size > 0:
        if classification_task:
            try:
                x_train_df, x_test_df, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size,
                                                                                          stratify=y,
                                                                                          random_state=random_state)
            except ValueError:
                x_train_df, x_test_df, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size,
                                                                                          random_state=random_state)
        else:
            x_train_df, x_test_df, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size,
                                                                                      random_state=random_state)

        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)

        train_initial_indices = x_train_df.index
        test_initial_indices = x_test_df.index

        train_indices = range(x_train_df.shape[0])
        test_indices = range(x_test_df.shape[0])

        x_train_df.index = train_indices
        x_test_df.index = test_indices
        y_train_df.index = train_indices
        y_test_df.index = test_indices
    else:
        x_train_df = x
        y_train_df = pd.DataFrame(y)
        train_initial_indices = x_train_df.index
        [x_test_df, y_test_df, test_initial_indices] = 3 * [None]

    if classification_task:
        y_train_df, ordinal_encoding_target_dict = ord_encoding(y_train_df[output_column_name], output_column_name,
                                                                target='_target')

    return [x_train_df, x_test_df, y_train_df, y_test_df, ordinal_encoding_target_dict, train_initial_indices,
            test_initial_indices]


def read_test_dataset(dataset_name, csv_file_name, predicted_column, metric):
    test_df = read_dataset('/odinstorage/automl_data/datasets/' + dataset_name + '/' + csv_file_name + '.csv')
    shape_info = 'Initial shape of the dataset: {}'.format(test_df.shape)

    if predicted_column not in list(test_df.columns):
        print('\n{} column wasn\'t found in {} dataset. {} won\'t be computed.'.format(predicted_column,
                                                                                       csv_file_name,
                                                                                       str.capitalize(metric)))

        x_test_initial_df = test_df
        y_test_initial_df = None

    else:
        x_test_initial_df = test_df.drop(predicted_column, axis=1)
        y_test_initial_df = pd.DataFrame(test_df[predicted_column])

    return x_test_initial_df, y_test_initial_df, shape_info
