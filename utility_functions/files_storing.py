import datetime
import json
import os
import pickle
import shutil
from os import path

import numpy as np

import config


def type_conv(value):
    if isinstance(value, datetime.datetime):
        conv_value = str(value)
    elif isinstance(value, np.int) or isinstance(value, np.int64):
        conv_value = int(value)
    elif isinstance(value, np.float) or isinstance(value, np.float64):
        conv_value = float(value)
    else:
        conv_value = value

    return conv_value


def gen_config_folders(dataset_name, target_column_name, training):
    folders_names = []
    new_folder_name = None

    if training:
        folders_names.append(
            '/odinstorage/automl_data/training_results/preprocess_data/' + dataset_name + '/' + target_column_name + '/')
        folders_names.append(
            '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/')
        folders_names.append(
            '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/logs/')
        folders_names.append(
            '/odinstorage/automl_data/training_results/trained_models/' + dataset_name + '/' + target_column_name + '/')
        folders_names.append('/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/')
    else:
        file_path = '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/'
        folders_names_ = [f for f in os.listdir(file_path) if path.isdir(path.join(file_path, f))]

        if len(folders_names_) == 0:
            new_folder_name = '0'
        else:
            new_folder_name = str(max([int(folder_name) for folder_name in folders_names_]) + 1)

        folders_names.append(
            '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/' + new_folder_name + '/')

    for folder_name in folders_names:
        try:
            os.makedirs(folder_name)
        except FileExistsError:
            if 'testing' not in folder_name:
                shutil.rmtree(folder_name)
                os.makedirs(folder_name)

    if not training:
        return new_folder_name


def write_log_file(dataset_name, target_column_name, config_name, version_name):
    file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + \
                '/logs/' + config_name + '_' + version_name + '_log.txt'

    with open(file_name, 'w', encoding='utf-8') as log_file:
        if version_name == '0':
            log_file.write(config.log_text_1)
        elif version_name == '1':
            log_file.write(config.log_text_2)


def file_writer(dataset_name, target_column_name, config_name, version_name, data):
    if config_name == 'training_report':
        file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + \
                    '/training_report.txt'
    elif config_name == 'testing_report':
        file_name = '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/' + version_name + \
                    '/testing_report.txt'
    elif config_name == 'model_config_mapping':
        file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/' + \
                    config_name + '.txt'
    else:
        file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/' + \
                    config_name + '_' + version_name + '_conf.txt'

    if config_name in ['training_report', 'testing_report']:
        with open(file_name, 'w') as file:
            file.write(data)
    else:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def serialize_prep_models(dataset_name, target_column_name, config_name, version_name, models_dict):
    file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + \
                '/' + config_name + '_' + version_name + '.pickle'

    with open(file_name, 'wb') as pickle_file:
        pickle.dump(models_dict, pickle_file)


def save_preprocessed_dataset(dataset_name, target_column_name, config_name, version_name, x_df, y_df, store_y,
                              training=True):
    if training:
        main_path = '/odinstorage/automl_data/training_results/preprocess_data/'
        x_train_name = main_path + dataset_name + '/' + target_column_name + '/x_train_' + \
                       config_name + '_' + version_name + '.csv'
        x_df.to_csv(x_train_name, index=None)

        if store_y:
            y_train_name = main_path + dataset_name + '/' + target_column_name + '/y_train.csv'
            y_df.to_csv(y_train_name, index=None)

    else:
        if not store_y:
            file_name = '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/' + \
                        version_name + '/x_test_' + config_name + '.csv'
            x_df.to_csv(file_name, index=None)
        else:
            file_name = '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column_name + '/' + \
                        version_name + '/' + config_name + '.csv'
            y_df.to_csv(file_name)


def serialize_trained_models(dataset_name, target_column_name, trained_models):
    file_name = '/odinstorage/automl_data/training_results/trained_models/' + dataset_name + '/' + target_column_name + '/' + \
                'models.pickle'

    with open(file_name, 'wb') as pickle_file:
        pickle.dump(trained_models, pickle_file)


def serialize_param_space(folder_name, param_space_name, param_space):
    file_name = folder_name + param_space_name + '_param_space.pickle'

    with open(file_name, 'wb') as f:
        pickle.dump(param_space, f)


def create_folder_store_train_data(path, file_name, loaded_file):
    try:
        os.mkdir(path)
    except OSError:
        pass

    with open(path + file_name, 'wb+') as written_file:
        for chunk in loaded_file.chunks():
            written_file.write(chunk)


def store_test_with_predictions(dataset_name, target_column, predictions):
    results_path = '/odinstorage/automl_data/testing_results/' + dataset_name + '/' + target_column
    subfolders = os.listdir(results_path)
    last_subfolder = str(max([int(subfolder_number) for subfolder_number in subfolders]))

    predictions.to_csv(results_path + '/' + last_subfolder + '/' + 'test_with_predictions.csv')

    return results_path, last_subfolder