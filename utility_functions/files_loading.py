import json
import base64
import pickle

import pandas as pd
from . import data_manipulation


def read_json_file(dataset_name, target_column_name, config_name, version_name):
    if config_name == 'model_config_mapping':
        file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/' + \
                    config_name + '.txt'
    else:
        file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/' + \
                    config_name + '_' + version_name + '_conf.txt'

    with open(file_name, 'r') as json_file:
        config_dict = json.load(json_file)

    return config_dict


def load_prep_models(dataset_name, target_column_name, config_name, version_name):
    file_name = '/odinstorage/automl_data/training_results/config_files/' + dataset_name + '/' + target_column_name + '/' + \
                config_name + '_' + version_name + '.pickle'

    with open(file_name, 'rb') as pickle_file:
        models_dict = pickle.load(pickle_file)

    return models_dict


def load_configuration(dataset_name, target_column_name, config_number, index):
    config_name = str(config_number)
    version_name = str(index)

    config_dict = read_json_file(
        dataset_name, target_column_name, config_name, version_name)
    models_dict = load_prep_models(
        dataset_name, target_column_name, config_name, version_name)

    return config_dict, models_dict


def load_preprocessed_dataset(dataset_name, target_column_name, config_name, version_name):
    loaded_datasets = []
    datasets_path = '/odinstorage/automl_data/training_results/preprocess_data/' + \
        dataset_name + '/' + target_column_name + '/'

    dataset_id = config_name + '_' + version_name
    loaded_datasets.append(pd.read_csv(
        datasets_path + 'x_train_' + dataset_id + '.csv'))

    return loaded_datasets


def load_targets(dataset_name, target_column_name):
    targets = {}
    datasets_path = '/odinstorage/automl_data/training_results/preprocess_data/' + \
        dataset_name + '/' + target_column_name + '/'
    targets['y_train'] = pd.read_csv(datasets_path + 'y_train.csv')

    return targets


def load_trained_models(dataset_name, target_column_name):
    file_name = '/odinstorage/automl_data/training_results/trained_models/' + dataset_name + '/' + target_column_name + '/' + \
                'models.pickle'

    with open(file_name, 'rb') as pickle_file:
        trained_models = pickle.load(pickle_file)

    return trained_models


def load_model_data_mappings(dataset_name, predicted_column):
    config_data = {}

    model_config_mapping = read_json_file(
        dataset_name, predicted_column, 'model_config_mapping', None)
    configs_names = [config_name for _, [
        _, config_name] in model_config_mapping.items()]

    for config_name in configs_names:
        split_config_name = config_name.split('_')
        config_data[config_name] = load_preprocessed_dataset(dataset_name, predicted_column, split_config_name[0],
                                                             split_config_name[1])

    targets = load_targets(dataset_name, predicted_column)

    config_data = {**config_data, **targets}
    model_config_mapping = {model_name: config_name for model_name, [
        _, config_name] in model_config_mapping.items()}

    return model_config_mapping, config_data


def deserialize_param_space(folder_name, model_name):
    file_name = folder_name + model_name + '_param_space.pickle'

    with open(file_name, 'rb') as f:
        param_space = pickle.load(f)

    return param_space


def load_test_with_predictions(results_path, last_subfolder):
    with open(results_path + '/' + last_subfolder + '/' + 'test_with_predictions.csv', 'rb') as file:
        test_with_predictions = file.read()

    test_with_predictions_encoded = base64.b64encode(
        test_with_predictions).decode('ascii')

    return test_with_predictions_encoded


def load_features_report(path):
    with open(path, 'rb') as file:
        features_report = file.read()

    features_report_encoded = base64.b64encode(
        features_report).decode('ascii')

    return features_report_encoded


def load_task_type(dataset_name, target_column):
    path = '/odinstorage/automl_data/training_results/config_files/' + \
        dataset_name + '/' + target_column + '/' + 'task_type.txt'

    with open(path, 'r') as file:
        task_type = file.read()

    return task_type


def load_intermediary_reports(dataset_name, target_column):
    main_path = '/odinstorage/automl_data/training_results/config_files/'
    txt_log_path = main_path + '{}/{}/logs/general_info_report_text.txt'
    json_log_path = main_path + '{}/{}/logs/general_info_report_json.txt'

    with open(txt_log_path.format(dataset_name, target_column), 'r') as file:
        txt_report = file.readlines()

    with open(json_log_path.format(dataset_name, target_column), 'r') as file:
        json_report = json.load(file)

    return txt_report, json_report


def load_feature_importance_data(dataset_name, target_column):
    json_config_path = '/odinstorage/automl_data/training_results/config_files/' + \
        '{}/{}/logs/general_info_report_json.txt'

    with open(json_config_path.format(dataset_name, target_column), 'r') as file:
        json_config_data = json.load(file)

    task_type = load_task_type(dataset_name, target_column)

    model_config_mapping, config_data = load_model_data_mappings(
        dataset_name, target_column)
    trained_models = load_trained_models(dataset_name, target_column)

    data_set = data_manipulation.read_dataset(
        '/odinstorage/automl_data/datasets/' + dataset_name + '/' + dataset_name + '.csv', disable_logging=True)

    stacking_type = list(trained_models.keys())[0]

    models_performance = read_json_file(
        dataset_name, target_column, 'model_config_mapping', None)
    models_performance = dict(zip(models_performance.keys(), [
                              value[0] * 100 for value in models_performance.values()]))

    return [json_config_data, task_type, model_config_mapping, config_data, trained_models, data_set, stacking_type, models_performance]


def load_feature_importance_img(dataset_name, target_column):
    path = '/odinstorage/automl_data/training_results/config_files/{}/{}/feature_importance.png'

    with open(path.format(dataset_name, target_column), 'rb') as file:
        img = file.read()

    img = base64.b64encode(img).decode('ascii')

    return img
