import time
from os import listdir
from os.path import isdir, join

import numpy as np
import pandas as pd

from ensemble_learning import models_stacking, standard_stacking, prob_based_stacking
from utility_functions import files_loading, files_storing, data_manipulation
from . import preprocessing_testing


def testing_input_validity(dataset_name, predicted_column):
    path = '/odinstorage/automl_data/training_results/trained_models/' + dataset_name + '/'

    try:
        folders_names = [f for f in listdir(path) if isdir(join(path, f))]
    except FileNotFoundError:
        print('There aren\'t trained models using {} dataset'.format(dataset_name))
        return False

    if predicted_column not in folders_names:
        print('There aren\'t trained models that predict {} column'.format(predicted_column))
        return False

    return True


def check_construct(x_train_df, test_continuous_df, test_categorical_df):
    x_test_df = None

    if test_continuous_df is None or test_categorical_df is None:
        if test_continuous_df is None:
            x_test_df = test_categorical_df
        if test_categorical_df is None:
            x_test_df = test_continuous_df
    else:
        x_test_df = pd.concat([test_continuous_df, test_categorical_df], axis=1)

    if (x_train_df.columns != x_test_df.columns).sum() != 0:
        print('Training and testing datasets have different naming for columns.')
        print('Train dataset columns:\n{}'.format(list(x_train_df.columns)))
        print('Test dataset columns:\n{}'.format(list(x_test_df.columns)))
        return None

    return x_test_df


def get_used_models_configs(dataset_name, predicted_column):
    models_configs_names, config_dataset = files_loading.load_model_data_mappings(dataset_name, predicted_column)

    trained_models = files_loading.load_trained_models(dataset_name, predicted_column)
    models_used = list(trained_models.values())[0][0]

    if isinstance(models_used, str):
        models_used = [models_used]

    all_models = set(models_configs_names.keys())
    models_to_remove = all_models - set(models_used)

    for model_to_remove in list(models_to_remove):
        corresponding_config = models_configs_names[model_to_remove]
        indices = [i for i, config in enumerate(list(models_configs_names.values())) if config == corresponding_config]

        del models_configs_names[model_to_remove]

        if len(indices) == 1:
            del config_dataset[corresponding_config]

    models_configs = {}

    for config_name in list(models_configs_names.values()):
        conf_name = config_name.split('_')

        config_dict, models_dict = files_loading.load_configuration(dataset_name, predicted_column, conf_name[0],
                                                                    conf_name[1])
        models_configs[config_name] = [config_dict, models_dict]

    return models_configs_names, config_dataset, trained_models, models_configs


def predict_test_data(encoded_y_test_df, predicted_column, models_configs_names, models_data, trained_models,
                      models_dict, metric, classification_task):
    score = None
    method_used = list(trained_models.keys())[0]

    if 'without_stacking' == method_used:
        trained_model = trained_models[method_used][1][1]
        x_train_df = list(models_data.values())[0]
        predictions = trained_model.predict(x_train_df)

        if encoded_y_test_df is not None:
            true_values = list(encoded_y_test_df[encoded_y_test_df.columns[-1]].values)
            score = models_stacking.get_score(true_values, predictions, metric)

    else:
        predictions = None
        meta_models = trained_models[method_used][1][0]
        models_names = trained_models[method_used][0]
        stacked_models = trained_models[method_used][1][1]
        optimized_models = dict(zip(models_names, stacked_models))

        datasets_models = models_stacking.prepare_stacking(models_configs_names, models_data, optimized_models,
                                                           testing=True)
        if 'standard' in method_used:
            predictions, score = standard_stacking.stacked_prediction(datasets_models, meta_models, encoded_y_test_df,
                                                                      metric)
        elif 'prob_based' in method_used:
            predictions, score = prob_based_stacking.stacked_prediction_prob(datasets_models, meta_models,
                                                                             encoded_y_test_df, metric)

    predictions_df = pd.DataFrame({predicted_column: np.array(predictions).flatten()})
    if classification_task:
        target_encoder = models_dict['2']
        try:
            del target_encoder['mode']
        except KeyError:
            pass

        encoder_keys = list(target_encoder.keys())
        encoder_values = list(target_encoder.values())

        inverse_encoder = dict(zip(encoder_values, encoder_keys))
        predictions_df[predicted_column] = predictions_df[predicted_column].map(inverse_encoder)

    return predictions_df, score


def predict_data(dataset_name, csv_file_name, predicted_column, metric, classification_task, handle_unknown_classes):
    start_time = time.time()

    if not testing_input_validity(dataset_name, predicted_column):
        return

    new_folder_name = files_storing.gen_config_folders(dataset_name, predicted_column, False)
    x_test_initial_df, y_test_initial_df, shape_info = data_manipulation.read_test_dataset(dataset_name, csv_file_name,
                                                                                           predicted_column, metric)
    models_configs_names, config_dataset, trained_models, models_configs = get_used_models_configs(dataset_name,
                                                                                                   predicted_column)
    models_data, encoded_y_test_df, models_dict = preprocessing_testing.preprocess_test_datasets(x_test_initial_df,
                                                                                                 y_test_initial_df,
                                                                                                 models_configs,
                                                                                                 config_dataset,
                                                                                                 dataset_name,
                                                                                                 predicted_column,
                                                                                                 new_folder_name,
                                                                                                 handle_unknown_classes)

    if models_data is None:
        return

    predictions_df, score = predict_test_data(encoded_y_test_df, predicted_column, models_configs_names, models_data,
                                              trained_models, models_dict, metric, classification_task)
    files_storing.save_preprocessed_dataset(dataset_name, predicted_column, 'predictions', new_folder_name, None,
                                            predictions_df, True, False)

    testing_info = '\n*Details of AutoML testing run*\n\n'
    testing_info += shape_info + '\n'

    if score is not None and x_test_initial_df.shape[0] > 1:
        testing_info += 'Testing {} score: {}\n'.format(metric, score)

    testing_info += 'Time required by testing: {} s'.format(round(time.time() - start_time, 4))
    testing_info = testing_info.replace('neg_root_mean_squared_error', 'nrmse')
    files_storing.file_writer(dataset_name, predicted_column, 'testing_report', new_folder_name, testing_info)

    print(testing_info)

    if y_test_initial_df is None:
        test_predictions_df = pd.concat([x_test_initial_df, pd.DataFrame(predictions_df).rename({predicted_column: 'predicted_' + predicted_column}, axis=1)], axis=1)
    else:
        test_predictions_df = pd.concat([x_test_initial_df, y_test_initial_df, pd.DataFrame(predictions_df).rename({predicted_column: 'predicted_' + predicted_column}, axis=1)], axis=1)

    return test_predictions_df, metric, score
