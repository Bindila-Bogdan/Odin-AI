import os
import base64
import automl_backend
from utility_functions import files_loading, files_storing, data_manipulation
from report_creation import reporting, features_importances
from data_cleaning import datatypes_conversion
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework import status
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import TrainResults, TestResults


import sys
sys.path.append("..")


def index(request):
    return HttpResponse("Odin AI backend")


@api_view(["POST"])
def train(request):
    """
    Gets the parameters needed and performs the trainig.
    """

    request_data = JSONParser().parse(request)
    request_keys = request_data.keys()

    if 'file_content' in request_keys and 'file_name' in request_keys and 'target_column' in request_keys:
        file_content = request_data['file_content']
        file_name = request_data['file_name']
        target_column = request_data['target_column']

        loaded_file = base64.b64decode(file_content)
        dataset_name = file_name[:-4]

        path = '/odinstorage/automl_data/datasets/' + dataset_name + '/'
        files_storing.create_folder_store_train_data(
            path, file_name, loaded_file)

        train_data_set = data_manipulation.read_dataset(path + file_name, True)
        categorical = datatypes_conversion.find_target_column_type(
            train_data_set, target_column)

        try:
            task_type = request_data["task_type"]
        except KeyError:
            if categorical:
                task_type = 'classification'

            else:
                task_type = 'regression'

        print("train", dataset_name, target_column, task_type)

        try:
            automl = automl_backend.AutoML(
                "train", dataset_name, target_column, task_type, 2, None)
            metric, score = automl.train()

            path_ = '/odinstorage/automl_data/training_results/config_files/' + \
                dataset_name + '/' + target_column + '/'
            files_storing.store_task_type(path_, task_type)

            reporting_info = reporting.create_general_info_report(
                dataset_name, target_column)
            [sizes, useless_columns, duplicated_rows,
                outlier_missing_data] = reporting_info

            features_report_path = files_storing.store_features_report(
                dataset_name, target_column, 'features_report', outlier_missing_data)
            features_report_encoded = files_loading.load_features_report(
                features_report_path)

            features_importances.compute_feature_importance(
                dataset_name, target_column)
            features_importances_img = files_loading.load_feature_importance_img(
                dataset_name, target_column)

            train_results = TrainResults(metric, score, sizes, useless_columns, duplicated_rows,
                                         features_report_encoded, features_importances_img)

        except:
            print('internal training error')
            error_message = {'error': 'internal training error'}

            return JsonResponse(error_message, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return JsonResponse(train_results.__dict__, status=status.HTTP_200_OK)

    else:
        print('invalid form')
        error_message = {'error': 'wrong request format'}

        return JsonResponse(error_message, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def test(request):
    """
    Predicts the target for received test data set.
    """

    request_data = JSONParser().parse(request)
    request_keys = request_data.keys()

    if 'file_content' in request_keys and 'file_name' in request_keys and 'target_column' in request_keys and 'train_file_name' in request_keys:
        file_content = request_data['file_content']
        file_name = request_data['file_name']
        train_file_name = request_data['train_file_name']
        target_column = request_data['target_column']

        loaded_file = base64.b64decode(file_content)
        dataset_name = train_file_name[:-4]

        path = '/odinstorage/automl_data/datasets/' + dataset_name + '/'
        files_storing.create_folder_store_train_data(
            path, file_name, loaded_file)

        try:
            task_type = request_data["task_type"]
        except KeyError:
            task_type = files_loading.load_task_type(
                dataset_name, target_column)

        test_dataset_name = file_name[:-4]
        print("test", task_type, dataset_name,
              test_dataset_name, target_column)

        try:
            automl = automl_backend.AutoML(
                "test", dataset_name, target_column, task_type, 2, test_dataset_name)
            predictions, metric, score = automl.predict()

            print(metric, score)
            results_path, last_subfolder = files_storing.store_test_with_predictions(
                dataset_name, target_column, predictions)
            test_with_predictions_encoded = files_loading.load_test_with_predictions(
                results_path, last_subfolder)

            test_results = TestResults(score, test_with_predictions_encoded)
        except:
            print('internal testing error')
            error_message = {'error': 'internal testing error'}

            return JsonResponse(error_message, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return JsonResponse(test_results.__dict__, status=status.HTTP_200_OK)

    else:
        print('invalid form')
        error_message = {'error': 'wrong request format'}

        return JsonResponse(error_message, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def delete_data(request):
    parsed_request = JSONParser().parse(request)

    if 'dataset_name' in parsed_request.keys():
        dataset_name = parsed_request['dataset_name']

        if dataset_name not in os.listdir('/odinstorage/automl_data/datasets/'):
            return JsonResponse({'error': f'{dataset_name} data set does not exist'}, status=status.HTTP_400_BAD_REQUEST)

        if files_storing.delete_data(dataset_name) == 5:
            return JsonResponse({'succeded': f'all data related to {dataset_name} data set is deleted'}, status=status.HTTP_200_OK)

        else:
            return JsonResponse({'error': f'failed to detele all data related to {dataset_name} data set'}, status=status.HTTP_501_NOT_IMPLEMENTED)

    else:
        error_message = {'error': 'wrong request format'}

        return JsonResponse(error_message, status=status.HTTP_400_BAD_REQUEST)
