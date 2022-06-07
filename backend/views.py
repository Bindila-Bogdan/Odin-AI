import os
import base64
import automl_backend
from utility_functions import files_loading, files_storing, data_manipulation
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
    file_content = request_data['file_content']
    file_name = request_data['file_name']
    target_column = request_data['target_column']

    loaded_file = base64.b64decode(file_content)
    dataset_name = file_name[:-4]

    path = './automl_data/datasets/' + dataset_name + '/'
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

    automl = automl_backend.AutoML(
        "train", dataset_name, target_column, task_type, 2, None)
    metric, score = automl.train()

    path_ = './automl_data/training_results/config_files/' + \
        dataset_name + '/' + target_column + '/'
    files_storing.store_task_type(path_, task_type)
    train_results = TrainResults(metric, score)

    return JsonResponse(train_results.__dict__, status=status.HTTP_200_OK)


@api_view(['POST'])
def test(request):
    """
    Predicts the target for received test data set.
    """

    form = FileFormTest(request.POST, request.FILES)

    if form.is_valid():
        loaded_file = request.FILES['file']
        file_name = loaded_file.name
        dataset_name = request.POST.dict()['dataset_name']

        path = './automl_data/datasets/' + dataset_name + '/'
        files_storing.create_folder_store_train_data(
            path, file_name, loaded_file)

        target_column = request.POST.dict()["target_column"]

        try:
            task_type = request.POST.dict()["task_type"]
        except KeyError:
            path_ = './automl_data/training_results/config_files/' + \
                dataset_name + '/' + target_column + '/'
            task_type = files_loading.load_task_type(path_)

        test_dataset_name = file_name[:-4]
        print("test", task_type, dataset_name,
              test_dataset_name, target_column)

        automl = automl_backend.AutoML(
            "test", dataset_name, target_column, task_type, 2, test_dataset_name)
        predictions, metric, score = automl.predict()

        print(metric, score)
        results_path, last_subfolder = files_storing.store_test_with_predictions(
            dataset_name, target_column, predictions)
        test_with_predictions_encoded = files_loading.load_test_with_predictions(
            results_path, last_subfolder)

        test_results = TestResults(score, test_with_predictions_encoded)

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

        if dataset_name not in os.listdir('./automl_data/datasets/'):
            return JsonResponse({'error': f'{dataset_name} data set does not exist'}, status=status.HTTP_400_BAD_REQUEST)

        if files_storing.delete_data(dataset_name) == 5:
            return JsonResponse({'succeded': f'all data related to {dataset_name} data set is deleted'}, status=status.HTTP_200_OK)

        else:
            return JsonResponse({'error': f'failed to detele all data related to {dataset_name} data set'}, status=status.HTTP_501_NOT_IMPLEMENTED)

    else:
        error_message = {'error': 'wrong request format'}

        return JsonResponse(error_message, status=status.HTTP_400_BAD_REQUEST)
