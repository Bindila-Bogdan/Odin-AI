import os
import base64
import automl_backend
from utility_functions import files_loading, files_storing
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework import status
from django.http.response import JsonResponse
from django.http import HttpResponse, FileResponse
from .models import FileFormTrain, FileFormTest, TrainResults, TestResults

import sys
sys.path.append("..")


def index(request):
    return HttpResponse("Odin AI backend")


@api_view(["POST"])
def train(request):
    """
    Gets the parameters needed and performs the trainig.
    """

    form = FileFormTrain(request.POST, request.FILES)

    if form.is_valid():
        loaded_file = request.FILES['file']
        file_name = loaded_file.name
        dataset_name = file_name[:-4]

        path = '/odinstorage/automl_data/datasets/' + dataset_name + '/'
        files_storing.create_folder_store_train_data(
            path, file_name, loaded_file)

        target_column = request.POST.dict()["target_column"]
        task_type = request.POST.dict()["task_type"]
        print("train", dataset_name, target_column, task_type)

        automl = automl_backend.AutoML(
            "train", dataset_name, target_column, task_type, 2, None)
        metric, score = automl.train()
        train_results = TrainResults(metric, score)

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

    form = FileFormTest(request.POST, request.FILES)

    if form.is_valid():
        loaded_file = request.FILES['file']
        file_name = loaded_file.name
        dataset_name = request.POST.dict()['dataset_name']

        path = '/odinstorage/automl_data/datasets/' + dataset_name + '/'
        files_storing.create_folder_store_train_data(
            path, file_name, loaded_file)

        target_column = request.POST.dict()["target_column"]
        task_type = request.POST.dict()["task_type"]
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
