import os
import automl_backend
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework import status
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import FileFormTrain, TrainResults, Prediction, TestResults
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

        try:
            os.mkdir(path)
        except OSError:
            pass

        with open(path + file_name, 'wb+') as written_file:
            for chunk in loaded_file.chunks():
                written_file.write(chunk)

        target_column = request.POST.dict()["target_column"]
        task_type = request.POST.dict()["task_type"]

        print("train", dataset_name, target_column, task_type)

        automl = automl_backend.AutoML(
            "train", dataset_name, target_column, task_type, 5, None)
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

    test_params = JSONParser().parse(request)
    print(test_params)

    run_type = "test"
    task_type = test_params["task_type"]
    dataset_name = test_params["dataset_name"][:-4]
    test_dataset_name = test_params["test_dataset_name"][:-4]
    target_column = test_params["target_column"]

    print(run_type, task_type, dataset_name, test_dataset_name, target_column)

    automl = automl_backend.AutoML(
        run_type, dataset_name, target_column, task_type, 1, test_dataset_name)
    predictions, metric, score = automl.predict()
    print(metric, score)

    predictions_obj = []

    for i, prediction in enumerate(predictions):
        j = i + 1

        prediction_obj = Prediction(prediction, j)
        predictions_obj.append(prediction_obj.__dict__)

    test_results = TestResults(metric, score, predictions_obj)

    return JsonResponse(test_results.__dict__, status=status.HTTP_200_OK)
