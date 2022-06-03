from django import forms
from django.db import models


class FileFormTrain(forms.Form):
    target_column = forms.CharField()
    task_type = forms.CharField()
    file = forms.FileField()


class TrainResults():
    def __init__(self, metric, score):
        self.metric = metric
        self.score = score


class FileFormTest(forms.Form):
    target_column = forms.CharField()
    task_type = forms.CharField()
    dataset_name = forms.CharField()
    file = forms.FileField()


class Prediction():
    def __init__(self, predicted_value, index):
        self.index = index
        self.predicted_value = predicted_value


class TestResults():
    def __init__(self, metric, score, predictions):
        self.metric = metric
        self.score = score
        self.predictions = predictions
