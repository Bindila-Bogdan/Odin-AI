from django.db import models

class TrainResults():
    def __init__(self, metric, score):
        self.metric = metric
        self.score = score

class Prediction():
    def __init__(self, predicted_value, index):
        self.index = index
        self.predicted_value = predicted_value


class TestResults():
    def __init__(self, metric, score, predictions):
        self.metric = metric
        self.score = score
        self.predictions = predictions
