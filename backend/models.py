from django import forms


class TrainResults():
    def __init__(self, metric, score):
        self.metric = metric
        self.score = score


class TestResults():
    def __init__(self, score, predictions):
        self.score = score
        self.file = predictions
