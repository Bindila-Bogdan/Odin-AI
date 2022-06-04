from django import forms


class FileFormTrain(forms.Form):
    target_column = forms.CharField()
    file = forms.FileField()


class TrainResults():
    def __init__(self, metric, score):
        self.metric = metric
        self.score = score


class FileFormTest(forms.Form):
    target_column = forms.CharField()
    dataset_name = forms.CharField()
    file = forms.FileField()


class TestResults():
    def __init__(self, score, predictions):
        self.score = score
        self.file = predictions
