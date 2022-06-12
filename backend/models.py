class TrainResults():
    def __init__(self, metric, score, sizes, useless_columns, duplicated_rows, features_report, feature_importance):
        self.metric = metric
        self.score = score
        self.sizes = sizes
        self.useless_columns = useless_columns
        self.duplicated_rows = duplicated_rows
        self.features_report_file = features_report
        self.feature_importance_image = feature_importance


class TestResults():
    def __init__(self, score, predictions):
        self.score = score
        self.file = predictions
