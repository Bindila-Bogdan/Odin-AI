import sys


class ArgumentsChecker():
    def __init__(self, run_type, dataset_name, target_column, task_type, training_time, test_dataset_name):
        self._run_type = run_type
        self._dataset_name = dataset_name
        self._target_column = target_column
        self._task_type = task_type
        self._training_time = training_time
        self._test_dataset_name = test_dataset_name

        self._metric_name = '-'
        self._unknown_classes = False
        self._max_search_time = training_time * 60 // 2 # sys.maxsize 
        self._configurations_no = 10
        self._iterations_bo = None
        self._max_opt_time = training_time * 60 // 2 # sys.maxsize 
        self._early_stopping_bo = 50
        self._folds_number_bo = None
        self._iterations_meta = None
        self._folds_number_meta = None

        self._validate_arguments()

    def _get_bool(self, value):
        if isinstance(value, bool):
            return value
        elif value == 'True':
            return True
        else:
            return False

    def _lower_limit_checker(self, argument, limit, argument_name):
        if argument is not None:
            if argument < limit:
                raise ValueError(argument_name + ' must be >= ' + str(limit))

    def _between_interval_checker(self, argument, lower_limit, higher_limit, argument_name):
        if argument is not None:
            if argument < lower_limit or argument > higher_limit:
                raise ValueError(
                    argument_name + ' must be inside [' + str(lower_limit) + ', ' + str(higher_limit) + ']')

    def _validate_arguments(self):
        if self._run_type is None:
            raise ValueError('type of run is not valid')
        elif self._run_type not in ['train', 'test', 'train_test']:
            raise ValueError(self._run_type + ' is not a valid type of run')
        elif self._run_type in ['test', 'train_test'] and self._test_dataset_name is None:
            raise ValueError('test_dataset_name is not valid')

        if self._task_type is None:
            raise ValueError('task type is not valid')
        elif self._task_type == 'classification':
            if self._metric_name not in ['-', 'f1_weighted', 'balanced_accuracy', 'precision_weighted',
                                        'recall_weighted']:
                raise ValueError(self._metric_name +
                                 ' is not a valid classification metric')
        elif self._task_type == 'regression':
            if self._metric_name not in ['-', 'r2', 'nrmse']:
                raise ValueError(self._metric_name +
                                 ' is not a valid regression metric')
        else:
            raise ValueError(self._task_type + ' is an unknown task')

        self._lower_limit_checker(self._training_time, 1, 'training_time')
        self._lower_limit_checker(self._max_search_time, 30, 'max_search_time')
        self._between_interval_checker(
            self._configurations_no, 2, 10000, 'configurations_no')
        self._lower_limit_checker(self._iterations_bo, 2, 'iterations_bo')
        self._lower_limit_checker(self._max_opt_time, 30, 'max_opt_time')
        self._lower_limit_checker(
            self._early_stopping_bo, 2, 'early_stopping_bo')
        self._between_interval_checker(
            self._folds_number_bo, 3, 10, 'folds_number_bo')
        self._lower_limit_checker(self._iterations_meta, 2, 'iterations_meta')
        self._between_interval_checker(
            self._folds_number_meta, 3, 10, 'folds_number_meta')
