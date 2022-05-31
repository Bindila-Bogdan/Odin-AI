import sys

import config
from data_prediction import test_data_prediction
from ensemble_learning import models_stacking
from models_optimization import hyperparameters_optimization, hyperparameters_spaces
from preprocessing_searching import configuration_searching
from utility_functions import arguments_parser, files_loading, reporting, files_storing


class AutoML:
    def __init__(self):
        arguments = arguments_parser.get_arguments()

        self._run_type = arguments.run_type
        self._dataset_name = arguments.dataset_name
        self._target_column = arguments.target_column
        self._metric = arguments.metric_name

        if arguments.task_type == 'classification':
            self._classification_task = True
            if self._metric == '-':
                self._metric = 'f1_weighted'
        else:
            self._classification_task = False
            if self._metric == '-':
                self._metric = 'r2'
            elif self._metric == 'nrmse':
                self._metric = 'neg_root_mean_squared_error'

        self._handle_unknown_classes = arguments.unknown_classes
        self._include_gbc = True
        self._max_search_time = arguments.max_search_time

        if self._max_search_time != sys.maxsize:
            self._configurations_no = 10000
        else:
            self._configurations_no = arguments.configurations_no

        self._iterations_bo = arguments.iterations_bo
        self._max_optimization_time = arguments.max_opt_time

        if self._max_optimization_time == sys.maxsize and self._iterations_bo is None:
            self._time_limited_optimization = False
        elif self._max_optimization_time != sys.maxsize:
            self._time_limited_optimization = True
        elif self._iterations_bo is not None:
            self._time_limited_optimization = False

        self._early_stopping_bo = arguments.early_stopping_bo
        self._folds_number_bo = arguments.folds_number_bo
        self._iterations_meta = arguments.iterations_meta
        self._folds_number_meta = arguments.folds_number_meta
        self._max_optimization_time_copy = None
        self._test_dataset = arguments.test_dataset_name

        if self._run_type == 'train':
            self.train()
        elif self._run_type == 'test':
            self.predict()
        elif self._run_type == 'train_test':
            self.train()
            self.predict()

    def _configure_optimization(self, rows_no_train_dataset):
        hyperparameters_spaces.write_param_spaces(True)
        hyperparameters_spaces.write_param_spaces(False)

        if self._max_optimization_time == sys.maxsize and self._iterations_bo is None:
            if rows_no_train_dataset <= 10000:
                if self._classification_task:
                    self._iterations_bo = 20
                else:
                    self._iterations_bo = 30
            else:
                if self._classification_task:
                    self._iterations_bo = 10
                else:
                    self._iterations_bo = 20

        config.TIME_LIMITED_OPTIMIZATION = self._time_limited_optimization

        if config.TIME_LIMITED_OPTIMIZATION:
            config.max_optimization_time = self._max_optimization_time
            self._max_optimization_time_copy = config.max_optimization_time
            config.EARLY_STOPPING_BO = self._early_stopping_bo
            self._iterations_bo = 999
        else:
            config.max_optimization_time = -1

        config.max_model_optimization_time = 0.0
        config.max_optimization_time_mlpc = 0.0

        if self._folds_number_bo is None:
            if rows_no_train_dataset <= 1000:
                self._folds_number_bo = 10
            else:
                self._folds_number_bo = 5

    def _configure_stacking(self, rows_no_train_dataset):
        if self._folds_number_meta is None:
            if rows_no_train_dataset <= 1000:
                self._folds_number_meta = 10
            else:
                self._folds_number_meta = 5

        if self._iterations_meta is None:
            if self._classification_task:
                self._iterations_meta = 10
            else:
                self._iterations_meta = 20

    def train(self):
        print('*Details of AutoML training run*\n')

        config.initial_opt_time = 0.0

        random_state, search_info = configuration_searching.search_preprocessing_steps(self._dataset_name,
                                                                                       self._target_column,
                                                                                       self._classification_task,
                                                                                       self._metric,
                                                                                       self._max_search_time,
                                                                                       self._configurations_no,
                                                                                       self._include_gbc)

        model_config_mapping, config_data = files_loading.load_model_data_mappings(self._dataset_name,
                                                                                   self._target_column)
        rows_no_train_dataset = list(config_data.values())[0][0].shape[0]

        self._configure_optimization(rows_no_train_dataset)

        optimization_results = hyperparameters_optimization.optimize_models(model_config_mapping, config_data,
                                                                            self._classification_task, self._metric,
                                                                            self._iterations_bo, self._folds_number_bo,
                                                                            random_state, self._include_gbc)

        [optimized_models, _, optimization_times, obtained_metric_values, total_time_required] = optimization_results

        optimization_info = reporting.get_optimization_info(optimization_results, optimization_times,
                                                            obtained_metric_values, self._time_limited_optimization,
                                                            self._max_optimization_time_copy, self._iterations_bo,
                                                            self._early_stopping_bo, self._folds_number_bo,
                                                            self._metric, total_time_required)

        self._configure_stacking(rows_no_train_dataset)

        result = models_stacking.create_models_groups(optimized_models, self._metric)
        [best_model, stacking_model_setups, models_ranking_scores, models_ranking] = result

        stacking_info = models_stacking.generate_best_ensemble(model_config_mapping, config_data, self._metric,
                                                               config.META_MODEL_TYPE, self._iterations_meta,
                                                               config.ITERATION_META_PROB, self._folds_number_meta,
                                                               config.META_MODEL_PROB_TYPE, best_model, models_ranking,
                                                               models_ranking_scores, stacking_model_setups,
                                                               self._classification_task, self._dataset_name,
                                                               self._target_column, random_state)

        report_text = search_info + optimization_info + stacking_info
        report_text = report_text.replace('neg_root_mean_squared_error', 'nrmse')
        files_storing.file_writer(self._dataset_name, self._target_column, 'training_report', None, report_text)

    def predict(self):
        test_data_prediction.predict_data(self._dataset_name, self._test_dataset, self._target_column, self._metric,
                                          self._classification_task, self._handle_unknown_classes)


if __name__ == '__main__':
    automl = AutoML()
