import argparse
import sys


def get_bool(value):
    if isinstance(value, bool):
        return value
    elif value == 'True':
        return True
    else:
        return False


def lower_limit_checker(argument, limit, parser, argument_name):
    if argument is not None:
        if argument < limit:
            parser.error(argument_name + ' must be >= ' + str(limit))


def between_interval_checker(argument, lower_limit, higher_limit, parser, argument_name):
    if argument is not None:
        if argument < lower_limit or argument > higher_limit:
            parser.error(argument_name + ' must be inside [' + str(lower_limit) + ', ' + str(higher_limit) + ']')


def validate_arguments(arguments, parser):
    if arguments.run_type is None:
        parser.error('type of run is not valid')
    elif arguments.run_type not in ['train', 'test', 'train_test']:
        parser.error(arguments.run_type + ' is not a valid type of run')
    elif arguments.run_type in ['test', 'train_test'] and arguments.test_dataset_name is None:
        parser.error('test_dataset_name is not valid')

    if arguments.task_type is None:
        parser.error('task type is not valid')
    elif arguments.task_type == 'classification':
        if arguments.metric_name not in ['-', 'f1_weighted', 'balanced_accuracy', 'precision_weighted',
                                         'recall_weighted']:
            parser.error(arguments.metric_name + ' is not a valid classification metric')
    elif arguments.task_type == 'regression':
        if arguments.metric_name not in ['-', 'r2', 'nrmse']:
            parser.error(arguments.metric_name + ' is not a valid regression metric')
    else:
        parser.error(arguments.task_type + ' is an unknown task')

    lower_limit_checker(arguments.max_search_time, 30, parser, 'max_search_time')
    between_interval_checker(arguments.configurations_no, 2, 10000, parser, 'configurations_no')
    lower_limit_checker(arguments.iterations_bo, 2, parser, 'iterations_bo')
    lower_limit_checker(arguments.max_opt_time, 60, parser, 'max_opt_time')
    lower_limit_checker(arguments.early_stopping_bo, 2, parser, 'early_stopping_bo')
    between_interval_checker(arguments.folds_number_bo, 3, 10, parser, 'folds_number_bo')
    lower_limit_checker(arguments.iterations_meta, 2, parser, 'iterations_meta')
    between_interval_checker(arguments.folds_number_meta, 3, 10, parser, 'folds_number_meta')


def get_arguments():
    parser = argparse.ArgumentParser(description="*AutoML Tool*")

    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--run_type', type=str, metavar='',
                                    help='Valid types of run are train, test or train_test')
    required_arguments.add_argument('--dataset_name', type=str, metavar='',
                                    help='Name of the dataset used for training')
    required_arguments.add_argument('--target_column', type=str, metavar='',
                                    help='Name of the column that will be predicted')
    required_arguments.add_argument('--task_type', type=str, metavar='',
                                    help='Could be classification or regression')

    parser.add_argument('--test_dataset_name', type=str, metavar='', default=None,
                        help='Name of the dataset which is required for testing')
    parser.add_argument('--metric_name', type=str, metavar='', default='-',
                        help='Metric computed in order to measure the performance')
    parser.add_argument('--unknown_classes', type=get_bool, metavar='', default=False,
                        help='True in order to ignore unknown classes at test time')
    parser.add_argument('--max_search_time', type=int, metavar='', default=sys.maxsize,
                        help='Maximum number of seconds allocated to try preprocessing configurations')
    parser.add_argument('--configurations_no', type=int, metavar='', default=10,
                        help='Number of preprocessing configurations that will be tried')
    parser.add_argument('--iterations_bo', type=int, metavar='', default=None,
                        help='Number of iterations used by bayesian optimization')
    parser.add_argument('--max_opt_time', type=int, metavar='', default=sys.maxsize,
                        help='Maximum number of seconds allocated to optimize all models')
    parser.add_argument('--early_stopping_bo', type=int, metavar='', default=50,
                        help='Maximum number of iterations before early stopping while'
                             ' optimizing linear models')
    parser.add_argument('--folds_number_bo', type=int, metavar='', default=None,
                        help='Value of k used in cross-validation during models optimization')
    parser.add_argument('--iterations_meta', type=int, metavar='', default=None,
                        help='Number of iterations used by bayesian optimization for meta-model')
    parser.add_argument('--folds_number_meta', type=int, metavar='', default=None,
                        help='Value of k used in cross-validation during models stacking')

    arguments = parser.parse_args()
    validate_arguments(arguments, parser)

    return arguments
