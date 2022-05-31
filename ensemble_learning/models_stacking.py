import time

from sklearn import base, metrics

from utility_functions import logging, files_storing, reporting
from . import standard_stacking, prob_based_stacking


def get_score(true_y, pred_y, metric):
    if len(true_y) != len(pred_y):
        print('There might be unknown classes.')
        return None

    score = None

    if metric == 'balanced_accuracy':
        score = metrics.balanced_accuracy_score(true_y, pred_y)
    elif metric == 'precision_weighted':
        score = metrics.precision_score(true_y, pred_y, average='weighted')
    elif metric == 'recall_weighted':
        score = metrics.recall_score(true_y, pred_y, average='weighted')
    elif metric == 'f1_weighted':
        score = metrics.f1_score(true_y, pred_y, average='weighted')
    elif metric == 'neg_root_mean_squared_error':
        score = metrics.mean_squared_error(true_y, pred_y, squared=False)
        score /= (max(true_y) - min(true_y))
    elif metric == 'r2':
        score = metrics.r2_score(true_y, pred_y)

    return score


def prepare_stacking(model_config_mapping, config_data, optimized_models, testing=False):
    models_training = []
    datasets_training = []

    trained_models = []

    for model_name, optimized_model in optimized_models.items():
        if not testing:
            models_training.append(base.clone(optimized_model[2]))
            trained_models.append(optimized_model[2])
            datasets_training.append(config_data[model_config_mapping[model_name]][0])
        else:
            models_training.append(optimized_model)
            datasets_training.append(config_data[model_config_mapping[model_name]])

    datasets_models = list(zip(list(optimized_models.keys()), datasets_training, models_training))

    if not testing:
        return datasets_models, trained_models
    else:
        return datasets_models


def create_models_groups(optimized_models, metric):
    if metric != 'neg_root_mean_squared_error':
        reverse = True
    else:
        reverse = False

    models_ranking_scores = {model_name: data[0] for model_name, data in
                             sorted(optimized_models.items(), key=lambda x: x[1][0], reverse=reverse)}
    models_ranking = list(models_ranking_scores.keys())

    stacking_model_setups = [{models_ranking[0]: optimized_models[models_ranking[0]],
                              models_ranking[1]: optimized_models[models_ranking[1]],
                              models_ranking[2]: optimized_models[models_ranking[2]]},
                             {models_ranking[0]: optimized_models[models_ranking[0]],
                              models_ranking[1]: optimized_models[models_ranking[1]]}]

    best_model = optimized_models[models_ranking[0]]

    return [best_model, stacking_model_setups, models_ranking_scores, models_ranking]


def generate_best_ensemble(model_config_mapping, config_data, metric, meta_model_type, iterations_meta,
                           iteration_meta_prob, folds_number_meta, meta_model_prob_type, best_model, models_ranking,
                           models_ranking_scores, stacking_model_setups, classification_task, dataset_name,
                           target_column_name, rand_state):
    print('*Stacking models*')
    initial_time = time.time()
    stacking_info = '\n*Information regarding ensembling*\n' + 53 * '*' + '\n'
    stacking_scores = {'best_model': (None, best_model[2], list(models_ranking_scores.values())[0])}

    score = stacking_scores['best_model'][2]

    if score == 1.0 or score == 0.0:
        stacking_info += 'Best model has already a perfect score ({}).'.format(score)
        logging.display('Best model has already a perfect score.', p=4)

        if metric == 'neg_root_mean_squared_error':
            print('nrmse: {}'.format(score))
        else:
            print('{}: {}'.format(metric, score))

        files_storing.serialize_trained_models(dataset_name, target_column_name,
                                               {'without_stacking': [models_ranking[0], stacking_scores['best_model']]})

        return stacking_info, metric, score
    else:
        stacking_info += reporting.get_stacking_info(classification_task, meta_model_type, meta_model_prob_type,
                                                     iterations_meta, iteration_meta_prob, folds_number_meta)
        y_train_df = config_data['y_train']

        for i in range(2):
            train_datasets_models, trained_models = prepare_stacking(model_config_mapping, config_data,
                                                                     stacking_model_setups[i])

            status = 53 * '*' + '\nStandard stacking using {} models'.format(len(train_datasets_models))
            print(status)
            stacking_info += status + '\n'
            start_time = time.time()
            meta_model_standard, meta_param_space_standard = standard_stacking.configure_standard_stacking(
                classification_task, meta_model_type, rand_state)
            meta_model_standard, training_standard_stacking_score, info = standard_stacking.stack_models(
                classification_task, train_datasets_models, meta_model_standard, y_train_df, metric, rand_state,
                meta_param_space_standard, iterations_meta, folds_number_meta)

            stacking_info += info + '\n'
            stacking_info += 'Stacking time: {} s\n'.format(round(time.time() - start_time, 4))
            stacking_scores['standard_' + str(len(train_datasets_models))] = (
                meta_model_standard, trained_models, training_standard_stacking_score)

            if classification_task:
                status = 53 * '*' + '\nProb based stacking using {} models'.format(len(train_datasets_models))
                print(status)
                stacking_info += status + '\n'
                start_time = time.time()
                meta_model_prob, meta_param_space_prob = prob_based_stacking.configure_prob_based_stacking(
                    meta_model_prob_type, rand_state)
                meta_model_prob, training_prob_based_stacking_score, info = prob_based_stacking.stack_models_prob(
                    classification_task, train_datasets_models, meta_model_prob, y_train_df, 'r2', metric, rand_state,
                    meta_param_space_prob, iteration_meta_prob, folds_number_meta)

                stacking_info += info + '\n'
                stacking_info += 'Stacking time: {} s\n'.format(round(time.time() - start_time, 4))
                stacking_scores['prob_based_' + str(len(train_datasets_models))] = (
                    meta_model_prob, trained_models, training_prob_based_stacking_score)

        if metric != 'neg_root_mean_squared_error':
            reverse = True
        else:
            reverse = False

        stacking_scores_sorted = {ensemble_name: ensemble for ensemble_name, ensemble in
                                  sorted(stacking_scores.items(), key=lambda x: x[1][2], reverse=reverse)}
        best_method = list(stacking_scores_sorted.keys())[0]

        models_no = 1
        if best_method == 'best_model':
            files_storing.serialize_trained_models(dataset_name, target_column_name,
                                                   {'without_stacking': [models_ranking[0],
                                                                         stacking_scores['best_model']]})
        else:
            models_no = best_method.split('_')[-1]
            best_models_trained = {best_method: [models_ranking[:int(models_no)], stacking_scores_sorted[best_method]]}
            files_storing.serialize_trained_models(dataset_name, target_column_name, best_models_trained)

        if metric == 'neg_root_mean_squared_error':
            final_score = 53 * '*' + '\nFinal nrmse: {}'.format(list(stacking_scores_sorted.values())[0][2])
        else:
            final_score = 53 * '*' + '\nFinal {}: {}'.format(metric, list(stacking_scores_sorted.values())[0][2])

        print(final_score)
        stacking_info += final_score

        if best_method == 'best_model':
            stacking_info += '\nBest performance is obtained without stacking\nBest model is {}\n'.format(
                models_ranking[0])
        else:
            stacking_info += '\nBest type of stacking: {} using {}' \
                             ' models\n'.format(' '.join(best_method.split('_')[:-1]), models_no)
        time_required_info = 'Time required by stacking: {} s'.format(round(time.time() - initial_time, 4))
        print(time_required_info)
        stacking_info += time_required_info

        return stacking_info, metric, score
