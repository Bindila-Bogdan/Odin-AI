import json


def get_searching_info(passed_time, config_number, folds_number, metric, scores):
    search_info = '*Details of AutoML training run*\n\n'
    search_info += '*Information regarding config search*\n' + 53 * '*' + '\n'
    search_info += 'Folds number: {}\nMetric: {}\nNumber of configurations tried: {}\n'.format(folds_number, metric,
                                                                                               config_number + 1)
    search_info += 'Scores:\n'

    info = {model_name: [metric + ': ' + str(score), 'config: ' + conf_id] for model_name, [score, conf_id] in
            scores.items()}
    formatted_info = json.dumps(info, indent=4)
    search_info += formatted_info + '\n'
    search_info += 'Time required by searching: {} s\n'.format(round(passed_time, 4))

    return search_info


def get_optimization_info(optimization_results, optimization_times, obtained_metric_values, time_limited_optimization,
                          max_optimization_time, iterations_bo, early_stopping_bo, folds_number_bo, metric,
                          total_time_required):
    optimization_info = '\n*Information regarding optimization*\n' + 53 * '*' + '\n'
    optimization_info += 'Folds number: {}\n'.format(folds_number_bo)

    if time_limited_optimization:
        optimization_info += 'Max optimization time: {} s\n'.format(max_optimization_time)
        optimization_info += 'Early stopping for linear models after {} iterations\n'.format(early_stopping_bo)
    else:
        optimization_info += 'Number of iterations for each model: {}\n'.format(iterations_bo)

    opt_info = dict(zip(list(optimization_results[0].keys()),
                        zip(['optimization time: ' + str(round(opt_time, 4)) + ' s' for opt_time in optimization_times],
                            [metric + ': ' + str(score) for score in obtained_metric_values])))
    formatted_opt_info = json.dumps(opt_info, indent=4)
    optimization_info += formatted_opt_info
    optimization_info += '\n'
    optimization_info += 'Time required by optimization: {} s\n'.format(total_time_required)

    return optimization_info


def get_stacking_info(classification_task, meta_model_type, meta_model_prob_type, iterations_meta, iteration_meta_prob,
                      folds_number_meta):
    stacking_info = ''
    stacking_info += 'Folds number:{}\n'.format(folds_number_meta)

    if classification_task:
        if meta_model_type == 0:
            stacking_info += 'Meta-model for standard stacking: log_reg\n'
        elif meta_model_type == 1:
            stacking_info += 'Meta-model for standard stacking: rfc\n'

        if meta_model_prob_type == 0:
            stacking_info += 'Meta-model for prob-based stacking: ridge_reg\n'
        elif meta_model_prob_type == 1:
            stacking_info += 'Meta-model for prob-based stacking: rfr\n'
    else:
        if meta_model_type == 0:
            stacking_info += 'Meta-model for standard stacking: ridge_reg\n'
        elif meta_model_type == 1:
            stacking_info += 'Meta-model for standard stacking: rfr\n'

    stacking_info += 'Iterations for optimizing standard meta-model: {}\n'.format(iterations_meta)

    if classification_task:
        stacking_info += 'Iterations for optimizing prob based meta-model: {}\n'.format(iteration_meta_prob)

    return stacking_info
