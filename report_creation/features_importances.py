from time import perf_counter
from utility_functions import files_storing
from utility_functions.files_loading import load_feature_importance_data
from sklearn import feature_selection
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.1)


def extract_needed_info(dataset_name, target_column):
    loaded_data = load_feature_importance_data(dataset_name, target_column)
    json_config_data = loaded_data[0]
    data_set = loaded_data[-3]

    cat_columns = json_config_data['4']
    created_features = json_config_data['7g']
    dropped_columns = json_config_data['1a'] + json_config_data['1c']

    features_mapping = {}

    for index, feature in enumerate(created_features):
        features_mapping['new_feat_' + str(index)] = [feature[0], feature[1]]

    all_columns = list(data_set.columns)
    all_columns.remove(target_column)
    cont_columns = list(set(all_columns).difference(set(cat_columns)))
    all_columns = list(set(all_columns).difference(set(dropped_columns)))

    return [cat_columns, cont_columns, all_columns, features_mapping], loaded_data[1:]


def separate_columns(x_train, cont_columns, cat_columns):
    current_all_columns = list(x_train.columns)
    current_all_columns_copy = current_all_columns.copy()
    current_cat_columns = []
    current_cont_columns = []

    for column in current_all_columns:
        for cont_column in cont_columns:
            if cont_column in column and '_is_missing' not in column:
                try:
                    current_all_columns_copy.remove(column)
                    current_cont_columns.append(column)
                except:
                    pass

            elif '_is_missing' in column:
                try:
                    current_all_columns_copy.remove(column)
                    current_cat_columns.append(column)
                except:
                    pass

    for column in current_all_columns:
        for cat_column in cat_columns:
            if cat_column in column:
                try:
                    current_all_columns_copy.remove(column)
                    current_cat_columns.append(column)
                except:
                    pass

    current_cont_columns.extend(current_all_columns_copy)

    categorical_mask = []

    for column in current_all_columns:
        if column in current_cont_columns:
            categorical_mask.append(False)

        elif column in current_cat_columns:
            categorical_mask.append(True)

    return categorical_mask


def compute_importances(task_type, model_name, model, x_train, y_train, categorical_mask):
    if model_name in ['rfr', 'gbr', 'rfc', 'gbc']:
        model.fit(x_train, y_train)
        importance = model.feature_importances_

    else:
        if task_type == 'classification':
            importance = feature_selection.mutual_info_classif(
                x_train, y_train, discrete_features=categorical_mask, random_state=42)
        else:
            importance = feature_selection.mutual_info_regression(
                x_train, y_train, discrete_features=categorical_mask, random_state=42)

    importances = list(zip(x_train.columns, importance))

    return importances


def post_process_importances(importances, new_features_mapping, multiplier=1):
    features = [value[0] for value in importances]
    importances = [value[1] * multiplier for value in importances]

    overall_features_importances = dict(zip(features, importances))

    for feature in overall_features_importances:
        for created_feature in new_features_mapping:
            if feature == created_feature:
                for parrent_feature in new_features_mapping[created_feature]:
                    for existing_feature in overall_features_importances.keys():
                        if parrent_feature in existing_feature:
                            overall_features_importances[existing_feature] += overall_features_importances[feature]
                            break

    return overall_features_importances


def aggregate_feature_importance(all_columns, features_importances):
    aggregated_features_importances = defaultdict()

    for feature in all_columns:
        aggregated_features_importances[feature] = 0

    for feature in features_importances.keys():
        for feature_ in aggregated_features_importances.keys():
            if feature_ in feature:
                aggregated_features_importances[feature_] += features_importances[feature]

    return aggregated_features_importances


def plot_feature_importance(aggregated_features_importances, dataset_name, target_column, language):
    if language == 'ro':
        importance = 'importanță procentuală'
        columns = 'nume coloană'

    else:
        importance = 'percentage importance'
        columns = 'column name'

    feature_importance_df = pd.DataFrame(
        aggregated_features_importances, index=[importance])
    feature_importance_df = feature_importance_df.T.reset_index().rename(
        {'index': columns}, axis=1).sort_values(importance, ascending=False)
    feature_importance_df[importance] = feature_importance_df[importance] / \
        feature_importance_df[importance].sum()
    feature_importance_df[importance] = feature_importance_df[importance] * 100

    plt.figure(figsize=(8, feature_importance_df.shape[0]))

    plot = sns.barplot(data=feature_importance_df, y=columns, x=importance, palette=[
        '#203568'] * 3 + ['#01adee'] * 3 + ['#656b7d'] * 1000)

    files_storing.store_feature_importance_img(
        dataset_name, target_column, plot)


def compute_feature_importance(dataset_name, target_column, language):
    extracted_data, loaded_data = extract_needed_info(
        dataset_name, target_column)
    [cat_columns, cont_columns, all_columns, features_mapping] = extracted_data
    [task_type, model_config_mapping, config_data,
        trained_models, _, stacking_type, models_performance] = loaded_data

    all_columns_combined = set()

    if stacking_type == 'without_stacking':
        model_name = list(trained_models.values())[0][0]
        model = list(trained_models.values())[0][1][1]

        x_train = config_data[model_config_mapping[model_name]][0]
        y_train = pd.DataFrame(config_data['y_train'])

        categorical_mask = separate_columns(x_train, cont_columns, cat_columns)
        importances = compute_importances(
            task_type, model_name, model, x_train, y_train, categorical_mask)

        features_importances = post_process_importances(
            importances, features_mapping)
        aggregated_features_importances = aggregate_feature_importance(
            all_columns, features_importances)

    else:
        all_models_importances = []

        models_names = list(trained_models.values())[0][0]
        models = list(trained_models.values())[0][1][1]

        for index in range(len(models_names)):
            model_name = models_names[index]
            model = models[index]

            x_train = config_data[model_config_mapping[model_name]][0]
            y_train = pd.DataFrame(config_data['y_train'])
            performance = models_performance[model_name]

            categorical_mask = separate_columns(
                x_train, cont_columns, cat_columns)
            importances = compute_importances(
                task_type, model_name, model, x_train, y_train, categorical_mask)

            features_importances = post_process_importances(
                importances, features_mapping, performance)

            importances = list(
                zip(features_importances.keys(), list(features_importances.values())))

            all_models_importances.append(importances)
            all_columns_combined = all_columns_combined.union(
                set([value[0] for value in importances]))

        all_columns_combined = list(all_columns_combined)

        features_importances = defaultdict(lambda: 0)

        for importances in all_models_importances:
            for column, importance in importances:
                features_importances[column] += importance

        aggregated_features_importances = aggregate_feature_importance(
            all_columns, features_importances)

    plot_feature_importance(
        aggregated_features_importances, dataset_name, target_column, language)
