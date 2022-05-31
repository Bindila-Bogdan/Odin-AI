import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_selection

import feature_engineering
from utility_functions import logging


def add_missing_columns(train_continuous_df, train_categorical_df, continuous_outliers_info_5a):
    for column_is_missing in continuous_outliers_info_5a['removed_columns']:
        column_name = column_is_missing + '_is_missing'

        train_categorical_df.insert(0, column_name, list(train_continuous_df[column_name].values))
        train_continuous_df.drop(column_name, axis=1, inplace=True)

    return [train_categorical_df, train_continuous_df]


def generate_date_features(date_column, column_name, max_month=None, max_day=None):
    year = date_column.dt.year.values
    month = date_column.dt.month.values
    day = date_column.dt.day.values

    if max_month is None or max_day is None:
        max_month = max(month - 1)
        max_day = max(day - 1)

    month_sin = np.sin(2 * np.pi * (month - 1) / max_month)
    month_cos = np.cos(2 * np.pi * (month - 1) / max_month)

    day_sin = np.sin(2 * np.pi * (day - 1) / max_day)
    day_cos = np.cos(2 * np.pi * (day - 1) / max_day)

    date_dict = {column_name + '_year': year, column_name + '_month_sin': month_sin,
                 column_name + '_month_cos': month_cos, column_name + '_day_sin': day_sin,
                 column_name + '_day_cos': day_cos}

    date_features_df = pd.DataFrame(date_dict)

    return date_features_df, max_month, max_day


def process_dates(continuous_df, categorical_df):
    logging.display('7. Feature engineering', p=2)
    logging.display('7.a. Generate cyclical date features', p=3)
    date_transform_info = {}

    for column in categorical_df.columns:
        if categorical_df[column].dtype == 'datetime64[ns]':
            date_features_df, max_month, max_day = generate_date_features(categorical_df[column], column)
            date_transform_info[column] = [int(max_month), int(max_day)]
            categorical_df.drop([column], axis=1, inplace=True)
            continuous_df = pd.concat([continuous_df, date_features_df], axis=1)
            logging.display('Column name: {} with max_month = {} and max_day = {}'.format(column, max_month, max_day),
                            p=4)

    return [continuous_df, categorical_df, date_transform_info]


def create_text_features(column, column_name, added_columns=None, display_message=False):
    if display_message:
        logging.display('6.a. Creating additional features for text columns', p=4)

    text_features_dict = {}
    column_copy = column.map(lambda x: str(x))

    try:
        split_words_lengths = list(column_copy.map(lambda text: len(text.split(' '))).values)
        texts_lengths = list(column_copy.map(lambda text: len(text)).values)

        if (len(set(texts_lengths)) != 1 and (
                column_copy.nunique() / column_copy.shape[0] > 0.1)) or added_columns is not None:
            text_features_dict[column_name + '_chars_no'] = texts_lengths

        if (len(set(split_words_lengths)) != 1 and (
                column_copy.nunique() / column_copy.shape[0] > 0.1)) or added_columns is not None:
            text_features_dict[column_name + '_avg_word_len'] = list(
                np.array(texts_lengths) / np.array(split_words_lengths))

    except AttributeError:
        if added_columns is None:
            return None
        else:
            columns_to_check = [column_name + '_chars_no', column_name + '_avg_word_len']
            for column in columns_to_check:
                if column in added_columns:
                    text_features_dict[column] = len(column[column_name].values) * [1]

            if len(list(text_features_dict.keys())) == 0:
                return None
            else:
                return pd.DataFrame(text_features_dict)

    if added_columns is not None:
        columns_to_remove = []

        for column_name in list(text_features_dict.keys()):
            if column_name not in added_columns:
                columns_to_remove.append(column_name)

        for column_name in columns_to_remove:
            del text_features_dict[column_name]

    text_features_df = pd.DataFrame(text_features_dict)

    return text_features_df


def transform_skewed_features(train_continuous_df, y_train_df):
    skew_dict = {}
    values_dict = {}
    target_values = list(y_train_df[y_train_df.columns[-1]].values)

    for column_name in train_continuous_df.columns:
        values = np.array(list(train_continuous_df[column_name].values))
        skew_value = float(stats.skew(values))
        skew_dict[column_name] = [skew_value, 0, 0]

        min_value = min(list(values))

        if min_value < 1:
            values += (1 - min_value)
            if isinstance(1 - min_value, np.integer):
                skew_dict[column_name][1] = int(1 - min_value)
            elif isinstance(1 - min_value, np.floating):
                skew_dict[column_name][1] = float(1 - min_value)
            else:
                skew_dict[column_name][1] = 1 - min_value

        if skew_value < 0:
            max_value = max(list(values))
            values = (max_value + 1) - values
            if isinstance(max_value + 1, np.integer):
                skew_dict[column_name][2] = int(max_value + 1)
            elif isinstance(max_value + 1, np.floating):
                skew_dict[column_name][2] = float(max_value + 1)
            else:
                skew_dict[column_name][2] = max_value + 1

        abs_skew_value = abs(skew_value)

        if 0.0 <= abs_skew_value < 0.5:
            values = np.sqrt(values)
        elif 0.5 <= abs_skew_value < 1:
            values = np.log(values)
        else:
            values = np.log10(values)

        initial_corr = np.corrcoef(list(train_continuous_df[column_name].values), target_values)[0][1]
        transformed_corr = np.corrcoef(list(values), target_values)[0][1]

        if np.abs(initial_corr) < np.abs(transformed_corr):
            values_dict[column_name + '_wo_skew'] = list(values)
            train_continuous_df.drop(column_name, axis=1, inplace=True)

    values_df = pd.DataFrame(values_dict)
    train_continuous_df.index = list(range(train_continuous_df.shape[0]))
    train_continuous_df = pd.concat([train_continuous_df, values_df], axis=1)
    remained_columns = list(train_continuous_df.columns)

    return [train_continuous_df, skew_dict, remained_columns]


def select_parent_columns(columns_importance, train_continuous_df, train_categorical_df, y_train_df, cont_feat_no,
                          cat_feat_no, classification_task, related_to_all, rand_state):
    if columns_importance is None:
        _, columns_importance = feature_engineering.feature_selection.mutual_info_imp(train_continuous_df,
                                                                                      train_categorical_df, y_train_df,
                                                                                      0.0, classification_task,
                                                                                      rand_state)

    if isinstance(columns_importance, dict):
        columns_importance = list(zip(columns_importance.keys(), columns_importance.values()))

    columns_importance.sort(key=lambda x: x[1], reverse=True)
    min_importance = columns_importance[-1][1]

    if related_to_all:
        parent_feat_no = min(cont_feat_no, np.ceil(np.sqrt(cont_feat_no + cat_feat_no)) + 1)
    else:
        parent_feat_no = min(cont_feat_no, np.ceil(np.sqrt(cont_feat_no)) + 1)

    count = 0
    used_columns = []
    cont_columns = train_continuous_df.columns

    for column_name, importance in columns_importance:
        if column_name in cont_columns:
            used_columns.append(column_name)
            count += 1

        if count >= parent_feat_no:
            break

    return [columns_importance, min_importance, used_columns]


def construct_select_features(used_columns, train_continuous_df):
    count = 0
    new_features_dict = {}
    new_features_df = pd.DataFrame({})

    for feature in used_columns:
        train_continuous_df[feature] = train_continuous_df[feature].map(
            lambda x: x + sys.float_info.epsilon if x == 0 else x)

    for i, first_feature in enumerate(used_columns):
        for j, second_feature in enumerate(used_columns):
            if i < j:
                new_feat_add = train_continuous_df[first_feature] + train_continuous_df[second_feature]
                new_feat_sub_one = train_continuous_df[first_feature] - train_continuous_df[second_feature]
                new_feat_sub_two = train_continuous_df[second_feature] - train_continuous_df[first_feature]

                new_feat_mul = train_continuous_df[first_feature] * train_continuous_df[second_feature]
                new_feat_div_one = train_continuous_df[first_feature] / train_continuous_df[second_feature]
                new_feat_div_two = train_continuous_df[second_feature] / train_continuous_df[first_feature]

                new_features = [new_feat_add, new_feat_sub_one, new_feat_sub_two, new_feat_mul, new_feat_div_one,
                                new_feat_div_two]
                operators = ['+', '-', '-rev', '*', '/', '/rev']
                new_features_operators = list(zip(new_features, operators))

                for new_feature, operator in new_features_operators:
                    corr_first_parent = abs(
                        np.corrcoef(np.array(new_feature), train_continuous_df[first_feature].values)[0][1])
                    corr_second_parent = abs(
                        np.corrcoef(np.array(new_feature), train_continuous_df[second_feature].values)[0][1])

                    if max([corr_first_parent, corr_second_parent]) <= 0.2:
                        corr_with_original_features = abs(train_continuous_df.corrwith(new_feature).values).max()

                        if new_features_df.shape[1] != 0:
                            corr_with_new_features = abs(new_features_df.corrwith(new_feature).values).max()
                        else:
                            corr_with_new_features = 0.0

                        corr = max(corr_with_original_features, corr_with_new_features)

                        if corr <= 0.2:
                            new_features_dict['new_feat_' + str(count)] = (
                                first_feature, second_feature, operator, corr)
                            new_features_df = pd.concat(
                                [new_features_df, pd.DataFrame({'new_feat_' + str(count): new_feature})], axis=1)
                            count += 1

    return [new_features_dict, new_features_df]


def select_new_features(new_features_df, y_train_df, new_features_dict, min_importance, select_max, classification_task,
                        rand_state, first_log):
    if classification_task:
        new_features_importance = feature_selection.mutual_info_classif(new_features_df, y_train_df,
                                                                        discrete_features=False,
                                                                        random_state=rand_state)
    else:
        new_features_importance = feature_selection.mutual_info_regression(new_features_df, y_train_df,
                                                                           discrete_features=False,
                                                                           random_state=rand_state)

    new_columns_importance = list(zip(new_features_df.columns, new_features_importance))
    new_columns_importance.sort(key=lambda x: x[1], reverse=True)

    count = 0
    selected_new_features = []
    selected_new_features_df = pd.DataFrame()

    for (column_name, importance) in new_columns_importance:
        if importance >= min_importance:
            logging.display('({} {} {}) => {}'.format(new_features_dict[column_name][0],
                                                      new_features_dict[column_name][1],
                                                      new_features_dict[column_name][2],
                                                      'new_feat_' + str(count)), p=4, first_log=first_log)
            selected_new_features += [new_features_dict[column_name]]
            selected_new_features_df = pd.concat([selected_new_features_df, new_features_df[column_name]], axis=1)
            count += 1

            if count >= select_max:
                break

    selected_new_features_df.columns = ['new_feat_' + str(i) for i in range(selected_new_features_df.shape[1])]

    return [selected_new_features, selected_new_features_df]


def construct_features(train_continuous_df, train_categorical_df, y_train_df, columns_importance, classification_task,
                       rand_state, related_to_all, select_max, first_log):
    logging.display('7.g. Feature construction', p=3, first_log=first_log)

    cont_feat_no = train_continuous_df.shape[1]
    cat_feat_no = train_categorical_df.shape[1]

    if cont_feat_no < 2 or select_max == 0:
        return [None, None]

    result = select_parent_columns(columns_importance, train_continuous_df, train_categorical_df, y_train_df,
                                   cont_feat_no, cat_feat_no, classification_task, related_to_all, rand_state)
    [_, min_importance, used_columns] = result

    result = construct_select_features(used_columns, train_continuous_df)
    [new_features_dict, new_features_df] = result

    if new_features_df.shape[1] == 0:
        return [None, None]

    result = select_new_features(new_features_df, y_train_df, new_features_dict, min_importance, select_max,
                                 classification_task, rand_state, first_log)

    return result


def add_missing_columns_test(test_continuous_df, test_categorical_df):
    for column in test_continuous_df.columns:
        if '_is_missing' in column:
            test_categorical_df.insert(0, column, list(test_continuous_df[column].values))
            test_continuous_df.drop(column, axis=1, inplace=True)

    return [test_continuous_df, test_categorical_df]


def process_dates_test(test_categorical_df, test_continuous_df, config_dict):
    date_dict = config_dict['7a']

    for column, [max_month, max_day] in date_dict.items():
        date_features_df, _, _ = generate_date_features(test_categorical_df[column], column, max_month, max_day)

        test_categorical_df.drop([column], axis=1, inplace=True)
        test_continuous_df = pd.concat([test_continuous_df, date_features_df], axis=1)

    return [test_categorical_df, test_continuous_df]


def transform_skewed_features_test(test_continuous_df, config_dict):
    skew_dict = config_dict['7d'][0]
    remained_columns = config_dict['7d'][1]
    values_dict = {}

    if skew_dict is None:
        return test_continuous_df

    for column_name in list(skew_dict.keys()):
        if column_name + '_wo_skew' in remained_columns:
            values = np.array(list(test_continuous_df[column_name].values))
            skew_value = skew_dict[column_name][0]
            to_add = skew_dict[column_name][1]

            if skew_value > 0:
                if np.issubdtype(values[0], np.integer) and isinstance(to_add, float):
                    to_add = int(to_add)
                elif np.issubdtype(values[0], np.floating) and isinstance(to_add, int):
                    to_add = float(to_add)

                values += to_add
                values[values < 1] = 1

            elif skew_value < 0:
                if np.issubdtype(values[0], np.integer) and isinstance(to_add, float):
                    to_add = int(to_add)
                elif np.issubdtype(values[0], np.floating) and isinstance(to_add, int):
                    to_add = float(to_add)

                values += to_add
                max_value = skew_dict[column_name][2] - 1
                values[values > max_value] = max_value
                values = (max_value + 1) - values

            if skew_value != 0:
                abs_skew_value = abs(skew_value)

                if 0.0 <= abs_skew_value < 0.5:
                    values = np.sqrt(values)
                elif 0.5 <= abs_skew_value < 1:
                    values = np.log(values)
                else:
                    values = np.log10(values)

            values_dict[column_name + '_wo_skew'] = list(values)

        if column_name not in remained_columns:
            test_continuous_df.drop(column_name, axis=1, inplace=True)

    values_df = pd.DataFrame(values_dict)
    test_continuous_df = pd.concat([test_continuous_df, values_df], axis=1)

    return test_continuous_df


def construct_features_test(test_continuous_df, config_dict, models_dict):
    selected_new_features = config_dict['7g']
    scaler_new_features = models_dict['9a_nf']

    if selected_new_features is None:
        return test_continuous_df

    count = 0
    new_feat = None
    new_features_dict = {}

    for (first_feature, second_feature, operator, _) in selected_new_features:
        if operator == '+':
            new_feat = test_continuous_df[first_feature] + test_continuous_df[second_feature]
        elif operator == '-':
            new_feat = test_continuous_df[first_feature] - test_continuous_df[second_feature]
        elif operator == '-rev':
            new_feat = test_continuous_df[second_feature] - test_continuous_df[first_feature]
        elif operator == '*':
            new_feat = test_continuous_df[first_feature] * test_continuous_df[second_feature]
        elif operator == '/':
            test_continuous_df[second_feature] = test_continuous_df[second_feature].map(
                lambda x: x + sys.float_info.epsilon if x == 0 else x)
            new_feat = test_continuous_df[first_feature] / test_continuous_df[second_feature]
        elif operator == '/rev':
            test_continuous_df[first_feature] = test_continuous_df[first_feature].map(
                lambda x: x + sys.float_info.epsilon if x == 0 else x)
            new_feat = test_continuous_df[second_feature] / test_continuous_df[first_feature]

        new_features_dict['new_feat_' + str(count)] = new_feat
        count += 1

    new_features_df = pd.DataFrame(new_features_dict)
    new_features_names = list(new_features_df.columns)

    scaled_new_features = scaler_new_features.transform(new_features_df)
    new_scaled_features_df = pd.DataFrame(scaled_new_features, columns=new_features_names)

    test_continuous_df = pd.concat([test_continuous_df, new_scaled_features_df], axis=1)

    return test_continuous_df
