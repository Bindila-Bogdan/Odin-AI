import pandas as pd
from sklearn import ensemble, feature_selection

from utility_functions import logging


def remove_features(continuous_df, categorical_df, discarded_columns):
    continuous_columns = continuous_df.columns
    categorical_columns = categorical_df.columns

    for column_name in discarded_columns:
        if column_name in continuous_columns:
            continuous_df.drop(column_name, axis=1, inplace=True)
        elif column_name in categorical_columns:
            categorical_df.drop(column_name, axis=1, inplace=True)

    return [continuous_df, categorical_df]


def tree_features_imp(train_continuous_df, train_categorical_df, y_df, min_importance, classification_task,
                      tree_based_model, random_state):
    model = None

    if tree_based_model == 0:
        if classification_task:
            model = ensemble.RandomForestClassifier(n_estimators=128, max_features=None, max_samples=None,
                                                    min_samples_leaf=1, class_weight='balanced', n_jobs=-1,
                                                    random_state=random_state)
        else:
            model = ensemble.RandomForestRegressor(n_estimators=128, max_features=None, max_samples=None,
                                                   min_samples_leaf=1, n_jobs=-1, random_state=random_state)

    elif tree_based_model == 1:
        if classification_task:
            model = ensemble.GradientBoostingClassifier(n_estimators=128, min_samples_leaf=1, random_state=random_state)
        else:
            model = ensemble.GradientBoostingRegressor(n_estimators=128, min_samples_leaf=1, random_state=random_state)

    x_train_df = pd.concat([train_continuous_df, train_categorical_df], axis=1)
    model.fit(x_train_df, y_df)

    feature_importance = model.feature_importances_
    feature_importance = dict(zip(x_train_df.columns, feature_importance))

    discarded_columns = []

    for column_name, importance in feature_importance.items():
        if importance <= min_importance:
            discarded_columns.append(column_name)

    return discarded_columns, feature_importance


def mutual_info_imp(train_continuous_df, train_categorical_df, y_df, min_importance, classification_task, rand_state):
    cont_features_no = train_continuous_df.shape[1]
    cat_features_no = train_categorical_df.shape[1]

    concat_df = pd.concat([train_continuous_df, train_categorical_df], axis=1)

    if cat_features_no == 0:
        discrete_features = False
    elif cont_features_no == 0:
        discrete_features = True
    else:
        discrete_features = list(range(cont_features_no, cont_features_no + cat_features_no))

    if classification_task:
        importance = feature_selection.mutual_info_classif(concat_df, y_df, discrete_features=discrete_features,
                                                           random_state=rand_state)
    else:
        importance = feature_selection.mutual_info_regression(concat_df, y_df, discrete_features=discrete_features,
                                                              random_state=rand_state)

    columns_importance = list(zip(concat_df.columns, importance))

    max_mi = importance.max()
    importance = importance / max_mi

    discarded_columns = []

    for index, imp in enumerate(importance):
        if imp <= min_importance:
            discarded_columns.append(concat_df.columns[index])

    return discarded_columns, columns_importance


def select_features(train_continuous_df, train_categorical_df, y_df, classification_task, min_importance,
                    tree_based_model, rand_state, first_log):
    logging.display('7.f. Feature selection', p=3, first_log=first_log)
    continuous_columns = train_continuous_df.columns
    categorical_columns = train_categorical_df.columns

    if tree_based_model == -1:
        discarded_columns, columns_importance = mutual_info_imp(train_continuous_df, train_categorical_df, y_df,
                                                                min_importance, classification_task, rand_state)
    else:
        discarded_columns, columns_importance = tree_features_imp(train_continuous_df, train_categorical_df, y_df,
                                                                  min_importance, classification_task, tree_based_model,
                                                                  rand_state)

    logging.display('Discarded features are: {}'.format(discarded_columns), p=4, first_log=first_log)

    if len(discarded_columns) == (len(continuous_columns) + len(categorical_columns)):
        selected_data = [train_continuous_df, train_categorical_df, [], columns_importance]
    else:
        selected_data = remove_features(train_continuous_df, train_categorical_df, discarded_columns)
        selected_data += [discarded_columns]
        selected_data += [columns_importance]

    return selected_data
