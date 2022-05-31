import numpy as np
import pandas as pd
from sklearn import decomposition, feature_extraction

from utility_functions import logging


def extract_text_features(text_column, column_name, min_freq=0.001, max_freq=0.999, input_tfidf=None, svd=None,
                          reduce_dim=False, rand_state=None):
    logging.display('', p=4)
    corpus = list(text_column.fillna('').values)
    tfidf = feature_extraction.text.TfidfVectorizer()

    if input_tfidf is None:
        index = 0
        tf_idf_applied = False

        first_list = list(zip([round(x, 3) for x in np.linspace(min_freq - min_freq / 10, 0.001, 9)], 9 * [0.99]))
        second_list = list(zip([round(x, 3) for x in np.linspace(min_freq - min_freq / 10, 0.001, 9)], 9 * [0.999]))
        generated_min_max = first_list + second_list

        min_max_values = [(min_freq, max_freq)] + generated_min_max + [(1, 1.0)]

        while not tf_idf_applied:
            logging.display('{} trying with min_df={} max_df={}'.format(column_name, min_max_values[index][0],
                                                                        min_max_values[index][1]), p=4)
            try:
                tfidf = feature_extraction.text.TfidfVectorizer(min_df=min_max_values[index][0],
                                                                max_df=min_max_values[index][1])
                tfidf.fit(corpus)
                tf_idf_applied = True
                logging.display('for {} tf-idf with min_df={} max_df={}'.format(column_name, min_max_values[index][0],
                                                                                min_max_values[index][1]), p=4)
            except ValueError:
                pass

            index += 1
    else:
        tfidf = input_tfidf

    tf_idf = tfidf.transform(corpus)

    if reduce_dim:
        reduced_data, svd, dim_reduced = reduce_dimensionality(tf_idf, svd, rand_state, column_name)

        if svd is not None:
            column_names = [column_name + '_svd_' + str(i) for i in list(range(reduced_data.shape[1]))]
        else:
            column_names = [column_name + '_long_text_' + str(feature_name) for feature_name in
                            tfidf.get_feature_names()]

        if dim_reduced:
            text_features_df = pd.DataFrame(reduced_data, columns=column_names)
        else:
            text_features_df = pd.DataFrame(reduced_data.toarray(), columns=column_names)
    else:
        column_names = [column_name + '_long_text_' + str(feature_name) for feature_name in
                        tfidf.get_feature_names()]
        text_features_df = pd.DataFrame(tf_idf.toarray(), columns=column_names)

    return text_features_df, tfidf, svd


def reduce_dimensionality(tf_idf, svd, rand_state, column_name=''):
    max_components = tf_idf.shape[1]

    if svd is None:
        if max_components > 2:
            logging.display('tf_idf for {} has initially {} columns'.format(column_name, max_components), p=4)
            min_components = 1
            components = (min_components + max_components) // 2
            greater_explained_var = False
            explained_var = 1.0

            while greater_explained_var is False:
                svd = decomposition.TruncatedSVD(n_components=components, random_state=rand_state)
                svd.fit(tf_idf)
                explained_var = svd.explained_variance_ratio_.sum()
                logging.display('Try {} components. Explained variance: {}'.format(components, explained_var), p=4)
                if explained_var >= 0.9 or min_components == components:
                    greater_explained_var = True
                else:
                    min_components = components
                    components = (min_components + max_components) // 2

            if explained_var < 0.9:
                return tf_idf, None, False

        else:
            return tf_idf, None, False

    reduced_data = svd.transform(tf_idf)
    logging.display('Explained variance for {}: {}'.format(column_name, svd.explained_variance_ratio_.sum()), p=4)

    return reduced_data, svd, True
