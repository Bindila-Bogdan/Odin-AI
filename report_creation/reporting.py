import pandas as pd

from utility_functions import files_loading


def get_uselss_data(index, line, txt_report, useless_columns, duplicated_rows):
    if '1.a. Remove useless columns' in line:
        aux_index = index + 1

        while '1.b. Remove duplicated rows' not in txt_report[aux_index]:
            useless_columns.append(
                txt_report[aux_index].replace('\n', '').lstrip())
            aux_index += 1

        else:
            duplicated_rows.append(txt_report[aux_index +
                                              1].replace(':', ' is').lstrip().replace('\n', ''))

    if '1.c. Remove columns with missing data (after splitting)' in line:
        if 'Every column has < 30% missing data.' in txt_report[index + 1]:
            useless_columns.append(
                'Columns do not have too many missing values to be insignificant.')

        else:
            missing_values_columns = [txt_report[index] for index in range(
                len(txt_report)) if 'of missing data' in txt_report[index]]
            missing_values_columns_ = [col.replace('\n', '').lstrip().split(' has')[
                0][7:] for col in missing_values_columns]

            if len(missing_values_columns_) == 1:
                useless_columns.append('Column {} has too many missing values to be statistically significant.'.format(
                    missing_values_columns_[0]))
            else:
                useless_columns.append('Columns {} have too many missing values to be statistically significant.'.format(
                    ', '.join(missing_values_columns_)).replace('\'', ''))


def get_cat_outliers(index, line, txt_report, cat_outlier_missing_data):
    if '5.b. Outliers detection in categorical columns' in line:
        aux_index = index + 4

        while 'Lower boundary for outliers:' not in txt_report[aux_index]:
            if 'was inserted.' not in txt_report[aux_index] or 'was dropped.' not in txt_report[aux_index]:
                outlier_missing_data = txt_report[aux_index].replace(
                    '\n', '').lstrip().split(' ')
                outlier_missing_data_ = [
                    data for data in outlier_missing_data if len(data) > 0]
                missing_no = outlier_missing_data_.pop()
                outliers_no = outlier_missing_data_.pop()
                column_name = ' '.join(outlier_missing_data_)

                cat_outlier_missing_data.append(
                    [column_name, outliers_no, missing_no])

            aux_index += 1


def get_cont_outliers(index, line, txt_report, cont_outlier_missing_data):
    if '5.a. Outliers detection in continuous columns' in line:
        aux_index = index + 3

        while 'Lower and upper boundaries for outliers:' not in txt_report[aux_index]:
            if 'was inserted.' not in txt_report[aux_index] or 'was dropped.' not in txt_report[aux_index]:
                outlier_missing_data = txt_report[aux_index].replace(
                    '\n', '').lstrip().split(' ')
                outlier_missing_data_ = [
                    data for data in outlier_missing_data if len(data) > 0]
                missing_no = outlier_missing_data_.pop()
                outliers_no = outlier_missing_data_.pop()
                column_name = ' '.join(outlier_missing_data_)

                cont_outlier_missing_data.append(
                    [column_name, outliers_no, missing_no])

            aux_index += 1


def add_outliers_boundaries(json_report, cont_outlier_missing_data, cat_outlier_missing_data):
    cont_outliers_boundaries = json_report['5a']['outliers_boundaries']
    cat_outliers_boundaries = json_report['5b']['outliers_boundaries']

    for index, [col_name, outliers_no, missing_no] in enumerate(cont_outlier_missing_data):
        if outliers_no != '0':
            lower_boundary = cont_outliers_boundaries[col_name][0]
            higher_boundary = cont_outliers_boundaries[col_name][1]

            if lower_boundary == -9223372036854775807:
                lower_boundary = '-'

            if higher_boundary == 9223372036854775807:
                higher_boundary = ' -'

            cont_outlier_missing_data[index].extend(
                [str(lower_boundary), str(higher_boundary)])

        else:
            cont_outlier_missing_data[index].extend(['-', '-'])

    for index, [col_name, outliers_no, missing_no] in enumerate(cat_outlier_missing_data):
        if outliers_no != '0':
            cat_outlier_missing_data[index].extend([cat_outliers_boundaries[col_name], '-'])

        else:
            cat_outlier_missing_data[index].extend(['-', '-'])


def format_info(sizes, useless_columns, duplicated_rows, cont_outlier_missing_data, cat_outlier_missing_data):
    sizes = sizes[0] + ' rows x ' + sizes[1] + ' columns'
    useless_columns = '\n'.join(useless_columns)
    duplicated_rows = '\n'.join(duplicated_rows)

    renaming = {0: 'colum', 1: 'outliers number', 2: 'missing values',
                     3: 'lower outlier boundary', 4: 'higher outlier boundary'}


    cont_outlier_missing_data = pd.DataFrame(
        cont_outlier_missing_data).rename(renaming, axis=1)
    cat_outlier_missing_data = pd.DataFrame(
        cat_outlier_missing_data).rename(renaming, axis=1)

    outlier_missing_data = pd.concat([cont_outlier_missing_data, cat_outlier_missing_data])
    outlier_missing_data.index = list(range(outlier_missing_data.shape[0]))

    return [sizes, useless_columns, duplicated_rows, outlier_missing_data]


def create_general_info_report(dataset_name, target_column):
    txt_report, json_report = files_loading.load_intermediary_reports(
        dataset_name, target_column)

    duplicated_rows = []
    useless_columns = []

    cont_outlier_missing_data = []
    cat_outlier_missing_data = []

    for index, line in enumerate(txt_report):
        if 'Initial shape of the dataset:' in line:
            sizes = line.split(': ')[-1][1:-2].split(', ')

        get_uselss_data(index, line, txt_report,
                        useless_columns, duplicated_rows)
        get_cont_outliers(index, line, txt_report, cont_outlier_missing_data)
        get_cat_outliers(index, line, txt_report, cat_outlier_missing_data)


    add_outliers_boundaries(json_report, cont_outlier_missing_data, cat_outlier_missing_data)

    return format_info(sizes, useless_columns, duplicated_rows, cont_outlier_missing_data, cat_outlier_missing_data)
