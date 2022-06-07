import json

info = [['insurance__prob_2', 'charges'],
        ['click_through_rate_3', 'Clicked']]

main_path = '/odinstorage/automl_data/training_results/config_files/'
txt_log_path = main_path + '{}/{}/logs/general_info_report_text.txt'
json_log_path = main_path + '{}/{}/logs/general_info_report_json.txt'

with open(txt_log_path.format(info[0][0], info[0][1]), 'r') as file:
    txt_report = file.readlines()

useless_columns = []
cont_outlier_missing_data = []
cat_outlier_missing_data = []

for index, line in enumerate(txt_report):
    # print(line)
    if 'Initial shape of the dataset:' in line:
        sizes = line.split(': ')[-1][1:-2].split(', ')

    if '1.a. Remove useless columns' in line:
        aux_index = index + 1

        while '1.b. Remove duplicated rows' not in txt_report[aux_index]:
            useless_columns.append(
                txt_report[aux_index].replace('\n', '').lstrip())
            aux_index += 1

        else:
            duplicated_rows = txt_report[index +
                                         1].replace(':', ' is ').lstrip()

    if '1.c. Remove columns with missing data (after splitting)' in line:
        if 'Every column has < 30% missing data.' in txt_report[index + 1]:
            useless_columns.append(
                'Columns do not too many missing values to be insignificant.')

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

    
    if '5.a. Outliers detection in continuous columns' in line:
        aux_index = index + 3

        while 'Lower and upper boundaries for outliers:' not in txt_report[aux_index]:
            if 'was inserted.' not in txt_report[aux_index] or 'was dropped.' not in txt_report[aux_index]:
                outlier_missing_data = txt_report[aux_index].replace('\n', '').lstrip().split(' ')
                outlier_missing_data_ = [data for data in outlier_missing_data if len(data) > 0]
                missing_no = outlier_missing_data_.pop()
                outliers_no = outlier_missing_data_.pop()
                column_name = ' '.join(outlier_missing_data_)

                cont_outlier_missing_data.append([column_name, outliers_no, missing_no])

            aux_index += 1

    if '5.b. Outliers detection in categorical columns' in line:
        aux_index = index + 4

        while 'Lower boundary for outliers:' not in txt_report[aux_index]:
            if 'was inserted.' not in txt_report[aux_index] or 'was dropped.' not in txt_report[aux_index]:
                outlier_missing_data = txt_report[aux_index].replace('\n', '').lstrip().split(' ')
                outlier_missing_data_ = [data for data in outlier_missing_data if len(data) > 0]
                missing_no = outlier_missing_data_.pop()
                outliers_no = outlier_missing_data_.pop()
                column_name = ' '.join(outlier_missing_data_)

                cat_outlier_missing_data.append([column_name, outliers_no, missing_no]) 

            aux_index += 1 

# print(sizes, useless_columns, duplicated_rows, cont_outlier_missing_data, cat_outlier_missing_data)

with open(json_log_path.format(info[0][0], info[0][1]), 'r') as file:
    json_report = json.load(file)

cont_outliers_boundaries = json_report['5a']['outliers_boundaries']
cat_outliers_boundaries = json_report['5b']['outliers_boundaries']

print(cont_outlier_missing_data)
print(cont_outliers_boundaries)

for index, [col_name, outliers_no, missing_no] in enumerate(cont_outlier_missing_data):
    print(index)
    if outliers_no != 0:
        cont_outlier_missing_data[index].extend(cont_outliers_boundaries[col_name])

print(cont_outlier_missing_data)


for index, [col_name, outliers_no, missing_no] in enumerate(cat_outlier_missing_data):
    if outliers_no != '0':
        cat_outlier_missing_data[index].append(cat_outliers_boundaries[col_name])

print(cat_outlier_missing_data)
