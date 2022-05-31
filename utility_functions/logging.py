import json

import config


def display(message, p=0, end_line='\n', first_log=None):
    if message == 'clean':
        config.log_text_1 = ''
        config.log_text_2 = ''
        return

    if config.PRIORITY >= p:
        if p == 3:
            if config.VERBOSE:
                print('\t', end='')
            if first_log is None or first_log is True:
                config.log_text_1 += '\t'
            if first_log is None or first_log is False:
                config.log_text_2 += '\t'
        elif p == 2:
            if config.VERBOSE:
                print('')
            if first_log is None or first_log is True:
                config.log_text_1 += '\n'
            if first_log is None or first_log is False:
                config.log_text_2 += '\n'
        elif p == 0:
            raise Exception(message)
        else:
            if first_log is None or first_log is True:
                config.log_text_1 += '\t\t'
            if first_log is None or first_log is False:
                config.log_text_2 += '\t\t'
            if config.VERBOSE:
                print('\t\t', end='')
        if first_log is None or first_log is True:
            config.log_text_1 += (str(message) + end_line)
        if first_log is None or first_log is False:
            config.log_text_2 += (str(message) + end_line)

        if config.VERBOSE:
            print(message, end=end_line)


def display_json(input_dict):
    text = json.dumps(input_dict, indent=4).split('\n')

    for line in text:
        display(line, p=4)


def display_dict(input_dict):
    for key in input_dict.keys():
        first_index = key.find('_')
        last_index = key.rfind('_')

        display(key[:first_index].ljust(32) + key[first_index + 1:last_index].ljust(32) + key[last_index + 1:], p=4)
