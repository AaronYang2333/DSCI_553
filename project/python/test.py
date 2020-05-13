__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '5/3/2020 9:33 AM'

import json


def combineTwoDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = (value + dict1[key]) / 2
    return dict3


def export2File(data_type, data, file_path):
    """
    export json content to a file
    :param json_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    if data_type == "array":
        with open(file_path, 'w+') as output_file:
            for item in data:
                output_file.writelines(json.dumps(item) + "\n")
    else:
        with open(file_path, 'w+') as output_file:
            pairs = json.dumps(data)
            output_file.write(pairs)


if __name__ == '__main__':
    # # Create first dictionary
    # dict1 = {'Ritika': 5, 'Sam': 7, 'John': 10, 'A': 22}
    # # Create second dictionary
    # dict2 = {'Ritika': 8, 'Sam': 20, 'Mark': 11}
    # # Merge dictionaries and add values of common keys in a list
    # dict3 = dict1.update(dict2)
    # print(dict1)
    ss = 3.2543209876
    print(round(ss))