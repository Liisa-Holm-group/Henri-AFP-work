"""initial location for utility functions."""

import os
import math
import re
from collections import Counter
import itertools


from pdb import set_trace as bp

def read_file(path, function, *args, header=True, header_len=1):
    """Read and process generic feature files.

    :param function: function for processing a sigle line
    :type function: function

    :param args: arguments of the line processing function
    :type args: tuple
    """
    with open(path) as f:
        if not header:
            header_len=0
        for line in itertools.islice(f, header_len, None):
            try:
                # process line
                function(line, *args)
            except IndexError:
                # reached the program info lines at the end of the goa file.
                continue

def update_feature_counter(line, counter):
    fields = line.split('\t')
    name = fields[0]
    features = fields[2].strip().split(';')
    counter.update(features)

def count_features(path):
    """Count the number of occurrences of each feature."""
    counter = Counter()
    read_file(path, update_feature_counter, counter)
    return counter.most_common()

def process_taxonomy_line(line, data, selected_features):
    """update data dict by the contents of a line"""
    fields = line.split('\t')
    name = fields[0]
    features = fields[3].strip().split(';')
    data[name] = features # [feature for feature in features if feature in selected_features]

def process_taxonomy_data(path, min_count):
    """Select the features from a file that are present in at least min_count samples.

    Return a dictionary mapping the sequences to a corresponding feature list.

    :param path: location of the data file
    :type path: string

    :param min_count: the minimun number of occurrences allowed
    :type score_type: int

    :returns dict
    """
    data = {}
    counts = count_features(path)
    selected_features = {feature[0] for feature in counts if feature[1] > min_count}
    read_file(path, process_taxonomy_line, data, selected_features)
    return data


def update_target_counter(line, counter):
    fields = line.split('\t')
    name = fields[0]
    classes = re.findall(r'\t(\d{7}.*?)\t', line)[0].split(' ')
    counter.update(classes)

def count_targets(path):
    """Count the number of occurrences of each target class."""
    counter = Counter()
    read_file(path, update_target_counter, counter)
    return counter.most_common()

def process_target_line(line, data, selected_targets):
    fields = line.split('\t')
    name = fields[0]
    classes = re.findall(r'\t(\d{7}.*?)\t', line)[0].split(' ')

    selected_classes = [target for target in classes if target in selected_targets]

    if name in data:
        data[name] = data[name].union(selected_classes)
    else:
        data[name] = set(selected_classes)

def process_target_data(path, min_count):
    data = {}

    counts = count_targets(path)
    selected_targets = {target[0] for target in counts if target[1] > min_count}

    read_file(path, process_target_line, data, selected_targets)
    return data

def process_ipr_line(line, data):
    """Extract contents of a line and store them to a dict.

    :param line: pfam data line
    :type line: string

    :param dict: data dict
    :type dict: dict
    """

    fields = line.split('\t')
    name = fields[0]
    feature = f'{fields[3]}|{fields[4]}'
    e_value = fields[8] # log transform this later
    try:
        cluster = fields[11]
    except IndexError:
        cluster = 'missing'
    if name in data:
        if feature not in data[name]:
            data[name][feature] = (e_value, cluster)
        else:
            if data[name][feature][0] > e_value:
                data[name][feature] = (e_value, cluster)
    else:
        data[name] = {feature:(e_value, cluster)}


def process_ipr_data(ipr_dir):
    """Read ipr data files.

    Construct a dictionary containing sequence names as keys and cluster names as values.

    :param path: location of the pfam data file
    :type path: string

    :returns dict
    """
    data = {}
    for file_path in os.scandir(ipr_dir):
        read_file(file_path, process_ipr_line, data, header=False)
    return data

def process_wps_line(line, data):
    """Extract contents of a line and store them to a dict.

    :param line: pfam data line
    :type line: string

    :param dict: data dict
    :type dict: dict
    """
    fields = line.split(' ', 1)
    name = fields[0]
    features = fields[1].split(',')
    data[name] = {feature.strip().split(' ')[0]:feature.strip().split(' ')[1] for feature in features}

def process_wps_data(wps_dir):
    """Read wps data files.

    Construct a dictionary containing sequence names as keys and cluster names as values.

    :param path: location of the wdps data files
    :type path: string

    :returns dict
    """
    data = {}
    for file_path in os.scandir(wps_dir):
        read_file(file_path, process_wps_line, data, header=True)
    return data

def process_targetp_line(line, data):
    """Extract contents of a line and store them to a dict.

    :param line: targetp data line
    :type line: string

    :param dict: data dict
    :type dict: dict
    """
    fields = line.split('\t')
    name = fields[0]
    features = fields[2:5]
    data[name] = [float(feature) for feature in features]

def process_targetp_data(targetp_dir):
    """Read targetp data files.

    Construct a dictionary containing sequence names as keys and cluster names as values.

    :param path: location of the targetp data files
    :type path: string

    :returns dict
    """
    data = {}
    for file_path in os.scandir(targetp_dir):
        read_file(file_path, process_targetp_line, data, header=True, header_len=2)
    return data

def process_sans_line(line, data):
    fields = line.split('\t')
    name = fields[2]
    feature = fields[3]
    score = fields[6]
    if name in data:
        data[name][feature] = float(score)
    else:
        data[name] = {feature:float(score)}

def process_sans_data(path):
    data = {}
    read_file(path, process_sans_line, data, header=True, header_len=1)
    return data

def process_string_line(line, data):
    fields = line.split('\t')
    name = fields[0]
    feature = fields[5]
    score = fields[4]
    if name in data:
        data[name][feature] = float(score)
    else:
        data[name] = {feature:float(score)}

def process_string_data(path):
    data = {}
    read_file(path, process_string_line, data, header=True, header_len=1)
    return data

def main():
    pass

if __name__ == '__main__':
    main()
