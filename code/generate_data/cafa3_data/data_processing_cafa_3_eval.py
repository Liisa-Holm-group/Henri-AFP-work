"""initial location for utility functions."""

import os
import math
import re
from collections import Counter
import itertools
from operator import itemgetter


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

def process_taxonomy_line(line, data):
    """update data dict by the contents of a line"""
    fields = line.split('\t')
    name = fields[0]
    features = fields[3].strip().split(';')
    data[name] = [feature for feature in features]

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
    read_file(path, process_taxonomy_line, data, header=True)
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

def process_ipr_line2(line, data):
    """Extract contents of a line and store them to a dict.

    :param line: pfam data line
    :type line: string

    :param dict: data dict
    :type dict: dict
    """

    fields = line.split('\t')
    name = fields[0]
    feature = f'{fields[3]}|{fields[4]}'
    e_value = fields[8]
    start = int(fields[6])
    stop = int(fields[7])
    length = int(fields[2])
    try:
        cluster = fields[11]
    except IndexError:
        cluster = 'missing'
    if name in data:
        if feature in data[name].keys():
            data[name][feature].append((e_value, cluster, start, stop, length))
        else:
            t = list()
            t.append((e_value, cluster, start, stop, length))
            data[name][feature] = t
    else:
        t = list()
        t.append((e_value, cluster, start, stop, length))
        data[name] = {feature:t}
    # t = [len(list(d.values())[0]) for d in self.data.values()]


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
    e_value = fields[8]
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


def non_overlapping(feature, max_overlap=0.2):
    """Extract occurrences of a feature where overlap is below max_overlap."""
    features = []
    t = sorted(feature, key=itemgetter(2))
    j = 0
    features.append(t[j])
    for i in range(1, len(t)):
        if (1-max_overlap)*t[j][3] < t[i][2]:
            features.append(t[i])
            j = i
        else:
            continue
    return features

def calc_counts(feature):
    return len(non_overlapping(feature, 0.2))

def calc_localisation(feature, max_tail_len=100):
    no = non_overlapping(feature, 0.5)
    length = no[0][4]
    result = [0, 0, 0]
    if length < 300:
        max_tail_len = length // 3
        start = set(range(max_tail_len))
        middle = set(range(max_tail_len,2*max_tail_len))
        end = set(range(2*max_tail_len,length))
    else:
        start = set(range(max_tail_len))
        middle = set(range(max_tail_len,length-max_tail_len))
        end = set(range(length-max_tail_len, length))
    try:
        for n in no:
            values = set(range(n[2], n[3]))
            result[0] += len(values.intersection(start))/len(values)
            result[1] += len(values.intersection(middle))/len(values)
            result[2] += len(values.intersection(end))/len(values)
        return [r/sum(result) for r in result]
    except ZeroDivisionError: # lenght 1 sequence
        return [1, 1, 1]

def process_ipr_data(ipr_dir):
    """Read ipr data files.

    Construct a dictionary containing sequence names as keys and cluster names as values.

    :param path: location of the pfam data file
    :type path: string

    :returns dict
    """
    data = {}
    for file_path in os.scandir(ipr_dir):
        read_file(file_path, process_ipr_line2, data, header=False)

    # count non overlapping occurrences of each feature
    for t1, sequence in data.items():
        for t2, feature in sequence.items():
            count = calc_counts(feature)
            max_e = max(feature, key=itemgetter(0))[0]
            cluster = feature[0][1]
            localisation = calc_localisation(feature)
            data[t1][t2] = (cluster, count, max_e, localisation[0], localisation[1], localisation[2])
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
    # assert calc_counts([(2.06e-29, 'IPR010982', 531, 614, 1499), (2.5e-32, 'IPR010982', 925, 1013, 1499), (2.06e-34, 'IPR010982', 1101, 1198, 1499)]) == 2
    data = process_ipr_data('test_dir')
    bp()
    pass

if __name__ == '__main__':
    main()
