"""Read and process data.

Read features and targets and convert them to sparse matrices. Save results.
"""

import argparse
import itertools
from pdb import set_trace as bp
import sys

import joblib
from joblib import Parallel, delayed
import numpy as np

import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


if sys.argv[-1] == '-p':
    import data_processing_cafa as data_processing
else:
    import data_processing

def transform_features(sequences, features, empty_structure):
    """Extract features corresponding to sequences and transfrom them to be
    compatible with MultilabelBinarizer.

    :param sequences: list of sequences
    :type sequences: list

    :param features: dictionary mapping sequences to features
    :type args: dict

    :param empty_structure: structure type used to store the features of a particular sequence
    :type args: container

    :returns list
    """
    new_features = []
    for seq in sequences:
        try:
            new_features.append(features[seq])
        except KeyError:
            new_features.append(empty_structure())
    return new_features

def process_ipr(ipr_list):
    """Divide list of dicts containing tuples as values into two separate dicts.
    results: log transformed e values, clusters, and feature without e-values"""
    e_values = []
    clusters = []
    e_missing = []
    count_and_loc = [[],[],[],[]]
    for item in ipr_list:
        e_dict = {}
        c_dict = {}
        em_set = set()
        count_and_loc_dict = [{},{},{},{}]
        for key in item.keys():
            if item[key][2] != '-':
                float_val = float(item[key][2])
                if float_val == 0:
                    float_val = 7.5e-305 # global ipr data min
                e_dict[key] = -np.log(float_val)
            elif item[key][0] != 'missing':
                c_dict[key] = item[key][0]
            else:
                em_set.add(key)
            count_and_loc_dict[0][key] = item[key][1]
            count_and_loc_dict[1][key] = item[key][3]
            count_and_loc_dict[2][key] = item[key][4]
            count_and_loc_dict[3][key] = item[key][5]
        e_values.append(e_dict)
        clusters.append(c_dict)
        e_missing.append(em_set)
        for i in range(len(count_and_loc)):
            count_and_loc[i].append(count_and_loc_dict[i])
    return e_values, clusters, e_missing, count_and_loc

def load_ipscan(path, sequences, pretrained, output_path):
    """Load InterProScan data."""
    print('loading interproscan features')
    cluster_binarizer = MultiLabelBinarizer(sparse_output=True)
    e_value_vectorizer = DictVectorizer()
    e_missing_binarizer = MultiLabelBinarizer(sparse_output=True)
    cv = [DictVectorizer() for i in range(4)]

    features = data_processing.process_ipr_data(path)
    tr = transform_features(sequences, features, dict)
    e_values, clusters, e_missing, count_and_loc = process_ipr(tr)
    clusters = cluster_binarizer.fit_transform(clusters)
    e_values = e_value_vectorizer.fit_transform(e_values)
    e_missing = e_missing_binarizer.fit_transform(e_missing)
    cl = [cv[i].fit_transform(count_and_loc[i]) for i in range(len(count_and_loc))]

    cluster_names = ['ipscan_cluster'+name for name in cluster_binarizer.classes_]
    e_value_names = ['ipscan_e_value'+name for name in e_value_vectorizer.feature_names_]
    e_missing_names = ['ipscan_e_missing'+name for name in e_missing_binarizer.classes_]
    cl_names = ['ipscan_count'+name for name in cv[0].feature_names_] + ['ipscan_start'+name for name in cv[1].feature_names_] + ['ipscan_middle'+name for name in cv[2].feature_names_] + ['ipscan_end'+name for name in cv[3].feature_names_]

    feature_names = cluster_names + e_value_names + e_missing_names + cl_names
    features = sp.hstack([clusters, e_values, e_missing]+cl).tocsr()

    if pretrained:
        old_names = joblib.load(output_path)
        feature_index = [i for i, f in enumerate(feature_names) if f in old_names]
        features, feature_names = filter_features(features[:, np.array(feature_index)],
                                                  [feature_names[i] for i in feature_index],
                                                  old_names)
    else:
        joblib.dump(feature_names, output_path)
    return features, feature_names

def load_wps(path, sequences, pretrained, output_path):
    """Load WolfPSORT data."""
    print('loading wolfpsort features')
    vectorizer = DictVectorizer()
    features = data_processing.process_wps_data(path)
    tr = transform_features(sequences, features, dict)
    values = vectorizer.fit_transform(tr)
    feature_names = vectorizer.get_feature_names()
    if pretrained:
        old_names = joblib.load(output_path)
        feature_index = [i for i, f in enumerate(feature_names) if f in old_names]
        features, feature_names = filter_features(values[:, np.array(feature_index)],
                                                      [feature_names[i] for i in feature_index],
                                                      old_names)
    else:
        joblib.dump(feature_names, output_path)
        features = values
    return features, ['wps_' + name for name in feature_names]

def load_targetp(path, sequences):
    """load TargetP data."""
    print('loading targetp features')
    features = data_processing.process_targetp_data(path)
    values = np.array(transform_features(sequences, features, list))
    res = np.zeros((len(values), 3))
    for i, line in enumerate(values):
        if len(line) == 3:
            res[i] = np.array(line)
    return res, ['targetp_SP', 'targetp_mTP', 'targetp_CS_position']

def load_targets(path, sequences):
    """load target data."""
    print('loading targets')
    binarizer = MultiLabelBinarizer(sparse_output=True)
    targets = data_processing.process_target_data(path, min_count=20)
    bin_targets = binarizer.fit_transform(targets[seq] for seq in sequences)
    return bin_targets, binarizer.classes_

def filter_features(features, feature_names, old_names):
    final_features = sp.csr_matrix((features.shape[0], len(old_names)))
    feature_map = {}
    old_names_list = list(old_names)
    for f in feature_names:
        try:
            feature_map[f] = old_names_list.index(f)
        except ValueError:
            continue
    for i, name in enumerate(feature_names):
        if i % 1000 == 0:
            print(i)
        try:
            try:
                final_features[:, feature_map[name]] = sp.csr_matrix(np.squeeze(features[:, i].toarray())).reshape(-1, 1)
            except ValueError:
                bp()
        except KeyError:
            continue
    return sp.csr_matrix(final_features), old_names

def filter_features2(features, feature_names, old_names):
    """Transform a feature set to the same shape a second, smaller one.

    Features missing from old names are removed, features present in the old
    names but missing from the new are set to zero. Resulting feature matrix
    contains the features present in old_names.

    :param features: feature matrix to be transformed
    :type features: matrix

    :param feature_names: list of feature names of features matrix
    :type feature_names: list

    :param old_names: list of feature names of the transformed matrix
    :type old_names: list
    """
    common_names = sorted(list(set(old_names).intersection(feature_names)))
    missing_features = sp.csr_matrix((features.shape[0], len(common_names)))
    new_set = set(feature_names)
    feature_names = np.array(list(common_names) + [name for name in old_names if name not in new_set])
    index = np.argsort(feature_names)
    assert len(feature_names) == len(old_names)

    features = sp.csr_matrix(sp.hstack((features, missing_features)))
    features = features[:, index]
    feature_names = feature_names[index]

    return sp.csr_matrix(features), feature_names

def load_taxonomy(path, sequences, pretrained, output_path):
    """load taxonomy data."""
    print('loading taxonomy features')
    binarizer = MultiLabelBinarizer(sparse_output=True)
    features = data_processing.process_taxonomy_data(path, min_count=0)
    features = binarizer.fit_transform(transform_features(sequences, features, list))
    features = features.tocsr()
    if pretrained:
        old_names = joblib.load(output_path)
        feature_names = binarizer.classes_
        features, names = filter_features(features, feature_names, old_names)
    else:
        counts = features.sum(axis=0)
        counts = np.squeeze(np.asarray(counts))
        index = counts > 10
        names = binarizer.classes_[index]
        joblib.dump(names, output_path)
        features = features[:, index]
        names = binarizer.classes_[index]
    return features, ['taxonomy_' + name for name in names]

def reduce_sans_dim(features, vectorizer, output_path, pretrained, n_clusters=50000, filter_size=5):
    print('reducing dimenasionality')
    feature_names = vectorizer.get_feature_names()
    if pretrained:
        srp, old_names = joblib.load(output_path)
        old_set = set(old_names)
        feature_index = [i for i, f in enumerate(feature_names) if f in old_set]
        common_names = [feature_names[i] for i in feature_index]
        common_features = features[:, np.array(feature_index)]
        features, names = filter_features2(common_features, common_names, old_names)
    else:
        counts = np.diff(features.tocsc().indptr)
        index = counts > filter_size
        features = features[:, index]
        names = np.array(feature_names)[index]
        density = 1/features.shape[1]
        srp = SparseRandomProjection(n_components=n_clusters, random_state=42, density=density).fit(features)
        joblib.dump((srp, names), output_path)
    new_features = srp.transform(features)
    return new_features

def reduce_string_dim(features, vectorizer, output_path, pretrained):
    print('reducing string dimensionality')
    return reduce_sans_dim(features, vectorizer, output_path, pretrained, n_clusters=70000, filter_size=10)

def load_sans(path, sequences, output_path, pretrained=False):
    print('reading sans data')
    data = data_processing.process_sans_data(path)
    print('vectorizing sans data')
    data = transform_features(sequences, data, dict)
    vectorizer = DictVectorizer()
    vec_sans = vectorizer.fit_transform(data)
    vec_sans = reduce_sans_dim(vec_sans, vectorizer, output_path=output_path, pretrained=pretrained)
    return vec_sans, [str(i) for i in range(vec_sans.shape[1])]

def load_string(path, sequences, output_path, pretrained=False):
    print('reading string data')
    data = data_processing.process_string_data(path)
    print('vectorizing string data')
    data = transform_features(sequences, data, dict)
    vectorizer = DictVectorizer()
    vec_string = vectorizer.fit_transform(data)
    vec_string = reduce_string_dim(vec_string, vectorizer, output_path=output_path, pretrained=pretrained)
    return vec_string, [str(i) for i in range(vec_string.shape[1])]

def load_profet(path, sequences):
    print('reading profet data')
    profet = pd.read_csv(path, sep='\t', header=0, index_col=0)
    sequences = pd.Series(sequences)
    print('processing profet data')
    common = profet.index.intersection(sequences)
    other = pd.Series([s for s in sequences if s not in profet.index])
    common_sequences = profet.loc[common, :]
    missing_sequences = pd.DataFrame(np.zeros((len(other), profet.shape[1])), index=other, columns=profet.columns)
    res = common_sequences.append(missing_sequences)
    res.sort_index(inplace=True, ascending=False)
    data = sp.csr_matrix(res.values)
    imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    data = imp_zero.fit_transform(data)
    density = 1/data.shape[1]
    n_clusters = 500
    features = SparseRandomProjection(n_components=n_clusters, random_state=42, density=density).fit_transform(data)
    return features, ['profet_' + str(i) for i in range(features.shape[1])]


class DatasetGenerator:
    """Generate datasets."""

    def __init__(self, data_path, ontology, output_dir, pretrained=False):
        """
        :param data_path: directory containing datasets
        :type args: string

        :param ontology: ontology used to generate the data. possible values MF, BP, CC
        :type ontology: string

        :param output_dir: directory where results are saved. if pretrained==True, transformation information is read from this directory
        :type output_dir: string

        :param pretrained: generate data using transformations of a previously constructed dataset
        :type pretrained: bool
        """

        self.ontology = ontology
        self.output_dir = output_dir
        self.ipr_dir = f'{data_path}/ipr_dir/'
        self.taxonomy_path = f'{data_path}/taxonomy.txt'
        self.wps_dir = f'{data_path}/wps_dir'
        self.profet_data = f'{data_path}/profet_2.txt'
        self.targetp_dir = f'{data_path}/targetp_data'
        self.sans_path = f'{data_path}/sans.txt'
        self.string_path = f'{data_path}/string.txt'
        self.pretrained = pretrained

        if not self.pretrained:
            ontology_paths = {'MF':f'{data_path}/MF_targets.txt',
                              'BP':f'{data_path}/BP_targets.txt',
                              'CC':f'{data_path}/CC_targets.txt'}
            self.target_path = ontology_paths[ontology]
            targets = data_processing.process_target_data(self.target_path, min_count=20)
            self.sequences = sorted(targets.keys())

            with open(f'{data_path}/{self.ontology}_eval_sequences.txt', 'r') as f:
                seq = f.readlines()
                eval_sequences = set([s[:-1] for s in seq])
            self.sequences = [s for s in self.sequences if s not in eval_sequences]
        else:
            with open(f'{data_path}/{self.ontology}_eval_sequences.txt', 'r') as f:
                seq = f.readlines()
                self.sequences = sorted(list(set([s[:-1] for s in seq])))


    def generate_targets(self):
        if not self.pretrained:
            targets, names = load_targets(self.target_path, self.sequences)
            sp.save_npz(f'{self.output_dir}/{self.ontology}_targets.npz', targets)
            joblib.dump(names, f'{self.output_dir}/{self.ontology}_target_names.joblib')
            joblib.dump(self.sequences, f'{self.output_dir}/{self.ontology}_sequences.joblib')
        else:
            joblib.dump(self.sequences, f'{self.output_dir}/{self.ontology}_sequences.joblib')

    def generate_features(self, feature_names, dataset_name):
        data_loader = {'ipscan':lambda: load_ipscan(self.ipr_dir, self.sequences, self.pretrained,f'{self.output_dir}/{self.ontology}_ipscan_index.joblib' ),
                'taxonomy':lambda:load_taxonomy(self.taxonomy_path, self.sequences, self.pretrained, f'{self.output_dir}/{self.ontology}_tax_index.joblib'),
                'profet':lambda:load_profet(self.profet_data, self.sequences),
                'wps':lambda:load_wps(self.wps_dir, self.sequences, self.pretrained, f'{self.output_dir}/{self.ontology}_wps_index.joblib'),
                'targetp':lambda:load_targetp(self.targetp_dir, self.sequences),
                'sans':lambda:load_sans(self.sans_path, self.sequences, f'{self.output_dir}/{self.ontology}_sans_projector.joblib', self.pretrained),
                'string':lambda:load_string(self.string_path, self.sequences, f'{self.output_dir}/{self.ontology}_string_projector.joblib', self.pretrained)}


        parallel = Parallel(n_jobs=len(feature_names))
        datasets = parallel(delayed(lambda name: data_loader[name]())(name) for name in feature_names)
        features = [dataset[0] for dataset in datasets]
        names = list(itertools.chain.from_iterable([dataset[1] for dataset in datasets]))

        prefix = 'CAFA_' if self.pretrained else ''

        results = sp.hstack(features)
        assert results.shape[0] == len(self.sequences)
        sp.save_npz(f'{self.output_dir}/{prefix}{self.ontology}_{dataset_name}_features.npz', results)
        joblib.dump(names, f'{self.output_dir}/{prefix}{self.ontology}_{dataset_name}_feature_names.joblib')

    def ips_plus_ta(self):
        """Generate interpro + taxonomy"""

        self.generate_targets()
        self.generate_features(['ipscan', 'taxonomy', 'wps', 'targetp'], 'ips_plus_ta')

    def no_ips_or_sans(self):
        self.generate_features(['profet', 'taxonomy', 'wps', 'targetp'], 'no_ips_or_sans')

    def sans_plus_others(self):
        self.generate_features(['sans', 'taxonomy', 'wps', 'targetp'], 'sans_plus_others')

    def string(self):
        self.generate_features(['string', 'taxonomy', 'wps', 'targetp'], 'string')

    def ipscan(self):
        self.generate_targets()
        self.generate_features(['ipscan','taxonomy'], 'ipscan')

def main():
    """CLI for generating datasets."""

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',type=str)
    parser.add_argument('output_path',type=str)
    parser.add_argument('ontology', help='Options: MF, CC or BP', type=str)
    parser.add_argument('dataset_name', help='Options: ips_plus_ta, no_ips_or_sans or sans_plus_others, string, ipscan', type=str)
    parser.add_argument('-p', '--pretrained', action='store_true', help='Use previously constructed dimensionality reductors')
    args = parser.parse_args()


    dataset_generator = DatasetGenerator(args.data_path, args.ontology,
                                         args.output_path, args.pretrained)

    getattr(dataset_generator, args.dataset_name)()

if __name__ == '__main__':
    main()
