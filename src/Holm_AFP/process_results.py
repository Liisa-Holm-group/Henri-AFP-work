"""Process classification results."""

from functools import reduce
import itertools
from pathlib import Path
from pdb import set_trace as bp
import pickle
import os
import pandas as pd
import sys

from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import MultiLabelBinarizer
import h5py
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.sparse as sp

mpl.use('Agg')


def save_cafa_format(predictions, sequences, go_names, filename):
    nonzero = predictions.nonzero()
    nonzero_indices = zip(list(nonzero[0]), list(nonzero[1]))
    # Get filename dir
    directory = os.path.dirname(filename)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w+') as f:
        for row, column in nonzero_indices:
            f.write(f'{sequences[row]}\t{list(go_names)[column]}\t{predictions[row, column]:.5f}\t\n')

def targets_to_cafa(dir_, ontology, eval_data=True):
    eval_str = 'eval_' if eval_data else ''
    data = sp.load_npz(f'{dir_}/{ontology}_{eval_str}targets.npz')
    sequences = joblib.load(f'{dir_}/{ontology}_{eval_str}sequences.joblib')
    names = joblib.load(f'{dir_}/{ontology}_{eval_str}target_names.joblib')
    save_cafa_format(data, sequences, names, f'{dir_}/{ontology}_{eval_str}targets.cafa')

def cafa2h5(path, sequences, target_names):

    sequences = dict(zip((sequences),range(len(sequences))))
    target_names = dict(zip((target_names),range(len(target_names))))

    res = np.zeros((len(sequences), len(target_names)))
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        try:
            fields = line.split('\t')
            fields[1] = fields[1][3:]
            res[sequences[fields[0]], target_names[fields[1]]] = fields[2]
        except KeyError:
            print(line)
    predictions = res.T
    with h5py.File(f'{path}.h5', 'w', libver='latest') as f:
        for i, arr in enumerate(predictions):
            dset = f.create_dataset(str(i), shape=(predictions[0].shape), data=arr, compression='gzip', compression_opts=9)

def h52cafa(path, ontology):
    data_path='.'
    names = joblib.load(f'{data_path}/{ontology}_ipscan_feature_names.joblib')
    n_go = sp.load_npz(f'{data_path}/datasets/{ontology}_targets.npz').shape[1]
    sequences = joblib.load(f'{data_path}/datasets/{ontology}_sequences.joblib')
    go_names = joblib.load(f'{data_path}/datasets/{ontology}_target_names.joblib')

    predictions = np.zeros((len(sequences),n_go))
    for fname in os.listdir(path):
        if ontology in fname and 'h5' in fname:
            if os.path.exists(f'{path}/{fname}'.replace('h5', 'cafa')):
                    continue

            print(f'reading {path}/{fname}')
            with h5py.File(f'{path}/{fname}', 'r') as f:
                for i in range(n_go):
                    if i % 1000 == 0:
                        print(i)
                        print('\a')
                        print('\a')
                        print('\a')
                    predictions[:,i] =  f[str(i)][:]

                new_name = fname.replace('h5','cafa')
                save_cafa_format(np.round(predictions, 5), sequences, go_names, f'{path}/{new_name}')

def combine_results(prediction_results, n_samples):
    """Combine a list of cv fold predictions and indices into a correctly ordered array."""
    results = np.zeros(n_samples)
    for prediction, index in prediction_results:
        results[index] = prediction[:,1].ravel() #NOTE prediction.ravel()
    return results

def print_scores(path, method):
    for f in os.listdir(path):
        if (method in f) and ('scores' in f) and ('joblib' in f):
            data = joblib.load(f'{path}/{f}')
            print(f'{f}: {data.mean().round(3)}')

if __name__ == '__main__':
    print_scores(sys.argv[1], '.')
