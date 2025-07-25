"""Run CV experiments. Save scores, models, and predictions."""

import argparse
from datetime import timedelta
from pdb import set_trace as bp
import time
import random

import h5py
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

import models
import process_results as pr

np.random.seed(1)

def select_features(names, features, feature_type, part, base_method='e_value'):
    """Return cluster features and feature names

    :param names: list containing feature names
    :type names: list

    :param features: features
    :type features: sp.csr_matrix

    :param feature_type: feature names substring specifying which feature subset to select
    :type feature_type: string

    :param part: 1: ipscan preprocessing experiment. 2: additional features experiments
    :type part: int

    """
    print('selecting features')
    print(f'all features shape: {features.shape}')
    if part == 1:
        if feature_type == 'e_value':
            index = np.array([i for i, n in enumerate(names) if 'e_value' in n])
        elif feature_type == 'binary':
            index = np.array([i for i, n in enumerate(names) if ('e_value' in n) or ('e_missing' in n)])
        elif feature_type == 'e_value_plus_cluster':
            index = np.array([i for i, n in enumerate(names) if ('cluster' in n) or ('e_value' in n)])
        elif feature_type == 'binary_plus_cluster':
            index = np.array([i for i, n in enumerate(names) if ('cluster' in n) or ('e_missing' in n) or ('e_value' in n)])
        res = features[:, index]
        if 'binary' in feature_type:
            res[res > 0] = 1
    if part == 2 or part == 3:
        if base_method == 'binary':
            test = lambda n: ('e_value' in n) or ('e_missing' in n)
        else:
            test = lambda n: ('e_value' in n)

        if feature_type == 'location':
            index = np.array([i for i, n in enumerate(names) if test(n) or any(position in n for position in ['start','middle','end'])])
        elif feature_type == 'count':
            index = np.array([i for i, n in enumerate(names) if test(n) or 'count' in n])
        elif feature_type == 'taxonomy':
            index = np.array([i for i, n in enumerate(names) if test(n) or 'taxonomy' in n])
        elif feature_type == 'all':
            index = np.array([i for i, n in enumerate(names) if test(n) or ('taxonomy' in n) or ('wps' in n) or ('targetp' in n)])

        res = features[:, index]

        if 'binary' in base_method:
            res[res > 0] = 1

    return res

class Experiment:

    def __init__(self, name, model, feature_path, target_path, output_path, feature_names_path, part):
        """ :param name: function for processing a sigle line
        :type name: string

        :param model: model function for a single go class
        :type model: function

        :param feature_path: feature file path
        :type feature_path: string

        :param target_path: prediction target file path
        :type target_path: string

        :param output_path: output path
        :type output_path: string

        :param base_method: e-value preprocessing method to be used in part 2.
        :type base_method: string

        """

        self.name = name
        self.model = model
        self.feature_path = feature_path
        self.feature_names_path = feature_names_path
        self.target_path = target_path
        self.output_path = output_path
        self.feature_importances = None
        self.class_sizes = None
        self.n_samples = None
        self.part = part

    def run(self, n_jobs, debug=False):

        if self.part == 1:
            feature_options = ['e_value', 'binary', 'e_value_plus_cluster', 'binary_plus_cluster']
        elif self.part == 2:
            feature_options = ['taxonomy', 'location', 'count']
        elif self.part == 3:
            feature_options = ['all']

        for feature_type in feature_options :
            print('loading matrices')
            names = joblib.load(self.feature_names_path)


            if self.part == 1:
                X = select_features(names, sp.load_npz(self.feature_path).tocsr(), feature_type, self.part)
            else:
                base_method = 'binary' if (('svm_test' in str(self.model)) or ('fm_test' in str(self.model)) or ('ann' in str(self.model))) else 'e_value'
                X = select_features(names, sp.load_npz(self.feature_path).tocsr(), feature_type, self.part, base_method)

            print(f'feature shape: {X.shape}')
            y = sp.load_npz(self.target_path)

            if debug:
                y = y[:, :20]
                n_jobs = 20

            start = time.time()
            self.class_sizes = np.array(y.sum(axis=0)).ravel()
            self.n_samples = y.shape[0]

            parallel = Parallel(n_jobs=n_jobs)
            results = parallel(delayed(self.model)(X, y, go_class) for go_class in range(y.shape[1]))

            elapsed = time.time()-start
            print(f'Time: {str(timedelta(seconds=elapsed))}')

            predictions = [result['predictions'].round(5) for result in results]
            cv_results = [result['cv_results'] for result in results]

            self.feature_importances = [[fold['feature_importances'] for fold in result]
                                        for result in cv_results]
            cv_models = [[fold['model'] for fold in result] for result in cv_results]

            print('evaluating results')
            scores = self.evaluate(predictions, y, n_jobs, feature_type)

            print('saving results')
            with h5py.File(f'{self.output_path}/{self.name}_{feature_type}_predictions.h5', 'w', libver='latest') as f:
                for i, arr in enumerate(predictions):
                    dset = f.create_dataset(str(i), shape=(predictions[0].shape), data=arr, compression='gzip', compression_opts=9)

            joblib.dump(self.feature_importances, f'{self.output_path}/{self.name}_{feature_type}_feature_importances.joblib', pickle.HIGHEST_PROTOCOL)

    def evaluate(self, predictions, targets, n_jobs, feature_type):
        """Evaluate predictions scores vs. target labels.
        predictions: list of prediction arrays, targets: target matrix"""

        scores = np.array([average_precision_score(targets[:, i].toarray(), prediction) for i, prediction in enumerate(predictions)])
        print(f'mean average precision score for all go-classes: {scores.mean()}')

        joblib.dump(scores, f'{self.output_path}/{self.name}_{feature_type}_scores.joblib')
        return scores


def main():
    """CLI for ipscan experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name',type=str)
    parser.add_argument('part', type=int,
                        help='part 1: select basic ipscan processing method. part 2: evaluate additional processing options')
    parser.add_argument('model', help='model funtion from models.py')
    parser.add_argument('feature_path',type=str)
    parser.add_argument('target_path',type=str)
    parser.add_argument('feature_names_path',type=str)
    parser.add_argument('output_path',type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug mode: run the experiment for only 20 classes')
    args = parser.parse_args()

    experiment = Experiment(args.experiment_name, getattr(models, args.model),
                            args.feature_path, args.target_path, args.output_path,
                            args.feature_names_path, args.part)

    experiment.run(n_jobs=args.n_jobs, debug=args.debug)

if __name__ == '__main__':
    main()
