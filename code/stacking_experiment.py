import argparse
import itertools
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path


import h5py
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.sparse as sp
from functools import reduce
from joblib import Parallel, delayed
from pdb import set_trace as bp
from scipy.stats import rankdata
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier

import numpy as np
import pandas as pd


import models

def combine_results(prediction_results, n_samples):
    """Combine a list of cv fold predictions and indices into a correctly ordered array."""
    results = np.zeros(n_samples)
    for prediction, index in prediction_results:
        results[index] = prediction[:,1].ravel()
    return results


def rank_tr_data(tr_data):
    ranks = rankdata(tr_data, axis=0, method='dense').astype(np.int)
    tr_data = np.column_stack((tr_data, ranks))
    return tr_data

def rank_test_data(test_data, tr_data, timing=False):
    start = time.time()
    res = np.zeros(test_data.shape)
    unique =  [np.unique(tr_data[:,i], axis=0) for i in range(tr_data.shape[1])]
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i,j] = np.searchsorted(unique[j], test_data[i, j])

    elapsed = time.time()-start
    if timing:
        print(f'Ranking time: {str(timedelta(seconds=elapsed))}')
    return np.column_stack((test_data, res))

def load_data(filenames, go_class):
    features = []
    for filename in filenames:
        try:
            with h5py.File(filename, 'r') as f:
                features.append(f[str(go_class)][:])
        except (OSError, KeyError):
            print(f'Error reading {filename} go class: {go_class}')
    features = np.column_stack(features)
    return features

def classifier_types(filenames):
    res = []
    map_ = {'xgb':1, 'svm':2, 'fm':3, 'elasticnet':4}
    for f in filenames:
        for name in map_.keys():
            if name in f:
                res.append(map_[name])
    return res


def train_stacking(filenames, targets, go_class, ontology, model, output_path, cluster_index=None, ranking=False, additional=False, taxonomy=False):
    """Train a stacking model for a single go-class and return cv predictions."""

    features = load_data(filenames, go_class)

    # load additional features
    if additional:
        additional_features = np.load(f'{output_path}/datasets/{ontology}_stacking_features.npy')

    if taxonomy:
        taxonomy_features = sp.load_npz(f'{output_path}/datasets/{ontology}_stacking_taxonomy.npz')


    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    if go_class % 10 == 0:
        print(f'started training GO class {go_class}')
    predictions = []


    for i, (train, test) in enumerate(cv.split(features, targets[:,go_class].toarray())):

        # combine features
        if ranking:
            training_features = rank_tr_data(features[train])
            test_features = rank_test_data(features[test], features[train])
        else:
            training_features = features[train]
            test_features = features[test]

        if additional:
            training_features = np.column_stack((training_features, additional_features[train]))
            test_features = np.column_stack((test_features, additional_features[test]))
            n_additional = additional_features.shape[1]
        else:
            n_additional = 0

        scaled = ['svm', 'ann', 'cv', 'sgdc']
        if any(s in str(model) for s in scaled):
            scaler = StandardScaler().fit(training_features)
            training_features = scaler.transform(training_features)
            test_features = scaler.transform(test_features)

        if taxonomy:
            training_features = sp.hstack((sp.csr_matrix(training_features), taxonomy_features[train]/taxonomy_features.shape[1]))
            test_features = sp.hstack((sp.csr_matrix(test_features), taxonomy_features[test]/taxonomy_features.shape[1]))
            n_additional += taxonomy_features.shape[1]

        n = training_features.shape[1]
        if go_class == 0 and i == 0:
            print(f'feature shape: {training_features.shape}')

        if cluster_index:
            # read predictions and calculate cv folds into a list. access with i
            res = []
            res_targets = []
            for go_index in cluster_index[i][go_class]:
                cluster_targets = targets[:,go_index].toarray().ravel()
                cluster_data = load_data(filenames, go_index)
                cluster_data_index = list(cv.split(cluster_data, cluster_targets))

                data_index = np.setdiff1d(cluster_data_index[i][0], test)

                res_targets.append(cluster_targets[data_index])
                res.append(cluster_data[data_index,:])

                # remove sequences in the main model training set

            final_features = np.concatenate(res + [features[train]])
            final_targets = np.concatenate(res_targets + [targets[train, go_class].toarray().ravel()])

        if cluster_index:
            final_model = model(final_features, final_targets, n_additional)
        else:
            final_model = model(training_features, targets[train, go_class].toarray().ravel(), n_additional)

        if 'ranking' in str(model):
            final_model.max_rank = training_features.max()

        res = final_model.predict_proba(test_features)

        if res.shape[0] != test_features.shape[0]:
            t = np.zeros((test_features.shape[0], 2))
            t[:,1] = res[res.shape[0]//2:]
            res = t

        predictions.append((res, test))

    predictions = combine_results(predictions, targets.shape[0])
    return predictions

def stack_models(model: str, target_path, output_path, ontology, ranking=False, additional=False, taxonomy=False, n_jobs=1, level_3=False,
        pool=False, level2_data_path=None, level3_data_path=None, cluster_index_path=None):


    path = level2_data_path

    methods = [['xgb'], ['svm'], ['fm'], ['elasticnet']]

    stacking_model = getattr(models, model)
    print(f'processing {ontology}')

    if level_3:
        additional = False
        taxonomy = False
        path = level3_data_path
        methods = [[f'{ontology}_xgb',], [f'{ontology}_LTR_xgb',], [f'{ontology}_sgdc',],
                   [f'{ontology}_ann',], [f'{ontology}_mean',],[f'{ontology}_ranking_mean',],]

    if pool:
        cluster_index = joblib.load(f'{cluster_index_path}/{ontology}_cluster_indices.joblib')
    else:
        cluster_index = None

    targets = sp.load_npz(f'{target_path}/{ontology}_targets.npz')

    files = []
    for method in methods:
        try:
            name = [f for f in os.listdir(path) if (ontology in f) and all(m in f for m in method) and ('h5' in f)]
            for i in range(len(name)): #NOTE
                files.append(path + name[i])
        except IndexError:
            print(f'{method} file not found')


    import multiprocessing as mp
    mp.set_start_method('forkserver')

    parallel = Parallel(n_jobs=n_jobs)
    results = np.array(parallel(delayed(train_stacking)(files, targets, i, ontology, stacking_model, output_path, ranking=ranking, additional=additional, taxonomy=taxonomy, cluster_index=cluster_index) for i in range(targets.shape[1])))

    print('evaluating results')
    scores = np.array([average_precision_score(targets[:, i].toarray(), result) for i, result in enumerate(results)])
    print(f'mean average precision score for all go-classes: {scores.mean()}')

    # save results
    print('saving results')
    ranking_str = '_ranking' if ranking else ''
    pool_str = '_pooled' if pool else ''
    additional_str = '_additional' if additional else ''
    taxonomy_str = '_taxonomy' if taxonomy else ''
    level_3_str = '_level_3' if level_3 else ''
    joblib.dump(scores, f'{output_path}/{ontology}_{model}{ranking_str}{additional_str}{taxonomy_str}_stacking{level_3_str}{pool_str}_scores.joblib')
    with h5py.File(f'{output_path}/{ontology}_{model}{ranking_str}{additional_str}{taxonomy_str}_stacking{level_3_str}{pool_str}_results.h5', 'w', libver='latest') as f:
        for i, arr in enumerate(results):
            dset = f.create_dataset(str(i), shape=(results[0].shape), data=arr, compression='gzip', compression_opts=9)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('output_path')
    parser.add_argument('target_path')
    parser.add_argument('level2_data_path')
    parser.add_argument('cluster_index_path')
    parser.add_argument('ontology', type=str)
    parser.add_argument('n_jobs', type=int,)
    parser.add_argument('level3', type=str)
    parser.add_argument('ranking', type=str)
    parser.add_argument('additional', type=str)
    parser.add_argument('taxonomy', type=str)
    parser.add_argument('pool', type=str)
    args = parser.parse_args()

    tf = lambda x: True if x =='True' else False

    level_3 = tf(args.level3)
    stack_models(args.model, target_path=args.target_path, output_path=args.output_path, ontology=args.ontology, level_3=level_3, ranking=tf(args.ranking), additional=tf(args.additional), taxonomy=tf(args.taxonomy), pool=tf(args.pool), level2_data_path=args.level2_data_path, level3_data_path='', cluster_index_path=args.cluster_index_path, n_jobs=args.n_jobs, )
