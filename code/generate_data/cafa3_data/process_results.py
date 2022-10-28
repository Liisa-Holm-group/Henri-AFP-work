"""Process classification results."""

from functools import reduce
import itertools
from pathlib import Path
from pdb import set_trace as bp
import pickle
import os
import pandas as pd

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

# from load_data import transform_features
import load_data
import data_processing


mpl.use('Agg')

def save_cafa_format(predictions, sequences, go_names, filename):
    nonzero = predictions.nonzero()
    nonzero_indices = zip(list(nonzero[0]), list(nonzero[1]))
    with open(filename, 'w+') as f:
        for row, column in nonzero_indices:
            f.write(f'{sequences[row]}\tGO:{go_names[column]}\t{predictions[row, column]:.5f}\t\n')

def targets_to_cafa(dir_, ontology, eval_data=True):
    eval_str = 'eval_' if eval_data else ''
    data = sp.load_npz(f'{dir_}/{ontology}_{eval_str}targets.npz')
    sequences = joblib.load(f'{dir_}/{ontology}_{eval_str}sequences.joblib')
    names = joblib.load(f'{dir_}/{ontology}_{eval_str}target_names.joblib')
    save_cafa_format(data, sequences, names, f'{dir_}/{ontology}_{eval_str}targets.cafa')

