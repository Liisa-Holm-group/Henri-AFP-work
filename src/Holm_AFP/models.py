"""Feature selection and classification methods."""
from typing import Optional, List

from sklearn.neural_network import MLPClassifier
from datetime import timedelta
import os
import time



import h5py
import joblib
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from joblib import Parallel, delayed
import process_results as pr

np.random.seed(42)

def select_features(names, features, feature_type, base_method='taxonomy', only_names=False):
    """
    cafa3 experiment feature selection

    :param names: list containing feature names
    :type names: list

    :param features: features
    :type features: sp.csr_matrix

    :param feature_type: feature names substring specifying which feature subset to select
    :type feature_type: string

    """
    print('selecting features')
    if base_method == 'binary':
        test = lambda n: ('e_value' in n) or ('e_missing' in n)
    else:
        test = lambda n: ('e_value' in n)


    if feature_type == 'taxonomy':
        index = np.array([i for i, n in enumerate(names) if test(n) or 'taxonomy' in n])
        names2 = [names[i] for i in index]
        bin_index = range(len(index))
    elif feature_type == 'all':
        index = np.array([i for i, n in enumerate(names) if test(n) or ('taxonomy' in n) or ('wps' in n) or ('targetp' in n)])
        names2 = [names[i] for i in index]
        bin_index = [i for i, n in enumerate(names2) if ('wps' not in n) and ('targetp' not in n)]
    elif feature_type == 'ipscan':
        index = np.array([i for i, n in enumerate(names) if test(n)])
        names2 = [names[i] for i in index]
        bin_index = range(len(index))
    else: # location
        index = np.array([i for i, n in enumerate(names) if test(n) or any(position in n for position in ['start','middle','end'])])
        names2 = [names[i] for i in index]
        bin_index = range(len(index))

    if only_names:
        return names2
    res = features[:, index]
    if 'binary' in base_method:
        res[res[:, bin_index] > 0] = 1
    return res, names2


def predict(X, model, i):
    """Apply a model to a feature vector X."""

    if not i % 10:
        print(f'model {i}')

    prediction = model.predict_proba(X)
    try:
        prediction = prediction[:,1]
    except IndexError:
        pass
    return prediction.reshape(-1, 1).round(2)


def predict_on_go_class(X, model, go_class_index):
    """ A helper function to split data outside the main thread before predicting"""
    cols = list(range(10)) + [go_class_index * 2 + 10, go_class_index * 2 + 11]
    Xi = X[:, cols]
    return predict(Xi, model, go_class_index)


def train_on_go_class(X, y, random_state, model, go_class_index):
    """ A helper function to split data outside the main thread before training"""
    cols = list(range(10)) + [go_class_index * 2 + 10, go_class_index * 2 + 11]
    Xi = X[:, cols]
    return model(Xi, y, go_class_index, random_state)


class Predictor:
    """Predict all GO-classes using pretrained models."""

    def __init__(self, name, model_path, go_class_names, te_feature_path, te_sequences, output_path, feature_names, h5=False, n_jobs=1):
        self.name = name
        self.model_path = model_path
        self.te_feature_path = te_feature_path
        self.te_sequences = te_sequences
        self.output_path = output_path
        self.go_class_names = go_class_names
        self.n_jobs=n_jobs
        self.feature_names = feature_names
        self.h5 = h5

    def predict(self, X, models, n_jobs, feature_type = str):
        """Given sequence features X, predict prob for each go-class."""

        if feature_type == "string_search":
            parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing')
            res = parallel(delayed(predict_on_go_class)
                           (X, model, go_class_index) for go_class_index, model in enumerate(models))
            predictions = np.hstack(res)
            assert predictions.shape[0] == X.shape[0] and predictions.shape[1] == len(models)
            return predictions
        else:
            # TODO: Implement more types here!
            raise ValueError(f"Invalid feature type '{feature_type}'. Must be 'string_search'.")

    def run(self, feature_type: str = "string_search"):
        """Predict and save results."""

        print('loading models')
        models = joblib.load(self.model_path)
        print('loading data')
        X = sp.load_npz(self.te_feature_path).tocsr()

        sequences = joblib.load(self.te_sequences)
        go_names = joblib.load(self.go_class_names)
        print('predicting')
        predictions = self.predict(X, models, self.n_jobs, feature_type) # output: n_seq x n_go matrix

        print('saving predictions')
        if self.h5:
            predictions = predictions.T
            with h5py.File(f'{self.output_path}/{self.name}_{feature_type}_predictions.h5', 'w', libver='latest') as f:
                for i, arr in enumerate(predictions):
                    # TODO: Why is this unused?
                    dset = f.create_dataset(str(i), shape=(predictions[0].shape), data=arr, compression='gzip', compression_opts=9)
        else:
            pr.save_cafa_format(predictions, sequences, go_names, f'{self.output_path}/{self.name}_{feature_type}_predictions')

class ModelTrainer:
    """Train models for each GO-class on a given data"""

    def __init__(self, name, model, go_class_names, tr_feature_path, tr_target_path, feature_names, output_path, h5=False, classes=None, random_state=None):
        """Experiment: string name, go class model function, features and targets paths. output pah"""
        self.name = name
        self.model = model
        self.tr_feature_path = tr_feature_path  # A csr matrix with shape (N, p) (p is the number of features) #NOTE: Was (p,N)
        self.tr_target_path = tr_target_path    # A csr matrix with shape (N, k) (k is the amount of go classes)
        self.output_path = output_path
        self.go_class_names = go_class_names    # A joblib with 1D (k,) ndarray of go class names
        self.feature_names = feature_names  # A joblib with a list of all feature names (p)
        self.debug=False
        self.h5 = h5
        self.classes = classes
        self.random_state = random_state

    def train_models(self, n_jobs: int, feature_type: str):

        print('loading matrices')

        if feature_type == "string_search":
            X = sp.load_npz(self.tr_feature_path).tocsr()

            y = sp.load_npz(self.tr_target_path).tocsr()
            print(f'target shape: {y.shape}')
            print(f'feature shape: {X.shape}')

            if self.classes is not None:
                y = y[:, self.classes.ravel()]

            start = time.time()
            parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing')
            print('training models')
            models = parallel(
                delayed(train_on_go_class)(
                    X=X,
                    y=y,
                    model=self.model,
                    random_state=self.random_state,
                    go_class_index=go_class_index,
                )
                for go_class_index in range(y.shape[1])
            )
            elapsed = time.time() - start
            print(f'Training time: {str(timedelta(seconds=elapsed))}')
            return models
        else:
            # TODO: Implement more types here!
            raise ValueError(f"Invalid feature type '{feature_type}'. Must be 'string_search'.")


    def run(self, n_jobs, feature_types: Optional[List[str]]=None):
        # TODO: Add an argument here for the
        if feature_types is None or len(feature_types) == 0:
            feature_types = ['string_search']
        for feature_type in feature_types:
            models = self.train_models(n_jobs, feature_type)
            if not self.random_state and self.random_state != 0:
                print('saving models')
                if not os.path.exists(f'{self.output_path}'):
                    os.mkdir(f'{self.output_path}')
                joblib.dump(models, f'{self.output_path}/{self.name}_{feature_type}_full_models.joblib',
                                       protocol=pickle.HIGHEST_PROTOCOL)
                return None
            else:
                return models



class StackingModelTrainer:
    """Train models for each GO-class on a given data"""

    def __init__(self, name, model, tr_feature_path, tr_target_path, output_path, ontology, additional_features, taxonomy_features, level_3=False):
        """Experiment: string name, go class model function, features and targets paths. output pah"""
        self.name = name
        self.model = model
        self.tr_feature_path = tr_feature_path
        self.tr_target_path = tr_target_path
        self.output_path = output_path
        self.debug = False
        self.ontology = ontology
        self.level_3 = level_3
        self.additional_features=additional_features
        self.taxonomy_features = taxonomy_features

        if 'svm' in str(model) or 'ann' in str(model) or 'sgdc' in str(model):
            self.scaler = True
        else:
            self.scaler = False

    def train_stacking(self, filenames, targets, go_class, ontology, model):
        features = load_predictions(filenames, go_class)
        if not self.level_3:
            additional_features = np.load(self.additional_features)
            taxonomy_features = sp.load_npz(self.taxonomy_features)

            if self.scaler:
                scaler = MinMaxScaler(feature_range=(0, 1)).fit(features)
                features = scaler.transform(features)
            if not go_class:
                print(features.shape)

            if 'ranking' in str(model):
                features = stacking_experiment.rank_tr_data(features)

            if 'mean' not in str(model):
                features = sp.hstack((sp.csr_matrix(features), sp.csr_matrix(additional_features), taxonomy_features))
            else:
                features = sp.csr_matrix(features)
        else:
            features = sp.csr_matrix(features)

        ts = targets.toarray().ravel()

        if not go_class % 10:
            print(f'training GO class {go_class}')
        final_model = model(features, ts, additional_features.shape[1])

        if self.scaler:
            return final_model, scaler
        else:
            return final_model, None

    def train_models(self, n_jobs, debug=False):
        if not self.level_3:
            methods = [['xgb', 'taxonomy'], ['xgb', 'location'],
                       ['svm', 'taxonomy'], ['svm', 'location'],
                       ['fm', 'taxonomy'], ['fm', 'location'],
                       ['elasticnet', 'taxonomy'], ['elasticnet', 'location']]
        else:
            methods = [['sgdc'], ['xgb'], ['LTR_xgb'], ['svm'], ['ann'], ['mean'], ['ranking_mean']]

        files = find_files(self.tr_feature_path, methods, self.ontology)
        stacking_model = self.model
        y = sp.load_npz(self.tr_target_path)
        if debug:
            y = y[:, :20]
            n_jobs = 1
        start = time.time()
        parallel = Parallel(n_jobs=n_jobs)
        print('training models')
        results = parallel(delayed(self.train_stacking)(files, y[:,go_class], go_class, self.ontology, stacking_model) for go_class in range(y.shape[1]))

        elapsed = time.time()-start
        print(f'Training time: {str(timedelta(seconds=elapsed))}')
        return results

    def run(self, n_jobs, debug=False):
        results = self.train_models(n_jobs, debug)
        models, scalers = zip(*results)
        print('saving models')
        level_3_str = '_level_3' if self.level_3 else ''
        joblib.dump(models, f'{self.output_path}/{self.name}_{self.ontology}{level_3_str}_stacking_models.joblib',
                                   protocol=pickle.HIGHEST_PROTOCOL)
        if self.scaler:
            print('saving scalers')
            joblib.dump(scalers, f'{self.output_path}/{self.name}_{self.ontology}{level_3_str}_stacking_scalers.joblib',
                                       protocol=pickle.HIGHEST_PROTOCOL)

class StackingPredictor:
    """Predict all GO-classes using pretrained models."""

    def __init__(self, name, model_path, go_class_names, te_feature_path, te_sequences, output_path, ontology, additional_features, taxonomy_features, n_jobs=1, tr_feature_path=None,):
        self.name = name
        self.model_path = model_path
        self.te_feature_path = te_feature_path
        self.te_sequences = te_sequences
        self.output_path = output_path
        self.go_class_names = go_class_names
        self.n_jobs=n_jobs
        self.ontology=ontology
        self.tr_feature_path = tr_feature_path

        self.additional_features=additional_features
        self.taxonomy_features=taxonomy_features

        if 'svm' in str(model_path) or 'ann' in str(model_path) or 'sgdc' in str(model_path):
            self.scaler = True
        else:
            self.scaler = False

    def predict_stacking(self, filenames, go_class, ontology, model):

        features = load_predictions(filenames, go_class)

        additional_features_tr = np.load(self.additional_features)
        taxonomy_features_tr = sp.load_npz(self.taxonomy_features)

        if self.scaler:
            scaler = model[1]
            features = scaler.transform(features)


        if 'ranking' in self.model_path or 'LTR' in self.model_path:
            features_train = load_predictions(self.tr_files, go_class)
            if 'ranking' in self.model_path:
                features = stacking_experiment.rank_test_data(features, features_train)
            else:

                features_train = sp.hstack((sp.csr_matrix(features_train), sp.csr_matrix(additional_features_tr), taxonomy_features_tr))
                t = model.predict_proba(features_train)
                scaler = MinMaxScaler().fit(t)

        if 'mean' not in self.model_path:
            features = sp.hstack((sp.csr_matrix(features), sp.csr_matrix(additional_features), taxonomy_features))
        else:
            features = sp.csr_matrix(features)


        if not go_class % 10:
            print(f'training go class {go_class}')

        if not go_class:
            print(features.shape)


        if self.scaler:
            return model[0].predict_proba(features)
        else:
            if 'LTR' in self.model_path:
                t = scaler.transform(model.predict_proba(features))
                t[t>1] = 1
                t[t<0] = 0
                return t

            else:
                return model.predict_proba(features)

    def predict(self, feature_files, models, n_jobs):
        """Given sequence features X, predict prob for each go-class."""


        res = []
        parallel = Parallel(n_jobs=int(n_jobs))

        res = parallel(delayed(self.predict_stacking)(feature_files, i, self.ontology, models[i]) for i, model in enumerate(models))

        predictions = np.array(res)[:,:,1]
        return predictions

    def run(self):
        """Predict and save results."""

        print('loading models')
        models = joblib.load(self.model_path)
        if self.scaler:
            scalers = joblib.load(self.model_path.replace('models', 'scalers'))
            models = [(m, s) for m, s in zip(models, scalers)]

        print('loading data')

        methods = [['xgb', 'taxonomy'], ['xgb', 'location'],
                   ['svm', 'taxonomy'], ['svm', 'location'],
                   ['fm', 'taxonomy'], ['fm', 'location'],
                   ['lasso', 'taxonomy'], ['lasso', 'location']]

        files = find_files(self.te_feature_path, methods, self.ontology)
        if self.tr_feature_path:
            self.tr_files = find_files(self.tr_feature_path, methods, self.ontology)

        print('predicting')

        predictions = self.predict(files, models, self.n_jobs) # output: n_seq x n_go matrix

        print('saving predictions')
        pr.save_cafa_format(predictions.T, joblib.load(self.te_sequences), joblib.load(self.go_class_names), f'{self.output_path}/{self.name}_stacking_predictions')

def load_predictions(filenames, go_class, train=True):
    """Load predictions for a specific go-class from distinct h5 files"""
    features = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            if train:
                features.append(np.array(f[str(go_class)][:]))
            else:
                n = len(list(f.keys()))
                features.append(np.array([f[str(i)][:][go_class] for i in range(n)]))

    features = np.column_stack(features)

    return features

def find_files(path, methods, ontology, filetype='h5'):
    """Find the h5 filenames corresponding to the methods list in directory path"""
    files = []
    for method in methods:
        try:
            test = lambda f, method, ontology: (ontology in f) and all(m in f for m in method) and (filetype in f)
            files.append(path + [f for f in os.listdir(path) if test(f, method, ontology)][0])
        except IndexError:
            print(f'{method} file not found')
    return files

def xgb_stacking(X, y, n_additional):
    n = X.shape[1]
    model = XGBClassifier(objective='binary:logistic',
            n_estimators=75,
            tree_method='hist',
            random_state=42,).fit(X, y)
    return model

def LTR_xgb_stacking(X, y, n_additional):
    n = X.shape[1]
    model = XGBClassifier(objective='rank:pairwise',
            n_estimators=100,
            tree_method='hist',
            max_depth=4,
            random_state=42,).fit(X, y)
    return model

def sgdc_stacking(X, y, n_additional=0):
    model = SGDClassifier(random_state=42, loss='log', penalty='l2', alpha=0.1).fit(X, y)
    return model

def sgdc_stacking_lr(X, y, n_additional=0):
    class_weight = np.sqrt(len(y) / (y.sum() * np.bincount(y)))
    # class_weight[class_weight > 20] = 20
    class_weight = {0:class_weight[0], 1:class_weight[1]}
    model = LogisticRegression(random_state=42, penalty='l2', max_iter=1000, class_weight=class_weight).fit(X, y)
    return model

def lr_stacking(X, y, n_additional=0):
    model = SGDClassifier(random_state=42, loss='log', penalty='None', alpha=0.1).fit(X, y)
    return model

def elasticnet_stacking(X, y, n_additional=0):
    model = SGDClassifier(random_state=42, loss='log', penalty='elasticnet', alpha=0.1).fit(X, y)
    return model

def ann_stacking(X, y, n_additional=0):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', alpha=0.01, batch_size=100, tol=1e-3, random_state=42).fit(X, y)
    return model


class Svm(SVC):
    def predict_proba(self, x):
        res = self.decision_function(x)
        return np.vstack((np.zeros(res.shape), res)).T

def svm_stacking(X, y, n_additional=0):
    X = X.tocsr()
    index = np.random.choice(np.argwhere(y==0).ravel(), int(0.1*len(y==0)))
    index = np.append(np.argwhere(y).ravel(),index)
    X = X[index,:]
    y = y[index]
    warnings.filterwarnings('ignore')
    model = SVC(probability=True, tol=0.01,random_state=42,max_iter=1000).fit(X, y.ravel())
    return model

class MeanStacking:

    def __init__(self, ranking=False, weighting=False):
        self.ranking=ranking
        self.weighting=weighting
        self.weights = None

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def fit(self, X, y):
        self.max_rank = X.max()
        if self.weighting:
            self.weights = np.array([average_precision_score(y, X[:,i]) for i in range(X.shape[1])])


            print(self.weights.argmax())
        return self

    def predict_proba(self, X):
        if self.ranking:
            #NOTE: ranking=True, additional=False, taxonomy=False
            n = X.shape[1]
            res = np.mean(X[:,(n//2):], axis=1)
            res = res / self.max_rank
            result = np.hstack((np.zeros(res.shape), res))
            return result
        else:
            res = np.average(X.toarray(), axis=1)
            # res = X[:,np.argmax(self.weights)]
            res = np.vstack((np.zeros(res.shape), res)).T
            return res

def mean_stacking(X, y, n_additional=0):
    return MeanStacking().fit(X, y)

def ranking_mean_stacking(X, y, n_additional=0):
    return MeanStacking(ranking=True).fit(X, y)

def xgb_train(X: sp.csr_matrix, y: sp.csr_matrix, go_class: int, random_state: int = 42):
    """Train the model on full data"""
    print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()

    model = XGBClassifier(n_estimators=25, max_depth=7, learning_rate=0.5,
            alpha=0.1, objective='binary:logistic', random_state=random_state).fit(X, y.ravel())
    return model

def lasso_train(X, y, go_class, random_state=42):
    """Train the model on full data"""
    print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()

    model = SGDClassifier(tol=1e-1, loss='log_loss', penalty='elasticnet', random_state=random_state).fit(X, y.ravel())
    return model

def elasticnet_train(X, y, go_class, random_state=42):
    return lasso_train(X, y, go_class, random_state=random_state)



# @delayed
# @wrap_non_picklable_objects
def fm_train(X, y, go_class, random_state=42):
    """train this function on a particular go class"""
    np.random.seed(random_state)
    import pyfms
    import pyfms.regularizers
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()
    classifier_dims = X.shape[1]
    model = pyfms.Classifier(classifier_dims, k=2, X_format="csr")
    model.fit(X, y.ravel(), nb_epoch=6,batch_size=250)
    return model

def svm_train(X, y, go_class):
    """Train the model on full data"""
    print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()

    index = np.random.choice(np.argwhere(y==0).ravel(), int(0.1*len(y==0)))
    index = np.append(np.argwhere(y).ravel(),index)
    X = X[index,:]
    y = y[index]

    # model = SGDClassifier(tol=1e-1, loss='log', penalty='elasticnet',random_state=42).fit(X_sample, y_sample.ravel())
    warnings.filterwarnings('ignore')
    model = SVC(probability=True,tol=0.1,random_state=42,max_iter=750).fit(X, y.ravel())
    return model

def xgboost_test(X, y, go_class, random_state=42):
    """train this function on a particular go class"""
    n_folds=5
    cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()

    cv_results = []
    cv_predictions = []
    for train, test in cv.split(X, y):
        model = XGBClassifier(n_estimators=25, max_depth=7, learning_rate=0.5, alpha=0.1, objective='binary:logistic', random_state=random_state).fit(X[train], y[train].ravel())
        prediction = model.predict_proba(X[test])
        # cv_results.append({'model':model, 'feature_importances':pr.process_feature_importances(model)})
        cv_results.append({'model':model, 'feature_importances':[]})
        cv_predictions.append((prediction, test))

    predictions = pr.combine_results(cv_predictions, y.shape[0])
    # if go_class == 10:
    #     joblib.dump([m['model'] for m in cv_results], 'test_model')

    return {'predictions':predictions, 'cv_results':cv_results}

def fm_test(X, y, go_class, random_state=42):
    """train this function on a particular go class"""
    n_folds=5
    cv = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')

    y = y[:, go_class].toarray()


    cv_results = []
    cv_predictions = []

    for train, test in cv.split(X, y):

        import pyfms
        import pyfms.regularizers

        classifier_dims = X[train].shape[1]

        model = pyfms.Classifier(classifier_dims, k=2, X_format="csr")
        model.fit(X[train], y[train].ravel(), nb_epoch=6,batch_size=250)
        prediction = model.predict_proba(X[test])
        cv_predictions.append((np.vstack((np.zeros(len(prediction)), prediction)).T, test))
        indices = np.zeros(2)
        values = np.zeros(2)
        cv_results.append({'model':1, 'feature_importances':{'indices':indices, 'values':values}})

    predictions = pr.combine_results(cv_predictions, y.shape[0])
    return {'predictions':predictions, 'cv_results':cv_results}

def ann_test(X, y, go_class):
    """train this function on a particular go class"""
    n_folds=5
    cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')

    y = y[:, go_class].toarray()

    cv_results = []
    cv_predictions = []

    for train, test in cv.split(X, y):
        X_tr = X[train,:]
        y_tr = y[train]

        model = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=3,  activation='relu', alpha=0.01, batch_size=100, tol=1e-3, random_state=42).fit(X_tr, y_tr.ravel())
        prediction = model.predict_proba(X[test])
        cv_predictions.append((prediction, test))

    predictions = pr.combine_results(cv_predictions, y.shape[0])
    return {'predictions':predictions, 'cv_results':cv_results}

def svm_test(X, y, go_class):
    """train this function on a particular go class"""
    n_folds=5
    cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')

    y = y[:, go_class].toarray()

    cv_results = []
    cv_predictions = []

    for train, test in cv.split(X, y):

        X_tr, y_tr = X[train], y[train]
        index = np.random.choice(np.argwhere(y_tr==0).ravel(), int(0.1*len(y_tr==0)))
        index = np.append(np.argwhere(y_tr).ravel(),index)
        X_sample = X_tr[index,:]
        y_sample = y_tr[index]

        from sklearn.svm import SVC
        import warnings
        warnings.filterwarnings('ignore')
        model = SVC(probability=True,tol=0.1,random_state=42,max_iter=750).fit(X_sample, y_sample.ravel())
        prediction = model.predict_proba(X[test])
        cv_predictions.append((prediction, test))

    predictions = pr.combine_results(cv_predictions, y.shape[0])
    return {'predictions':predictions, 'cv_results':cv_results}


def lasso_test(X, y, go_class):
    """train this function on a particular go class"""
    n_folds=5
    cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    if go_class % 100 == 0:
        print(f'started training GO class {go_class}')
    y = y[:, go_class].toarray()

    cv_results = []
    cv_predictions = []

    for train, test in cv.split(X, y):
        model = SGDClassifier(tol=1e-1, loss='log', penalty='elasticnet',random_state=42).fit(X[train], y[train].ravel())
        prediction = model.predict_proba(X[test])
        indices = model.coef_.nonzero()
        values = model.coef_[indices]
        cv_results.append({'model':model, 'feature_importances':{'indices':indices, 'values':values}})
        cv_predictions.append((prediction, test))
    predictions = pr.combine_results(cv_predictions, y.shape[0])
    return {'predictions':predictions, 'cv_results':cv_results}

