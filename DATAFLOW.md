# Data Flow Documentation

This document describes how data flows through the Henri-AFP protein function prediction pipeline, from raw input features to final predictions.

## Table of Contents

1. [Overview](#overview)
2. [Input Data Sources](#input-data-sources)
3. [InterProScan Feature Processing](#interproscan-feature-processing)
4. [Other Feature Processing](#other-feature-processing)
5. [Gene Ontology Annotation Processing](#gene-ontology-annotation-processing)
6. [Feature Matrix Assembly](#feature-matrix-assembly)
7. [First-Level Classifier Training](#first-level-classifier-training)
8. [Cross-Validation Flow](#cross-validation-flow)
9. [Second-Level Stacking](#second-level-stacking)
10. [Prediction Pipeline](#prediction-pipeline)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA GENERATION PHASE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw InterProScan Output    Taxonomy File    WolfPSort    TargetP    GO     │
│          │                       │               │           │       │      │
│          ▼                       ▼               ▼           ▼       ▼      │
│   ┌─────────────┐         ┌───────────┐    ┌────────┐  ┌────────┐ ┌──────┐ │
│   │ Parse IPS   │         │  Parse    │    │ Parse  │  │ Parse  │ │Parse │ │
│   │ Extract:    │         │ Taxonomy  │    │  WPS   │  │TargetP│ │  GO  │ │
│   │ - e-values  │         │ hierarchy │    │ scores │  │ probs  │ │annot.│ │
│   │ - clusters  │         └─────┬─────┘    └───┬────┘  └───┬────┘ └──┬───┘ │
│   │ - positions │               │              │           │         │      │
│   │ - counts    │               │              │           │         │      │
│   └──────┬──────┘               │              │           │         │      │
│          │                      │              │           │         │      │
│          ▼                      ▼              ▼           ▼         ▼      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SPARSE MATRIX CONSTRUCTION                        │   │
│   │    scipy.sparse.hstack([IPS_features, taxonomy, wps, targetp])      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│                    ┌──────────────────────────────┐                         │
│                    │  {ontology}_features.npz     │                         │
│                    │  {ontology}_targets.npz      │                         │
│                    └──────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIRST-LEVEL CLASSIFICATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Feature Selection (per feature_type: taxonomy, location, ipscan, all)     │
│                                    │                                         │
│           ┌────────────────────────┼────────────────────────┐               │
│           ▼                        ▼                        ▼               │
│      ┌─────────┐             ┌─────────┐              ┌─────────┐           │
│      │   XGB   │             │   SVM   │              │   FM    │           │
│      │(e-value)│             │(binary) │              │(binary) │           │
│      └────┬────┘             └────┬────┘              └────┬────┘           │
│           │                       │                        │                 │
│           └───────────────────────┼────────────────────────┘                │
│                                   ▼                                          │
│                    5-Fold Stratified CV per GO class                        │
│                                   │                                          │
│                                   ▼                                          │
│                    {method}_{feature_type}_predictions.h5                   │
│                    (n_samples predictions per GO class)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECOND-LEVEL STACKING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Load 8 H5 files (4 classifiers × 2 feature types)                        │
│                    │                                                         │
│                    ▼                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ Features = [xgb_tax, xgb_loc, svm_tax, svm_loc, fm_tax, fm_loc,   │    │
│   │             elasticnet_tax, elasticnet_loc] + additional + taxonomy│    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                    │                                                         │
│                    ▼                                                         │
│             XGB Stacking (or other stacking classifier)                     │
│                    │                                                         │
│                    ▼                                                         │
│              Final Predictions (CAFA format)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Data Sources

| Source | Description | Output Format |
|--------|-------------|---------------|
| InterProScan | Domain/motif predictions with e-values | Tab-delimited TSV |
| NCBI Taxonomy | Species hierarchy for each protein | Tab-delimited text |
| WolfPSort | Subcellular localization scores | Space-delimited text |
| TargetP | Signal peptide/transit peptide probabilities | Tab-delimited TSV |
| GO Annotations | Ground truth Gene Ontology labels | Tab-delimited text |

---

## InterProScan Feature Processing

### Raw IPS Parsing

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:105-136`

```python
def process_ipr_line2(line, data):
    fields = line.split('\t')
    name = fields[0].split('|')[1]           # Protein ID
    feature = f'{fields[3]}|{fields[4]}'     # InterPro ID | Name
    e_value = fields[8]                       # E-value (string)
    start = int(fields[6])                    # Domain start position
    stop = int(fields[7])                     # Domain stop position
    length = int(fields[2])                   # Protein length
    cluster = fields[11]                      # IPS cluster (optional)

    # Store: data[protein][feature] = [(e_value, cluster, start, stop, length), ...]
```

**IPS File Format (tab-delimited):**
```
field[0]: sp|P12345|PROT_HUMAN    (protein ID)
field[2]: 450                      (protein length)
field[3]: PF00069                  (InterPro/Pfam ID)
field[4]: Protein kinase domain    (feature name)
field[6]: 50                       (start position)
field[7]: 300                      (stop position)
field[8]: 1.2e-45                  (e-value, or '-' if none)
field[11]: IPR000719               (cluster, optional)
```

### Position and Count Calculations

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:169-207`

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROTEIN SEQUENCE                              │
│  ├─────────┬──────────────────────────────┬─────────┤           │
│  │N-terminal│          Middle              │C-terminal│          │
│  │ (100 aa) │                              │ (100 aa) │          │
│  └─────────┴──────────────────────────────┴─────────┘           │
│                                                                  │
│  For proteins < 300 aa: sequence divided into thirds            │
│  For proteins >= 300 aa: fixed 100 aa tails                     │
└─────────────────────────────────────────────────────────────────┘
```

**Non-overlapping filter:** Keeps domain hits with < 20% overlap
**Count:** Number of non-overlapping occurrences of each feature
**Localization:** Proportion of domain in N-terminal, middle, C-terminal regions

### E-value Transformation

**File:** `src/Holm_AFP/generate_data/in-house_data/load_data.py:54-85`

```python
def process_ipr(ipr_list):
    for item in ipr_list:
        for key in item.keys():
            if item[key][2] != '-':                    # Has e-value
                float_val = float(item[key][2])
                if float_val == 0:
                    float_val = 7.5e-305              # Global minimum
                e_dict[key] = -np.log(float_val)     # Log transform
            elif item[key][0] != 'missing':
                c_dict[key] = item[key][0]           # Cluster only
            else:
                em_set.add(key)                       # Missing flag
```

**Transformation:** `feature_value = -log(e-value)`

- Lower e-values (better matches) → Higher transformed values
- E-value of 0 → Use minimum value 7.5e-305
- Missing e-values → Stored separately as binary flags

### Sparse Matrix Construction

**File:** `src/Holm_AFP/generate_data/in-house_data/load_data.py:87-131`

```python
# Vectorizers for different feature types
cluster_binarizer = MultiLabelBinarizer(sparse_output=True)    # Binary: cluster presence
e_value_vectorizer = DictVectorizer()                          # Continuous: log(e-values)
e_missing_binarizer = MultiLabelBinarizer(sparse_output=True)  # Binary: missing flag
cv = [DictVectorizer() for i in range(4)]                      # count, start, middle, end

# Final assembly
features = sp.hstack([clusters, e_values, e_missing] + cl).tocsr()
```

**Final IPS Feature Matrix Structure:**

| Feature Type | Prefix | Values | Example |
|--------------|--------|--------|---------|
| Cluster presence | `ipscan_cluster` | 0/1 | `ipscan_clusterIPR000719` |
| Log e-value | `ipscan_e_value` | float | `ipscan_e_valuePF00069\|Protein kinase` |
| Missing e-value | `ipscan_e_missing` | 0/1 | `ipscan_e_missingPF00069\|Protein kinase` |
| Hit count | `ipscan_count` | int | `ipscan_countPF00069\|Protein kinase` |
| N-terminal prop. | `ipscan_start` | 0.0-1.0 | `ipscan_startPF00069\|Protein kinase` |
| Middle prop. | `ipscan_middle` | 0.0-1.0 | `ipscan_middlePF00069\|Protein kinase` |
| C-terminal prop. | `ipscan_end` | 0.0-1.0 | `ipscan_endPF00069\|Protein kinase` |

---

## Other Feature Processing

### Taxonomy Features

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:45-69`

```python
def process_taxonomy_data(path, min_count):
    counts = count_features(path)
    selected_features = {f[0] for f in counts if f[1] > min_count}
    # Returns: dict[protein] = [taxonomy_term1, taxonomy_term2, ...]
```

**File:** `src/Holm_AFP/generate_data/in-house_data/load_data.py:222-241`

```python
def load_taxonomy(path, sequences, pretrained, output_path):
    binarizer = MultiLabelBinarizer(sparse_output=True)
    features = data_processing.process_taxonomy_data(path, min_count=1)
    features = binarizer.fit_transform(...)
    # Filter: keep only features appearing in > 10 sequences
    counts = features.sum(axis=0)
    index = counts > 10
    features = features[:, index]
    # Returns: sparse binary matrix with prefix 'taxonomy_'
```

**Input format:** `protein_id \t species_name \t Eukaryota;Metazoa;Chordata;...`

**Output:** Binary sparse matrix where each column is a taxonomy term

### WolfPSort Localization

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:233-260`

```python
def process_wps_line(line, data):
    fields = line.split(' ', 1)
    name = fields[0].split('|')[1]
    features = fields[1].split(',')
    data[name] = {loc.split(' ')[0]: loc.split(' ')[1] for loc in features}
    # Returns: dict[protein] = {'nucl': '12.5', 'cyto': '8.2', ...}
```

**Input format:** `sp|P12345|PROT_HUMAN nucl 12.5, cyto 8.2, mito 5.1`

**Output:** Sparse matrix with continuous scores, prefix `wps_`

### TargetP Localization

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:262-288`

```python
def process_targetp_line(line, data):
    fields = line.split('\t')
    name = fields[0].split('|')[1]
    features = fields[2:5]  # 3 probability scores
    data[name] = [float(f) for f in features]
```

**Output:** Dense array with 3 columns:
- `targetp_SP` - Signal peptide probability
- `targetp_mTP` - Mitochondrial transit peptide probability
- `targetp_CS_position` - Cleavage site position

---

## Gene Ontology Annotation Processing

### Target Matrix Creation

**File:** `src/Holm_AFP/generate_data/in-house_data/data_processing.py:72-103`

```python
def count_targets(path):
    """Count occurrences of each GO class"""
    counter = Counter()
    # Parses: regex r'\t(\d{7}.*?)\t' to extract GO IDs

def process_target_data(path, min_count):
    counts = count_targets(path)
    selected_targets = {t[0] for t in counts if t[1] > min_count}
    # Returns: dict[protein] = {GO:0000001, GO:0000002, ...}
```

**File:** `src/Holm_AFP/generate_data/in-house_data/load_data.py:164-170`

```python
def load_targets(path, sequences):
    binarizer = MultiLabelBinarizer(sparse_output=True)
    targets = data_processing.process_target_data(path, min_count=20)  # FILTER!
    bin_targets = binarizer.fit_transform(targets[seq] for seq in sequences)
    return bin_targets, binarizer.classes_
```

### GO Class Filtering

**Minimum count threshold: 20** (configurable)

```
┌──────────────────────────────────────────────────────────────────┐
│                    GO CLASS FILTERING                             │
│                                                                   │
│  All GO annotations                                               │
│        │                                                          │
│        ▼                                                          │
│  Count occurrences per GO class                                  │
│        │                                                          │
│        ▼                                                          │
│  Keep only GO classes with count > min_count (20)                │
│        │                                                          │
│        ▼                                                          │
│  Binary target matrix: (n_proteins × n_GO_classes)               │
│  - 1 if protein has GO annotation                                │
│  - 0 otherwise                                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Note:** GO hierarchy/propagation is NOT implemented. GO classes are treated as independent labels.

---

## Feature Matrix Assembly

**File:** `src/Holm_AFP/generate_data/in-house_data/load_data.py:379-407`

```python
class DatasetGenerator:
    def generate_features(self, feature_names, dataset_name):
        data_loader = {
            'ipscan': lambda: load_ipscan(...),
            'taxonomy': lambda: load_taxonomy(...),
            'wps': lambda: load_wps(...),
            'targetp': lambda: load_targetp(...),
        }

        # Load all feature types in parallel
        datasets = parallel(delayed(lambda n: data_loader[n]())(n) for n in feature_names)

        # Horizontal concatenation
        results = sp.hstack(features)

        # Save
        sp.save_npz(f'{output_dir}/{ontology}_{dataset_name}_features.npz', results)
        joblib.dump(names, f'{output_dir}/{ontology}_{dataset_name}_feature_names.joblib')
```

**Output Files:**

| File | Description |
|------|-------------|
| `{ontology}_features.npz` | Sparse feature matrix (n_proteins × n_features) |
| `{ontology}_feature_names.joblib` | List of feature names |
| `{ontology}_targets.npz` | Sparse target matrix (n_proteins × n_GO_classes) |
| `{ontology}_target_names.joblib` | List of GO class names |
| `{ontology}_sequences.joblib` | List of protein IDs |

---

## First-Level Classifier Training

### Feature Selection

**File:** `src/Holm_AFP/models.py:38-81`

```python
def select_features(names, features, feature_type, base_method='taxonomy'):
    if base_method == 'binary':
        test = lambda n: ('e_value' in n) or ('e_missing' in n)
    else:
        test = lambda n: ('e_value' in n)  # e-value only

    if feature_type == 'taxonomy':
        # e-value features + taxonomy
        index = [i for i, n in enumerate(names) if test(n) or 'taxonomy' in n]
    elif feature_type == 'location':
        # e-value features + position info (start/middle/end)
        index = [i for i, n in enumerate(names) if test(n) or 'start' in n or 'middle' in n or 'end' in n]
    elif feature_type == 'ipscan':
        # e-value features only
        index = [i for i, n in enumerate(names) if test(n)]
    elif feature_type == 'all':
        # e-value + taxonomy + wps + targetp
        index = [i for i, n in enumerate(names) if test(n) or 'taxonomy' in n or 'wps' in n or 'targetp' in n]

    # Binary conversion for SVM/FM
    if 'binary' in base_method:
        res[res[:, bin_index] > 0] = 1
```

**Feature Type Matrix:**

| Feature Type | IPS e-value | IPS e-missing | IPS position | Taxonomy | WPS | TargetP |
|--------------|-------------|---------------|--------------|----------|-----|---------|
| `ipscan` | Yes | No | No | No | No | No |
| `taxonomy` | Yes | No | No | Yes | No | No |
| `location` | Yes | No | Yes | No | No | No |
| `all` | Yes | No | No | Yes | Yes | Yes |

**Base Method:**
- `e_value` (default): Use continuous log-transformed e-values (XGB, Elastic Net)
- `binary`: Convert e-values to 0/1 presence (SVM, FM)

### ModelTrainer Class

**File:** `src/Holm_AFP/models.py:153-209`

```python
class ModelTrainer:
    def train_models(self, n_jobs, feature_type, base_method):
        X = sp.load_npz(self.tr_feature_path).tocsr()   # Load features
        y = sp.load_npz(self.tr_target_path).tocsr()    # Load targets
        X, _ = select_features(names, X, feature_type, base_method)

        # Parallel training: one model per GO class
        parallel = Parallel(n_jobs=n_jobs, backend='multiprocessing')
        models = parallel(delayed(self.model)(X, y, go_class)
                         for go_class in range(y.shape[1]))
        return models
```

### First-Level Classifiers

**File:** `src/Holm_AFP/models.py`

| Classifier | Function | Parameters | Base Method |
|------------|----------|------------|-------------|
| XGBoost | `xgb_train()` :536-542 | n_estimators=25, max_depth=7, lr=0.5 | e_value |
| Elastic Net | `lasso_train()` :544-549 | loss='log', penalty='elasticnet' | e_value |
| SVM | `svm_train()` :571-584 | probability=True, tol=0.1, max_iter=750 | binary |
| Factorization Machine | `fm_train()` :558-569 | k=2, nb_epoch=6, batch_size=250 | binary |

**SVM Negative Sampling:** To handle class imbalance, SVM uses only 10% of negative samples:

```python
def svm_train(X, y, go_class):
    y = y[:, go_class].toarray()
    # Sample 10% of negatives
    index = np.random.choice(np.argwhere(y==0).ravel(), int(0.1*len(y==0)))
    index = np.append(np.argwhere(y).ravel(), index)  # All positives + sampled negatives
    X, y = X[index,:], y[index]
    model = SVC(probability=True, ...).fit(X, y.ravel())
```

---

## Cross-Validation Flow

### CV Experiment Functions

**File:** `src/Holm_AFP/models.py:586-713`

```python
def xgboost_test(X, y, go_class):
    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    y = y[:, go_class].toarray()

    cv_predictions = []
    for train, test in cv.split(X, y):
        model = XGBClassifier(...).fit(X[train], y[train].ravel())
        prediction = model.predict_proba(X[test])
        cv_predictions.append((prediction, test))  # Store predictions with indices

    # Combine fold predictions
    predictions = pr.combine_results(cv_predictions, y.shape[0])
    return {'predictions': predictions, 'cv_results': cv_results}
```

### Result Combination

**File:** `src/Holm_AFP/process_results.py:91-96`

```python
def combine_results(prediction_results, n_samples):
    results = np.zeros(n_samples)
    for prediction, index in prediction_results:
        results[index] = prediction[:,1].ravel()  # Positive class probability
    return results
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    5-FOLD CROSS-VALIDATION FLOW                              │
│                                                                              │
│   Full Dataset (n samples)                                                  │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────┬─────┬─────┬─────┬─────┐                                          │
│   │Fold1│Fold2│Fold3│Fold4│Fold5│  ← Stratified split per GO class         │
│   └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                                          │
│      │     │     │     │     │                                              │
│      ▼     ▼     ▼     ▼     ▼                                              │
│   Train on 4 folds, predict on 1                                            │
│      │     │     │     │     │                                              │
│      └─────┴─────┴─────┴─────┘                                              │
│                  │                                                           │
│                  ▼                                                           │
│   combine_results() → predictions array (n samples)                         │
│   Each sample has prediction from its test fold                             │
│                  │                                                           │
│                  ▼                                                           │
│   Save to HDF5: dataset[go_class] = predictions                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HDF5 Output Structure

**File:** `src/Holm_AFP/ipscan_experiment.py:145-158`

```python
with h5py.File(f'{path}/{name}_{feature_type}_predictions.h5', 'w') as f:
    for i, arr in enumerate(predictions):
        f.create_dataset(str(i), data=arr, compression='gzip')
```

**HDF5 Structure:**
```
{method}_{feature_type}_predictions.h5
├── "0"  → array(n_samples,)  # GO class 0 predictions
├── "1"  → array(n_samples,)  # GO class 1 predictions
├── "2"  → array(n_samples,)  # GO class 2 predictions
...
└── "N"  → array(n_samples,)  # GO class N predictions
```

---

## Second-Level Stacking

### Loading First-Level Predictions

**File:** `src/Holm_AFP/models.py:410-434`

```python
def load_predictions(filenames, go_class, train=True):
    """Load predictions for a specific GO class from multiple H5 files"""
    features = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            features.append(np.array(f[str(go_class)][:]))
    features = np.column_stack(features)  # Shape: (n_samples, n_methods)
    return features

def find_files(path, methods, ontology, filetype='h5'):
    """Find H5 files matching method combinations"""
    # methods = [['xgb', 'taxonomy'], ['xgb', 'location'], ...]
    files = []
    for method in methods:
        test = lambda f: (ontology in f) and all(m in f for m in method)
        files.append(path + [f for f in os.listdir(path) if test(f)][0])
    return files
```

### Stacking Feature Assembly

**File:** `src/Holm_AFP/models.py:233-264`

```python
def train_stacking(self, filenames, targets, go_class, ontology, model):
    # Load first-level predictions as features
    features = load_predictions(filenames, go_class)  # Shape: (n, 8)

    # Load additional features
    additional_features = np.load(self.additional_features)
    taxonomy_features = sp.load_npz(self.taxonomy_features)

    # Scale for SVM/ANN/SGDC
    if self.scaler:
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(features)
        features = scaler.transform(features)

    # Optional ranking transformation
    if 'ranking' in str(model):
        features = stacking_experiment.rank_tr_data(features)

    # Combine all features
    if 'mean' not in str(model):
        features = sp.hstack((
            sp.csr_matrix(features),           # 8 first-level predictions
            sp.csr_matrix(additional_features), # sequence features
            taxonomy_features                   # taxonomy (sparse)
        ))

    final_model = model(features, targets, additional_features.shape[1])
    return final_model, scaler
```

### Stacking Feature Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STACKING FEATURE MATRIX                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  First-Level Predictions (8 columns)                                        │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐ │
│  │xgb_tax │xgb_loc │svm_tax │svm_loc │fm_tax  │fm_loc  │en_tax  │en_loc  │ │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘ │
│                                    +                                         │
│  Additional Features (~15 columns)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ seq_noise | seq_length | ips_coverage | ips_count | max_e_value |...│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    +                                         │
│  Taxonomy Features (sparse, ~1000+ columns)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Eukaryota | Metazoa | Chordata | Mammalia | Primates | ... (binary) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    =                                         │
│  Combined Stacking Features (dense + sparse)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stacking Classifiers

**File:** `src/Holm_AFP/models.py:436-534`

| Classifier | Function | Parameters |
|------------|----------|------------|
| XGB Stacking | `xgb_stacking()` :436 | n_estimators=75, tree_method='hist' |
| LTR XGB | `LTR_xgb_stacking()` :444 | objective='rank:pairwise', n_estimators=100 |
| SGDC | `sgdc_stacking()` :453 | loss='log', penalty='l2' |
| ANN | `ann_stacking()` :472 | hidden_layer_sizes=(5,), activation='relu' |
| SVM | `svm_stacking()` :483 | probability=True, tol=0.01 |
| Mean | `mean_stacking()` :530 | Simple average of predictions |
| Ranking Mean | `ranking_mean_stacking()` :533 | Average of ranked predictions |

---

## Prediction Pipeline

### First-Level Prediction

**File:** `src/Holm_AFP/models.py:97-151`

```python
class Predictor:
    def run(self):
        models = joblib.load(self.model_path)           # Load trained models
        X = sp.load_npz(self.te_feature_path).tocsr()   # Load test features
        X, names = select_features(names, X, feature_type, base_method)

        # Parallel prediction
        predictions = self.predict(X, models, self.n_jobs)

        # Save as HDF5 or CAFA format
        if self.h5:
            with h5py.File(output_path, 'w') as f:
                for i, arr in enumerate(predictions.T):
                    f.create_dataset(str(i), data=arr)
        else:
            pr.save_cafa_format(predictions, sequences, go_names, output_path)
```

### Stacking Prediction

**File:** `src/Holm_AFP/models.py:302-408`

```python
class StackingPredictor:
    def predict_stacking(self, filenames, go_class, ontology, model):
        # Load test first-level predictions
        features = load_predictions(filenames, go_class)

        # Apply same scaling as training
        if self.scaler:
            features = model[1].transform(features)  # scaler from training

        # Combine with additional features
        features = sp.hstack((
            sp.csr_matrix(features),
            sp.csr_matrix(additional_features),
            taxonomy_features
        ))

        return model[0].predict_proba(features)

    def run(self):
        # Find first-level prediction H5 files
        files = find_files(self.te_feature_path, methods, self.ontology)

        # Parallel prediction per GO class
        predictions = self.predict(files, models, self.n_jobs)

        # Save in CAFA format
        pr.save_cafa_format(predictions.T, sequences, go_names, output_path)
```

### CAFA Output Format

**File:** `src/Holm_AFP/process_results.py:31-36`

```python
def save_cafa_format(predictions, sequences, go_names, filename):
    nonzero = predictions.nonzero()
    with open(filename, 'w+') as f:
        for row, column in zip(nonzero[0], nonzero[1]):
            f.write(f'{sequences[row]}\tGO:{go_names[column]}\t{predictions[row, column]:.5f}\t\n')
```

**CAFA Format:**
```
P12345    GO:0000001    0.95412
P12345    GO:0000002    0.84521
P67890    GO:0000001    0.72103
...
```

---

## Summary: Complete Data Flow

```
RAW DATA
   │
   ├── InterProScan TSV ─────────────────────────────────────────────────┐
   │        │                                                             │
   │        ▼                                                             │
   │   Parse: e-values, clusters, positions, counts                      │
   │        │                                                             │
   │        ▼                                                             │
   │   Transform: -log(e-value), localization proportions                │
   │                                                                      │
   ├── Taxonomy file ────────────────────────────────────────────────────│
   │        │                                                             │
   │        ▼                                                             │
   │   Binary encoding (min_count > 10)                                  │
   │                                                                      │
   ├── WolfPSort ────────────────────────────────────────────────────────│
   │        │                                                             │
   │        ▼                                                             │
   │   DictVectorizer → sparse scores                                    │
   │                                                                      │
   ├── TargetP ──────────────────────────────────────────────────────────│
   │        │                                                             │
   │        ▼                                                             │
   │   Dense array (3 columns)                                           │
   │                                                                      │
   └── GO Annotations ───────────────────────────────────────────────────┤
            │                                                             │
            ▼                                                             │
       Filter: min_count > 20                                            │
       Binary encoding → target matrix                                   │
                                                                         │
                         ▼                                               │
              ┌──────────────────────┐                                   │
              │  FEATURE MATRICES    │ ◄──────────────────────────────────┘
              │  {ont}_features.npz  │
              │  {ont}_targets.npz   │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │   XGB   │    │   SVM   │    │   FM    │    ← LEVEL 1
    │tax + loc│    │tax + loc│    │tax + loc│      (4 classifiers × 2 feature types)
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
              5-fold CV per GO class
                        │
                        ▼
              ┌──────────────────────┐
              │  H5 PREDICTION FILES │
              │  (8 files per ont)   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  STACKING FEATURES   │
              │  8 preds + taxonomy  │
              │  + additional feats  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   XGB STACKING       │    ← LEVEL 2
              │   (per GO class)     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  FINAL PREDICTIONS   │
              │   (CAFA format)      │
              └──────────────────────┘
```
