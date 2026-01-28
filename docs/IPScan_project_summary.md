# Project A: InterProScan-based Protein Function Prediction

## Overview
**Purpose**: Automated Protein Function Prediction (AFP) using optimized InterProScan (IPS) features  
**Authors**: Henri Tiittanen, Liisa Holm, Petri Törönen  
**Institution**: University of Helsinki, Institute of Biotechnology  
**Publication**: bioRxiv preprint, August 26, 2022

## Main Contribution
This project demonstrates that optimizing InterProScan feature processing with proper classifiers and stacking can outperform complex multi-source AFP methods. Surprisingly, the best methods outperformed all CAFA3 competition participants in most tests.

---

## Data Types

### Input Data
1. **InterProScan (IPS) features** - Primary input
   - Protein domains, sequence motifs, protein families
   - From IPS version 5.38-76.0 (in-house) and 5.22-61.0 (CAFA3)

2. **Species Taxonomy**
   - NCBI taxonomy hierarchy
   - Converted to binary vectors

3. **Cellular Localization Predictions**
   - TargetP and WolfPSort predictions
   - Rough predictions of cellular location

### Target Data
- **Gene Ontology (GO) annotations**
  - Three ontologies: Biological Process (BP), Molecular Function (MF), Cellular Component (CC)
  - Only manually curated annotations used (excluded IEA, ND, ISS evidence codes)
  - Classes must be specific enough (< 5% of root node size)

### Datasets
1. **In-house dataset** (2019.10.16)
   - CC: 577,424 proteins, 1,688 GO classes
   - MF: 637,552 proteins, 3,452 GO classes  
   - BP: 666,349 proteins, 11,288 GO classes

2. **CAFA3 dataset** (training: 2016.11.29, evaluation: separate)
   - Used for comparison against international competition
   - No Knowledge (NK) data for evaluation

---

## Feature Engineering

### Base Feature Sets

#### 1. **e-value**
- Negative log-transformed IPS E-values
- Preserves strength of signal

#### 2. **binary**
- All non-negative log e-values converted to 1
- Simple presence/absence encoding

### Enhanced Feature Sets

#### 3. **taxonomy**
- Base features + species taxonomy binary vector
- Allows different predictions for same sequence feature in different species

#### 4. **location**
- Base features + sequence position information
- Sequence divided into three parts: start, middle, end
- For each IPS feature, proportion in each part included

#### 5. **count**
- Base features + occurrence count for each IPS feature
- How many times each feature appears in sequence

#### 6. **cluster**
- Base features + IPS feature cluster information
- Groups related features, uses strongest signal per cluster

### Feature Dimensionality
- e-value/binary: ~32,000-36,000 features
- location: ~130,000-147,000 features  
- count: ~65,000-73,000 features
- taxonomy: adds species-specific dimensions

---

## Classification Architecture

### Two-Level Stacking Approach

```
Input Sequence
    ↓
InterProScan + Taxonomy + Localization
    ↓
[LEVEL 1: Multiple Classifiers × Multiple Feature Sets]
    ├── XGB (e-value, binary, taxonomy, location)
    ├── FM (e-value, binary, taxonomy, location)
    ├── SVM (e-value, binary, taxonomy, location)
    └── e.net (e-value, binary, taxonomy, location)
    ↓
First Level Predictions (8 predictions per protein per GO class)
    ↓
[LEVEL 2: Stacking Classifier]
    ├── XGB (best overall)
    ├── LR (logistic regression)
    ├── ANN (artificial neural network)
    └── XGB-ltr (learning to rank)
    ↓
Final Predictions
```

### Training Strategy
- **Class-specific models**: Separate classifier trained for each GO class
- **Stratified cross-validation**: 5-fold, stratified separately per class
- **Time constraint**: ~8 minutes per class to handle thousands of GO classes

---

## First-Level Classifiers

### 1. **XGB (Extreme Gradient Boosting)**
- Tree-based classifier
- Best overall performer at first level
- Especially benefits from taxonomy features
- Non-linear, handles high-dimensional data well

### 2. **FM (Factorization Machine)**
- Related to matrix factorization and SVM
- Very scalable for sparse, high-dimensional data
- Designed for binary data, performs best with binary features
- Second best overall at first level

### 3. **SVM (Support Vector Machine)**
- Radial basis function (RBF) kernel
- Required downsampling due to memory constraints
- Better with binary features than e-values

### 4. **e.net (Elastic Net Logistic Regression)**
- Linear model with elasticnet regularization
- Suitable for high-dimensional sparse data
- Better with e-value features than binary

### 5. **ANN (Artificial Neural Network)**
- Small architecture: 2 hidden layers of size 5
- ReLU activation, batch size 100, 3 iterations
- Limited use due to computational constraints

---

## Second-Level Classifiers

Used for stacking first-level predictions:

### 1. **XGB**
- Best second-level performer
- Increased number of trees vs first level

### 2. **LR (Logistic Regression with L2)**
- Non-sparse loss (unlike first level e.net)
- Good performance, especially for CC and BP

### 3. **ANN**
- One hidden layer of size 5
- Better performance at second level due to lower dimensionality

### 4. **XGB-ltr (XGB with Learning to Rank)**
- Pairwise ranking loss function

### Baseline Methods
- **Mean**: Simple average of first-level predictions
- **r.mean**: Rank-averaged mean

---

## Key Results

### First-Level Performance (In-house data, AUC-PR)

| Ontology | Best Classifier | Best Feature Set | Score |
|----------|----------------|------------------|-------|
| CC | XGB | taxonomy | 0.723 |
| MF | XGB | taxonomy | 0.874 |
| BP | XGB | taxonomy | 0.597 |

**Key Findings**:
- Taxonomy features give largest improvement (especially for XGB)
- Location features consistently beneficial
- Count features provide moderate improvement
- Different classifiers best for different GO classes (shown in heatmap analysis)

### Second-Level Performance (In-house data, AUC-PR)

| Ontology | Best Classifier | Best Features | Score | Improvement |
|----------|----------------|---------------|-------|-------------|
| CC | XGB | + taxonomy + additional | 0.781 | +8.0% |
| MF | XGB | + taxonomy + additional | 0.873 | -0.1% |
| BP | XGB | + taxonomy + additional | 0.655 | +9.7% |

**Key Findings**:
- Stacking improves CC and BP significantly
- MF already near ceiling at first level
- Taxonomy still beneficial at second level
- Additional features (sequence length, IPS coverage) provide marginal gains

### CAFA3 Competition Comparison

**Molecular Function (MF)**:
- XGB and ANN at level 2 rank #1 in Smin, nSmin, wFmax
- Outperformed ALL CAFA3 competition methods in 3/5 metrics

**Biological Process (BP)**:
- XGB at level 2 ranks #1 in wFmax, Smin, nSmin, TC-AUC
- Outperformed ALL CAFA3 methods in 4/5 metrics

**Cellular Component (CC)**:
- Results more variable across metrics
- Top performance in Smin and TC-AUC (#1-5)
- Lower ranks in other metrics (suggests evaluation data quality issues)

---

## Evaluation Metrics

### Term-Centric (TC)
- **TC-AUCPR**: Area under precision-recall curve, averaged across GO classes
- **TC-AUC**: Area under ROC curve, averaged across GO classes
- Insensitive to class size bias

### Protein-Centric (PrC)
- **Fmax**: Maximum F-score (biased metric, not reliable)
- **Smin**: Minimum semantic distance
- **nSmin**: Normalized semantic distance  
- **wFmax**: Weighted F-score

**Important**: Multiple metrics used in parallel because each has biases. Fmax in particular is known to be unreliable.

---

## Code Architecture Details

### Training Pipeline
1. Load sequences and GO annotations
2. Run InterProScan on sequences
3. Extract taxonomy from species IDs
4. Predict cellular localization (TargetP, WolfPSort)
5. Generate feature sets (binary, e-value, taxonomy, location, count)
6. Train first-level classifiers (separate model per GO class)
7. Generate first-level predictions via cross-validation
8. Train second-level classifiers on first-level predictions
9. Generate final predictions

### Cross-Validation Strategy
- **Stratified 5-fold CV per GO class**
- Ensures positive and negative samples in each fold
- Enables reliable evaluation of small classes (some have only 10+ members)
- **Limitation**: Cannot use multilabel classifiers (each class done separately)

### Computational Constraints
- Training time limited to ~8 minutes per GO class
- Total models for in-house data:
  - CC: 8,440 models (1,688 classes × 5 folds)
  - MF: 17,260 models (3,452 classes × 5 folds)
  - BP: 56,440 models (11,288 classes × 5 folds)

---

## Key Technologies

### Python Libraries
- **scikit-learn**: LR, SVM, ANN implementations
- **XGBoost**: Gradient boosting
- **pyfms**: Factorization machines
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### External Tools
- **InterProScan v5.38-76.0**: Feature extraction
- **TargetP**: Localization prediction
- **WolfPSort**: Localization prediction
- **NCBI Taxonomy**: Species hierarchy

### Data Formats
- **Input**: FASTA sequences, UniProt annotations
- **Intermediate**: Sparse matrices for high-dimensional features
- **Output**: GO class predictions with scores

---

## Notable Design Decisions

### 1. Class-Specific Everything
- Separate classifier per GO class (not multilabel)
- Separate cross-validation per class
- Allows handling of very small classes (10+ positives)
- Enables class-specific hyperparameter tuning

### 2. Two-Level Stacking
- First level: Diverse classifiers capture different patterns
- Second level: Learns optimal combination per GO class
- Shown to outperform simple averaging

### 3. Handling Class Imbalance
- Extreme imbalance: >20,000 negatives per positive in small classes
- Stratified CV ensures both classes in each fold
- Metrics chosen for imbalanced data (AUCPR > AUROC)

### 4. Positive/Negative Selection
- **Positives**: Only manually curated GO annotations
- **Negatives**: Proteins with other annotations but not target class
- Filters both false positives and false negatives

### 5. Feature Philosophy
- Start simple (binary, e-value)
- Add biological knowledge (taxonomy, location)
- Let classifiers extract complex patterns
- High dimensionality acceptable (up to ~150k features)

---

## Known Limitations

### 1. Cross-Validation Approach
- Class-specific CV prevents multilabel methods
- Cannot evaluate with protein-centric metrics on in-house data
- CAFA3 evaluation compensates for this

### 2. Computational Constraints
- 8-minute training limit per class
- Could achieve better results with more time
- Neural networks particularly affected

### 3. Deep Learning
- DNNs excluded due to:
  - Requires training all classes together (conflicts with class-specific CV)
  - Small class sizes problematic for DNNs
  - Computational constraints

### 4. Single Data Source
- Only uses IPS features (no sequence similarity, PPI, expression data)
- Intentional design to study IPS optimization
- Room for improvement by adding other sources

---

## Future Directions (from Discussion)

1. **Improved Cross-Validation**: Develop stratified multilabel CV for large GO datasets
2. **Additional Data Sources**: Sequence similarity, PPI networks, gene expression
3. **Hierarchical Structure**: Incorporate GO hierarchy relationships
4. **Third-Level Stacking**: Tested but showed no benefit (abandoned)
5. **Deep Learning**: Once CV issues resolved

---

## File Organization (Inferred)

```
project_A/
├── data/
│   ├── training/
│   │   ├── sequences.fasta
│   │   ├── go_annotations.tsv
│   │   └── interproscan_results/
│   └── evaluation/
│       └── cafa3_evaluation_set/
├── features/
│   ├── interproscan/
│   ├── taxonomy/
│   ├── localization/
│   └── feature_matrices/
├── models/
│   ├── level1/
│   │   ├── xgb/
│   │   ├── fm/
│   │   ├── svm/
│   │   └── enet/
│   └── level2/
│       ├── xgb/
│       ├── lr/
│       └── ann/
├── predictions/
│   ├── level1/
│   └── level2/
└── evaluation/
    ├── cross_validation_results/
    └── cafa3_comparison/
```

---

## Contact & Resources
- **Contact**: petri.toronen(AT)helsinki.fi
- **Web**: http://ekhidna2.biocenter.helsinki.fi/AFP/
- **Funding**: Novo Nordisk Foundation (NNF20OC0065157)
- **Computing**: Biocenter Finland, FCCI

---

## Summary for Claude Code

**What this project does**: Predicts protein functions (GO terms) from InterProScan features using optimized preprocessing and two-level classifier stacking.

**Key innovation**: Shows that proper feature engineering (taxonomy, location) and stacking of simple classifiers can outperform complex multi-source methods.

**Architecture**: First level has multiple classifiers (XGB, FM, SVM, e.net) trained on different feature sets. Second level combines these predictions with another classifier (XGB, LR, ANN).

**Best approach**: XGB with taxonomy features at level 1, then XGB stacking at level 2.

**Critical characteristic**: Everything is class-specific (one model per GO term), which is both a strength (handles small classes) and limitation (cannot use multilabel methods).
