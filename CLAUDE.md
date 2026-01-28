# Henri-AFP - Automated Function Prediction using InterProScan

## Project Overview

This is a protein function prediction system that predicts Gene Ontology (GO) annotations from InterProScan (IPS) features using a two-level classifier stacking approach. The work is associated with the bioRxiv preprint from August 2022.

**Authors:** Henri Tiittanen, Liisa Holm, Petri Törönen (University of Helsinki)

**Key Achievement:** Outperformed all CAFA3 competition participants in most evaluation metrics using only InterProScan features.

## Repository Structure

```
Henri-AFP-work/
├── docs/                          # Documentation
│   ├── IPScan_project_summary.md  # Detailed technical summary (read this first)
│   └── henri_proj_preprint.pdf    # Research preprint
├── src/Holm_AFP/                   # Main source code
│   ├── models.py                  # Core ML models (XGB, SVM, FM, ElasticNet, ANN)
│   ├── ipscan_experiment.py       # Cross-validation experiment runner
│   ├── stacking_experiment.py     # Second-level stacking experiments
│   ├── train_models.py            # Model training CLI
│   ├── predict.py                 # Prediction script
│   ├── process_results.py         # Result processing utilities
│   └── generate_data/             # Data generation scripts
│       ├── in-house_data/         # In-house dataset processing
│       ├── cafa3_data/            # CAFA3 challenge dataset processing
│       └── stacking_data/         # Stacking-specific data generation
└── pyproject.toml                 # Python dependencies (Python 3.11+)
```

## Technology Stack

- **Python 3.11+**
- **ML Libraries:** xgboost, scikit-learn, lightgbm, pyfms (factorization machines)
- **Data:** pandas, numpy, scipy (sparse matrices), h5py (HDF5 storage)
- **External Tools:** InterProScan v5, TargetP, WolfPSort, NCBI Taxonomy

## Architecture

### Two-Level Classifier Stacking

```
Input Features (InterProScan + Taxonomy + Localization)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
   XGB             SVM             FM        ← Level 1 (per GO class)
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
              XGB Stacking              ← Level 2 (combines predictions)
                    ↓
           Final Predictions
```

### Feature Types (6 variants)

1. **e-value**: Log-transformed InterProScan e-values
2. **binary**: Presence/absence encoding of IPS features
3. **taxonomy**: Base + species taxonomy vectors
4. **location**: Base + sequence position proportions
5. **count**: Base + feature occurrence counts
6. **cluster**: Base + IPS cluster information

## Key Files

| File | Purpose |
|------|---------|
| `models.py` | All classifier implementations and training infrastructure |
| `ipscan_experiment.py` | Run CV experiments for different feature types |
| `stacking_experiment.py` | Second-level stacking logic |
| `process_results.py` | CAFA format conversion, HDF5 utilities |
| `train_models.py` | CLI for training first-level models |
| `predict.py` | Generate predictions from trained models |

## Key Classes in models.py

- **`ModelTrainer`**: Trains separate models for each GO class
- **`Predictor`**: Makes predictions using trained models
- **`StackingModelTrainer`**: Trains second-level stacking models
- **`StackingPredictor`**: Makes stacking predictions

## Running Experiments

### First-Level Experiments
```bash
cd src/Holm_AFP
./ipscan_experiments.sh
```

### Training Models
```bash
python train_models.py --data_path <path> --model_type xgb --feature_type e-value
```

### Generating Predictions
```bash
python predict.py --model_path <models> --data_path <data> --output <out.h5>
```

## Data Scale

| Ontology | Proteins | GO Classes | Models (5-fold CV) |
|----------|----------|------------|-------------------|
| CC       | 577K     | 1,688      | 8,440            |
| MF       | 637K     | 3,452      | 17,260           |
| BP       | 666K     | 11,288     | 56,440           |

## Important Design Decisions

1. **Class-specific models**: One model trained per GO class (allows handling very small classes with 10+ positives)
2. **Sparse matrices**: Uses scipy.sparse throughout for memory efficiency (features can be 32K-150K dimensions)
3. **HDF5 storage**: Predictions stored in HDF5 format for efficient I/O
4. **Stratified 5-fold CV**: Cross-validation done separately per class to handle class imbalance
5. **8-minute timeout**: Per-class training limited to prevent runaway computation

## Common Commands

```bash
# Install dependencies
pip install -e .

# Run unit tests (if available)
pytest tests/

# Generate data for in-house experiments
cd src/Holm_AFP/generate_data/in-house_data
./generate_features.sh

# Generate CAFA3 data
cd src/Holm_AFP/generate_data/cafa3_data
./generate_cafa3_data.sh
```

## Contact

- **Email:** petri.toronen@helsinki.fi
- **Website:** http://ekhidna2.biocenter.helsinki.fi/AFP/
- **Preprint:** https://www.biorxiv.org/content/10.1101/2022.08.10.503467v1
