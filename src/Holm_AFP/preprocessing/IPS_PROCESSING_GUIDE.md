# IPS Feature Processing Pipeline - Usage Guide

## Overview

This pipeline processes InterProScan (IPS) output files and generates feature matrices with multiple preprocessing options. All outputs are saved in HDF5 format with comprehensive metadata.

## Installation

Required packages:
```bash
pip install numpy scipy scikit-learn h5py
```

## Quick Start

```bash
# Basic usage with default e-value preprocessing
python ips_feature_pipeline.py /path/to/ips_dir output.h5

# Use different preprocessing method
python ips_feature_pipeline.py /path/to/ips_dir output.h5 --preprocessing counts

# Process all variants
for method in binary e-value counts location location_b clusters; do
    python ips_feature_pipeline.py /path/to/ips_dir output_${method}.h5 --preprocessing $method
done
```

## Preprocessing Methods

### 1. **binary** - Binary Encoding
- Converts all IPS features to 0/1 values
- Reports only presence/absence of features
- **Use case**: When you only care about which features are present, not their strength

### 2. **e-value** (DEFAULT) - E-value Transformation
- E-value features: Y = -log(E-value)
- Binary features: For features without E-values
- Selects strongest (smallest) E-value when feature occurs multiple times
- **Use case**: Standard processing that captures feature significance

### 3. **counts** - With Occurrence Counts
- All features from `e-value` preprocessing
- PLUS: Count of non-overlapping feature occurrences
- Uses 20% overlap threshold
- **Use case**: When feature frequency matters (e.g., repeated domains)

### 4. **location** - 3-Part Localization
- All features from `e-value` preprocessing
- PLUS: Feature proportions in:
  - N-terminal region
  - Middle region
  - C-terminal region
- For proteins <300aa: divide into thirds
- For proteins ≥300aa: fixed 100aa tails
- **Use case**: When protein topology and domain location are important

### 5. **location_b** - Center Position
- All features from `e-value` preprocessing
- PLUS: Relative center position (0-1) of strongest feature
- Scaled by sequence length
- **Use case**: Simple position-aware encoding

### 6. **clusters** - With IPS Clusters
- All features from `e-value` preprocessing
- PLUS: IPS cluster scores
- For each cluster, returns strongest feature score from all linked features
- **Use case**: Capturing higher-level domain family relationships

## Input Format

The pipeline expects a directory containing InterProScan TSV files with the following format:

```
Column 0:  sp|P12345|PROT_HUMAN (protein ID, supports sp|ID|NAME format)
Column 2:  450                   (protein length)
Column 3:  PF00069               (InterPro/Pfam ID)
Column 4:  Protein kinase domain (feature name)
Column 6:  50                    (start position)
Column 7:  300                   (stop position)
Column 8:  1.2e-45               (E-value, or '-' if none)
Column 11: IPR000719             (cluster ID, optional)
```

The pipeline will process all `.tsv` and `.txt` files in the specified directory.

## Output Format

### HDF5 File Structure

```
output.h5
├── /features/csr/
│   ├── data      (float32 array - sparse matrix values)
│   ├── indices   (int32 array - column indices)
│   └── indptr    (int32 array - row pointers)
├── /features [attributes: shape, dtype, format]
├── /feature_names (string array - feature names)
├── /protein_names (string array - protein IDs)
├── /metadata/ [attributes: all statistics and parameters]
└── /readme (human-readable description)
```

### Summary Report

A text file `output_summary.txt` is automatically generated with:
- Number of proteins processed
- Number of unique IPS features (raw)
- Number of resulting features (processed)
- Number of non-zero values
- Processing time breakdown
- Feature type distribution
- Sample data

## Loading the Data

### Python Example

```python
import h5py
import scipy.sparse as sp
import numpy as np

# Load HDF5 file
with h5py.File('output.h5', 'r') as f:
    # Load sparse matrix
    data = f['features/csr/data'][:]
    indices = f['features/csr/indices'][:]
    indptr = f['features/csr/indptr'][:]
    shape = f['features'].attrs['shape']
    
    # Reconstruct sparse matrix
    X = sp.csr_matrix((data, indices, indptr), shape=shape)
    
    # Load names
    feature_names = f['feature_names'][:].astype(str)
    protein_names = f['protein_names'][:].astype(str)
    
    # Load metadata
    n_proteins = f['metadata'].attrs['n_proteins']
    preprocessing = f['metadata'].attrs['preprocessing_mode']
    
    # Read README
    readme = f['readme'][()].decode('utf-8')
    print(readme)

print(f"Loaded matrix: {X.shape}")
print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
print(f"First protein: {protein_names[0]}")
print(f"First feature: {feature_names[0]}")
```

### Working with Specific Proteins

```python
# Get features for a specific protein
protein_idx = list(protein_names).index('P12345')
protein_features = X[protein_idx, :].toarray().flatten()

# Find non-zero features
nonzero_idx = np.nonzero(protein_features)[0]
for idx in nonzero_idx:
    print(f"{feature_names[idx]}: {protein_features[idx]:.4f}")
```

### Feature Selection

```python
# Select only E-value features
e_value_mask = np.array(['e_value' in name for name in feature_names])
X_evalue = X[:, e_value_mask]

# Select only binary features
binary_mask = np.array(['binary' in name for name in feature_names])
X_binary = X[:, binary_mask]

# Select features for specific domains
pfam_mask = np.array(['PF00069' in name for name in feature_names])
X_pfam = X[:, pfam_mask]
```

## Performance Characteristics

### Typical Statistics (120,000 proteins)

| Preprocessing | Features | Non-zeros | Sparsity | File Size |
|--------------|----------|-----------|----------|-----------|
| binary       | ~15,000  | ~12M      | 99.3%    | ~50 MB    |
| e-value      | ~30,000  | ~18M      | 99.5%    | ~75 MB    |
| counts       | ~45,000  | ~24M      | 99.6%    | ~100 MB   |
| location     | ~75,000  | ~36M      | 99.6%    | ~150 MB   |
| location_b   | ~45,000  | ~24M      | 99.6%    | ~100 MB   |
| clusters     | ~35,000  | ~20M      | 99.5%    | ~85 MB    |

*Actual numbers depend on your specific dataset*

### Processing Time

- Parsing: ~10-30 seconds
- Transformation: ~5-15 seconds  
- Matrix construction: ~20-60 seconds
- **Total: ~1-2 minutes for typical datasets**

## Integration with Existing Pipeline

This module can replace the IPS processing portion of the existing Henri-AFP pipeline:

```python
# Old way (from load_data.py)
from load_data import load_ipscan
features, feature_names = load_ipscan(ipr_dir, sequences, pretrained, output_path)

# New way
import h5py
import scipy.sparse as sp

with h5py.File('ips_features.h5', 'r') as f:
    data = f['features/csr/data'][:]
    indices = f['features/csr/indices'][:]
    indptr = f['features/csr/indptr'][:]
    shape = f['features'].attrs['shape']
    
    features = sp.csr_matrix((data, indices, indptr), shape=shape)
    feature_names = f['feature_names'][:].astype(str)
    protein_names = f['protein_names'][:].astype(str)
```

## Troubleshooting

### Common Issues

1. **"No TSV/TXT files found"**
   - Check that your directory contains `.tsv` or `.txt` files
   - Verify the file extensions

2. **"Malformed line" warnings**
   - Some IPS lines may have missing fields
   - These are skipped automatically
   - Check your IPS output format

3. **Memory errors**
   - For very large datasets (>500K proteins), process in batches
   - Consider using the `counts` preprocessing (most compact)

4. **Feature name encoding issues**
   - All feature names are stored as UTF-8 strings
   - Use `.astype(str)` when loading from HDF5

## Advanced Usage

### Custom Processing

```python
from ips_feature_pipeline import (
    process_ipr_data, 
    build_feature_matrix,
    save_to_hdf5
)

# Load raw data
ipr_data = process_ipr_data('/path/to/ips_dir')

# Build matrix
matrix, features, proteins, stats = build_feature_matrix(
    '/path/to/ips_dir',
    preprocessing='e-value'
)

# Custom post-processing
# ... your transformations ...

# Save
save_to_hdf5('output.h5', matrix, features, proteins, stats, 
             '/path/to/ips_dir', 'custom')
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple datasets
for dataset in dataset1 dataset2 dataset3; do
    python ips_feature_pipeline.py \
        data/${dataset}/ips_dir \
        results/${dataset}_features.h5 \
        --preprocessing e-value
done
```

## Contact & Support

For questions or issues:
1. Check the README in the HDF5 file: `f['readme'][()]`
2. Review the summary report
3. Refer to the Henri-AFP project documentation
4. Check feature names for expected prefixes:
   - `ipscan_e_value_*`: E-value features
   - `ipscan_binary_*`: Binary features
   - `ipscan_count_*`: Count features
   - `ipscan_loc_start_*`: N-terminal features
   - `ipscan_loc_middle_*`: Middle region features
   - `ipscan_loc_end_*`: C-terminal features
   - `ipscan_center_pos_*`: Center position features
   - `ipscan_cluster_*`: Cluster features

## References

- Henri-AFP Project: Tiittanen et al. (manuscript in preparation)
- InterProScan: Jones et al. (2014) Bioinformatics
- E-values: HMMER documentation (http://hmmer.org/)
