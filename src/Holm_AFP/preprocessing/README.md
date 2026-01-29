# InterProScan Feature Processing Pipeline

**Complete Python pipeline for processing InterProScan features with HDF5 output**

## 📦 Package Contents

This package contains three files:

1. **ips_feature_pipeline.py** - Main processing pipeline
2. **IPS_PROCESSING_GUIDE.md** - Comprehensive documentation
3. **example_usage.py** - Usage examples and demonstrations

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn h5py

# Run the pipeline (default: e-value preprocessing)
python ips_feature_pipeline.py /path/to/ips_directory output.h5

# Run with different preprocessing
python ips_feature_pipeline.py /path/to/ips_directory output.h5 --preprocessing counts

# View examples and test output
python example_usage.py output.h5
```

## 📊 What This Pipeline Does

The pipeline processes InterProScan output files and generates:

✅ **Sparse feature matrices** in efficient HDF5 format  
✅ **6 preprocessing options** for different analysis needs  
✅ **Comprehensive metadata** including all processing parameters  
✅ **Detailed summary reports** with statistics and breakdowns  
✅ **Complete provenance** tracking of all inputs and settings

## 🎯 Six Preprocessing Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **binary** | Binary encoding (0/1) | Presence/absence only |
| **e-value** ⭐ | -log(E-value) transformation | Standard processing (DEFAULT) |
| **counts** | E-values + occurrence counts | Repeated domain analysis |
| **location** | E-values + N/M/C-term split | Topology-aware features |
| **location_b** | E-values + center position | Simple position encoding |
| **clusters** | E-values + IPS clusters | Domain family relationships |

⭐ = Recommended default

## 📥 Input Requirements

**Directory structure:**
```
ips_directory/
├── protein_batch1.tsv
├── protein_batch2.tsv
└── protein_batch3.tsv
```

**File format:** InterProScan TSV output with columns:
- Column 0: Protein ID (sp|P12345|NAME format supported)
- Column 2: Protein length
- Column 3: InterPro/Pfam ID
- Column 4: Feature name
- Column 6: Start position
- Column 7: Stop position
- Column 8: E-value (or '-')
- Column 11: Cluster ID (optional)

## 📤 Output Files

### 1. HDF5 Feature Matrix (`output.h5`)

```
output.h5
├── /features/csr/          # Sparse matrix (CSR format)
│   ├── data                # Float32 values
│   ├── indices             # Int32 column indices
│   └── indptr              # Int32 row pointers
├── /feature_names          # Array of feature names
├── /protein_names          # Array of protein IDs
├── /metadata/              # All processing parameters
└── /readme                 # Human-readable description
```

**Key features:**
- Compressed storage (gzip level 6)
- Float32 precision for values
- Variable-length UTF-8 strings for names
- Complete metadata and provenance

### 2. Summary Report (`output_summary.txt`)

Contains:
- Processing parameters
- Dataset statistics (proteins, features, sparsity)
- Feature type breakdown
- Processing time analysis
- Sample data preview

## 📊 Expected Output Statistics

For typical proteome (120,000 proteins):

| Metric | Typical Value |
|--------|---------------|
| Proteins | ~120,000 |
| Unique IPS features | ~15,000-20,000 |
| Resulting features | 30,000-75,000 (depends on method) |
| Non-zero values | 12-36 million |
| Sparsity | 99.3-99.7% |
| File size | 50-150 MB |
| Processing time | 1-2 minutes |

## 💻 Loading and Using the Output

### Basic Loading

```python
import h5py
import scipy.sparse as sp

# Load HDF5 file
with h5py.File('output.h5', 'r') as f:
    # Reconstruct sparse matrix
    data = f['features/csr/data'][:]
    indices = f['features/csr/indices'][:]
    indptr = f['features/csr/indptr'][:]
    shape = f['features'].attrs['shape']
    
    X = sp.csr_matrix((data, indices, indptr), shape=shape)
    
    # Load identifiers
    feature_names = f['feature_names'][:].astype(str)
    protein_names = f['protein_names'][:].astype(str)
    
    # Access metadata
    preprocessing = f['metadata'].attrs['preprocessing_mode']
    n_proteins = f['metadata'].attrs['n_proteins']

print(f"Matrix: {X.shape}, Sparsity: {1 - X.nnz/(X.shape[0]*X.shape[1]):.2%}")
```

### Querying Specific Proteins

```python
# Get features for protein P12345
protein_idx = list(protein_names).index('P12345')
protein_features = X[protein_idx, :].toarray().flatten()

# Find active features
active_idx = np.nonzero(protein_features)[0]
for idx in active_idx[:5]:
    print(f"{feature_names[idx]}: {protein_features[idx]:.4f}")
```

### Feature Selection

```python
# Select only E-value features
e_value_mask = np.array(['e_value' in name for name in feature_names])
X_evalue = X[:, e_value_mask]

# Select specific domains
pfam_mask = np.array(['PF00069' in name for name in feature_names])
X_pfam = X[:, pfam_mask]
```

## 🔧 Integration with Existing Pipeline

This pipeline can replace IPS processing in Henri-AFP:

```python
# Instead of:
from load_data import load_ipscan
features, names = load_ipscan(ipr_dir, sequences, pretrained, output_path)

# Use:
with h5py.File('ips_features.h5', 'r') as f:
    data = f['features/csr/data'][:]
    indices = f['features/csr/indices'][:]
    indptr = f['features/csr/indptr'][:]
    shape = f['features'].attrs['shape']
    
    features = sp.csr_matrix((data, indices, indptr), shape=shape)
    names = f['feature_names'][:].astype(str)
    sequences = f['protein_names'][:].astype(str)
```

## 📋 Command-Line Options

```bash
python ips_feature_pipeline.py --help

positional arguments:
  ips_dir              Directory containing InterProScan TSV files
  output               Output HDF5 file path

optional arguments:
  --preprocessing {binary,e-value,counts,location,location_b,clusters}
                       Preprocessing method (default: e-value)
```

## 🎓 Understanding Preprocessing Methods

### E-value Method (Default)
Best for: General protein function prediction
- Transforms E-values: Y = -log(E-value)
- Captures feature significance
- Binary encoding for features without E-values
- Selects strongest signal when features occur multiple times

### Counts Method
Best for: Detecting domain repeats
- All e-value features PLUS
- Count of non-overlapping occurrences
- Uses 20% overlap threshold

### Location Methods
Best for: Topology-aware analysis
- **location**: Splits signal into N-terminal/Middle/C-terminal
  - Proteins <300aa: divide into thirds
  - Proteins ≥300aa: 100aa tails
- **location_b**: Simpler relative center position (0-1)

### Clusters Method
Best for: Domain family analysis
- Groups features by InterPro clusters
- Returns strongest signal per cluster

## ⚙️ Advanced Usage

### Batch Processing Multiple Datasets

```bash
#!/bin/bash
for dataset in dataset1 dataset2 dataset3; do
    python ips_feature_pipeline.py \
        data/${dataset}/ips_dir \
        results/${dataset}_features.h5 \
        --preprocessing e-value
done
```

### Processing All Methods

```bash
for method in binary e-value counts location location_b clusters; do
    python ips_feature_pipeline.py \
        ips_dir \
        output_${method}.h5 \
        --preprocessing $method
done
```

## 🐛 Troubleshooting

**Problem:** "No TSV/TXT files found"  
**Solution:** Check directory contains `.tsv` or `.txt` files

**Problem:** "Malformed line" warnings  
**Solution:** Some lines may have missing fields; they're automatically skipped

**Problem:** Memory errors  
**Solution:** For very large datasets (>500K proteins), process in batches

**Problem:** Feature names not loading correctly  
**Solution:** Use `.astype(str)` when loading from HDF5

## 📚 Documentation

- **IPS_PROCESSING_GUIDE.md**: Complete documentation with examples
- **example_usage.py**: Runnable examples demonstrating all features
- **HDF5 /readme dataset**: Built-in documentation in output file

## 🔍 Quality Checks

The pipeline includes built-in validation:
- ✅ Verifies input directory exists
- ✅ Checks for TSV/TXT files
- ✅ Validates matrix dimensions
- ✅ Computes and reports statistics
- ✅ Generates comprehensive summary

## 📈 Performance

Typical performance on standard hardware:
- **Parsing:** 10-30 seconds
- **Transformation:** 5-15 seconds  
- **Matrix construction:** 20-60 seconds
- **Total:** 1-2 minutes for 100K-150K proteins

## 🎯 Key Features

1. **Reproducible:** All parameters saved in metadata
2. **Efficient:** Compressed sparse storage
3. **Flexible:** 6 preprocessing options
4. **Complete:** Includes all relevant information
5. **Documented:** Built-in README and summary reports
6. **Validated:** Comprehensive statistics and checks

## 📞 Support

For questions:
1. Check the README in HDF5 file: `f['readme'][()]`
2. Review the summary report
3. Run example_usage.py for demonstrations
4. Consult IPS_PROCESSING_GUIDE.md

## 🏆 Advantages Over Previous Implementation

| Feature | Old (`load_data.py`) | New Pipeline |
|---------|---------------------|--------------|
| Output format | NPZ (separate files) | HDF5 (single file) |
| Metadata | Minimal | Comprehensive |
| Documentation | External | Built-in |
| Validation | None | Full statistics |
| Preprocessing | Fixed | 6 options |
| Provenance | Manual | Automatic |

## 📝 Citation

If you use this pipeline, please cite:
- Henri-AFP Project: Tiittanen et al. (manuscript in preparation)
- InterProScan: Jones et al. (2014) Bioinformatics

---

**Version:** 1.0  
**Date:** 2026-01-28  
**Author:** Petri Toronen (University of Helsinki)  
**License:** Same as Henri-AFP project
