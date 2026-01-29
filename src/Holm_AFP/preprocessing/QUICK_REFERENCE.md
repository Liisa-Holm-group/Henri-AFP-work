# IPS Feature Pipeline - Quick Reference Card

## One-Line Commands

```bash
# Standard processing (e-value)
python ips_feature_pipeline.py /path/to/ips_dir output.h5

# With counts
python ips_feature_pipeline.py /path/to/ips_dir output.h5 --preprocessing counts

# With location
python ips_feature_pipeline.py /path/to/ips_dir output.h5 --preprocessing location

# Test/inspect output
python example_usage.py output.h5
```

## Quick Load Pattern

```python
import h5py, scipy.sparse as sp

with h5py.File('output.h5', 'r') as f:
    X = sp.csr_matrix((f['features/csr/data'][:], 
                       f['features/csr/indices'][:], 
                       f['features/csr/indptr'][:]),
                      shape=f['features'].attrs['shape'])
    features = f['feature_names'][:].astype(str)
    proteins = f['protein_names'][:].astype(str)
```

## Preprocessing Methods Cheat Sheet

| Method | Features Generated | Use When |
|--------|-------------------|----------|
| binary | Binary only | Only need presence/absence |
| e-value | E-values + binary | Standard processing ⭐ |
| counts | e-value + counts | Care about domain repeats |
| location | e-value + N/M/C | Need topology info |
| location_b | e-value + center | Simple position matters |
| clusters | e-value + clusters | Want domain families |

⭐ = Recommended default

## Output Statistics Reference

| Metric | Where to Find |
|--------|---------------|
| Number of proteins | `f['metadata'].attrs['n_proteins']` |
| Number of features | `f['metadata'].attrs['n_resulting_features']` |
| Non-zero values | `f['metadata'].attrs['n_nonzero_values']` |
| Sparsity | `f['metadata'].attrs['sparsity']` |
| Matrix shape | `f['features'].attrs['shape']` |

## Common Operations

### Get features for one protein
```python
idx = list(proteins).index('P12345')
protein_feats = X[idx, :].toarray().flatten()
```

### Select feature type
```python
import numpy as np
mask = np.array(['e_value' in n for n in features])
X_subset = X[:, mask]
```

### Find top features
```python
nonzero = np.nonzero(protein_feats)[0]
top5 = nonzero[np.argsort(protein_feats[nonzero])[-5:][::-1]]
for i in top5:
    print(f"{features[i]}: {protein_feats[i]:.4f}")
```

## Feature Name Prefixes

| Prefix | Meaning |
|--------|---------|
| ipscan_e_value_ | E-value transformed features |
| ipscan_binary_ | Binary presence features |
| ipscan_count_ | Occurrence count features |
| ipscan_loc_start_ | N-terminal features |
| ipscan_loc_middle_ | Middle region features |
| ipscan_loc_end_ | C-terminal features |
| ipscan_center_pos_ | Center position features |
| ipscan_cluster_ | IPS cluster features |

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| No files found | Check .tsv/.txt extension |
| Memory error | Process in batches |
| String decode error | Use `.astype(str)` |
| Wrong shape | Check preprocessing method |

## File Locations

```
outputs/
├── README.md                    ← Full documentation
├── IPS_PROCESSING_GUIDE.md      ← Detailed guide
├── ips_feature_pipeline.py      ← Main script
└── example_usage.py             ← Examples
```

## Performance Expectations

| Dataset Size | Processing Time | File Size |
|--------------|----------------|-----------|
| 10K proteins | ~15 seconds | ~5-10 MB |
| 50K proteins | ~45 seconds | ~25-50 MB |
| 120K proteins | ~2 minutes | ~50-150 MB |

## Key Metadata Fields

```python
with h5py.File('output.h5', 'r') as f:
    meta = dict(f['metadata'].attrs)
    
    # Useful fields:
    meta['n_proteins']              # Number of proteins
    meta['n_unique_ips_features']   # Raw IPS features
    meta['n_resulting_features']    # Processed features
    meta['n_nonzero_values']        # Sparse entries
    meta['sparsity']                # Fraction of zeros
    meta['preprocessing_mode']      # Method used
    meta['total_time_sec']          # Processing time
```

## Integration with Existing Code

```python
# Replace load_ipscan() with:
def load_ips_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = sp.csr_matrix((f['features/csr/data'][:],
                           f['features/csr/indices'][:],
                           f['features/csr/indptr'][:]),
                          shape=f['features'].attrs['shape'])
        names = f['feature_names'][:].astype(str)
    return X, names
```

## Validation Checklist

✅ Output file exists  
✅ File size reasonable (not 0, not huge)  
✅ Matrix shape = (n_proteins, n_features)  
✅ Sparsity > 95% (typical)  
✅ Summary report generated  
✅ Feature names match preprocessing method  

---

**Quick Help:** `python ips_feature_pipeline.py --help`  
**Full Docs:** See `README.md` and `IPS_PROCESSING_GUIDE.md`  
**Examples:** Run `python example_usage.py output.h5`
