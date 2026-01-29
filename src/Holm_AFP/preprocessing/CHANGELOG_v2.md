# IPS Feature Pipeline - Version 2.0 Changes

## What's New

Version 2.0 adds flexible input handling to support real-world InterProScan batch processing workflows.

## Major Changes

### 1. Directory + Pattern Matching Support ⭐

**Old (v1):**
```bash
# Only accepted directory path
python ips_feature_pipeline.py /path/to/ips_dir output.h5
# Would process *.tsv and *.txt files
```

**New (v2):**
```bash
# Directory with default pattern (*.out)
python ips_feature_pipeline.py --input-dir /path/to/ips_output output.h5

# Directory with custom pattern
python ips_feature_pipeline.py --input-dir /path/to/ips_output --pattern "EBI_GO*.out" output.h5

# Single file mode
python ips_feature_pipeline.py --input-file protein.out output.h5
```

**Benefits:**
- ✅ Process `.out` files (InterProScan default extension)
- ✅ Automatically exclude README files
- ✅ Custom patterns with wildcards (e.g., `"EBI_GO_no_preds_*.out"`)
- ✅ Multiple exclusion patterns
- ✅ Both directory and single-file modes

### 2. Automatic README Exclusion

**Problem:** InterProScan output directories often contain README files
**Solution:** Automatically excludes files matching `README*` pattern

```bash
# These files will be skipped automatically:
# - README
# - README.txt
# - README.md
```

### 3. Enhanced File Discovery

**Features:**
- Lists all files that will be processed
- Shows excluded files
- Validates files exist before processing
- Sorts files alphabetically for reproducibility

**Example output:**
```
Searching for IPS files in: /data/ips_output
  Pattern: *.out
  Exclude: README*
  Found 14 files matching pattern
  Excluded 1 files: README
  Will process 14 files:
    1. EBI_GO_no_preds_1.IPScan.out
    2. EBI_GO_no_preds_2.IPScan.out
    ...
```

### 4. Strict Format Validation

**Old behavior:** Skipped malformed lines with warnings
**New behavior:** Stops with clear error message if format is invalid

**Why?** Ensures data integrity. If one file has wrong format, better to fail fast than silently skip data.

**Example error:**
```
ERROR: Invalid IPS file format
═══════════════════════════════════════
Invalid IPS format in EBI_GO_3.IPScan.out, line 42:
  Missing field 8 (E-value)
  Expected at least 9 tab-separated fields, found 7
  Line content: sp|P12345|PROT_HUMAN 450 PF00069...
═══════════════════════════════════════
```

### 5. Improved Provenance Tracking

**New in HDF5 output:**
- `/input_files` dataset listing all processed files
- `n_input_files` metadata attribute
- Input file list in README

**Access processed files:**
```python
with h5py.File('output.h5', 'r') as f:
    input_files = f['input_files'][:].astype(str)
    print(f"Processed {len(input_files)} files:")
    for filepath in input_files:
        print(f"  - {filepath}")
```

### 6. Better Error Messages

All errors now include:
- Clear description of the problem
- File name and line number (if applicable)
- Suggestions for fixing

**Examples:**

**No files found:**
```
ERROR
═══════════════════════════════════════
No IPS files found in /data/ips_output
  Pattern: *.out
  Exclude: README*
  Make sure files match the pattern and are not excluded.
═══════════════════════════════════════
```

**Invalid path:**
```
ERROR
═══════════════════════════════════════
Input directory not found: /data/wrong_path
═══════════════════════════════════════
```

## Migration Guide

### From v1 to v2

**If you used:**
```bash
python ips_feature_pipeline.py /path/to/ips_dir output.h5
```

**Now use:**
```bash
python ips_feature_pipeline.py --input-dir /path/to/ips_dir output.h5
```

**If files were `.tsv` or `.txt`:**
```bash
python ips_feature_pipeline.py --input-dir /path/to/ips_dir --pattern "*.tsv" output.h5
```

### Compatibility Notes

- ✅ Same HDF5 output format (fully compatible)
- ✅ Same preprocessing modes
- ✅ Same feature names and matrix structure
- ⚠️ Command-line interface changed (old scripts need updating)
- ⚠️ Now stops on format errors (was: skip with warning)

## Use Cases

### Use Case 1: InterProScan Batch Processing
```bash
# You have:
# - Multiple .out files from InterProScan batches
# - A README file you want to skip

python ips_feature_pipeline.py \
    --input-dir /data/interproscan/output \
    --pattern "*.out" \
    --exclude "README*" \
    output.h5
```

### Use Case 2: Specific File Selection
```bash
# You have:
# - Files named EBI_GO_no_preds_1.out through EBI_GO_no_preds_14.out
# - Want only these specific files

python ips_feature_pipeline.py \
    --input-dir /data/interproscan/output \
    --pattern "EBI_GO_no_preds_*.out" \
    output.h5
```

### Use Case 3: Single Protein Analysis
```bash
# Testing on one protein before full run

python ips_feature_pipeline.py \
    --input-file /data/test_protein.out \
    test_output.h5
```

### Use Case 4: Mixed Extensions
```bash
# You have both .tsv and .out files

# Option A: Process only .out
python ips_feature_pipeline.py --input-dir /data --pattern "*.out" output_out.h5

# Option B: Process only .tsv
python ips_feature_pipeline.py --input-dir /data --pattern "*.tsv" output_tsv.h5
```

## Detailed Feature Comparison

| Feature | v1 | v2 |
|---------|----|----|
| Directory input | ✓ | ✓ |
| Single file input | ✗ | ✓ |
| Pattern matching | Fixed (*.tsv, *.txt) | Flexible (--pattern) |
| File exclusion | ✗ | ✓ (--exclude) |
| README auto-exclude | ✗ | ✓ |
| File discovery report | ✗ | ✓ |
| Provenance tracking | Partial | Complete |
| Error on bad format | ✗ | ✓ |
| Error messages | Basic | Detailed |
| Input validation | Basic | Comprehensive |

## Summary

**Version 2.0 is production-ready** with:
- Real-world batch processing support
- Better error handling and validation
- Complete provenance tracking
- Flexible file selection

**Migration is straightforward:**
Just add `--input-dir` or `--input-file` flag to your existing commands.

---

**Questions?** Check README.md and IPS_PROCESSING_GUIDE.md for complete documentation.
