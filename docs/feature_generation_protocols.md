# Feature Generation Protocols

This document describes the exact protocols for generating the different feature types from InterProScan (IPS) output in the Henri-AFP protein function prediction system.

## Overview

The system transforms raw InterProScan output into five distinct feature representations:

| Feature Type | Values | Description |
|--------------|--------|-------------|
| E-value | Continuous | Log-transformed InterProScan e-values |
| Binary | 0/1 | Presence/absence of IPS features |
| Cluster | 0/1 | InterPro domain cluster membership |
| Count | Integer | Non-overlapping feature occurrence counts |
| Location | Continuous (0-1) | Sequence position proportions (N-term/middle/C-term) |

---

## 1. E-Value Features

**Source files:** `generate_data/in-house_data/load_data.py:54-85`

### Protocol

E-value features apply a negative log transformation to raw InterProScan e-values:

```
feature_value = -log(e-value)
```

### Algorithm

```python
def process_ipr(ipr_list):
    for item in ipr_list:
        for key in item.keys():
            if item[key][2] != '-':  # Has e-value
                float_val = float(item[key][2])
                if float_val == 0:
                    float_val = 7.5e-305  # Replace 0 with minimum value
                e_dict[key] = -np.log(float_val)  # Natural log transform
```

### Details

- **Input:** Raw e-values from InterProScan output (field[8])
- **Transformation:** Negative natural logarithm
- **Zero handling:** E-value of 0 is replaced with `7.5e-305` to avoid `log(0) = -∞`
- **Output type:** Continuous-valued sparse matrix
- **Feature prefix:** `ipscan_e_value`
- **Dimensionality:** ~32,000-36,000 features

### Interpretation

Lower e-values (stronger hits) produce higher transformed values:

| Raw E-value | Transformed Value | Interpretation |
|-------------|-------------------|----------------|
| 1e-45 | 103.6 | Strong signal |
| 1e-10 | 23.0 | Moderate signal |
| 0.01 | 4.6 | Weak signal |

---

## 2. Binary Features

**Source files:** `ipscan_experiment.py:40-51`

### Protocol

Binary features convert all non-zero e-value features to 1:

```
feature_value = 1 if e-value feature > 0 else 0
```

### Algorithm

```python
if feature_type == 'binary':
    index = np.array([i for i, n in enumerate(names)
                      if ('e_value' in n) or ('e_missing' in n)])
    res = features[:, index]
    res[res > 0] = 1  # Convert all non-zero values to 1
```

### Details

- **Input:** E-value features (after log transformation)
- **Transformation:** Threshold at zero
- **Output type:** Binary sparse matrix (0 or 1 only)
- **Features included:** Both `e_value` and `e_missing` columns
- **Dimensionality:** ~32,000-36,000 features (same as e-value)

### Rationale

Binary features represent simple presence/absence and work better with certain models (SVM, Factorization Machines) that don't benefit from continuous signal strength.

---

## 3. Cluster Features

**Source files:** `generate_data/in-house_data/data_processing.py:104-231`

### Protocol

Cluster features encode membership in InterPro domain clusters, grouping related domains/motifs:

```
feature_value = 1 if protein contains any feature from cluster else 0
```

### Algorithm

```python
def process_ipr_line2(line, data):
    fields = line.split('\t')
    feature = f'{fields[3]}|{fields[4]}'  # InterPro ID | Name
    cluster = fields[11]  # InterPro cluster (optional)

    # Store cluster for each feature
    data[name][feature].append((e_value, cluster, start, stop, length))

# Later processing extracts cluster from first hit
for t1, sequence in data.items():
    for t2, feature in sequence.items():
        cluster = feature[0][1]  # Get cluster from first hit
```

### Details

- **Input:** InterProScan field[11] containing cluster IDs (e.g., IPR000719)
- **Transformation:** Binary encoding via `MultiLabelBinarizer`
- **Missing handling:** Features without clusters assigned 'missing' label
- **Output type:** Binary sparse matrix
- **Feature prefix:** `ipscan_cluster`
- **Dimensionality:** Smaller than base features (clusters group features)

### Rationale

Clusters group related InterPro features, reducing dimensionality while preserving information about domain families. Useful when exact feature identity is less important than functional grouping.

---

## 4. Count Features

**Source files:** `generate_data/in-house_data/data_processing.py:168-184`

### Protocol

Count features record the number of **non-overlapping** occurrences of each InterProScan feature within a protein:

```
feature_value = number of non-overlapping domain hits
```

### Algorithm

```python
def non_overlapping(feature, max_overlap=0.2):
    """Extract occurrences where overlap is below max_overlap."""
    features = []
    t = sorted(feature, key=itemgetter(2))  # Sort by start position
    j = 0
    features.append(t[j])
    for i in range(1, len(t)):
        # Keep hit only if overlap with previous kept hit is < 20%
        if (1 - max_overlap) * t[j][3] < t[i][2]:
            features.append(t[i])
            j = i
    return features

def calc_counts(feature):
    return len(non_overlapping(feature, 0.2))
```

### Details

- **Input:** All hits of a feature with positions: `(e_value, cluster, start, stop, length)`
- **Overlap threshold:** 20% (hits overlapping more than 20% are merged)
- **Overlap calculation:** `(1 - 0.2) * previous_stop > current_start`
- **Output type:** Integer-valued sparse matrix
- **Feature prefix:** `ipscan_count`
- **Dimensionality:** ~65,000-73,000 features

### Example

```
Protein with 3 Pfam domains at positions:
  [10-50], [45-100], [150-200]

[10-50] and [45-100] overlap by >20% → only first is kept
[150-200] doesn't overlap significantly → kept

Count = 2 (2 non-overlapping occurrences)
```

### Rationale

Captures repeated domain architecture (e.g., multiple tandem repeats) which can be biologically significant for function prediction.

---

## 5. Location Features

**Source files:** `generate_data/in-house_data/data_processing.py:185-207`

### Protocol

Location features encode the proportional distribution of domain hits across three sequence regions (N-terminal, middle, C-terminal):

```
[N-terminal_proportion, middle_proportion, C-terminal_proportion]
```

### Region Definitions

| Protein Length | N-terminal | Middle | C-terminal |
|----------------|------------|--------|------------|
| < 300 aa | First 1/3 | Middle 1/3 | Last 1/3 |
| ≥ 300 aa | First 100 aa | aa 100 to (length-100) | Last 100 aa |

### Algorithm

```python
def calc_localisation(feature, max_tail_len=100):
    no = non_overlapping(feature, 0.5)  # 50% overlap threshold
    length = no[0][4]  # Protein length
    result = [0, 0, 0]  # [N-terminal, Middle, C-terminal]

    # Define region boundaries
    if length < 300:
        max_tail_len = length // 3
        start = set(range(max_tail_len))
        middle = set(range(max_tail_len, 2 * max_tail_len))
        end = set(range(2 * max_tail_len, length))
    else:
        start = set(range(max_tail_len))           # First 100 aa
        middle = set(range(max_tail_len, length - max_tail_len))
        end = set(range(length - max_tail_len, length))  # Last 100 aa

    # Calculate proportions
    for n in no:
        values = set(range(n[2], n[3]))  # Positions of this hit
        result[0] += len(values.intersection(start)) / len(values)
        result[1] += len(values.intersection(middle)) / len(values)
        result[2] += len(values.intersection(end)) / len(values)

    # Normalize
    return [r / sum(result) for r in result]
```

### Details

- **Input:** Non-overlapping hits (using 50% overlap threshold, different from counts)
- **Output:** Three normalized proportions summing to 1.0
- **Feature prefixes:** `ipscan_start`, `ipscan_middle`, `ipscan_end`
- **Dimensionality:** ~130,000-147,000 features (~3-4x base due to 3 values per feature)

### Example

```
Protein length = 400 aa
  N-terminal region: aa 0-100
  Middle region: aa 100-300
  C-terminal region: aa 300-400

Domain found at positions 80-120 (40 aa total):
  In N-terminal: 20 aa / 40 aa = 0.5
  In middle: 20 aa / 40 aa = 0.5
  In C-terminal: 0 aa / 40 aa = 0.0

Location features = [0.5, 0.5, 0.0]
```

### Rationale

Signal peptides, transmembrane regions, and other localization-relevant domains have characteristic positions. N-terminal signals are common for secreted/membrane proteins.

---

## Feature Combinations

The system supports combining base features with enhanced features:

### Part 1: Basic IPS Features
- `e_value` - Log-transformed e-values only
- `binary` - Binary presence/absence
- `e_value_plus_cluster` - E-values + cluster information
- `binary_plus_cluster` - Binary + cluster information

### Part 2: Enhanced Features
- `taxonomy` - Base + species taxonomy vector
- `location` - Base + sequence position proportions
- `count` - Base + feature occurrence counts

### Part 3: All Features
- `all` - E-values + taxonomy + WolfPSort + TargetP predictions

---

## Sparse Matrix Construction

All features are assembled into a single sparse matrix using scikit-learn vectorizers:

```python
# Vectorizers for different feature types
cluster_binarizer = MultiLabelBinarizer(sparse_output=True)
e_value_vectorizer = DictVectorizer()
e_missing_binarizer = MultiLabelBinarizer(sparse_output=True)
cv = [DictVectorizer() for i in range(4)]  # count, start, middle, end

# Transform and combine
features = sp.hstack([
    clusters,      # Cluster membership
    e_values,      # Log e-values
    e_missing,     # Missing e-value flags
    counts,        # Hit counts
    start_loc,     # N-terminal proportions
    middle_loc,    # Middle proportions
    end_loc        # C-terminal proportions
]).tocsr()
```

### Final Matrix Structure

| Component | Feature Prefix | Value Type | Vectorizer |
|-----------|---------------|------------|------------|
| Cluster presence | `ipscan_cluster` | 0/1 | MultiLabelBinarizer |
| Log e-value | `ipscan_e_value` | float | DictVectorizer |
| Missing e-value | `ipscan_e_missing` | 0/1 | MultiLabelBinarizer |
| Hit count | `ipscan_count` | int | DictVectorizer |
| N-terminal proportion | `ipscan_start` | 0.0-1.0 | DictVectorizer |
| Middle proportion | `ipscan_middle` | 0.0-1.0 | DictVectorizer |
| C-terminal proportion | `ipscan_end` | 0.0-1.0 | DictVectorizer |

---

## Dimensionality Summary

| Feature Type | Approximate Dimensions |
|--------------|------------------------|
| E-value / Binary | ~32,000-36,000 |
| Count | ~65,000-73,000 |
| Location | ~130,000-147,000 |
| Taxonomy | Base + ~1,000+ |

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `generate_data/in-house_data/data_processing.py` | Raw IPS parsing, count/location calculation |
| `generate_data/in-house_data/load_data.py` | Sparse matrix construction |
| `ipscan_experiment.py` | Feature type selection for experiments |
