#!/usr/bin/env python3
"""
InterProScan Feature Processing Pipeline (v2)

This module processes InterProScan output files and generates feature matrices
with multiple preprocessing options. Supports both single files and directories
with pattern matching. Outputs are saved in HDF5 format with comprehensive
metadata and statistics.

Author: Generated for Henri-AFP Project
Date: 2026-01-29
Version: 2.0
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
import json

import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================================
# FILE DISCOVERY AND VALIDATION
# ============================================================================

def find_ips_files(input_source, pattern='*.out', exclude_patterns=None, 
                   is_single_file=False, verbose=True):
    """
    Find IPS files to process.
    
    Parameters
    ----------
    input_source : Path
        Directory containing IPS files OR single file path
    pattern : str
        Glob pattern for matching files (only if is_single_file=False)
    exclude_patterns : list
        Patterns to exclude (e.g., ['README*'])
    is_single_file : bool
        If True, treat input_source as a single file
    verbose : bool
        Print discovery information
        
    Returns
    -------
    list of Path
        Sorted list of IPS files to process
        
    Raises
    ------
    ValueError
        If no suitable files are found
    FileNotFoundError
        If input path doesn't exist
    """
    exclude_patterns = exclude_patterns or ['README*']
    
    if is_single_file:
        # Single file mode
        if not input_source.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")
        if not input_source.is_file():
            raise ValueError(f"Input path is not a file: {input_source}")
        
        if verbose:
            print(f"Processing single file: {input_source}")
        
        return [input_source]
    
    else:
        # Directory mode
        if not input_source.exists():
            raise FileNotFoundError(f"Input directory not found: {input_source}")
        if not input_source.is_dir():
            raise ValueError(f"Input path is not a directory: {input_source}")
        
        if verbose:
            print(f"Searching for IPS files in: {input_source}")
            print(f"  Pattern: {pattern}")
            print(f"  Exclude: {', '.join(exclude_patterns)}")
        
        # Find files matching pattern
        all_files = sorted(input_source.glob(pattern))
        
        if verbose:
            print(f"  Found {len(all_files)} files matching pattern")
        
        # Filter out excluded files
        filtered_files = []
        excluded_files = []
        
        for file_path in all_files:
            # Check if file matches any exclude pattern
            excluded = False
            for exclude_pattern in exclude_patterns:
                if file_path.match(exclude_pattern):
                    excluded = True
                    excluded_files.append(file_path.name)
                    break
            
            if not excluded:
                filtered_files.append(file_path)
        
        if verbose and excluded_files:
            print(f"  Excluded {len(excluded_files)} files: {', '.join(excluded_files[:5])}")
            if len(excluded_files) > 5:
                print(f"    ... and {len(excluded_files) - 5} more")
        
        # Validate we found files
        if not filtered_files:
            raise ValueError(
                f"No IPS files found in {input_source}\n"
                f"  Pattern: {pattern}\n"
                f"  Exclude: {', '.join(exclude_patterns)}\n"
                f"  Make sure files match the pattern and are not excluded."
            )
        
        if verbose:
            print(f"  Will process {len(filtered_files)} files:")
            for i, f in enumerate(filtered_files[:10], 1):
                print(f"    {i}. {f.name}")
            if len(filtered_files) > 10:
                print(f"    ... and {len(filtered_files) - 10} more")
        
        return filtered_files


# ============================================================================
# CORE PARSING FUNCTIONS
# ============================================================================

def validate_ips_line(fields, filepath, line_num):
    """
    Validate that an IPS line has the required fields.
    
    Parameters
    ----------
    fields : list
        Split line fields
    filepath : Path
        Source file for error reporting
    line_num : int
        Line number for error reporting
        
    Raises
    ------
    ValueError
        If line format is invalid
    """
    required_fields = {
        0: "Protein ID",
        2: "Protein length",
        3: "InterPro/Pfam ID",
        4: "Feature name",
        6: "Start position",
        7: "Stop position",
        8: "E-value"
    }
    
    for idx, name in required_fields.items():
        if idx >= len(fields):
            raise ValueError(
                f"Invalid IPS format in {filepath.name}, line {line_num}:\n"
                f"  Missing field {idx} ({name})\n"
                f"  Expected at least 9 tab-separated fields, found {len(fields)}\n"
                f"  Line content: {' '.join(fields[:50])}"
            )
    
    # Validate numeric fields
    try:
        int(fields[2])  # length
        int(fields[6])  # start
        int(fields[7])  # stop
    except ValueError as e:
        raise ValueError(
            f"Invalid IPS format in {filepath.name}, line {line_num}:\n"
            f"  Non-numeric value in position field: {e}\n"
            f"  Fields 2,6,7 must be integers (length, start, stop)"
        )


def process_ipr_line2(line, data, filepath, line_num):
    """Extract contents of an IPS line and store them to a dict.
    
    Parses IPS TSV format and extracts:
    - Protein ID
    - Feature ID and name
    - E-value
    - Start/stop positions
    - Protein length
    - Cluster information
    
    Parameters
    ----------
    line : str
        Line from IPS file
    data : dict
        Dictionary to store parsed data
    filepath : Path
        Source file (for error messages)
    line_num : int
        Line number (for error messages)
        
    Raises
    ------
    ValueError
        If line format is invalid
    """
    fields = line.split('\t')
    
    # Validate format
    validate_ips_line(fields, filepath, line_num)
    
    # Parse fields
    name = fields[0].split('|')[1] if '|' in fields[0] else fields[0]
    feature = f'{fields[3]}|{fields[4]}'
    e_value = fields[8]
    start = int(fields[6])
    stop = int(fields[7])
    length = int(fields[2])
    
    try:
        cluster = fields[11].strip() if len(fields) > 11 else 'missing'
    except IndexError:
        cluster = 'missing'
    
    if name not in data:
        data[name] = {}
    
    if feature not in data[name]:
        data[name][feature] = []
    
    data[name][feature].append((e_value, cluster, start, stop, length))


def read_ips_file(filepath, data, verbose=True):
    """
    Read and parse a single IPS file.
    
    Parameters
    ----------
    filepath : Path
        Path to IPS file
    data : dict
        Dictionary to accumulate parsed data
    verbose : bool
        Print progress
        
    Raises
    ------
    ValueError
        If file format is invalid (stops processing)
    """
    if verbose:
        print(f"    Reading: {filepath.name}")
    
    line_count = 0
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse line - this will raise ValueError if format is invalid
            process_ipr_line2(line, data, filepath, line_num)
            line_count += 1
    
    if verbose:
        print(f"      Processed {line_count} lines")


def non_overlapping(feature_list, max_overlap=0.2):
    """Extract occurrences of a feature where overlap is below max_overlap."""
    from operator import itemgetter
    
    if not feature_list:
        return []
    
    features = []
    sorted_features = sorted(feature_list, key=itemgetter(2))  # Sort by start position
    features.append(sorted_features[0])
    j = 0
    
    for i in range(1, len(sorted_features)):
        # Check if overlap is acceptable
        if (1 - max_overlap) * sorted_features[j][3] < sorted_features[i][2]:
            features.append(sorted_features[i])
            j = i
    
    return features


def calc_counts(feature_list, max_overlap=0.2):
    """Calculate number of non-overlapping occurrences."""
    return len(non_overlapping(feature_list, max_overlap))


def calc_localisation(feature_list, max_tail_len=100):
    """Calculate proportions of feature in N-terminal, middle, C-terminal regions."""
    no = non_overlapping(feature_list, 0.5)
    if not no:
        return [0.0, 0.0, 0.0]
    
    length = no[0][4]
    result = [0.0, 0.0, 0.0]
    
    if length < 300:
        max_tail_len = length // 3
        start_region = set(range(max_tail_len))
        middle_region = set(range(max_tail_len, 2 * max_tail_len))
        end_region = set(range(2 * max_tail_len, length))
    else:
        start_region = set(range(max_tail_len))
        middle_region = set(range(max_tail_len, length - max_tail_len))
        end_region = set(range(length - max_tail_len, length))
    
    try:
        for occurrence in no:
            values = set(range(occurrence[2], occurrence[3]))
            if len(values) > 0:
                result[0] += len(values.intersection(start_region)) / len(values)
                result[1] += len(values.intersection(middle_region)) / len(values)
                result[2] += len(values.intersection(end_region)) / len(values)
        
        total = sum(result)
        if total > 0:
            return [r / total for r in result]
        else:
            return [0.0, 0.0, 0.0]
    except (ZeroDivisionError, ValueError):
        return [1.0, 1.0, 1.0]


def calc_center_position(feature_list):
    """Calculate relative center position (0-1) of the strongest feature."""
    if not feature_list:
        return 0.5
    
    # Find strongest (lowest e-value)
    strongest = min(feature_list, key=lambda x: float(x[0]) if x[0] != '-' else float('inf'))
    
    start = strongest[2]
    stop = strongest[3]
    length = strongest[4]
    
    if length == 0:
        return 0.5
    
    center = (start + stop) / 2
    return center / length


def process_ipr_data(ips_files, verbose=True):
    """
    Read IPS data files and process features.
    
    Parameters
    ----------
    ips_files : list of Path
        List of IPS files to process
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        data[protein][feature] = (cluster, count, max_e, loc_start, loc_middle, 
                                   loc_end, center_pos)
    
    Raises
    ------
    ValueError
        If any file has invalid format
    """
    data = {}
    
    if verbose:
        print(f"\n  Step 1: Parsing {len(ips_files)} IPS files")
    
    # Read all files
    for i, file_path in enumerate(ips_files, 1):
        if verbose:
            print(f"  File {i}/{len(ips_files)}:")
        
        # This will raise ValueError if format is invalid
        read_ips_file(file_path, data, verbose=verbose)
    
    # Calculate derived features
    if verbose:
        print(f"\n  Step 2: Calculating counts and localizations...")
    
    n_proteins = len(data)
    for protein_idx, (protein_id, features) in enumerate(data.items(), 1):
        if verbose and protein_idx % 10000 == 0:
            print(f"    Processed {protein_idx:,}/{n_proteins:,} proteins...")
        
        for feature_id, feature_list in features.items():
            count = calc_counts(feature_list)
            
            # Get strongest e-value
            max_e = max(feature_list, key=lambda x: float(x[0]) if x[0] != '-' else 0)[0]
            
            # Get cluster
            cluster = feature_list[0][1]
            
            # Calculate localization
            localisation = calc_localisation(feature_list)
            
            # Calculate center position
            center_pos = calc_center_position(feature_list)
            
            # Store processed data
            data[protein_id][feature_id] = (
                cluster,           # 0: cluster
                count,             # 1: count
                max_e,             # 2: max_e_value
                localisation[0],   # 3: start proportion
                localisation[1],   # 4: middle proportion
                localisation[2],   # 5: end proportion
                center_pos         # 6: center position
            )
    
    if verbose:
        print(f"    ✓ Processed all {n_proteins:,} proteins")
    
    return data


# ============================================================================
# FEATURE TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_features(sequences, features, empty_structure):
    """Extract features corresponding to sequences."""
    new_features = []
    for seq in sequences:
        try:
            new_features.append(features[seq])
        except KeyError:
            new_features.append(empty_structure())
    return new_features


def process_basic_features(ipr_list):
    """Basic processing: e-values and binary features."""
    e_values = []
    binary = []
    
    for item in ipr_list:
        e_dict = {}
        bin_set = set()
        
        for key, values in item.items():
            max_e = values[2]  # E-value
            
            if max_e != '-':
                float_val = float(max_e)
                if float_val == 0:
                    float_val = 7.5e-305
                e_dict[key] = -np.log(float_val)
            else:
                bin_set.add(key)
        
        e_values.append(e_dict)
        binary.append(bin_set)
    
    return e_values, binary


def process_cluster_features(ipr_list):
    """Process cluster information."""
    clusters = []
    
    for item in ipr_list:
        cluster_dict = {}
        
        for key, values in item.items():
            cluster = values[0]
            max_e = values[2]
            
            if cluster != 'missing' and max_e != '-':
                float_val = float(max_e)
                if float_val == 0:
                    float_val = 7.5e-305
                e_val_transformed = -np.log(float_val)
                
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = e_val_transformed
                else:
                    cluster_dict[cluster] = max(cluster_dict[cluster], e_val_transformed)
        
        clusters.append(cluster_dict)
    
    return clusters


def process_count_features(ipr_list):
    """Process count information."""
    counts = []
    
    for item in ipr_list:
        count_dict = {}
        for key, values in item.items():
            count_dict[key] = values[1]
        counts.append(count_dict)
    
    return counts


def process_location_features(ipr_list):
    """Process location information (3-part split)."""
    start_locs = []
    middle_locs = []
    end_locs = []
    
    for item in ipr_list:
        start_dict = {}
        middle_dict = {}
        end_dict = {}
        
        for key, values in item.items():
            start_dict[key] = values[3]
            middle_dict[key] = values[4]
            end_dict[key] = values[5]
        
        start_locs.append(start_dict)
        middle_locs.append(middle_dict)
        end_locs.append(end_dict)
    
    return start_locs, middle_locs, end_locs


def process_location_b_features(ipr_list):
    """Process location B: relative center position."""
    positions = []
    
    for item in ipr_list:
        pos_dict = {}
        for key, values in item.items():
            pos_dict[key] = values[6]
        positions.append(pos_dict)
    
    return positions


# ============================================================================
# MATRIX CONSTRUCTION
# ============================================================================

def build_feature_matrix(ips_files, preprocessing='e-value', verbose=True):
    """
    Build sparse feature matrix from IPS data.
    
    Parameters
    ----------
    ips_files : list of Path
        List of IPS files to process
    preprocessing : str
        One of: 'binary', 'e-value', 'counts', 'location', 'location_b', 'clusters'
    verbose : bool
        Print progress
    
    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        Feature matrix (n_proteins x n_features)
    feature_names : list
        Feature names
    protein_names : list
        Protein IDs
    stats : dict
        Statistics about the processing
    input_files : list
        List of processed file paths (for provenance)
        
    Raises
    ------
    ValueError
        If file format is invalid or preprocessing method is unknown
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"IPS Feature Processing")
        print(f"{'='*70}")
        print(f"Input files: {len(ips_files)}")
        print(f"Preprocessing: {preprocessing}")
        print(f"{'='*70}")
    
    start_time = time.time()
    
    # Parse IPS data - this will raise ValueError if format is invalid
    parse_start = time.time()
    try:
        ipr_data = process_ipr_data(ips_files, verbose=verbose)
    except ValueError as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Invalid IPS file format")
        print(f"{'='*70}")
        raise
    
    protein_names = sorted(ipr_data.keys())
    parse_time = time.time() - parse_start
    
    if verbose:
        print(f"  ✓ Parsed {len(protein_names):,} proteins in {parse_time:.2f}s")
    
    # Transform to list format
    if verbose:
        print(f"\n  Step 3: Transforming features...")
    
    transform_start = time.time()
    ipr_list = transform_features(protein_names, ipr_data, dict)
    
    # Count unique IPS features
    all_features = set()
    for protein_features in ipr_list:
        all_features.update(protein_features.keys())
    n_unique_ips_features = len(all_features)
    
    if verbose:
        print(f"  ✓ Found {n_unique_ips_features:,} unique IPS features")
    
    transform_time = time.time() - transform_start
    
    # Build matrix based on preprocessing type
    if verbose:
        print(f"\n  Step 4: Building feature matrix (mode: {preprocessing})...")
    
    matrix_start = time.time()
    
    if preprocessing == 'binary':
        e_values, binary = process_basic_features(ipr_list)
        binarizer = MultiLabelBinarizer(sparse_output=True)
        matrix = binarizer.fit_transform(binary)
        feature_names = ['ipscan_binary_' + name for name in binarizer.classes_]
    
    elif preprocessing == 'e-value':
        e_values, binary = process_basic_features(ipr_list)
        e_vectorizer = DictVectorizer()
        binarizer = MultiLabelBinarizer(sparse_output=True)
        
        e_matrix = e_vectorizer.fit_transform(e_values)
        b_matrix = binarizer.fit_transform(binary)
        
        matrix = sp.hstack([e_matrix, b_matrix]).tocsr()
        feature_names = (['ipscan_e_value_' + name for name in e_vectorizer.feature_names_] +
                        ['ipscan_binary_' + name for name in binarizer.classes_])
    
    elif preprocessing == 'counts':
        e_values, binary = process_basic_features(ipr_list)
        counts = process_count_features(ipr_list)
        
        e_vectorizer = DictVectorizer()
        b_binarizer = MultiLabelBinarizer(sparse_output=True)
        c_vectorizer = DictVectorizer()
        
        e_matrix = e_vectorizer.fit_transform(e_values)
        b_matrix = b_binarizer.fit_transform(binary)
        c_matrix = c_vectorizer.fit_transform(counts)
        
        matrix = sp.hstack([e_matrix, b_matrix, c_matrix]).tocsr()
        feature_names = (['ipscan_e_value_' + name for name in e_vectorizer.feature_names_] +
                        ['ipscan_binary_' + name for name in b_binarizer.classes_] +
                        ['ipscan_count_' + name for name in c_vectorizer.feature_names_])
    
    elif preprocessing == 'location':
        e_values, binary = process_basic_features(ipr_list)
        start_locs, middle_locs, end_locs = process_location_features(ipr_list)
        
        e_vectorizer = DictVectorizer()
        b_binarizer = MultiLabelBinarizer(sparse_output=True)
        s_vectorizer = DictVectorizer()
        m_vectorizer = DictVectorizer()
        e_loc_vectorizer = DictVectorizer()
        
        e_matrix = e_vectorizer.fit_transform(e_values)
        b_matrix = b_binarizer.fit_transform(binary)
        s_matrix = s_vectorizer.fit_transform(start_locs)
        m_matrix = m_vectorizer.fit_transform(middle_locs)
        e_loc_matrix = e_loc_vectorizer.fit_transform(end_locs)
        
        matrix = sp.hstack([e_matrix, b_matrix, s_matrix, m_matrix, e_loc_matrix]).tocsr()
        feature_names = (['ipscan_e_value_' + name for name in e_vectorizer.feature_names_] +
                        ['ipscan_binary_' + name for name in b_binarizer.classes_] +
                        ['ipscan_loc_start_' + name for name in s_vectorizer.feature_names_] +
                        ['ipscan_loc_middle_' + name for name in m_vectorizer.feature_names_] +
                        ['ipscan_loc_end_' + name for name in e_loc_vectorizer.feature_names_])
    
    elif preprocessing == 'location_b':
        e_values, binary = process_basic_features(ipr_list)
        positions = process_location_b_features(ipr_list)
        
        e_vectorizer = DictVectorizer()
        b_binarizer = MultiLabelBinarizer(sparse_output=True)
        p_vectorizer = DictVectorizer()
        
        e_matrix = e_vectorizer.fit_transform(e_values)
        b_matrix = b_binarizer.fit_transform(binary)
        p_matrix = p_vectorizer.fit_transform(positions)
        
        matrix = sp.hstack([e_matrix, b_matrix, p_matrix]).tocsr()
        feature_names = (['ipscan_e_value_' + name for name in e_vectorizer.feature_names_] +
                        ['ipscan_binary_' + name for name in b_binarizer.classes_] +
                        ['ipscan_center_pos_' + name for name in p_vectorizer.feature_names_])
    
    elif preprocessing == 'clusters':
        e_values, binary = process_basic_features(ipr_list)
        clusters = process_cluster_features(ipr_list)
        
        e_vectorizer = DictVectorizer()
        b_binarizer = MultiLabelBinarizer(sparse_output=True)
        c_vectorizer = DictVectorizer()
        
        e_matrix = e_vectorizer.fit_transform(e_values)
        b_matrix = b_binarizer.fit_transform(binary)
        c_matrix = c_vectorizer.fit_transform(clusters)
        
        matrix = sp.hstack([e_matrix, b_matrix, c_matrix]).tocsr()
        feature_names = (['ipscan_e_value_' + name for name in e_vectorizer.feature_names_] +
                        ['ipscan_binary_' + name for name in b_binarizer.classes_] +
                        ['ipscan_cluster_' + name for name in c_vectorizer.feature_names_])
    
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}. "
                        f"Choose from: binary, e-value, counts, location, location_b, clusters")
    
    matrix_time = time.time() - matrix_start
    total_time = time.time() - start_time
    
    # Compute statistics
    n_nonzero = matrix.nnz
    sparsity = 1.0 - (n_nonzero / (matrix.shape[0] * matrix.shape[1]))
    
    stats = {
        'n_proteins': len(protein_names),
        'n_unique_ips_features': n_unique_ips_features,
        'n_resulting_features': len(feature_names),
        'n_nonzero_values': n_nonzero,
        'sparsity': sparsity,
        'matrix_shape': matrix.shape,
        'preprocessing_mode': preprocessing,
        'parse_time_sec': parse_time,
        'transform_time_sec': transform_time,
        'matrix_build_time_sec': matrix_time,
        'total_time_sec': total_time,
        'timestamp': datetime.now().isoformat(),
        'n_input_files': len(ips_files)
    }
    
    if verbose:
        print(f"  ✓ Built matrix: {matrix.shape[0]:,} × {matrix.shape[1]:,}")
        print(f"  ✓ Non-zero values: {n_nonzero:,} ({(1-sparsity)*100:.2f}% dense)")
        print(f"  ✓ Matrix construction: {matrix_time:.2f}s")
        print(f"\n{'='*70}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"{'='*70}\n")
    
    # Store input files for provenance
    input_files = [str(f) for f in ips_files]
    
    return matrix, feature_names, protein_names, stats, input_files


# ============================================================================
# HDF5 OUTPUT
# ============================================================================

def save_to_hdf5(output_path, matrix, feature_names, protein_names, stats, 
                 input_files, input_source, preprocessing):
    """Save feature matrix and metadata to HDF5 file."""
    print(f"Saving to HDF5: {output_path}")
    
    # Convert to Path for name extraction
    output_path = Path(output_path)
    
    with h5py.File(str(output_path), 'w') as f:
        # Create groups
        features_grp = f.create_group('features')
        csr_grp = features_grp.create_group('csr')
        metadata_grp = f.create_group('metadata')
        
        # Save CSR matrix components
        csr_grp.create_dataset('data', data=matrix.data.astype(np.float32),
                               compression='gzip', compression_opts=6)
        csr_grp.create_dataset('indices', data=matrix.indices.astype(np.int32),
                               compression='gzip', compression_opts=6)
        csr_grp.create_dataset('indptr', data=matrix.indptr.astype(np.int32),
                               compression='gzip', compression_opts=6)
        
        # Add matrix shape as attributes
        features_grp.attrs['shape'] = matrix.shape
        features_grp.attrs['dtype'] = 'float32'
        features_grp.attrs['format'] = 'csr'
        
        # Save feature names
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('feature_names', data=np.array(feature_names, dtype=object),
                        dtype=dt, compression='gzip')
        
        # Save protein names
        f.create_dataset('protein_names', data=np.array(protein_names, dtype=object),
                        dtype=dt, compression='gzip')
        
        # Save input files list
        f.create_dataset('input_files', data=np.array(input_files, dtype=object),
                        dtype=dt, compression='gzip')
        
        # Save metadata and statistics
        for key, value in stats.items():
            if isinstance(value, (list, tuple)):
                metadata_grp.attrs[key] = str(value)
            else:
                metadata_grp.attrs[key] = value
        
        metadata_grp.attrs['input_source'] = str(input_source)
        metadata_grp.attrs['preprocessing_mode'] = preprocessing
        metadata_grp.attrs['creation_timestamp'] = datetime.now().isoformat()
        
        # Create README
        readme_text = f"""
InterProScan Feature Matrix
===========================

GENERAL INFORMATION
-------------------
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Preprocessing Mode: {preprocessing}
Input Source: {input_source}
Number of Input Files: {len(input_files)}

MATRIX DIMENSIONS
-----------------
Proteins: {stats['n_proteins']:,}
Features: {stats['n_resulting_features']:,}
Non-zero values: {stats['n_nonzero_values']:,}
Sparsity: {stats['sparsity']*100:.2f}%
Matrix shape: {stats['matrix_shape']}

FEATURE STATISTICS
------------------
Unique IPS features (raw): {stats['n_unique_ips_features']:,}
Resulting features (processed): {stats['n_resulting_features']:,}

PROCESSING TIME
---------------
Parsing: {stats['parse_time_sec']:.2f}s
Transformation: {stats['transform_time_sec']:.2f}s
Matrix construction: {stats['matrix_build_time_sec']:.2f}s
Total: {stats['total_time_sec']:.2f}s

INPUT FILES
-----------
{chr(10).join(f"  - {Path(f).name}" for f in input_files[:20])}
{'  ... and ' + str(len(input_files) - 20) + ' more' if len(input_files) > 20 else ''}

PREPROCESSING MODES
-------------------
- binary: Binary encoding (0/1) only
- e-value: E-values (-log transformed) + binary features
- counts: E-values + binary + non-overlapping occurrence counts
- location: E-values + binary + 3-part location split (N-term/middle/C-term)
- location_b: E-values + binary + relative center position (0-1)
- clusters: E-values + binary + IPS cluster aggregation

DATA STRUCTURE
--------------
/features/csr/data      - Sparse matrix data (float32)
/features/csr/indices   - Sparse matrix column indices (int32)
/features/csr/indptr    - Sparse matrix row pointers (int32)
/feature_names          - Array of feature name strings
/protein_names          - Array of protein ID strings
/input_files            - List of processed input files
/metadata/              - Processing statistics and parameters

USAGE EXAMPLE (Python)
----------------------
import h5py
import scipy.sparse as sp

with h5py.File('{output_path.name}', 'r') as f:
    # Load sparse matrix
    data = f['features/csr/data'][:]
    indices = f['features/csr/indices'][:]
    indptr = f['features/csr/indptr'][:]
    shape = f['features'].attrs['shape']
    
    matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
    
    # Load names
    feature_names = f['feature_names'][:].astype(str)
    protein_names = f['protein_names'][:].astype(str)
    
    # Check input files
    input_files = f['input_files'][:].astype(str)

For more information, see the Henri-AFP project documentation.
"""
        
        f.create_dataset('readme', data=readme_text, dtype=h5py.string_dtype(encoding='utf-8'))
    
    print(f"  ✓ HDF5 file saved successfully")
    
    # Calculate file size
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    print(f"  ✓ File size: {size_mb:.2f} MB")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(output_path, stats, feature_names, protein_names, input_files):
    """Generate a detailed summary report in text format."""
    report_path = output_path.replace('.h5', '_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("InterProScan Feature Processing Summary Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output file: {output_path}\n\n")
        
        f.write("PROCESSING PARAMETERS\n")
        f.write("-"*80 + "\n")
        f.write(f"Preprocessing mode: {stats['preprocessing_mode']}\n")
        f.write(f"Timestamp: {stats['timestamp']}\n")
        f.write(f"Number of input files: {stats['n_input_files']}\n\n")
        
        f.write("INPUT FILES\n")
        f.write("-"*80 + "\n")
        for i, filepath in enumerate(input_files[:20], 1):
            f.write(f"{i:3d}. {Path(filepath).name}\n")
        if len(input_files) > 20:
            f.write(f"... and {len(input_files) - 20} more files\n")
        f.write("\n")
        
        f.write("DATA STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of proteins:           {stats['n_proteins']:>12,}\n")
        f.write(f"Unique IPS features (raw):    {stats['n_unique_ips_features']:>12,}\n")
        f.write(f"Resulting features:           {stats['n_resulting_features']:>12,}\n")
        f.write(f"Non-zero feature values:      {stats['n_nonzero_values']:>12,}\n")
        f.write(f"Matrix shape:                 {str(stats['matrix_shape']):>12}\n")
        f.write(f"Sparsity:                     {stats['sparsity']*100:>11.2f}%\n")
        f.write(f"Density:                      {(1-stats['sparsity'])*100:>11.2f}%\n\n")
        
        f.write("FEATURE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        
        # Count features by type
        feature_types = Counter()
        for fname in feature_names:
            if 'e_value' in fname:
                feature_types['E-value'] += 1
            elif 'binary' in fname:
                feature_types['Binary'] += 1
            elif 'count' in fname:
                feature_types['Count'] += 1
            elif 'loc_start' in fname:
                feature_types['Location (start)'] += 1
            elif 'loc_middle' in fname:
                feature_types['Location (middle)'] += 1
            elif 'loc_end' in fname:
                feature_types['Location (end)'] += 1
            elif 'center_pos' in fname:
                feature_types['Center position'] += 1
            elif 'cluster' in fname:
                feature_types['Cluster'] += 1
        
        for ftype, count in sorted(feature_types.items()):
            f.write(f"{ftype:<25} {count:>12,}\n")
        
        f.write(f"\n{'Total':<25} {sum(feature_types.values()):>12,}\n\n")
        
        f.write("PROCESSING TIME\n")
        f.write("-"*80 + "\n")
        f.write(f"Parsing:                      {stats['parse_time_sec']:>11.2f}s\n")
        f.write(f"Transformation:               {stats['transform_time_sec']:>11.2f}s\n")
        f.write(f"Matrix construction:          {stats['matrix_build_time_sec']:>11.2f}s\n")
        f.write(f"Total:                        {stats['total_time_sec']:>11.2f}s\n\n")
        
        f.write("SAMPLE DATA\n")
        f.write("-"*80 + "\n")
        f.write(f"First 5 proteins:\n")
        for i, pname in enumerate(protein_names[:5], 1):
            f.write(f"  {i}. {pname}\n")
        f.write(f"\nFirst 5 features:\n")
        for i, fname in enumerate(feature_names[:5], 1):
            f.write(f"  {i}. {fname}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of report\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Summary report saved: {report_path}")
    
    return report_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline for IPS feature processing."""
    parser = argparse.ArgumentParser(
        description='Process InterProScan features and generate HDF5 output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preprocessing modes:
  binary      - Binary encoding (0/1) only
  e-value     - E-values (-log transformed) + binary features (DEFAULT)
  counts      - E-values + binary + non-overlapping occurrence counts
  location    - E-values + binary + 3-part location split (N-term/middle/C-term)
  location_b  - E-values + binary + relative center position (0-1)
  clusters    - E-values + binary + IPS cluster aggregation

Input modes:
  --input-dir      Process multiple files from directory (with pattern matching)
  --input-file     Process single file

Example usage:
  # Process directory (all .out files, excluding README)
  python ips_feature_pipeline.py --input-dir /path/to/ips_output output.h5
  
  # Process with custom pattern
  python ips_feature_pipeline.py --input-dir /path/to/ips_output --pattern "EBI_GO*.out" output.h5
  
  # Process single file
  python ips_feature_pipeline.py --input-file protein.out output.h5
  
  # Different preprocessing
  python ips_feature_pipeline.py --input-dir /path/to/ips_output output.h5 --preprocessing counts
        """
    )
    
    # Input group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=str,
                            help='Directory containing IPS files')
    input_group.add_argument('--input-file', type=str,
                            help='Single IPS file to process')
    
    parser.add_argument('output', type=str,
                       help='Output HDF5 file path')
    
    parser.add_argument('--preprocessing', type=str, default='e-value',
                       choices=['binary', 'e-value', 'counts', 'location', 'location_b', 'clusters'],
                       help='Preprocessing mode (default: e-value)')
    
    parser.add_argument('--pattern', type=str, default='*.out',
                       help='File pattern for matching (default: *.out, only used with --input-dir)')
    
    parser.add_argument('--exclude', type=str, nargs='+', default=['README*'],
                       help='Patterns to exclude (default: README*)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Determine input mode
    if args.input_dir:
        input_source = Path(args.input_dir)
        is_single_file = False
    else:
        input_source = Path(args.input_file)
        is_single_file = True
    
    # Validate and ensure output directory exists
    output_path = Path(args.output)
    if output_path.suffix != '.h5':
        output_path = output_path.with_suffix('.h5')
    
    if output_path.parent != Path('.') and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find IPS files
        ips_files = find_ips_files(
            input_source,
            pattern=args.pattern,
            exclude_patterns=args.exclude,
            is_single_file=is_single_file,
            verbose=verbose
        )
        
        # Build feature matrix
        matrix, feature_names, protein_names, stats, input_files = build_feature_matrix(
            ips_files,
            args.preprocessing,
            verbose=verbose
        )
        
        # Save to HDF5
        save_to_hdf5(
            str(output_path),
            matrix,
            feature_names,
            protein_names,
            stats,
            input_files,
            input_source,
            args.preprocessing
        )
        
        # Generate summary report
        report_path = generate_summary_report(
            str(output_path),
            stats,
            feature_names,
            protein_names,
            input_files
        )
        
        if verbose:
            print("\n" + "="*70)
            print("SUCCESS!")
            print("="*70)
            print(f"HDF5 file:      {output_path}")
            print(f"Summary report: {report_path}")
            print("="*70 + "\n")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"\n{'='*70}")
        print(f"ERROR")
        print(f"{'='*70}")
        print(f"{e}")
        print(f"{'='*70}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"UNEXPECTED ERROR")
        print(f"{'='*70}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
