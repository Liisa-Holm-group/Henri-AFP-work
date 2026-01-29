#!/usr/bin/env python3
"""
Example script demonstrating IPS feature processing pipeline usage.
This script shows how to process IPS files and load the results.
"""

import h5py
import numpy as np
import scipy.sparse as sp


def example_processing():
    """Example: Process IPS files and generate HDF5 output."""
    print("="*70)
    print("EXAMPLE 1: Processing IPS Files")
    print("="*70)
    
    # This is what you would run from command line:
    example_command = """
    python ips_feature_pipeline.py \\
        /path/to/your/ips_directory \\
        output.h5 \\
        --preprocessing e-value
    """
    
    print("\nCommand line usage:")
    print(example_command)
    
    print("\nPreprocessing options:")
    options = {
        'binary': 'Binary encoding only (0/1)',
        'e-value': 'E-values + binary (DEFAULT)',
        'counts': 'E-values + binary + occurrence counts',
        'location': 'E-values + binary + N-term/middle/C-term proportions',
        'location_b': 'E-values + binary + center position',
        'clusters': 'E-values + binary + IPS clusters'
    }
    
    for method, description in options.items():
        print(f"  --preprocessing {method:<12} : {description}")
    
    print("\n" + "="*70 + "\n")


def example_loading(h5_file='output.h5'):
    """Example: Load and inspect HDF5 output."""
    print("="*70)
    print("EXAMPLE 2: Loading HDF5 Output")
    print("="*70)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            print(f"\n✓ Successfully opened: {h5_file}")
            
            # Display file structure
            print("\nFile structure:")
            print("├── features/")
            print("│   ├── csr/")
            print("│   │   ├── data")
            print("│   │   ├── indices")
            print("│   │   └── indptr")
            print("│   └── [attributes: shape, dtype, format]")
            print("├── feature_names")
            print("├── protein_names")
            print("├── metadata/ [attributes]")
            print("└── readme")
            
            # Load sparse matrix
            print("\nLoading sparse matrix...")
            data = f['features/csr/data'][:]
            indices = f['features/csr/indices'][:]
            indptr = f['features/csr/indptr'][:]
            shape = f['features'].attrs['shape']
            
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
            print(f"  ✓ Matrix shape: {X.shape}")
            print(f"  ✓ Data type: {X.dtype}")
            print(f"  ✓ Non-zeros: {X.nnz:,}")
            print(f"  ✓ Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.2f}%")
            
            # Load names
            print("\nLoading names...")
            feature_names = f['feature_names'][:].astype(str)
            protein_names = f['protein_names'][:].astype(str)
            print(f"  ✓ {len(feature_names):,} features")
            print(f"  ✓ {len(protein_names):,} proteins")
            
            # Load metadata
            print("\nMetadata:")
            for key in f['metadata'].attrs.keys():
                value = f['metadata'].attrs[key]
                print(f"  {key}: {value}")
            
            # Show sample data
            print("\nSample proteins (first 5):")
            for i, pname in enumerate(protein_names[:5], 1):
                print(f"  {i}. {pname}")
            
            print("\nSample features (first 5):")
            for i, fname in enumerate(feature_names[:5], 1):
                print(f"  {i}. {fname}")
            
            # Analyze first protein
            print(f"\nAnalyzing first protein: {protein_names[0]}")
            protein_features = X[0, :].toarray().flatten()
            nonzero_idx = np.nonzero(protein_features)[0]
            print(f"  Non-zero features: {len(nonzero_idx)}")
            
            if len(nonzero_idx) > 0:
                print(f"  Top 5 features:")
                top_idx = nonzero_idx[np.argsort(protein_features[nonzero_idx])[-5:][::-1]]
                for idx in top_idx:
                    print(f"    {feature_names[idx]}: {protein_features[idx]:.4f}")
            
            # Read README
            print("\nREADME excerpt:")
            readme = f['readme'][()].decode('utf-8')
            readme_lines = readme.split('\n')[:15]
            for line in readme_lines:
                print(f"  {line}")
            print("  ...")
            
    except FileNotFoundError:
        print(f"\n✗ File not found: {h5_file}")
        print("  Run the processing pipeline first to generate the HDF5 file.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "="*70 + "\n")


def example_feature_selection(h5_file='output.h5'):
    """Example: Feature selection and filtering."""
    print("="*70)
    print("EXAMPLE 3: Feature Selection")
    print("="*70)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load matrix
            data = f['features/csr/data'][:]
            indices = f['features/csr/indices'][:]
            indptr = f['features/csr/indptr'][:]
            shape = f['features'].attrs['shape']
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
            
            feature_names = f['feature_names'][:].astype(str)
            
            print("\nFeature type breakdown:")
            
            # Count by type
            e_value_count = sum(1 for name in feature_names if 'e_value' in name)
            binary_count = sum(1 for name in feature_names if 'binary' in name)
            count_count = sum(1 for name in feature_names if 'count' in name)
            loc_count = sum(1 for name in feature_names if 'loc_' in name)
            center_count = sum(1 for name in feature_names if 'center' in name)
            cluster_count = sum(1 for name in feature_names if 'cluster' in name)
            
            print(f"  E-value features:        {e_value_count:>6,}")
            print(f"  Binary features:         {binary_count:>6,}")
            print(f"  Count features:          {count_count:>6,}")
            print(f"  Location features:       {loc_count:>6,}")
            print(f"  Center position features:{center_count:>6,}")
            print(f"  Cluster features:        {cluster_count:>6,}")
            print(f"  {'─'*35}")
            print(f"  Total:                   {len(feature_names):>6,}")
            
            # Example: Select only E-value features
            print("\nSelecting only E-value features...")
            e_value_mask = np.array(['e_value' in name for name in feature_names])
            X_evalue = X[:, e_value_mask]
            print(f"  Original shape: {X.shape}")
            print(f"  E-value only:   {X_evalue.shape}")
            
            # Example: Select specific domain features
            print("\nSelecting features containing 'PF00069'...")
            pfam_mask = np.array(['PF00069' in name for name in feature_names])
            if pfam_mask.any():
                X_pfam = X[:, pfam_mask]
                print(f"  Found {pfam_mask.sum()} features with PF00069")
                print(f"  Selected matrix shape: {X_pfam.shape}")
            else:
                print("  No PF00069 features found in this dataset")
            
    except FileNotFoundError:
        print(f"\n✗ File not found: {h5_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "="*70 + "\n")


def example_protein_query(h5_file='output.h5', protein_id=None):
    """Example: Query features for a specific protein."""
    print("="*70)
    print("EXAMPLE 4: Protein Query")
    print("="*70)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load data
            data = f['features/csr/data'][:]
            indices = f['features/csr/indices'][:]
            indptr = f['features/csr/indptr'][:]
            shape = f['features'].attrs['shape']
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
            
            feature_names = f['feature_names'][:].astype(str)
            protein_names = f['protein_names'][:].astype(str)
            
            # Use first protein if none specified
            if protein_id is None:
                protein_id = protein_names[0]
            
            print(f"\nQuerying protein: {protein_id}")
            
            # Find protein
            if protein_id not in protein_names:
                print(f"  ✗ Protein not found!")
                print(f"  Available proteins: {len(protein_names)}")
                print(f"  Sample: {', '.join(protein_names[:3])}...")
                return
            
            protein_idx = list(protein_names).index(protein_id)
            protein_features = X[protein_idx, :].toarray().flatten()
            
            # Find non-zero features
            nonzero_idx = np.nonzero(protein_features)[0]
            print(f"  ✓ Found {len(nonzero_idx)} active features")
            
            if len(nonzero_idx) > 0:
                # Sort by value
                sorted_idx = nonzero_idx[np.argsort(protein_features[nonzero_idx])[::-1]]
                
                print(f"\n  Top 10 features:")
                for i, idx in enumerate(sorted_idx[:10], 1):
                    feat_name = feature_names[idx]
                    feat_val = protein_features[idx]
                    
                    # Shorten long names
                    if len(feat_name) > 60:
                        feat_name = feat_name[:57] + '...'
                    
                    print(f"    {i:2d}. {feat_name:<60} {feat_val:8.4f}")
            else:
                print("  No active features found for this protein")
            
    except FileNotFoundError:
        print(f"\n✗ File not found: {h5_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70 + "\n")


def example_statistics(h5_file='output.h5'):
    """Example: Compute statistics on the feature matrix."""
    print("="*70)
    print("EXAMPLE 5: Matrix Statistics")
    print("="*70)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load matrix
            data = f['features/csr/data'][:]
            indices = f['features/csr/indices'][:]
            indptr = f['features/csr/indptr'][:]
            shape = f['features'].attrs['shape']
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
            
            print("\nMatrix statistics:")
            print(f"  Shape: {X.shape[0]:,} proteins × {X.shape[1]:,} features")
            print(f"  Total elements: {X.shape[0] * X.shape[1]:,}")
            print(f"  Non-zero elements: {X.nnz:,}")
            print(f"  Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.4f}%")
            
            # Per-protein statistics
            print("\nPer-protein statistics:")
            nonzeros_per_protein = np.diff(X.indptr)
            print(f"  Mean features/protein: {nonzeros_per_protein.mean():.1f}")
            print(f"  Median features/protein: {np.median(nonzeros_per_protein):.1f}")
            print(f"  Min features/protein: {nonzeros_per_protein.min()}")
            print(f"  Max features/protein: {nonzeros_per_protein.max()}")
            print(f"  Std dev: {nonzeros_per_protein.std():.1f}")
            
            # Per-feature statistics
            print("\nPer-feature statistics:")
            nonzeros_per_feature = np.array(X.sum(axis=0)).flatten()
            nonzeros_per_feature = nonzeros_per_feature[nonzeros_per_feature > 0]
            print(f"  Mean occurrences/feature: {nonzeros_per_feature.mean():.1f}")
            print(f"  Median occurrences/feature: {np.median(nonzeros_per_feature):.1f}")
            print(f"  Min occurrences/feature: {nonzeros_per_feature.min():.0f}")
            print(f"  Max occurrences/feature: {nonzeros_per_feature.max():.0f}")
            
            # Value statistics
            print("\nValue statistics:")
            print(f"  Mean value (non-zero): {X.data.mean():.4f}")
            print(f"  Median value (non-zero): {np.median(X.data):.4f}")
            print(f"  Min value: {X.data.min():.4f}")
            print(f"  Max value: {X.data.max():.4f}")
            print(f"  Std dev: {X.data.std():.4f}")
            
    except FileNotFoundError:
        print(f"\n✗ File not found: {h5_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Run all examples."""
    import sys
    
    # Check if HDF5 file provided
    h5_file = 'output.h5'
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
    
    print("\n" + "="*70)
    print("IPS Feature Processing Pipeline - Examples & Usage")
    print("="*70)
    print(f"\nUsing HDF5 file: {h5_file}")
    print("(Provide filename as argument to use a different file)")
    print("="*70 + "\n")
    
    # Run examples
    example_processing()
    example_loading(h5_file)
    example_feature_selection(h5_file)
    example_protein_query(h5_file)
    example_statistics(h5_file)
    
    print("="*70)
    print("All examples completed!")
    print("="*70)
    print("\nFor more information:")
    print("  - Read IPS_PROCESSING_GUIDE.md")
    print("  - Check the README in the HDF5 file")
    print("  - Review the summary report (.txt file)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
