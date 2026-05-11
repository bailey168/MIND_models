import pandas as pd
import numpy as np
import os

def reconstruct_matrices_from_csv(input_csv_path, output_dir, region_labels=None):
    """
    Reconstruct 68x68 matrices from flattened upper triangle data and save as individual CSV files.
    
    Parameters:
    input_csv_path: Path to the CSV file containing flattened data
    output_dir: Directory to save the reconstructed matrices
    region_labels: List of 68 region labels (if None, will try to extract from column names)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the master CSV file
    df = pd.read_csv(input_csv_path)
    
    # Extract EIDs
    eids = df['eid'].tolist()
    
    # If region labels not provided, try to extract from column names
    if region_labels is None:
        # Find columns that match the "regionA-regionB" pattern
        upper_triangle_cols = [col for col in df.columns if '-' in col and col != 'eid']
        
        # Extract unique region names
        regions = set()
        for col in upper_triangle_cols:
            parts = col.split('-')
            if len(parts) == 2:
                regions.update(parts)
        
        region_labels = sorted(list(regions))
        print(f"Extracted {len(region_labels)} region labels from column names")
    
    n_regions = len(region_labels)
    print(f"Reconstructing {n_regions}x{n_regions} matrices for {len(eids)} subjects")
    
    # Get upper triangle column names in the expected order
    upper_triangle_cols = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            col_name = f'{region_labels[i]}-{region_labels[j]}'
            upper_triangle_cols.append(col_name)
    
    # Check if all expected columns exist
    missing_cols = [col for col in upper_triangle_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols[:10]}...")  # Show first 10
        # Filter to only existing columns
        upper_triangle_cols = [col for col in upper_triangle_cols if col in df.columns]
    
    print(f"Using {len(upper_triangle_cols)} upper triangle values")
    
    # Process each EID
    for idx, eid in enumerate(eids):
        if idx % 100 == 0:
            print(f"Processing {idx+1}/{len(eids)}: EID {eid}")
        
        # Initialize symmetric matrix
        matrix = np.zeros((n_regions, n_regions))
        
        # Fill upper triangle
        upper_idx = np.triu_indices(n_regions, k=1)
        
        # Extract values for this EID
        values = df.loc[df['eid'] == eid, upper_triangle_cols].values.flatten()
        
        # Check if we have the right number of values
        if len(values) != len(upper_idx[0]):
            print(f"Warning: Expected {len(upper_idx[0])} values but got {len(values)} for EID {eid}")
            continue
        
        # Fill upper triangle
        matrix[upper_idx] = values
        
        # Make matrix symmetric by copying upper triangle to lower triangle
        matrix = matrix + matrix.T
        
        # Create DataFrame with region labels
        matrix_df = pd.DataFrame(matrix, index=region_labels, columns=region_labels)
        
        # Save as CSV
        output_file = os.path.join(output_dir, f'{eid}_20263_2_0_aparc_MIND_matrix.csv')
        matrix_df.to_csv(output_file)
    
    print(f"Reconstruction complete! Matrices saved to {output_dir}")

def load_region_labels(region_file_path):
    """Load region labels from text file."""
    with open(region_file_path, 'r') as f:
        region_labels = [line.strip() for line in f.readlines() if line.strip()]
    return region_labels

# Example usage
if __name__ == "__main__":
    # Load region labels from file
    region_file_path = '/Users/baileyng/MIND_models/region_names/MIND_avg_regions.txt'
    region_labels = load_region_labels(region_file_path)
    
    print(f"Loaded {len(region_labels)} region labels from {region_file_path}")
    print(f"First 5 regions: {region_labels[:5]}")
    
    input_csv = '/Users/baileyng/MIND_data/ukb_cog/ukb_master_all.csv'
    output_directory = '/Users/baileyng/MIND_data/reconstructed_matrices'
    
    # Run the reconstruction
    reconstruct_matrices_from_csv(input_csv, output_directory, region_labels)