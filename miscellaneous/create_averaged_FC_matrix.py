import numpy as np
import pandas as pd
import os

def get_expected_columns(prefix, matrix_size):
    """
    Generate the expected column names based on upper-triangular indices.
    E.g. FC25 (IC1-IC2), FC25 (IC1-IC3) ...
    """
    expected_cols = []
    # 1-indexed up to matrix_size
    for i in range(1, matrix_size + 1):
        for j in range(i + 1, matrix_size + 1):
            expected_cols.append(f"{prefix} (IC{i}-IC{j})")
    return expected_cols

def create_average_matrix(df, prefix, matrix_size, output_file=None):
    """
    Safely checks for required data, averages the values across all subjects,
    and returns an N x N symmetric DataFrame. Raises an error if any values are missing.
    """
    expected_cols = get_expected_columns(prefix, matrix_size)
    num_required_edges = len(expected_cols)
    
    # 1. Safely check if all required data columns are present in the DataFrame
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Cannot construct {prefix} matrix! Missing {len(missing_cols)} "
            f"required columns out of {num_required_edges}.\n"
            f"Examples of missing columns: {missing_cols[:5]}"
        )
    print(f"Validation successful: All {num_required_edges} columns found for {prefix}.")
    
    # 2. Strictly check that NO values are missing (NaN) in these columns
    if df[expected_cols].isna().any().any():
        num_missing = df[expected_cols].isna().sum().sum()
        raise ValueError(
            f"Missing data detected! Found {num_missing} missing values in the {prefix} columns. "
            "All participants must have completely intact data for these regions."
        )
        
    print(f"Calculating average for {prefix} across {len(df)} participants...")
    
    # 3. Calculate the column-wise arithmetic mean
    mean_values = df[expected_cols].mean()
    
    # 4. Create the empty N x N matrix
    matrix = np.zeros((matrix_size, matrix_size))
    
    # 5. Populate the upper triangle and mirror it to the lower triangle (symmetric)
    idx = 0
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            val = mean_values.iloc[idx]
            matrix[i, j] = val
            matrix[j, i] = val
            idx += 1
            
    # Diagonal represents self-correlation, usually 0 or 1. We'll set to 0.0.
    np.fill_diagonal(matrix, 0.0) 
    
    # 6. Label indices and columns properly
    labels = [f"IC{i}" for i in range(1, matrix_size + 1)]
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    
    # 7. Optionally save to CSV
    if output_file:
        matrix_df.to_csv(output_file)
        print(f"-> Successfully saved {prefix} averaged matrix to {output_file}\n")
        
    return matrix_df

if __name__ == "__main__":
    # Define file paths
    data_path = "/Users/baileyng/MIND_data/UKB_new_data/combined_data_no_outliers/combined_data_master_no_outliers.csv"
    
    # Define the new output directory
    output_dir = "/Users/baileyng/Documents/SOBP/SOBP Poster/Figures"
    
    # Safely create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the specific output file paths
    output_fc25_path = os.path.join(output_dir, "averaged_FC25_matrix.csv")
    output_fc100_path = os.path.join(output_dir, "averaged_FC100_matrix.csv")
    
    print(f"Loading participant data from {data_path}...")
    try:
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} total participants.")
        
        # Create averaged FC25 Matrix (21x21 regions requires 210 edges)
        fc25_matrix = create_average_matrix(
            df=df, 
            prefix="FC25", 
            matrix_size=21, 
            output_file=output_fc25_path
        )
        
        # Create averaged FC100 Matrix (55x55 regions requires 1485 edges)
        fc100_matrix = create_average_matrix(
            df=df, 
            prefix="FC100", 
            matrix_size=55, 
            output_file=output_fc100_path
        )
        
        print("All averaged matrices were successfully generated!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {data_path}. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")