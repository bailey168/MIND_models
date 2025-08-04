import numpy as np
import pandas as pd

def create_symmetric_matrix_from_txt(txt_file_path, output_csv_path=None):
    """
    Read 210 numbers from txt file and create a 21x21 symmetric matrix.
    The numbers represent the upper triangular part of a correlation matrix.
    """
    
    # Read the numbers from the text file
    with open(txt_file_path, 'r') as file:
        # Read all numbers, handling various formats (space, comma, or newline separated)
        content = file.read()
        numbers = [float(x) for x in content.replace(',', ' ').split()]
    
    # Verify we have exactly 210 numbers (21*20/2 for upper triangular)
    if len(numbers) != 210:
        raise ValueError(f"Expected 210 numbers, but got {len(numbers)}")
    
    # Create 21x21 empty matrix
    matrix_size = 21
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the upper triangular part (excluding diagonal)
    idx = 0
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            matrix[i, j] = numbers[idx]
            matrix[j, i] = numbers[idx]  # Make it symmetric
            idx += 1

    # Set diagonal to 0
    np.fill_diagonal(matrix, 0.0)
    
    # Create IC labels
    ic_labels = [f'IC{i+1}' for i in range(21)]
    
    # Convert to DataFrame with IC labels
    df = pd.DataFrame(matrix, index=ic_labels, columns=ic_labels)
    
    # Save as CSV if output path is provided
    if output_csv_path is None:
        output_csv_path = txt_file_path.replace('.txt', '_matrix.csv')
    
    df.to_csv(output_csv_path)
    print(f"Symmetric matrix saved to: {output_csv_path}")
    
    return df

# Main execution
if __name__ == "__main__":
    txt_file = "/Users/baileyng/MIND_data/MIND_results/1000177_25752_2_0.txt"
    output_file = "/Users/baileyng/MIND_data/MIND_results/1000177_25752_2_0_FC_matrix.csv"
    
    # Create the symmetric matrix
    matrix_df = create_symmetric_matrix_from_txt(txt_file, output_file)
    
    # Display basic info about the matrix
    print(f"Matrix shape: {matrix_df.shape}")
    print(f"Matrix preview:")
    print(matrix_df.iloc[:5, :5])  # Show first 5x5 corner



def create_symmetric_matrix_from_csv(csv_file_path, eid, fc_regions_file_path, output_csv_path=None):
    """
    Read FC values from CSV file for a specific eid and create a 21x21 symmetric matrix.
    The FC values are stored in columns with names matching the FC_regions.txt file.
    
    Args:
        csv_file_path: Path to the CSV file containing FC data
        eid: The eid value to filter for
        fc_regions_file_path: Path to FC_regions.txt file containing column names
        output_csv_path: Optional output path for the matrix CSV
    """
    
    # Read the FC regions (column names) from the text file
    with open(fc_regions_file_path, 'r') as file:
        fc_regions = [line.strip() for line in file.readlines()]
    
    # Verify we have exactly 210 region pairs
    if len(fc_regions) != 210:
        raise ValueError(f"Expected 210 FC regions, but got {len(fc_regions)}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Filter by eid
    subject_row = df[df['eid'] == eid]
    
    if subject_row.empty:
        raise ValueError(f"No data found for eid: {eid}")
    
    if len(subject_row) > 1:
        raise ValueError(f"Multiple rows found for eid: {eid}")
    
    # Extract FC values using the region names as column names
    fc_values = []
    for region in fc_regions:
        if region not in df.columns:
            raise ValueError(f"Column '{region}' not found in CSV file")
        fc_values.append(float(subject_row[region].iloc[0]))
    
    # Create 21x21 empty matrix
    matrix_size = 21
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the upper triangular part (excluding diagonal)
    idx = 0
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            matrix[i, j] = fc_values[idx]
            matrix[j, i] = fc_values[idx]  # Make it symmetric
            idx += 1

    # Set diagonal to 0
    np.fill_diagonal(matrix, 0.0)
    
    # Create IC labels
    ic_labels = [f'IC{i+1}' for i in range(21)]
    
    # Convert to DataFrame with IC labels
    matrix_df = pd.DataFrame(matrix, index=ic_labels, columns=ic_labels)
    
    # Save as CSV if output path is provided
    if output_csv_path is None:
        output_csv_path = f"{eid}_FC_matrix.csv"
    
    matrix_df.to_csv(output_csv_path)
    print(f"Symmetric matrix for eid {eid} saved to: {output_csv_path}")
    
    return matrix_df

# Main execution
if __name__ == "__main__":
    # # Original function example
    # txt_file = "/Users/baileyng/MIND_data/MIND_results/1000177_25752_2_0.txt"
    # output_file = "/Users/baileyng/MIND_data/MIND_results/1000177_25752_2_0_FC_matrix.csv"
    
    # # Create the symmetric matrix from txt file
    # matrix_df = create_symmetric_matrix_from_txt(txt_file, output_file)
    
    # New function example - uncomment and modify paths as needed
    csv_file = "/Users/baileyng/MIND_data/ukb_cog/ukb_master_PAL_no_outliers.csv"
    eid_value = 1392086  # or whatever eid you want to filter for
    fc_regions_file = "/Users/baileyng/MIND_models/region_names/FC_regions.txt"
    output_file_csv = f"/Users/baileyng/MIND_data/MIND_results/{eid_value}_25752_2_0_FC_matrix.csv"

    matrix_df = create_symmetric_matrix_from_csv(csv_file, eid_value, fc_regions_file, output_file_csv)
    
    # Display basic info about the matrix
    print(f"Matrix shape: {matrix_df.shape}")
    print(f"Matrix preview:")
    print(matrix_df.iloc[:5, :5])  # Show first 5x5 corner