import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os

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

def process_single_eid(args):
    """
    Wrapper function to process a single eid - needed for multiprocessing
    """
    eid_value, csv_file, fc_regions_file, output_directory = args
    
    try:
        output_file_csv = f"{output_directory}/{eid_value}_25752_2_0_FC_matrix.csv"
        matrix_df = create_symmetric_matrix_from_csv(csv_file, eid_value, fc_regions_file, output_file_csv)
        return {'success': True, 'eid': eid_value, 'message': f"Successfully processed {eid_value}"}
    except Exception as e:
        return {'success': False, 'eid': eid_value, 'message': str(e)}

# Main execution
if __name__ == "__main__":
    # Process all eids in the CSV file
    csv_file = "/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_master_GF_no_outliers.csv"
    fc_regions_file = "/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/FC_regions.txt"
    output_directory = "/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/FC_matrices/GF"
    
    # Read the CSV to get all unique eids
    df = pd.read_csv(csv_file)
    unique_eids = df['eid'].unique()
    
    print(f"Found {len(unique_eids)} unique eids to process")
    
    # Create arguments for each eid
    args_list = [(eid, csv_file, fc_regions_file, output_directory) for eid in unique_eids]
    
    # Determine number of processes (use all available cores, or specify a number)
    num_processes = cpu_count()  # Use all available cores
    # num_processes = 4  # Or specify a specific number
    
    print(f"Using {num_processes} processes")
    
    # Process in parallel
    successful_count = 0
    failed_eids = []
    
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = pool.imap(process_single_eid, args_list)
        
        for i, result in enumerate(results):
            if result['success']:
                successful_count += 1
            else:
                failed_eids.append(result['eid'])
                print(f"Error processing eid {result['eid']}: {result['message']}")
            
            # Print progress every 100 subjects
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(unique_eids)} subjects")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} subjects")
    print(f"Failed: {len(failed_eids)} subjects")
    
    if failed_eids:
        print(f"Failed eids: {failed_eids[:10]}...")  # Show first 10 failed eids