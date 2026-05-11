import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

def create_symmetric_matrix_from_txt(txt_file_path, matrix_size, fc_regions_file_path, output_csv_path=None):
    """
    Read FC values from txt file and create a symmetric matrix.
    The numbers represent the upper triangular part extracted COLUMNWISE (to match MATLAB/FC_colnames.py).
    
    Args:
        txt_file_path: Path to the txt file containing FC values
        matrix_size: Size of the square matrix (e.g., 21 for 21x21)
        fc_regions_file_path: Path to file containing column names for the FC regions
        output_csv_path: Optional output path for the matrix CSV
    """
    
    # Calculate expected number of values
    expected_values = matrix_size * (matrix_size - 1) // 2
    
    # Read the numbers from the text file
    with open(txt_file_path, 'r') as file:
        content = file.read()
        numbers = [float(x) for x in content.replace(',', ' ').split()]
    
    # Verify we have the correct number of values
    if len(numbers) != expected_values:
        raise ValueError(f"Expected {expected_values} numbers for {matrix_size}x{matrix_size} matrix, but got {len(numbers)}")
    
    # Read the FC regions (column names) from the text file if provided
    if fc_regions_file_path and os.path.exists(fc_regions_file_path):
        with open(fc_regions_file_path, 'r') as file:
            fc_regions = [line.strip() for line in file.readlines()]
        
        if len(fc_regions) != expected_values:
            print(f"Warning: FC regions file has {len(fc_regions)} entries, expected {expected_values}")
    
    # Create empty matrix
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the upper triangular part COLUMNWISE (to match MATLAB/FC_colnames.py ordering)
    # This uses the transpose with triu_indices to get columnwise ordering
    idx = 0
    for j in range(matrix_size):  # columns first
        for i in range(j):  # rows less than column index
            matrix[i, j] = numbers[idx]
            matrix[j, i] = numbers[idx]  # Make it symmetric
            idx += 1
    
    # Set diagonal to 0
    np.fill_diagonal(matrix, 0.0)
    
    # Create IC labels
    ic_labels = [f'IC{i+1}' for i in range(matrix_size)]
    
    # Convert to DataFrame with IC labels
    matrix_df = pd.DataFrame(matrix, index=ic_labels, columns=ic_labels)
    
    # Save as CSV if output path is provided
    if output_csv_path is None:
        output_csv_path = txt_file_path.replace('.txt', '_matrix.csv')
    
    matrix_df.to_csv(output_csv_path)
    print(f"Symmetric matrix saved to: {output_csv_path}")
    
    return matrix_df

def process_single_file(args):
    """
    Wrapper function to process a single txt file - needed for multiprocessing
    """
    txt_file, matrix_size, fc_regions_file, output_dir, file_type = args
    
    try:
        # Create output filename
        base_name = os.path.basename(txt_file)
        if file_type:
            output_file = os.path.join(output_dir, base_name.replace('.txt', f'_{file_type}_matrix.csv'))
        else:
            output_file = os.path.join(output_dir, base_name.replace('.txt', '_FC_matrix.csv'))
        
        # Process the file
        create_symmetric_matrix_from_txt(txt_file, matrix_size, fc_regions_file, output_file)
        return {'success': True, 'file': txt_file, 'message': f"Successfully processed {base_name}"}
    except Exception as e:
        return {'success': False, 'file': txt_file, 'message': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Convert txt files to symmetric FC matrices')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing txt files to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output CSV matrices')
    parser.add_argument('--matrix_size', type=int, default=21,
                        help='Size of the square matrix (default: 21 for 21x21)')
    parser.add_argument('--fc_regions_file', type=str, default=None,
                        help='Path to file containing FC region names/column names')
    parser.add_argument('--type', type=str, default=None,
                        help='Type label to add to output filename (e.g., "FC25", "FC100")')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of parallel processes (default: all available CPUs)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all txt files in input directory
    txt_files = list(Path(args.input_dir).glob('*.txt'))
    
    if not txt_files:
        print(f"No txt files found in {args.input_dir}")
        return
    
    print(f"Found {len(txt_files)} txt files to process")
    print(f"Matrix size: {args.matrix_size}x{args.matrix_size}")
    print(f"Expected values per file: {args.matrix_size * (args.matrix_size - 1) // 2}")
    if args.type:
        print(f"File type: {args.type}")
    
    # Determine number of processes
    num_processes = args.num_processes if args.num_processes else cpu_count()
    print(f"Using {num_processes} processes")
    
    # Create arguments for each file
    args_list = [(str(txt_file), args.matrix_size, args.fc_regions_file, args.output_dir, args.type) 
                 for txt_file in txt_files]
    
    # Process in parallel
    successful_count = 0
    failed_files = []
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap(process_single_file, args_list)
        
        for i, result in enumerate(results):
            if result['success']:
                successful_count += 1
            else:
                failed_files.append(result['file'])
                print(f"Error processing {result['file']}: {result['message']}")
            
            # Print progress every 100 files
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(txt_files)} files")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} files")
    print(f"Failed: {len(failed_files)} files")
    
    if failed_files:
        print(f"Failed files: {failed_files[:10]}...")  # Show first 10 failed files

if __name__ == "__main__":
    main()