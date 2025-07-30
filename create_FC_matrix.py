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