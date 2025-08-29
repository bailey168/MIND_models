import os
import pandas as pd
import numpy as np
from typing import List
import argparse

def create_training_ids_file(csv_file: str, output_path: str, train_ratio: float = 0.8, 
                           random_seed: int = 42, eid_col: str = 'eid') -> str:
    """
    Create a file containing training IDs for use in matrix processing.
    
    Args:
        csv_file: Path to CSV file containing EIDs and other data
        output_path: Path to save the training IDs file
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility
        eid_col: Name of the column containing EIDs
    
    Returns:
        Path to the created training IDs file
    """
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Check if EID column exists
    if eid_col not in df.columns:
        raise ValueError(f"Column '{eid_col}' not found in CSV file")
    
    # Get all available EIDs
    available_eids = df[eid_col].tolist()
    print(f"Found {len(available_eids)} unique EIDs")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the EIDs
    indices = np.random.permutation(len(available_eids))
    split_idx = int(len(available_eids) * train_ratio)
    train_indices = indices[:split_idx]
    
    # Extract training EIDs
    train_eids = [available_eids[i] for i in train_indices]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write training IDs to file
    with open(output_path, 'w') as f:
        for eid in train_eids:
            f.write(f"{eid}\n")
    
    print(f"Created training IDs file with {len(train_eids)} IDs: {output_path}")
    print(f"Training ratio: {train_ratio:.2f}")
    print(f"Random seed: {random_seed}")
    
    return output_path

def validate_training_ids_file(ids_file: str, csv_file: str, eid_col: str = 'eid') -> None:
    """
    Validate that all IDs in the training file exist in the CSV.
    
    Args:
        ids_file: Path to training IDs file
        csv_file: Path to CSV file
        eid_col: Name of the column containing EIDs
    """
    # Load training IDs
    with open(ids_file, 'r') as f:
        train_ids = set(int(line.strip()) for line in f if line.strip())
    
    # Load CSV EIDs
    df = pd.read_csv(csv_file)
    csv_eids = set(df[eid_col].tolist())
    
    # Check for missing IDs
    missing_ids = train_ids - csv_eids
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} training IDs not found in CSV:")
        print(f"Missing IDs (first 10): {list(missing_ids)[:10]}")
    else:
        print("✓ All training IDs found in CSV file")
    
    print(f"Training IDs: {len(train_ids)}")
    print(f"CSV EIDs: {len(csv_eids)}")
    print(f"Overlap: {len(train_ids & csv_eids)}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Create training IDs file for edge standardization')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file containing EIDs and labels')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the training IDs file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--eid_col', type=str, default='eid',
                       help='Name of the EID column in CSV (default: eid)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the created training IDs file')
    
    args = parser.parse_args()
    
    # Create training IDs file
    created_file = create_training_ids_file(
        csv_file=args.csv_file,
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        eid_col=args.eid_col
    )
    
    # Validate if requested
    if args.validate:
        print("\nValidating training IDs file...")
        validate_training_ids_file(created_file, args.csv_file, args.eid_col)

if __name__ == "__main__":
    # You can either use command line arguments or set these directly
    USE_CLI = False  # Set to True to use command line arguments
    
    if USE_CLI:
        main()
    else:
        # Direct configuration - modify these paths as needed
        csv_file = "/scratch/bng/cartbind/data/ukb_master_GF_no_outliers.csv"
        output_path = "/scratch/bng/cartbind/data/FC_graphs/processed/GF30/training_ids.txt"
        train_ratio = 0.8
        random_seed = 42
        eid_col = 'eid'
        
        print("Creating training IDs file...")
        created_file = create_training_ids_file(
            csv_file=csv_file,
            output_path=output_path,
            train_ratio=train_ratio,
            random_seed=random_seed,
            eid_col=eid_col
        )
        
        print("\nValidating training IDs file...")
        validate_training_ids_file(created_file, csv_file, eid_col)
        
        print(f"\n✓ Training IDs file created successfully!")
        print(f"Next steps:")
        print(f"1. Run matrix_to_graph_parallelized.py with TRAIN_IDS_FILE = '{created_file}'")
        print(f"2. Run create_dataset.py to create the final dataset")