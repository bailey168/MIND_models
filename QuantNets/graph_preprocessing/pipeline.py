import os
import time
from pathlib import Path
from training_ids import create_training_ids_file, validate_training_ids_file, validate_against_matrix_files
from matrix_to_graph_parallelized import process_all_matrices_with_standardization, process_all_matrices
from create_dataset import process_custom_graph_dataset
from config import (DATA_DIR, OUTPUT_DIR, CSV_FILE, TRAINING_IDS_FILE, 
                   GRAPH_DIR, MATRIX_DIR, PROCESSED_DIR, DATA_COL, EID_COL, DATASET_NAME,  # Add PROCESSED_DIR
                   TRAIN_RATIO, RANDOM_SEED, KEEP_PERCENT, KEEP_SELF_LOOPS,
                   USE_EDGE_STANDARDIZATION)

def run_pipeline():
    """Run the complete preprocessing pipeline"""

    print("=== Graph Preprocessing Pipeline ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CSV file: {CSV_FILE}")
    print(f"Matrix directory: {MATRIX_DIR}")
    print(f"Graph directory: {GRAPH_DIR}")  # This will now show the dynamic name
    print(f"Dataset name: {DATASET_NAME}")  # This will show the dynamic name
    print(f"Training ratio: {TRAIN_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Keep percent: {KEEP_PERCENT}")
    print(f"Keep self loops: {KEEP_SELF_LOOPS}")
    print(f"Use edge standardization: {USE_EDGE_STANDARDIZATION}")
    print()

    # Step 1: Create training IDs file
    print("Step 1: Creating training IDs file...")
    training_ids_file = create_training_ids_file(
        csv_file=CSV_FILE,
        output_path=TRAINING_IDS_FILE,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED,
        eid_col=EID_COL
    )
    print()

    # Step 2: Validate training IDs file
    print("Step 2: Validating training IDs file...")
    validate_training_ids_file(training_ids_file, CSV_FILE, EID_COL)
    print()

    # Step 3: Validate against matrix files
    print("Step 3: Validating against matrix files...")
    validate_against_matrix_files(training_ids_file, MATRIX_DIR)
    print()

    # Step 4: Process all matrices (with or without standardization)
    print("Step 4: Processing all matrices...")
    if USE_EDGE_STANDARDIZATION:
        process_all_matrices_with_standardization(
            input_dir=MATRIX_DIR,
            output_dir=GRAPH_DIR,
            train_ids_file=training_ids_file,
            keep_percent=KEEP_PERCENT,
            keep_self_loops=KEEP_SELF_LOOPS
        )
        print("Used per-edge standardization based on training data")
    else:
        process_all_matrices(
            input_dir=MATRIX_DIR,
            output_dir=GRAPH_DIR,
            keep_percent=KEEP_PERCENT,
            keep_self_loops=KEEP_SELF_LOOPS
        )
        print("No per-edge standardization applied")
    print()

    # Step 5: Create the dataset
    print("Step 5: Creating the dataset...")
    process_custom_graph_dataset(
        data_col=DATA_COL,
        graph_dir=GRAPH_DIR,
        csv_file=CSV_FILE,
        output_dir=PROCESSED_DIR,
        dataset_name=DATASET_NAME,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED
    )
    print()

    print("Pipeline execution completed successfully!")

def main():
    """Main function to run the entire preprocessing pipeline"""
    start_time = time.time()
    
    try:
        run_pipeline()
        end_time = time.time()
        print(f"Total pipeline runtime: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()