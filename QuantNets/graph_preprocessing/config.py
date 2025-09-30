import os
import argparse

dataset_dict = {"GF": "20016-2.0", "PAL": "20197-2.0", "DSST": "23324-2.0", "TMT": "trailmaking_score"}

def parse_args():
    parser = argparse.ArgumentParser(description='Graph preprocessing pipeline')
    parser.add_argument('--dataset', choices=['GF', 'PAL', 'DSST', 'TMT'], default='DSST',
                       help='Dataset to process (default: DSST)')
    parser.add_argument('--keep-percent', type=float, default=0.15,
                       help='Percentage of edges to keep (default: 0.15)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
DATASET = args.dataset
KEEP_PERCENT = args.keep_percent

# Paths
DATA_DIR = "/scratch/bng/cartbind/data"
OUTPUT_DIR = "/scratch/bng/cartbind/data"
CSV_FILE = os.path.join(DATA_DIR, f"ukb_master_{DATASET}_no_outliers.csv")
TRAINING_IDS_FILE = os.path.join(OUTPUT_DIR, f"training_ids/{DATASET}.txt")
MATRIX_DIR = os.path.join(DATA_DIR, f"FC_matrices/{DATASET}")

# Column names
DATA_COL = dataset_dict[DATASET]
EID_COL = "eid"

# Parameters
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
KEEP_SELF_LOOPS = True

# Standardization control
USE_EDGE_STANDARDIZATION = True

# Dynamic paths based on parameters
EDGE_PERCENT_STR = str(int(KEEP_PERCENT * 100))
GRAPH_DIR = os.path.join(OUTPUT_DIR, f"FC_graphs/raw/{DATASET}{EDGE_PERCENT_STR}")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, f"FC_graphs/processed/{DATASET}{EDGE_PERCENT_STR}")

# Dynamic dataset name
DATASET_NAME = f"custom_dataset_selfloops_{KEEP_SELF_LOOPS}_edgeft_None_norm_{USE_EDGE_STANDARDIZATION}"