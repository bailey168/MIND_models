import os

# Paths
DATA_DIR = "/scratch/bng/cartbind/data"
OUTPUT_DIR = "/scratch/bng/cartbind/data"
CSV_FILE = os.path.join(DATA_DIR, "ukb_master_GF_no_outliers.csv")
TRAINING_IDS_FILE = os.path.join(OUTPUT_DIR, "training_ids/GF.txt")
MATRIX_DIR = os.path.join(DATA_DIR, "FC_matrices/GF")

# Column names
DATA_COL = "20016-2.0"
EID_COL = "eid"

# Parameters
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
KEEP_PERCENT = 1.0
KEEP_SELF_LOOPS = True

# Standardization control
USE_EDGE_STANDARDIZATION = True

# Dynamic paths based on parameters
EDGE_PERCENT_STR = str(int(KEEP_PERCENT * 100))
GRAPH_DIR = os.path.join(OUTPUT_DIR, f"FC_graphs/raw/GF{EDGE_PERCENT_STR}")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, f"FC_graphs/processed/GF{EDGE_PERCENT_STR}")  # Add this

# Dynamic dataset name
DATASET_NAME = f"custom_dataset_selfloops_{KEEP_SELF_LOOPS}_edgeft_None_norm_{USE_EDGE_STANDARDIZATION}"