# File: /graph-preprocessing-pipeline/graph-preprocessing-pipeline/src/training_ids.py

import os
import pandas as pd
import numpy as np
from typing import List
import argparse

def create_training_ids_file(csv_file: str, output_path: str, train_ratio: float = 0.8, 
                           random_seed: int = 42, eid_col: str = 'eid') -> str:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    
    if eid_col not in df.columns:
        raise ValueError(f"Column '{eid_col}' not found in CSV file")
    
    available_eids = df[eid_col].tolist()
    print(f"Found {len(available_eids)} unique EIDs")
    
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(available_eids))
    split_idx = int(len(available_eids) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_eids = [available_eids[i] for i in train_indices]
    val_eids = [available_eids[i] for i in val_indices]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for eid in train_eids:
            formatted_id = f"{eid}_25752_2_0"
            f.write(f"{formatted_id}\n")
    
    print(f"Created training IDs file with {len(train_eids)} IDs: {output_path}")
    print(f"Training ratio: {train_ratio:.2f}")
    print(f"Random seed: {random_seed}")
    print(f"Sample formatted IDs:")
    for i, eid in enumerate(train_eids[:3]):
        formatted_id = f"{eid}_25752_2_0"
        print(f"  {i+1}: {eid} -> {formatted_id}")
    
    return output_path

def validate_training_ids_file(ids_file: str, csv_file: str, eid_col: str = 'eid') -> None:
    with open(ids_file, 'r') as f:
        train_formatted_ids = set(line.strip() for line in f if line.strip())
    
    train_eids = set()
    for formatted_id in train_formatted_ids:
        eid = int(formatted_id.split('_')[0])
        train_eids.add(eid)
    
    df = pd.read_csv(csv_file)
    csv_eids = set(df[eid_col].tolist())
    
    missing_ids = train_eids - csv_eids
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} training EIDs not found in CSV:")
        print(f"Missing EIDs (first 10): {list(missing_ids)[:10]}")
    else:
        print("✓ All training EIDs found in CSV file")
    
    print(f"Training formatted IDs: {len(train_formatted_ids)}")
    print(f"Training EIDs (extracted): {len(train_eids)}")
    print(f"CSV EIDs: {len(csv_eids)}")
    print(f"EID overlap: {len(train_eids & csv_eids)}")
    
    print("Sample formatted training IDs:")
    for i, formatted_id in enumerate(list(train_formatted_ids)[:3]):
        eid = formatted_id.split('_')[0]
        print(f"  {i+1}: {formatted_id} (EID: {eid})")

def validate_against_matrix_files(ids_file: str, matrix_dir: str) -> None:
    with open(ids_file, 'r') as f:
        train_ids = set(line.strip() for line in f if line.strip())
    
    import glob
    csv_files = glob.glob(os.path.join(matrix_dir, "*.csv"))
    
    actual_prefixes = set()
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        prefix = filename[:17]
        actual_prefixes.add(prefix)
    
    matched_ids = train_ids & actual_prefixes
    missing_in_files = train_ids - actual_prefixes
    
    print(f"\nValidation against matrix files in {matrix_dir}:")
    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Unique prefixes in files: {len(actual_prefixes)}")
    print(f"Training IDs: {len(train_ids)}")
    print(f"Matched training IDs: {len(matched_ids)}")
    print(f"Training IDs missing from files: {len(missing_in_files)}")
    
    if missing_in_files:
        print(f"Missing training IDs (first 10): {list(missing_in_files)[:10]}")
    
    print("Sample actual file prefixes:")
    for i, prefix in enumerate(list(actual_prefixes)[:3]):
        print(f"  {i+1}: {prefix}")
    
    print("Sample training IDs:")
    for i, train_id in enumerate(list(train_ids)[:3]):
        print(f"  {i+1}: {train_id}")