import os
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

def load_graph_data_with_labels(data_col: str, graph_dir: str, csv_file: str) -> List[Dict]:
    """
    Load graph data files and match them with labels from CSV file.
    
    Args:
        data_col: Column name in CSV file that contains y labels
        graph_dir: Directory containing graph files named '{eid}_25752_2_0_fc_graph.pt'
        csv_file: CSV file with 'eid' column and data_col column for labels

    Returns:
        List of dictionaries with processed data in the format expected by the models
    """
    
    # Load CSV with labels
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Create mapping from eid to label
    eid_to_label = dict(zip(df['eid'], df[f"{data_col}"]))
    print(f"Created label mapping for {len(eid_to_label)} eids")
    
    # Find all graph files
    graph_files = []
    for file in os.listdir(graph_dir):
        if file.endswith('_25752_2_0_fc_graph.pt'):
            graph_files.append(file)
    
    print(f"Found {len(graph_files)} graph files")
    
    # Load graphs and match with labels
    matched_data = []
    unmatched_eids = []
    
    for file in graph_files:
        # Extract eid from filename
        eid = int(file.split('_')[0])
        
        if eid in eid_to_label:
            # Load graph data
            graph_path = os.path.join(graph_dir, file)
            try:
                graph_data = torch.load(graph_path)
                
                # Set label
                label = eid_to_label[eid]
                graph_data.y = torch.tensor([label], dtype=torch.long)
                
                matched_data.append({
                    'eid': eid,
                    'graph': graph_data,
                    'label': label,
                    'file': file
                })
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        else:
            unmatched_eids.append(eid)
    
    print(f"Successfully matched {len(matched_data)} graphs with labels")
    if unmatched_eids:
        print(f"Warning: {len(unmatched_eids)} eids not found in CSV: {unmatched_eids[:5]}...")
    
    return matched_data

def create_train_test_split(matched_data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List]:
    """
    Split the matched data into train and test sets.
    
    Args:
        matched_data: List of dictionaries with graph data and labels
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    np.random.seed(random_seed)
    
    # Shuffle the data
    indices = np.random.permutation(len(matched_data))
    split_idx = int(len(matched_data) * train_ratio)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = [matched_data[i]['graph'] for i in train_indices]
    test_data = [matched_data[i]['graph'] for i in test_indices]
    
    print(f"Created train set with {len(train_data)} samples")
    print(f"Created test set with {len(test_data)} samples")
    
    return train_data, test_data

def save_dataset_in_standard_format(train_data: List, test_data: List, output_dir: str, dataset_name: str) -> None:
    """
    Save the dataset in the format expected by the existing code.
    
    Args:
        train_data: List of training graph data
        test_data: List of testing graph data
        output_dir: Directory to save the dataset
        dataset_name: Name for the dataset file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the structure
    struct = {
        "raw": {
            "x_train_data": [],
            "y_train_data": [],
            "x_test_data": [],
            "y_test_data": []
        },
        "geometric": {
            "gcn_train_data": train_data,
            "gcn_test_data": test_data,
            "sgcn_train_data": train_data,
            "sgcn_test_data": test_data
        }
    }
    
    # Save as pickle file
    output_path = os.path.join(output_dir, f"{dataset_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(struct, f)
    
    print(f"Successfully saved dataset to {output_path}")

def process_custom_graph_dataset(data_col: str, graph_dir: str, csv_file: str, output_dir: str, 
                                dataset_name: str = "custom_graph_dataset", 
                                train_ratio: float = 0.8, random_seed: int = 42):
    """
    Complete pipeline to process custom graph dataset.
    
    Args:
        data_col: Column name in CSV file that contains y labels
        graph_dir: Directory containing graph files
        csv_file: CSV file with labels
        output_dir: Output directory
        dataset_name: Name for the output dataset
        train_ratio: Ratio for train/test split
        random_seed: Random seed for reproducibility
    """
    
    # Load and match data
    matched_data = load_graph_data_with_labels(data_col, graph_dir, csv_file)
    
    if not matched_data:
        print("No matched data found. Please check your file paths and naming conventions.")
        return
    
    # Create train/test split
    train_data, test_data = create_train_test_split(matched_data, train_ratio, random_seed)
    
    # Create directory structure like in the original code
    num_train = len(train_data)
    num_test = len(test_data)
    dir_name = f"train_{num_train}_test_{num_test}"
    final_output_dir = os.path.join(output_dir, dir_name)
    
    # Save in standard format
    save_dataset_in_standard_format(train_data, test_data, final_output_dir, dataset_name)
    
    return final_output_dir

# Example usage
if __name__ == "__main__":
    # Set your paths here
    data_col = "23324-2.0"
    graph_directory = "/scratch/bng/cartbind/data/FC_graphs/raw/DSST"
    csv_file_path = "/scratch/bng/cartbind/data/ukb_master_DSST_no_outliers.csv"
    output_directory = "/scratch/bng/cartbind/data/FC_graphs/processed/DSST"
    dataset_name = "custom_dataset_selfloops_True_edgeft_None_norm_True"

    # Process the dataset
    result_dir = process_custom_graph_dataset(
        data_col=data_col,
        graph_dir=graph_directory,
        csv_file=csv_file_path,
        output_dir=output_directory,
        dataset_name=dataset_name,
        train_ratio=0.75,
        random_seed=42
    )
    
    print(f"Dataset processing complete. Files saved in: {result_dir}")