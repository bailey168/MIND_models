import os
import pandas as pd
import numpy as np
import torch
import pickle
import time
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_graph_data_with_labels(data_col: str, graph_dir: str, csv_file: str) -> List[Dict]:
    """
    Load graph data files and match them with labels and demographic data from CSV file.
    
    Args:
        data_col: Column name in CSV file that contains y labels
        graph_dir: Directory containing graph files named '{eid}_25752_2_0_fc_graph.pt'
        csv_file: CSV file with 'eid' column, data_col column for labels, and demographic columns

    Returns:
        List of dictionaries with processed data including demographics
    """
    
    # Load CSV with labels and demographics
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")

    # Define demographic columns
    age_col = "21003-2.0"
    sex_col = "31-0.0"
    assessment_centre_col = "54-2.0"
    
    # Check if demographic columns exist
    required_cols = [data_col, age_col, sex_col, assessment_centre_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return []

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
        
        # Find matching row in dataframe
        matching_rows = df[df['eid'] == eid]

        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            
            # Load graph data
            graph_path = os.path.join(graph_dir, file)
            try:
                graph_data = torch.load(graph_path, weights_only=False)
                
                # Set label (float for regression)
                label = row[data_col]
                graph_data.y = torch.tensor([label], dtype=torch.float32)
                
                # Store raw demographic data (we'll process after train/test split)
                matched_data.append({
                    'eid': eid,
                    'graph': graph_data,
                    'label': label,
                    'age': row[age_col],
                    'sex': row[sex_col],
                    'assessment_centre': row[assessment_centre_col],
                    'file': file
                })
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        else:
            unmatched_eids.append(eid)
    
    print(f"Successfully matched {len(matched_data)} graphs with labels and demographics")
    if unmatched_eids:
        print(f"Warning: {len(unmatched_eids)} eids not found in CSV: {unmatched_eids[:5]}...")
    
    return matched_data


def process_demographics(data_list, age_scaler, assessment_centre_encoder):
    processed_data = []

    for item in data_list:
        # Get the graph data
        graph_data = item['graph']

        # Process demographics
        age_scaled = age_scaler.transform([[item['age']]])[0]
        sex = [item['sex']]
        assessment_centre_onehot = assessment_centre_encoder.transform([[item['assessment_centre']]])[0]

        # Combine demographic features
        demo_features = np.concatenate([age_scaled, sex, assessment_centre_onehot])

        # Add demographic to graph data
        graph_data.demographics = torch.tensor(demo_features, dtype=torch.float32).unsqueeze(0)

        processed_data.append(graph_data)

    return processed_data


def create_train_test_split(matched_data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List, dict]:
    """
    Split the matched data into train and test sets, then fit demographic preprocessors on training data only.
    
    Args:
        matched_data: List of dictionaries with graph data, labels, and demographics
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data, preprocessors)
    """
    np.random.seed(random_seed)
    
    # Shuffle the data
    indices = np.random.permutation(len(matched_data))
    split_idx = int(len(matched_data) * train_ratio)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_raw = [matched_data[i] for i in train_indices]
    test_raw = [matched_data[i] for i in test_indices]
    
    print(f"Train set: {len(train_raw)} samples")
    print(f"Test set: {len(test_raw)} samples")

    # Fit preprocessors on training data only
    train_ages = np.array([item['age'] for item in train_raw]).reshape(-1, 1)
    train_assessment_centres = np.array([item['assessment_centre'] for item in train_raw]).reshape(-1, 1)

    # Fit age scaler on training data
    age_scaler = StandardScaler()
    age_scaler.fit(train_ages)

    # Fit one-hot encoder on training data (drop='first' to avoid multicollinearity)
    assessment_centre_encoder = OneHotEncoder(drop='first', sparse_output=False)
    assessment_centre_encoder.fit(train_assessment_centres)

    print(f"Age scaler fitted - mean: {age_scaler.mean_[0]:.2f}, std: {age_scaler.scale_[0]:.2f}")
    print(f"Assessment centre encoder fitted - categories: {assessment_centre_encoder.categories_[0]}")
    print(f"Assessment centre features after drop='first': {assessment_centre_encoder.get_feature_names_out()}")

    # Process train and test data
    train_data = process_demographics(train_raw, age_scaler, assessment_centre_encoder)
    test_data = process_demographics(test_raw, age_scaler, assessment_centre_encoder)

    # Store preprocessors
    preprocessors = {
        'age_scaler': age_scaler,
        'assessment_centre_encoder': assessment_centre_encoder
    }

    return train_data, test_data, preprocessors


def save_dataset_in_standard_format(train_data: List, test_data: List, output_dir: str, dataset_name: str,
                                   preprocessors: dict = None) -> None:
    """
    Save the dataset in the format expected by the existing code.
    
    Args:
        train_data: List of training graph data
        test_data: List of testing graph data
        output_dir: Directory to save the dataset
        dataset_name: Name for the dataset file
        preprocessors: Dictionary containing fitted preprocessors
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

    # Add preprocessors if provided
    if preprocessors:
        struct['preprocessing'] = preprocessors
    
    # Save as pickle file
    output_path = os.path.join(output_dir, f"{dataset_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(struct, f)
    
    print(f"Successfully saved dataset to {output_path}")

def process_custom_graph_dataset(data_col: str, graph_dir: str, csv_file: str, output_dir: str, 
                                dataset_name: str = "custom_graph_dataset", 
                                train_ratio: float = 0.8, random_seed: int = 42):
    """
    Complete pipeline to process custom graph dataset with demographics.
    
    Args:
        data_col: Column name in CSV file that contains y labels
        graph_dir: Directory containing graph files
        csv_file: CSV file with labels and demographics
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
    train_data, test_data, preprocessors = create_train_test_split(matched_data, train_ratio, random_seed)
    
    # Create directory structure like in the original code
    num_train = len(train_data)
    num_test = len(test_data)
    dir_name = f"train_{num_train}_test_{num_test}"
    final_output_dir = os.path.join(output_dir, dir_name)
    
    # Save in standard format
    save_dataset_in_standard_format(train_data, test_data, final_output_dir, dataset_name, preprocessors)
    
    return final_output_dir

# Example usage
if __name__ == "__main__":
    # Set variables and paths
    data_col = "23324-2.0"
    graph_dir = "/scratch/bng/cartbind/data/FC_graphs/raw/DSST30"
    csv_file = "/scratch/bng/cartbind/data/ukb_master_DSST_no_outliers.csv"
    output_dir = "/scratch/bng/cartbind/data/FC_graphs/processed/DSST30"
    dataset_name = "custom_dataset_selfloops_True_edgeft_None_norm_True"

    start_time = time.time()

    # Process the dataset
    result_dir = process_custom_graph_dataset(
        data_col=data_col,
        graph_dir=graph_dir,
        csv_file=csv_file,
        output_dir=output_dir,
        dataset_name=dataset_name,
        train_ratio=0.8,
        random_seed=42
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Dataset processing complete. Files saved in: {result_dir}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")