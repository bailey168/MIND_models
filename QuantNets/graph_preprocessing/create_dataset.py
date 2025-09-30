import os
import pandas as pd
import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_graph_data_with_labels(data_col: str, graph_dir: str, csv_file: str) -> List[Dict]:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")

    age_col = "21003-2.0"
    sex_col = "31-0.0"
    assessment_centre_col = "54-2.0"
    
    required_cols = [data_col, age_col, sex_col, assessment_centre_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return []

    graph_files = [file for file in os.listdir(graph_dir) if file.endswith('_fc_graph.pt')]
    print(f"Found {len(graph_files)} graph files")
    
    matched_data = []
    unmatched_eids = []
    
    for file in graph_files:
        eid = int(file.split('_')[0])
        matching_rows = df[df['eid'] == eid]

        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            graph_path = os.path.join(graph_dir, file)
            try:
                graph_data = torch.load(graph_path, weights_only=False)
                label = row[data_col]
                graph_data.y = torch.tensor([label], dtype=torch.float32)
                
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
        graph_data = item['graph']
        age_scaled = age_scaler.transform([[item['age']]])[0]
        sex = [item['sex']]
        assessment_centre_onehot = assessment_centre_encoder.transform([[item['assessment_centre']]])[0]

        demo_features = np.concatenate([age_scaled, sex, assessment_centre_onehot])
        graph_data.demographics = torch.tensor(demo_features, dtype=torch.float32).unsqueeze(0)

        processed_data.append(graph_data)

    return processed_data

def create_train_test_split(matched_data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List, dict]:
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(matched_data))
    split_idx = int(len(matched_data) * train_ratio)
    
    train_raw = [matched_data[i] for i in indices[:split_idx]]
    test_raw = [matched_data[i] for i in indices[split_idx:]]

    print(f"Train set: {len(train_raw)} samples")
    print(f"Test set: {len(test_raw)} samples")

    train_ages = np.array([item['age'] for item in train_raw]).reshape(-1, 1)
    train_assessment_centres = np.array([item['assessment_centre'] for item in train_raw]).reshape(-1, 1)

    age_scaler = StandardScaler()
    age_scaler.fit(train_ages)

    assessment_centre_encoder = OneHotEncoder(drop='first', sparse_output=False)
    assessment_centre_encoder.fit(train_assessment_centres)

    print(f"Age scaler fitted - mean: {age_scaler.mean_[0]:.2f}, std: {age_scaler.scale_[0]:.2f}")
    print(f"Assessment centre encoder fitted - categories: {assessment_centre_encoder.categories_[0]}")

    train_data = process_demographics(train_raw, age_scaler, assessment_centre_encoder)
    test_data = process_demographics(test_raw, age_scaler, assessment_centre_encoder)

    preprocessors = {
        'age_scaler': age_scaler,
        'assessment_centre_encoder': assessment_centre_encoder
    }

    return train_data, test_data, preprocessors

def save_dataset_in_standard_format(train_data: List, test_data: List, output_dir: str, dataset_name: str,
                                   preprocessors: dict = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
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

    if preprocessors:
        struct['preprocessing'] = preprocessors
    
    output_path = os.path.join(output_dir, f"{dataset_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(struct, f)
    
    print(f"Successfully saved dataset to {output_path}")

def process_custom_graph_dataset(data_col: str, graph_dir: str, csv_file: str, output_dir: str, 
                                dataset_name: str = "custom_graph_dataset", 
                                train_ratio: float = 0.8, random_seed: int = 42):
    
    matched_data = load_graph_data_with_labels(data_col, graph_dir, csv_file)
    
    if not matched_data:
        print("No matched data found. Please check your file paths and naming conventions.")
        return
    
    train_data, test_data, preprocessors = create_train_test_split(matched_data, train_ratio, random_seed)
    
    num_train = len(train_data)
    num_test = len(test_data)
    dir_name = f"train_{num_train}_test_{num_test}"
    final_output_dir = os.path.join(output_dir, dir_name)
    
    save_dataset_in_standard_format(train_data, test_data, final_output_dir, dataset_name, preprocessors)
    
    return final_output_dir

if __name__ == "__main__":
    data_col = "20016-2.0"
    graph_dir = "/scratch/bng/cartbind/data/FC_graphs/raw/GF30"
    csv_file = "/scratch/bng/cartbind/data/ukb_master_GF_no_outliers.csv"
    output_dir = "/scratch/bng/cartbind/data/FC_graphs/processed/GF30"
    dataset_name = "custom_dataset_selfloops_True_edgeft_None_norm_True"

    result_dir = process_custom_graph_dataset(
        data_col=data_col,
        graph_dir=graph_dir,
        csv_file=csv_file,
        output_dir=output_dir,
        dataset_name=dataset_name,
        train_ratio=0.8,
        random_seed=42
    )