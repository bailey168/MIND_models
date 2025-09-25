import pandas as pd

input_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/NEO_tabular.csv'
FC_dir = '/external/rprshnas01/external_data/uk_biobank/imaging/brain/correlation/rFMRI_par_corr_matrix_25'
output_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/NEO_tabular_all.csv'
col_labels_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/FC_colnames.txt'

df = pd.read_csv(input_path, dtype=str)
eids = df['eid'].tolist()

with open(col_labels_path, 'r') as f:
    col_labels = [line.strip() for line in f.readlines()]

# Pre-allocate a new DataFrame
matrix_values = pd.DataFrame(index=df.index, columns=col_labels)

# Process all EIDs
for idx, eid in enumerate(eids):
    matrix_path = f'{FC_dir}/{eid}_25752_2_0.txt'
    try:
        with open(matrix_path, 'r') as f:
            line = f.readline().strip()
            values = line.split()
            if len(values) != 210:
                raise ValueError(f"Expected 210 values, got {len(values)} for EID {eid}")
            matrix_values.iloc[idx] = values
    except Exception as e:
        print(f"Warning: Could not process {eid}: {e}")
        matrix_values.iloc[idx] = [None] * len(col_labels)

df = pd.concat([df.reset_index(drop=True), matrix_values.reset_index(drop=True)], axis=1)

# Only drop rows that are missing FC data (not missing values in other columns)
df = df.dropna(subset=col_labels)

print(f"Final DataFrame shape: {df.shape}")
df.to_csv(output_path, index=False)
print(f"Filtered DataFrame saved to '{output_path}'.")