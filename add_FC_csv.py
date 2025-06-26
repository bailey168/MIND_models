import pandas as pd

input_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_FIS.csv'
FC_dir = '/external/rprshnas01/external_data/uk_biobank/imaging/brain/correlation/rFMRI_par_corr_matrix_25'
output_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_FIS_all.csv'

df = pd.read_csv(input_path, dtype=str)

eids = df['eid'].tolist()

col_labels = pd.read_csv(first_matrix_path, index_col=0, header=None, skiprows=1).index.tolist()

# print(col_labels)

# Pre-allocate a new DataFrame
matrix_values = pd.DataFrame(index=df.index, columns=col_labels)

# Process all EIDs
for idx, eid in enumerate(eids):
    matrix_path = f'{FC_dir}/{eid}_25752_2_0.txt'
    try:
        matrix = pd.read_csv(matrix_path, index_col=0, header=None, skiprows=1)
        matrix_values.iloc[idx] = matrix.iloc[:, 0].values
    except Exception as e:
        print(f"Warning: Could not process {eid}: {e}")
        matrix_values.iloc[idx] = [None] * len(col_labels)

df = pd.concat([df.reset_index(drop=True), matrix_values.reset_index(drop=True)], axis=1)

print(f"Final DataFrame shape: {df.shape}")
df.to_csv(output_path, index=False)
print(f"Filtered DataFrame saved to '{output_path}'.")