import pandas as pd
import numpy as np
import glob
import os

input_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/aparc'
output_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/aparc_avg'

os.makedirs(output_dir, exist_ok=True)

for csv_file in glob.glob(os.path.join(input_dir, '*_MIND_matrix.csv')):
    df_matrix = pd.read_csv(csv_file, index_col=0)
    df_matrix_no_diag = df_matrix.mask(np.eye(len(df_matrix), dtype=bool))
    df_avg = pd.DataFrame({'average': df_matrix_no_diag.mean(axis=1)}, index=df_matrix.index)
    df_avg.index.name = df_matrix.index.name

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_path = os.path.join(output_dir, f'{base_name}.csv')
    df_avg.to_csv(output_path)