import os
import numpy as np
import pandas as pd

csv_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc'
csv_files = [f for f in os.listdir(csv_dir)]

sum_array = None
file_count = 0
row_index = None
col_names = None

for file in csv_files:
    df = pd.read_csv(os.path.join(csv_dir, file), header=0, index_col=0)
    if sum_array is None:
        sum_array = np.zeros(df.shape, dtype=np.float64)
        row_index = df.index
        col_names = df.columns
    sum_array += df.values
    file_count += 1

avg_array = sum_array / file_count
avg_df = pd.DataFrame(avg_array, index=row_index, columns=col_names)
output_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_average.csv'
avg_df.to_csv(output_path)
print(f"Averaged {file_count} files. Saved to {output_path}")
