import pandas as pd
import glob
import os

input_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc'

for csv_file in glob.glob(os.path.join(input_dir, '*_MIND_matrix.csv')):
    df_matrix = pd.read_csv(csv_file, index_col=0)
    if df_matrix.shape != (68, 68):
        print(f"File {os.path.basename(csv_file)} has shape {df_matrix.shape}, not (68, 68)")

print("Done checking sizes of MIND matrices!")