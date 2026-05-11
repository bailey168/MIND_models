import os
import pandas as pd

# Paths
old_dir = '/Users/baileyng/MIND_data/UKB_new_data/combined_data_no_outliers_old'
new_dir = '/Users/baileyng/MIND_data/UKB_new_data/combined_data_no_outliers'
exclude_file = '/Users/baileyng/MIND_data/exclude_participants/w61530_20260310.csv'

# Read exclusion list
exclude_df = pd.read_csv(exclude_file, header=None, names=['w6153'])
exclude_set = set(exclude_df['w6153'].astype(str))

# Ensure new directory exists
os.makedirs(new_dir, exist_ok=True)

for filename in os.listdir(old_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(old_dir, filename)
        df = pd.read_csv(file_path)
        eid_col = df['eid'].astype(str)
        matches = eid_col.isin(exclude_set)
        matched_eids = eid_col[matches]
        if not matched_eids.empty:
            print(f"{filename} matched eids:")
            for eid in matched_eids:
                print(eid)
        # Remove matched rows
        df_filtered = df[~matches]
        # Save to new directory
        new_file_path = os.path.join(new_dir, filename)
        df_filtered.to_csv(new_file_path, index=False)