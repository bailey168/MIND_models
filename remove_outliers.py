import pandas as pd
import numpy as np

input_path = '/Users/baileyng/MIND_data/ukb_FIS_all.csv'
output_path = '/Users/baileyng/MIND_data/ukb_FIS_all_no_outliers.csv'

df = pd.read_csv(input_path, index_col=0)
print("Initial shape of DataFrame:", df.shape)

# Exclude columns with non-continuous data
exclude_cols = ['31-0.0', '54-2.0']
continuous_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col not in exclude_cols]

means = df[continuous_cols].mean()
stds = df[continuous_cols].std()

outlier_mask = np.abs(df[continuous_cols] - means) > (5 * stds)
outlier_counts = outlier_mask.sum(axis=1)

# Threshold
threshold = len(continuous_cols) * 0.00001

# Keep rows with fewer than threshold % outliers
df_cleaned = df[outlier_counts < threshold]

print("Shape of DataFrame without outliers:", df_cleaned.shape)
df_cleaned.to_csv(output_path)