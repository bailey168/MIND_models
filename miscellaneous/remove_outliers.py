import pandas as pd
import numpy as np

input_path = '/Users/baileyng/MIND_data/UKB_new_data/combined_data_TMT.csv'
output_path = '/Users/baileyng/MIND_data/UKB_new_data/combined_data_no_outliers/combined_data_TMT_no_outliers.csv'

df = pd.read_csv(input_path)
print("Initial shape of DataFrame:", df.shape)

#############################################################################################
# # Remove rows where 'p6350_i2' column has value of 0
# df = df[df['p6350_i2'] != 0]
# print("Shape after removing rows with 0 in 'p6350_i2':", df.shape)

# # Remove rows where 'p6350_i2' or 'p6351_i2' are more than 5 stds from their respective means
# for col in ['p6350_i2', 'p6351_i2']:
#     if col in df.columns:
#         mean_val = df[col].mean()
#         std_val = df[col].std()
#         df = df[np.abs(df[col] - mean_val) <= (5 * std_val)]
#         print(f"Shape after removing outliers in '{col}':", df.shape)
#############################################################################################

# Exclude columns with non-continuous data
# exclude_cols = ['31-0.0', '54-2.0']

def load_columns_from_file(filepath):
    with open(filepath, 'r')as f:
        return [line.strip() for line in f if line.strip()]

demo_cols = load_columns_from_file('/Users/baileyng/MIND_models/region_names/demo_cols_dnanexus.txt')
cog_tests_cols = load_columns_from_file('/Users/baileyng/MIND_models/region_names/cog_tests_cols_dnanexus.txt')

exclude_cols = ['eid'] + demo_cols + cog_tests_cols
continuous_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col not in exclude_cols]

means = df[continuous_cols].mean()
stds = df[continuous_cols].std()

outlier_mask = np.abs(df[continuous_cols] - means) > (5 * stds)
outlier_counts = outlier_mask.sum(axis=1)

# Threshold
threshold = len(continuous_cols) * 0.005

# Keep rows with fewer than threshold % outliers
df_cleaned = df[outlier_counts < threshold]

print("Shape of DataFrame without outliers:", df_cleaned.shape)
df_cleaned.to_csv(output_path, index=False)