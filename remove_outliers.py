import pandas as pd
import numpy as np

input_path = '/Users/baileyng/MIND_data/ukb_master_PAL.csv'
output_path = '/Users/baileyng/MIND_data/ukb_master_PAL_no_outliers.csv'

df = pd.read_csv(input_path, index_col=0)
print("Initial shape of DataFrame:", df.shape)

#############################################################################################
# # Remove rows where '6350-2.0' column has value of 0
# df = df[df['6350-2.0'] != 0]
# print("Shape after removing rows with 0 in '6350-2.0':", df.shape)

# # Remove rows where '6350-2.0' or '6351-2.0' are more than 5 stds from their respective means
# for col in ['6350-2.0', '6351-2.0']:
#     if col in df.columns:
#         mean_val = df[col].mean()
#         std_val = df[col].std()
#         df = df[np.abs(df[col] - mean_val) <= (5 * std_val)]
#         print(f"Shape after removing outliers in '{col}':", df.shape)
#############################################################################################

# Exclude columns with non-continuous data
# exclude_cols = ['31-0.0', '54-2.0']
exclude_cols = ['21003-2.0', '31-0.0', '54-2.0', '25741-2.0', '6350-2.0', '6351-2.0', 
                '20016-2.0', '20197-2.0', '23324-2.0', 'trailmaking_score']
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
df_cleaned.to_csv(output_path)