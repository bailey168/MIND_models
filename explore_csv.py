import pandas as pd

df = pd.read_csv('/Users/baileyng/MIND_data/ukb_master_all.csv', index_col=0)

count = df['20016-2.0'].notna().sum()
print(count)

count = (df['6350-2.0'].notna() & df['6351-2.0'].notna()).sum()
print(count)

count = df['trailmaking_score'].notna().sum()
print(count)

count = df['20197-2.0'].notna().sum()
print(count)

count = df['23324-2.0'].notna().sum()
print(count)

all_cols = ['20016-2.0', '20197-2.0', '23324-2.0', '6350-2.0', '6351-2.0']
count_all = df[all_cols].notna().all(axis=1).sum()
print(f"Number of rows with values in ALL columns: {count_all}")

# # Create new trailmaking_score column
# df['trailmaking_score'] = pd.to_numeric(df['6350-2.0'], errors='coerce') + 5 * pd.to_numeric(df['6351-2.0'], errors='coerce')

# # Save the updated DataFrame
# df.to_csv('/Users/baileyng/MIND_data/ukb_master_all.csv')
# print("DataFrame with trailmaking_score column saved.")


# Create a separate DataFrame with rows that have values in all three trailmaking columns
trailmaking_cols = ['20016-2.0', '20197-2.0', '23324-2.0', '6350-2.0', '6351-2.0']
trailmaking_df = df[df[trailmaking_cols].notna().all(axis=1)].copy()

print(f"Number of rows with values in all trailmaking columns: {len(trailmaking_df)}")
print(f"Original DataFrame shape: {df.shape}")
print(f"Trailmaking DataFrame shape: {trailmaking_df.shape}")

trailmaking_df.to_csv('/Users/baileyng/MIND_data/ukb_master_allcols.csv')
print("Trailmaking DataFrame saved.")