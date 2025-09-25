import pandas as pd

df = pd.read_csv('/Users/baileyng/MIND_data/NEO_tabular_all.csv', index_col=0)

count = df['eid'].notna().sum()
print(count)

all_cols = ['eid', '20127-0.0']
count_all = df[all_cols].notna().all(axis=1).sum()
print(f"Number of rows with values in ALL columns: {count_all}")

# # Create new trailmaking_score column
# df['trailmaking_score'] = pd.to_numeric(df['6350-2.0'], errors='coerce') + 5 * pd.to_numeric(df['6351-2.0'], errors='coerce')

# # Save the updated DataFrame
# df.to_csv('/Users/baileyng/MIND_data/ukb_master_all.csv')
# print("DataFrame with trailmaking_score column saved.")


# Create a separate DataFrame with rows that have values in all three trailmaking columns
trailmaking_cols = ['eid', '20127-0.0']
trailmaking_df = df[df[trailmaking_cols].notna().all(axis=1)].copy()

print(f"Number of rows with values in all trailmaking columns: {len(trailmaking_df)}")
print(f"Original DataFrame shape: {df.shape}")
print(f"Trailmaking DataFrame shape: {trailmaking_df.shape}")

trailmaking_df.to_csv('/Users/baileyng/MIND_data/NEO_tabular_all_master.csv')
print("Trailmaking DataFrame saved.")