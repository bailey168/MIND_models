import pandas as pd

# Should I also include ['130895-0.0', '130897-0.0', '130907-0.0', '130911-0.0'] ?
exclude = ['130837-0.0', '130839-0.0', '130841-0.0', '130843-0.0',
        '130845-0.0', '130847-0.0', '130849-0.0', '130851-0.0', '130853-0.0', '130855-0.0',
        '130857-0.0', '130859-0.0', '130861-0.0', '130863-0.0', '130865-0.0', '130867-0.0',
        '130869-0.0', '130871-0.0', '130873-0.0', '130875-0.0', '130877-0.0', '130879-0.0',
        '130881-0.0', '130883-0.0', '130885-0.0', '130887-0.0', '130889-0.0', '130891-0.0',
        '130893-0.0', '130899-0.0', '130901-0.0', '130903-0.0',
        '130905-0.0', '130909-0.0', '130913-0.0', '130915-0.0',
        '130917-0.0', '130919-0.0', '130921-0.0', '130923-0.0', '130925-0.0', '130927-0.0',
        '130929-0.0', '130931-0.0', '130933-0.0', '130935-0.0', '130937-0.0', '130939-0.0',
        '130941-0.0', '130943-0.0', '130945-0.0', '130947-0.0', '130949-0.0', '130951-0.0',
        '130953-0.0', '130955-0.0', '130959-0.0', '130961-0.0', '130963-0.0', '130965-0.0',
        '130967-0.0', '130971-0.0', '130973-0.0', '130975-0.0', '130977-0.0', '130979-0.0',
        '130981-0.0', '130983-0.0', '130985-0.0', '130987-0.0', '130989-0.0', '130991-0.0',
        '130993-0.0', '130995-0.0', '130997-0.0', '130999-0.0', '131001-0.0', '131003-0.0',
        '131005-0.0', '131007-0.0', '131009-0.0', '131011-0.0', '131013-0.0', '131015-0.0',
        '131017-0.0', '131019-0.0', '131021-0.0', '131023-0.0', '131025-0.0', '131027-0.0',
        '131029-0.0', '131031-0.0', '131033-0.0', '131037-0.0', '131039-0.0', '131041-0.0',
        '131043-0.0', '131045-0.0', '131047-0.0', '131049-0.0', '131051-0.0', '131053-0.0',
        '131055-0.0', '131057-0.0', '131059-0.0', '131061-0.0', '131063-0.0', '131065-0.0',
        '131067-0.0', '131069-0.0', '131071-0.0', '131073-0.0', '131075-0.0', '131077-0.0',
        '131079-0.0', '131081-0.0', '131083-0.0', '131085-0.0', '131087-0.0', '131089-0.0',
        '131091-0.0', '131093-0.0', '131095-0.0', '131097-0.0', '131099-0.0', '131101-0.0',
        '131103-0.0', '131105-0.0', '131107-0.0', '131109-0.0', '131111-0.0', '131113-0.0',
        '131115-0.0', '131117-0.0', '131119-0.0', '131121-0.0', '131123-0.0', '131125-0.0',
        '131127-0.0']

include = ['eid', '31-0.0', '20016-0.0', '21003-0.0']

# input_path = '/Users/baileyng/MIND_models/ukb_tabular2.csv'
input_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/ukb_tabular2.csv'
MIND_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_unique'
output_path = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/ukb_FIS.csv'

df = pd.read_csv(input_path, dtype=str)

# print(df.shape)

# Remove rows where any exclude column has a value (not empty)
exclude_cols = df[exclude].apply(lambda x: x.isna() | (x.str.strip() == ''), axis=0)
df = df[exclude_cols.all(axis=1)]

# Remove rows where any include column is empty or missing
include_cols = df[include].apply(lambda x: x.notna() & (x.str.strip() != ''), axis=0)
df = df[include_cols.all(axis=1)]
df = df[include].copy()

# print(df.shape)
# print(type(df['21003-0.0'][3]))
# print(df['21003-0.0'][3])

eids = df['eid'].tolist()

# Read row labels from the first file
first_matrix_path = f'{MIND_dir}/{eids[0]}_20263_2_0_aparc_MIND_matrix.csv'
row_labels = pd.read_csv(first_matrix_path, index_col=0, header=None).index.tolist()

# Pre-allocate a new DataFrame
matrix_values = pd.DataFrame(index=df.index, columns=row_labels)

# Process all EIDs
for idx, eid in enumerate(eids):
    matrix_path = f'{MIND_dir}/{eid}_20263_2_0_aparc_MIND_matrix.csv'
    try:
        matrix = pd.read_csv(matrix_path, index_col=0, header=None)
        matrix_values.iloc[idx] = matrix.iloc[:, 1].values
    except Exception as e:
        print(f"Warning: Could not process {eid}: {e}")
        matrix_values.iloc[idx] = [None] * len(row_labels)

df = pd.cancat([df.reset_index(drop=True), matrix_values.reset_index(drop=True)], axis=1)

print(f"Final DataFrame shape: {df.shape}")
df.to_csv(output_path, index=False)
print(f"Filtered DataFrame saved to '{output_path}'.")