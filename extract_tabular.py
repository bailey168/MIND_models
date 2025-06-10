import pandas as pd
cols = ['eid', '130895-0.0', '130897-0.0', '130907-0.0', '130911-0.0', '2050-0.0', 
        '2050-1.0', '2050-2.0', '2050-3.0', '2060-0.0', '2060-1.0', '2060-2.0', 
        '2060-3.0', '2070-0.0', '2070-1.0', '2070-2.0', '2070-3.0', '2080-0.0',
        '2080-1.0', '2080-2.0', '2080-3.0', '25752-2.0', '25752-3.0', '20227-2.0',
        '20227-3.0', '6350-2.0', '6350-3.0', '6351-2.0', '6351-3.0', '20016-0.0',
        '20016-1.0', '20016-2.0', '20016-3.0', '20197-2.0', '20197-3.0', '23324-2.0',
        '23324-3.0', '21003-0.0', '21003-1.0', '21003-2.0', '21003-3.0', '34-0.0',
        '130894-0.0', '130906-0.0', '130910-0.0', '20434-0.0', '42039-0.0', '20433-0.0',
        '20442-0.0', '25741-2.0', '25741-3.0', '22021-0.0', '31-0.0', '21000-0.0',
        '21000-1.0', '21000-2.0']

with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/aparc_base_id.txt') as f:
    id_set = set(line.strip() for line in f)

df = pd.read_csv('/external/rprshnas01/external_data/uk_biobank/tabular_data_csv/ukb51007.csv', usecols=cols)
filtered_df = df[df['eid'].astype(str).isin(id_set)]
filtered_df.to_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/ukb_tabular.csv')

print("Filtered DataFrame saved to 'ukb_tabular.csv'.")

# duplicates = [item for item in cols if cols.count(item) > 1]
# if duplicates:
#     print("Duplicate columns found:", set(duplicates))
# else:
#     print("No duplicate columns found.")