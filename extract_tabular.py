import pandas as pd
cols = ['130895-0.0', '130897-0.0', '130907-0.0', '130911-0.0', '2050-0.0', '2060-0.0', '2070-0.0', '2080-0.0']
df = pd.read_csv('/external/rprshnas01/external_data/uk_biobank/tabular_data_csv/ukb51007.csv', usecols=cols)
df.to_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/ukb_tabular.csv')