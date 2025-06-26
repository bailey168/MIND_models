import pandas as pd

input_path = '/Users/baileyng/MIND_data/ukb_FIS.csv'
output_path = '/Users/baileyng/MIND_data/ukb_FIS_no_outliers.csv'

df = pd.read_csv(input_path, index_col=0)



