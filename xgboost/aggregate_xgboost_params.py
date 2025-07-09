import glob
import json
import pandas as pd

paths = glob.glob('/Users/baileyng/MIND_data/best_hyperparameters/*/split_*/best_hyperparameters.json')
best_params = [json.load(open(p)) for p in paths]
df = pd.DataFrame(best_params)

# numeric parameters → look at median (or mean)
print("\nHyperparameter statistics:")
print(df.describe())

# categorical/discrete parameters → look at mode
print("\nHyperparameter modes:")
for col in df.columns:
    mode = df[col].mode()[0]
    count = (df[col] == mode).sum()
    print(f"{col}: {mode} (appeared {count} times)")
