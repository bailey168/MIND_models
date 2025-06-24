import glob, json, pandas as pd

# point this at wherever you saved your split folders
paths = glob.glob('/…/best_hyperparameters/*/split_*/best_hyperparameters.json')
best_params = [json.load(open(p)) for p in paths]
df = pd.DataFrame(best_params)

# numeric parameters → look at median (or mean)
print(df.describe())

# categorical/discrete parameters → look at mode
for col in df.columns:
    print(col, df[col].mode()[0])