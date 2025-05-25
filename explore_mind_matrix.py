import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/baileyng/MIND/MIND_results/sub-001_MIND_matrix.csv", index_col=0)

print("Shape:", df.shape)
print("\nHead:\n", df.head())

# Print all column names
print("\nColumn names:\n", list(df.columns))

# Print all row names
print("\nRow names:\n", list(df.index))



plt.figure(figsize=(8, 8))
sns.heatmap(df, cmap="viridis", square=True, xticklabels=1, yticklabels=1)
plt.title("MIND Matrix Heatmap: sub-003")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.show()