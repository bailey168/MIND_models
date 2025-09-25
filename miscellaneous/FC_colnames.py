import numpy as np
import pandas as pd
from itertools import product

# Step 1: Define IC labels
ICs = [f'IC{i}' for i in range(1, 22)]  # ['IC1', ..., 'IC21']
D = len(ICs)

# Step 2: Generate all combinations: 'ICj' + 'ICi' 
combs = [i + '-' + j for j, i in product(ICs, ICs)]  # Now matches combs{i,2} + combs{i,1}

# Step 3: Reshape to 21 x 21 matrix
ICs_matrix = np.array(combs).reshape(D, D)

# Step 4: Extract upper triangle (excluding diagonal)
ICs_vector = ICs_matrix[np.triu_indices(D, k=1)]

# Step 5: Convert to DataFrame with each entry as a column name
df = pd.DataFrame(columns=ICs_vector)

# Step 6: Print or access the column names
column_names = df.columns.tolist()

print(column_names)

with open('/Users/baileyng/MIND_models/FC_colnames.txt', 'w') as f:
    for col in column_names:
        f.write(f"{col}\n")