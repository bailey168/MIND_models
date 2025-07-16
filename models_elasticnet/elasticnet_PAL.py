# %% [markdown]
# # Set up

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import time

# %%
start = time.time()

random.seed(42)
np.random.seed(42)

# %%
rename = pd.read_csv('/scratch/bng/cartbind/code/MIND_models/region_names/col_renames.csv')
rename_dict = dict(zip(rename['datafield_code'], rename['datafield_name']))

# %% [markdown]
# # ElasticNet Analysis Function

# %%
#inner parallelized
def elasticnet_analysis(X, y, continuous_vars, categorical_vars, n_splits=10):
    preprocessor = ColumnTransformer(transformers=[
        # scale continuous features
        ('num', StandardScaler(), continuous_vars),
        # one-hot encode the assessment centre (drop one level to avoid collinearity)
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars),
    ])

    # Cross-validation set-up
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    outer_mae, outer_rmse, outer_r2 = [], [], []
    best_params_per_fold = []
    nonzero_predictors = []
    coefs_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV
        pipe = make_pipeline(
            preprocessor,
            ElasticNetCV(
                l1_ratio=np.linspace(0.3,0.9,7),
                alphas=np.logspace(-4,1,11),
                cv=10, max_iter=30000, random_state=42,
                n_jobs=-1
            )
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # --- metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        outer_mae.append(mae)
        outer_rmse.append(rmse)
        outer_r2.append(r2)

        # --- store best α & l1_ratio for this fold ---
        est = pipe.named_steps['elasticnetcv']
        best_params_per_fold.append(
            {'alpha': est.alpha_, 'l1_ratio': est.l1_ratio_}
        )

        # --- predictors that survived ---
        coefs = est.coef_
        coefs_list.append(coefs)
        surviving = [col for col, c in zip(X.columns, coefs) if c != 0]
        nonzero_predictors.append(surviving)

        print(f'Fold {fold:02d} • MAE={mae:.3f} • RMSE={rmse:.3f} • R²={r2:.3f} '
            f'• α={est.alpha_:.4g} • l1_ratio={est.l1_ratio_:.2f}')
        

    # Aggregate results
    print('\n=== 10-fold CV summary ===')
    print(f'Mean MAE :  {np.mean(outer_mae):.3f}  ± {np.std(outer_mae):.3f}')
    print(f'Mean RMSE:  {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}')
    print(f'Mean R²  :  {np.mean(outer_r2):.3f}  ± {np.std(outer_r2):.3f}')



# %% [markdown]
# # PAL

# %%
# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/scratch/bng/cartbind/data/ukb_master_PAL_no_outliers.csv', index_col=0)

# %% [markdown]
# ## PAL vs. MIND

# %%
# Set X and y
with open('/scratch/bng/cartbind/code/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['20197-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
X = X.rename(columns=rename_dict)

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
print('Starting elastic net analysis...')
elasticnet_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

print(f'\nTotal time taken: {time.time() - start:.2f} seconds')


