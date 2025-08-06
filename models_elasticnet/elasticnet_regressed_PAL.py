# %% [markdown]
# # Set up

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import os
import time

# %%
start = time.time()

random.seed(42)
np.random.seed(42)

# %%
rename = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/col_renames.csv')
rename_dict = dict(zip(rename['datafield_code'], rename['datafield_name']))

weights_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_weights_reg'

# %% [markdown]
# # ElasticNet Analysis Function

# %%
def regress_out_covariates(data, covariates, categorical_cols=None):
    """
    Regress out covariates from data using vectorized operations.
    Handles categorical variables by treating them as dummy variables.
    
    Parameters:
    data: DataFrame or array - the data to regress covariates from
    covariates: DataFrame - the covariate data
    categorical_cols: list - explicit list of column names to treat as categorical
    
    Returns:
    residuals: DataFrame or array - data with covariates regressed out
    """
    # Convert to numpy arrays for computation
    Y = np.array(data, dtype=float)
    
    # Handle categorical variables if DataFrame is passed
    if isinstance(covariates, pd.DataFrame):
        if categorical_cols is not None:
            # Explicitly specify which columns are categorical
            X_cov = pd.get_dummies(covariates, columns=categorical_cols, drop_first=True, dtype=float).values
        else:
            # Auto-detect categorical columns (object/string types)
            X_cov = pd.get_dummies(covariates, drop_first=True, dtype=float).values
    else:
        X_cov = np.array(covariates, dtype=float)
    
    # Add intercept column
    design_matrix = np.column_stack([np.ones(X_cov.shape[0]), X_cov])
    
    # Vectorized regression: solve for all columns simultaneously
    beta_coeffs = np.linalg.lstsq(design_matrix, Y, rcond=None)[0]
    
    # Calculate predictions and residuals
    predicted = design_matrix @ beta_coeffs
    residuals = Y - predicted
    
    # Return as DataFrame if input was DataFrame
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(residuals, index=data.index, columns=data.columns)
    else:
        return residuals
    

def elasticnet_covariate_regression(X, y, brain_regions, X_covariates, y_covariates, 
                                                 categorical_cols=None, n_splits=10, weights_dir=None, data_name=None, target_name=None):
    """
    ElasticNet analysis with demographic covariates regressed out beforehand.
    """
    # Separate demographic and brain region data
    X_demographic = X[X_covariates]
    y_demographic = X[y_covariates]
    X_brain = X[brain_regions]
    
    # Determine which covariates are categorical
    X_categorical = [col for col in X_covariates if col in (categorical_cols or [])]
    y_categorical = [col for col in y_covariates if col in (categorical_cols or [])]
    
    # Regress out demographic variables from brain regions
    print("Regressing out demographic variables from brain regions...")
    X_brain_residuals = regress_out_covariates(X_brain, X_demographic, X_categorical)
    
    # Regress out demographic variables from target variable
    print("Regressing out demographic variables from target variable...")
    y_residuals = regress_out_covariates(y, y_demographic, y_categorical)
    
    # Use only the residualized brain regions as features
    X_final = X_brain_residuals
    y_final = y_residuals.squeeze() if y_residuals.ndim > 1 else y_residuals
    
    # Simple preprocessing (just scaling, no categorical encoding needed)
    scaler = StandardScaler()
    
    # Cross-validation set-up
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    outer_mae, outer_rmse, outer_r2 = [], [], []
    best_params_per_fold = []
    nonzero_predictors = []
    coefs_list = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_final, y_final), start=1):
        X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
        
        # Handle y_final indexing based on its type
        if isinstance(y_final, pd.DataFrame) or isinstance(y_final, pd.Series):
            y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]
        else:
            # y_final is a numpy array
            y_train, y_test = y_final[train_idx], y_final[test_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner CV
        elastic_net = ElasticNetCV(
            l1_ratio=np.linspace(0.3, 0.9, 7),
            alphas=np.logspace(-4, 1, 15),
            cv=10, max_iter=40000, random_state=42,
            n_jobs=-1
        )
        
        elastic_net.fit(X_train_scaled, y_train)
        y_pred = elastic_net.predict(X_test_scaled)
        
        # --- metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        outer_mae.append(mae)
        outer_rmse.append(rmse)
        outer_r2.append(r2)
        
        # --- store best α & l1_ratio for this fold ---
        best_params_per_fold.append(
            {'alpha': elastic_net.alpha_, 'l1_ratio': elastic_net.l1_ratio_}
        )
        
        # --- predictors that survived ---
        coefs = elastic_net.coef_
        coefs_list.append(coefs)
        surviving = [col for col, c in zip(X_final.columns, coefs) if c != 0]
        nonzero_predictors.append(surviving)
        
        print(f'Fold {fold:02d} • MAE={mae:.3f} • RMSE={rmse:.3f} • R²={r2:.3f} '
              f'• α={elastic_net.alpha_:.4g} • l1_ratio={elastic_net.l1_ratio_:.2f}')
    
    # Aggregate results
    print('\n=== 10-fold CV summary ===')
    print(f'Mean MAE :  {np.mean(outer_mae):.3f}  ± {np.std(outer_mae):.3f}')
    print(f'Mean RMSE:  {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}')
    print(f'Mean R²  :  {np.mean(outer_r2):.3f}  ± {np.std(outer_r2):.3f}')
    
    # Calculate mean best parameters from all folds
    mean_alpha = np.mean([p['alpha'] for p in best_params_per_fold])
    mean_l1_ratio = np.mean([p['l1_ratio'] for p in best_params_per_fold])
    
    # Fit final model on all data
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)
    
    final_model = ElasticNet(
        alpha=mean_alpha,
        l1_ratio=mean_l1_ratio,
        max_iter=40000,
        random_state=42
    )
    
    final_model.fit(X_final_scaled, y_final)
    
    # Create coefficients DataFrame
    coefs_df = pd.DataFrame({
        'Feature': X_final.columns,
        'Coefficient': final_model.coef_
    })
    
    # Save coefficients
    os.makedirs(weights_dir, exist_ok=True)
    weights_filename = f'ElasticNet_weights_{data_name}_{target_name}.csv'
    weights_filepath = os.path.join(weights_dir, weights_filename)
    coefs_df.to_csv(weights_filepath, index=False)
    
    print(f"\nElasticNet coefficients saved to: {weights_filepath}")
    print(f"Final model parameters: α={mean_alpha:.4g}, l1_ratio={mean_l1_ratio:.3f}")
    print(f"Number of non-zero coefficients: {np.sum(final_model.coef_ != 0)}")
    
    return {
        'mae': outer_mae,
        'rmse': outer_rmse,
        'r2': outer_r2,
        'best_params': best_params_per_fold,
        'nonzero_predictors': nonzero_predictors,
        'coefficients': coefs_list
    }

# %% [markdown]
# # PAL

# %%
# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_master_PAL_no_outliers.csv', index_col=0)
target_name = 'PAL'

# %% [markdown]
# ## PAL vs. MIND

# %%
data_name = 'MIND'

# Set X and y
with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define different covariates for X and y
X_covariates = ['31-0.0', '21003-2.0', '54-2.0']  
y_covariates = ['31-0.0', '21003-2.0', '54-2.0']  

# Combine all variables
all_vars = list(set(X_covariates + y_covariates + brain_regions))

X = df[all_vars]
y = df['20197-2.0']

print(f"Original shapes - X: {X.shape}, y: {y.shape}")

# rename columns
X = X.rename(columns=rename_dict)

# Update variable names after renaming
X_covariates_renamed = [rename_dict.get(var, var) for var in X_covariates]
y_covariates_renamed = [rename_dict.get(var, var) for var in y_covariates]
brain_regions_renamed = [rename_dict.get(var, var) for var in brain_regions]

# Specify which variables are categorical
categorical_variables = ['sex', 'assessment_centre']  # Use renamed column names

# %%
# Run analysis with covariate regression
print('Starting elastic net analysis (regressed)...')

results = elasticnet_covariate_regression(
    X, y, brain_regions_renamed, X_covariates_renamed, y_covariates_renamed, 
    categorical_cols=categorical_variables, n_splits=10,
    weights_dir=weights_dir, data_name=data_name, target_name=target_name
)

print(f'\nTotal time taken: {time.time() - start:.2f} seconds')


