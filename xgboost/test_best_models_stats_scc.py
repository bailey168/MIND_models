# %% [markdown]
# # Set Up

# %%
import pandas as pd
import json
import numpy as np
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

# %%
random.seed(42)
np.random.seed(42)

# %%
n_splits = 10

# %%
rename = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/col_renames.csv')
rename_dict = dict(zip(rename['datafield_code'], rename['datafield_name']))

# %%
with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/MIND_avg_regions.txt', 'r') as f:
    MIND_avg_regions = [line.strip() for line in f.readlines()]

with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    MIND_regions = [line.strip() for line in f.readlines()]

with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/CT_regions.txt', 'r') as f:
    CT_regions_base = [line.strip() for line in f.readlines()]
    CT_regions = [rename_dict[region] for region in CT_regions_base]

with open('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/FC_regions.txt', 'r') as f:
    FC_regions = [line.strip() for line in f.readlines()]

demo = []

# regions = [MIND_avg_regions, MIND_regions, CT_regions, FC_regions, demo]
# region_names = ['MIND_avg_regions', 'MIND_regions', 'CT_regions', 'FC_regions', 'demo']
# regions = [MIND_avg_regions, CT_regions, FC_regions, demo]
# region_names = ['MIND_avg_regions', 'CT_regions', 'FC_regions', 'demo']
regions = [MIND_regions]
region_names = ['MIND_regions']


# %%
numerical_variables = ['age']

categorical_variables = ['assessment_centre']

binary_variables = ['sex']

# %%
def train_test_metrics(hyperparameter_dir, categorical_variables, binary_variables, numerical_variables):
    all_results = {}

    for i, region_name in enumerate(region_names):
        print(f"\n{'='*60}")
        print(f"Running analysis for: {region_name}")
        print(f"{'='*60}")
        
        # Create a local copy for this iteration
        numerical_variables_copy = numerical_variables.copy()
        numerical_variables_copy = numerical_variables_copy + regions[i]

        if region_name == 'FC_regions':
            numerical_variables_copy = numerical_variables_copy + ['head_motion']

        region_hyperparameter_dir = os.path.join(hyperparameter_dir, region_name)
        
        # Check if the directory exists before running
        if not os.path.exists(region_hyperparameter_dir):
            print(f"Directory not found: {region_hyperparameter_dir}")
            continue
            
        mae_list, rmse_list, r2_list = [], [], []

        for split_idx in range(n_splits):
            split_dir = os.path.join(region_hyperparameter_dir, f'split_{split_idx}')

            # load data
            data = np.load(os.path.join(split_dir, 'train_test_data.npz'), allow_pickle=True)
            cols     = data['column_names']
            X_train  = pd.DataFrame(data=data['x_train'], columns=cols)
            X_test   = pd.DataFrame(data=data['x_test'],  columns=cols)
            y_train  = data['y_train']
            y_test   = data['y_test']

            # cast types
            for c in categorical_variables:
                    X_train[c] = X_train[c].astype('category')
                    X_test[c]  = X_test[c].astype('category')
            for b in binary_variables:
                    X_train[b] = pd.to_numeric(X_train[b], errors='coerce')
                    X_test[b]  = pd.to_numeric(X_test[b], errors='coerce')

            # load best hyperparams
            with open(os.path.join(split_dir, 'best_hyperparameters.json'), 'r') as f:
                params = json.load(f)

            # choose objective/metric
            if np.unique(y_train).shape[0] >= 3:
                params.update({'eval_metric':'rmse', 'objective':'reg:squarederror'})
            else:
                params.update({'eval_metric':'auc',  'objective':'binary:logistic'})

            # extract and remove n_estimators
            n_estimators = int(params.pop('n_estimators'))

            # scale numerics
            scaler = StandardScaler()
            num_vars = [v for v in cols if v in numerical_variables_copy]
            if num_vars:
                X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
                X_test[num_vars]  = scaler.transform(X_test[num_vars])

            # train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

            # predict
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            preds = booster.predict(dtest)

            # compute metrics
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            # r2 = r2_score(y_test, preds, force_finite=False)

            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)

            print(f"Split {split_idx:02d} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

        # after all splits, summary for this region
        print(f"\nOverall performance for {region_name}:")
        print(f"MAE  : {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
        print(f"RMSE : {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}")
        print(f"R²   : {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
        
        # Store results for this region
        all_results[region_name] = {
            'mae_list': mae_list,
            'rmse_list': rmse_list,
            'r2_list': r2_list,
            'mae_mean': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'rmse_mean': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'r2_mean': np.mean(r2_list),
            'r2_std': np.std(r2_list)
        }

    return all_results

# %% [markdown]
# # GF

# %%
hyperparameter_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_20016-2.0_reg_07-21'

results = train_test_metrics(
    hyperparameter_dir=hyperparameter_dir,
    categorical_variables=categorical_variables,
    binary_variables=binary_variables,
    numerical_variables=numerical_variables
)

# %% [markdown]
# # PAL

# %%
hyperparameter_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_20197-2.0_reg_07-21'

results = train_test_metrics(
    hyperparameter_dir=hyperparameter_dir,
    categorical_variables=categorical_variables,
    binary_variables=binary_variables,
    numerical_variables=numerical_variables
)

# %% [markdown]
# # DSST

# %%
hyperparameter_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_23324-2.0_reg_07-21'

results = train_test_metrics(
    hyperparameter_dir=hyperparameter_dir,
    categorical_variables=categorical_variables,
    binary_variables=binary_variables,
    numerical_variables=numerical_variables
)

# %% [markdown]
# # TMT

# %%
hyperparameter_dir = '/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_trailmaking_score_reg_07-21'

results = train_test_metrics(
    hyperparameter_dir=hyperparameter_dir,
    categorical_variables=categorical_variables,
    binary_variables=binary_variables,
    numerical_variables=numerical_variables
)


