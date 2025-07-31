# %%
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
import numpy as np
from os import makedirs
from os.path import join
import xgboost as xgb
import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import sklearn.model_selection
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial
import random

# %%
random.seed(42)
np.random.seed(42)

# %%
rename = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/region_names/col_renames.csv')
rename_dict = dict(zip(rename['datafield_code'], rename['datafield_name']))

# %%
# Load the dataset
df = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_master_PAL_no_outliers.csv', index_col=0)

# %% [markdown]
# ### Using MIND to predict Fluid Intelligence Score

# %%
# rename columns
df = df.rename(columns=rename_dict)

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

regions = [MIND_avg_regions, MIND_regions, CT_regions, FC_regions]
region_names = ['MIND_avg_regions', 'MIND_regions', 'CT_regions', 'FC_regions']


# %%
numerical_variables = ['age']

categorical_variables = ['assessment_centre']

binary_variables = ['sex']

output_variables = ['20197-2.0']

input_variables = list(set(numerical_variables + categorical_variables + binary_variables) - set(output_variables))
df[categorical_variables] = df[categorical_variables].astype('category')

# %%
lambda_values = np.logspace(-4, 4, num=17)
alpha_values = np.logspace(-4, 4, num=17)

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 20),
    'eta': hp.quniform('eta', 0.025, 0.8, 0.025),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth': hp.choice('max_depth', np.arange(1, 10, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.2, 1, 0.1),
    'gamma': hp.quniform('gamma', 0, 10, 0.2),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'lambda': hp.choice('lambda', lambda_values.tolist()),  # L2 regularization
    'alpha': hp.choice('alpha', alpha_values.tolist()),    # L1 regularization
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    # Increase this number if you have more cores. Otherwise, remove it and it will default
    # to the maximum number.
    # 'nthread': 12,
    'booster': 'gbtree',
    'tree_method': 'hist',
    'seed': 42
}

# %%
def score(params, data):
    data_x = data[0]
    data_y = data[1]
    train_features_instance = data_x[0]
    valid_features_instance = data_x[1]
    y_train_instance = data_y[0]
    y_valid_instance = data_y[1]

    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(train_features_instance, label=y_train_instance, enable_categorical=True)
    dvalid = xgb.DMatrix(valid_features_instance, label=y_valid_instance, enable_categorical=True)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=False)
    predictions = gbm_model.predict(dvalid)
    if y_train_instance.unique().shape[0] < 3:
        score = roc_auc_score(y_valid_instance, predictions)
    else:
        score = r2_score(y_valid_instance, predictions)
        # TODO: Add the importance for the selected features
        print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


def hyper_parameter_optimization(data_df, input_variables, output_variable, numeric_vars, 
                                 categorical_vars, binary_vars, region_name, space=space):
    number_of_splits = 10
    #     print(data_df.shape)
    # Data loading
    data_df = data_df.dropna(subset=[output_variable]).copy(deep=True)
    X = data_df[input_variables]
    y = data_df[output_variable]
    #     print(np.unique(y))
    if y.unique().shape[0] < 3:
        split_object = StratifiedShuffleSplit(n_splits=number_of_splits,
                               train_size=(number_of_splits - 1) / number_of_splits,
                               test_size=1 / number_of_splits,
                               random_state=42)
    else:
        split_object = ShuffleSplit(n_splits=number_of_splits,
                                              train_size=(number_of_splits - 1) / number_of_splits,
                                              test_size=1 / number_of_splits,
                                              random_state=42)

    blocks = np.arange(y.shape[0])
    for splt_idx, (train_idx, test_idx) in enumerate(split_object.split(blocks, y)):
        X_trainval = X.iloc[train_idx, :]
        y_trainval = y.iloc[train_idx]
        X_test = X.iloc[test_idx, :]
        y_test = y.iloc[test_idx]
        if y.unique().shape[0] < 3:
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, random_state=42,
                                                              stratify=y_trainval)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, random_state=42)
        X_values = (X_trainval.copy(deep=True), X_test.copy(deep=True))
        y_values = (y_trainval.copy(deep=True), y_test.copy(deep=True))

        if y_train.unique().shape[0] >= 3:
            space['eval_metric'] = 'rmse'
            space['objective'] = 'reg:squarederror'
        else:
            space['eval_metric'] = 'auc'
            space['objective'] = 'binary:logistic'
        normalizer = StandardScaler()
        to_normalize = X_trainval[numeric_vars].values

        # Make explicit copies to avoid SettingWithCopyWarning
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        X_train[numeric_vars] = normalizer.fit_transform(X_train[numeric_vars])
        X_val[numeric_vars] = normalizer.transform(X_val[numeric_vars])
        train_features_instance = X_train.copy(deep=True)
        y_train_instance = y_train.copy(deep=True)
        valid_features_instance = X_val.copy(deep=True)
        y_valid_instance = y_val.copy(deep=True)

        score_data = partial(score, data=((X_train, X_val), (y_train, y_val)))
        best_hyperparams = fmin(fn=score_data,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=500)

        columns = X_test.columns
        # print(data_x, data_y, best_hyperparameters, column_names)
        dir_name = f'/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_{output_variable}_reg_07-21/{region_name}/split_{splt_idx}'
        makedirs(dir_name, exist_ok=True)
        column_names = np.array(list(columns))
        np.savez(join(dir_name, 'train_test_data.npz'), x_train=X_trainval, y_train=y_trainval,
                 x_test=X_test, y_test=y_test, column_names=column_names)
        # json object getting serialised
        best_hyperparameters_json = json.dumps(best_hyperparams, indent=4, cls=NpEncoder)
        # Writing
        with open(join(dir_name, 'best_hyperparameters.json'), "w") as outfile:
            outfile.write(best_hyperparameters_json)

    return X_values, y_values, best_hyperparams, columns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# %%
df_input = df
base_input_vars = input_variables
output_var = output_variables[0]
base_numerical_vars = numerical_variables
categorical_vars = categorical_variables
binary_vars = binary_variables

start = time.time()
print("Starting hyperparameter optimization...")

for i, region in enumerate(regions):
    region_name = region_names[i]

    if region_name == 'MIND_avg_regions' or region_name == 'MIND_regions' or region_name == 'FC_regions':
        continue

    print(f"\n{'='*60}")
    print(f"Processing region: {region_name}")
    print(f"Number of features in region: {len(region)}")
    print(f"{'='*60}")

    input_vars = base_input_vars + region
    numerical_vars = base_numerical_vars + region

    if region_name == 'FC_regions':
        input_vars = input_vars + ['head_motion']
        numerical_vars = numerical_vars + ['head_motion']

    region_start = time.time()

    try:
        data_x, data_y, best_hyperparameters, column_names = hyper_parameter_optimization(
            df_input, 
            input_vars, 
            output_var, 
            numerical_vars, 
            categorical_vars, 
            binary_vars,
            region_name,
            space=space
        )

        region_end = time.time()
        print(f"{region_name} optimization completed in {region_end - region_start:.2f} seconds.")

    except Exception as e:
        print(f"Error processing {region_name}: {str(e)}")
        continue

end = time.time()
print(f"Hyperparameter optimization completed in {end - start} seconds.")


