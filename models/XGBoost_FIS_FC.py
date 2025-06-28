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
# Load the dataset
df = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_FIS_all_no_outliers.csv', index_col=0)

# %% [markdown]
# ### Using MIND to predict Fluid Intelligence Score

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    df = df.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

# %%
numerical_variables = ['age',
        
        'IC1IC2', 'IC1IC3', 'IC1IC4', 'IC1IC5', 'IC1IC6', 'IC1IC7', 'IC1IC8', 
        'IC1IC9', 'IC1IC10', 'IC1IC11', 'IC1IC12', 'IC1IC13', 'IC1IC14', 
        'IC1IC15', 'IC1IC16', 'IC1IC17', 'IC1IC18', 'IC1IC19', 'IC1IC20', 
        'IC1IC21', 'IC2IC3', 'IC2IC4', 'IC2IC5', 'IC2IC6', 'IC2IC7', 'IC2IC8', 
        'IC2IC9', 'IC2IC10', 'IC2IC11', 'IC2IC12', 'IC2IC13', 'IC2IC14', 'IC2IC15', 
        'IC2IC16', 'IC2IC17', 'IC2IC18', 'IC2IC19', 'IC2IC20', 'IC2IC21', 'IC3IC4', 
        'IC3IC5', 'IC3IC6', 'IC3IC7', 'IC3IC8', 'IC3IC9', 'IC3IC10', 'IC3IC11', 
        'IC3IC12', 'IC3IC13', 'IC3IC14', 'IC3IC15', 'IC3IC16', 'IC3IC17', 'IC3IC18', 
        'IC3IC19', 'IC3IC20', 'IC3IC21', 'IC4IC5', 'IC4IC6', 'IC4IC7', 'IC4IC8', 
        'IC4IC9', 'IC4IC10', 'IC4IC11', 'IC4IC12', 'IC4IC13', 'IC4IC14', 'IC4IC15', 
        'IC4IC16', 'IC4IC17', 'IC4IC18', 'IC4IC19', 'IC4IC20', 'IC4IC21', 'IC5IC6', 
        'IC5IC7', 'IC5IC8', 'IC5IC9', 'IC5IC10', 'IC5IC11', 'IC5IC12', 'IC5IC13', 
        'IC5IC14', 'IC5IC15', 'IC5IC16', 'IC5IC17', 'IC5IC18', 'IC5IC19', 'IC5IC20', 
        'IC5IC21', 'IC6IC7', 'IC6IC8', 'IC6IC9', 'IC6IC10', 'IC6IC11', 'IC6IC12', 
        'IC6IC13', 'IC6IC14', 'IC6IC15', 'IC6IC16', 'IC6IC17', 'IC6IC18', 'IC6IC19', 
        'IC6IC20', 'IC6IC21', 'IC7IC8', 'IC7IC9', 'IC7IC10', 'IC7IC11', 'IC7IC12', 
        'IC7IC13', 'IC7IC14', 'IC7IC15', 'IC7IC16', 'IC7IC17', 'IC7IC18', 'IC7IC19', 
        'IC7IC20', 'IC7IC21', 'IC8IC9', 'IC8IC10', 'IC8IC11', 'IC8IC12', 'IC8IC13', 
        'IC8IC14', 'IC8IC15', 'IC8IC16', 'IC8IC17', 'IC8IC18', 'IC8IC19', 'IC8IC20', 
        'IC8IC21', 'IC9IC10', 'IC9IC11', 'IC9IC12', 'IC9IC13', 'IC9IC14', 'IC9IC15', 
        'IC9IC16', 'IC9IC17', 'IC9IC18', 'IC9IC19', 'IC9IC20', 'IC9IC21', 'IC10IC11', 
        'IC10IC12', 'IC10IC13', 'IC10IC14', 'IC10IC15', 'IC10IC16', 'IC10IC17', 'IC10IC18', 
        'IC10IC19', 'IC10IC20', 'IC10IC21', 'IC11IC12', 'IC11IC13', 'IC11IC14', 'IC11IC15', 
        'IC11IC16', 'IC11IC17', 'IC11IC18', 'IC11IC19', 'IC11IC20', 'IC11IC21', 'IC12IC13', 
        'IC12IC14', 'IC12IC15', 'IC12IC16', 'IC12IC17', 'IC12IC18', 'IC12IC19', 'IC12IC20', 
        'IC12IC21', 'IC13IC14', 'IC13IC15', 'IC13IC16', 'IC13IC17', 'IC13IC18', 'IC13IC19', 
        'IC13IC20', 'IC13IC21', 'IC14IC15', 'IC14IC16', 'IC14IC17', 'IC14IC18', 'IC14IC19', 
        'IC14IC20', 'IC14IC21', 'IC15IC16', 'IC15IC17', 'IC15IC18', 'IC15IC19', 'IC15IC20', 
        'IC15IC21', 'IC16IC17', 'IC16IC18', 'IC16IC19', 'IC16IC20', 'IC16IC21', 'IC17IC18', 
        'IC17IC19', 'IC17IC20', 'IC17IC21', 'IC18IC19', 'IC18IC20', 'IC18IC21', 'IC19IC20', 
        'IC19IC21', 'IC20IC21']

categorical_variables = ['assessment centre']

binary_variables = ['sex']

output_variables = ['20016-2.0']

input_variables = list(set(numerical_variables + categorical_variables + binary_variables) - set(output_variables))
df[categorical_variables] = df[categorical_variables].astype('category')

# %%
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
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    # Increase this number if you have more cores. Otherwise, remove it and it will default
    # to the maximum number.
    'nthread': 12,
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
                                 categorical_vars, binary_vars, space=space):
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
        dir_name = f'/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/hyperparameters/best_hyperparameters_FC_06-28/{output_variable}/split_{splt_idx}'
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
input_vars = input_variables
output_var = output_variables[0]
numerical_vars = numerical_variables
categorical_vars = categorical_variables
binary_vars = binary_variables

start = time.time()
print("Starting hyperparameter optimization...")

data_x, data_y, best_hyperparameters, column_names = hyper_parameter_optimization(df_input, input_vars, output_var,
                            numerical_vars, categorical_vars, binary_vars, space=space)

end = time.time()
print(f"Hyperparameter optimization completed in {end - start} seconds.")


