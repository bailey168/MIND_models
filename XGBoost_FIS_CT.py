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
from sklearn.preprocessing import StandardScaler, RobustScaler
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
df = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_FIS.csv', index_col=0)

# %% [markdown]
# ### Using MIND to predict Fluid Intelligence Score

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0',
    
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']

datafield_name = ['sex', 'age',
    
        'lh_caudalanteriorcingulate_thickness', 'rh_caudalanteriorcingulate_thickness', 'lh_caudalmiddlefrontal_thickness',
        'rh_caudalmiddlefrontal_thickness', 'lh_cuneus_thickness', 'rh_cuneus_thickness', 'lh_entorhinal_thickness', 
        'rh_entorhinal_thickness', 'lh_fusiform_thickness', 'rh_fusiform_thickness', 'lh_inferiorparietal_thickness', 
        'rh_inferiorparietal_thickness', 'lh_inferiortemporal_thickness', 'rh_inferiortemporal_thickness', 'lh_insula_thickness', 
        'rh_insula_thickness', 'lh_isthmuscingulate_thickness', 'rh_isthmuscingulate_thickness', 'lh_lateraloccipital_thickness', 
        'rh_lateraloccipital_thickness', 'lh_lateralorbitofrontal_thickness', 'rh_lateralorbitofrontal_thickness', 
        'lh_lingual_thickness', 'rh_lingual_thickness', 'lh_medialorbitofrontal_thickness', 'rh_medialorbitofrontal_thickness', 
        'lh_middletemporal_thickness', 'rh_middletemporal_thickness', 'lh_paracentral_thickness', 'rh_paracentral_thickness', 
        'lh_parahippocampal_thickness', 'rh_parahippocampal_thickness', 'lh_parsopercularis_thickness', 'rh_parsopercularis_thickness', 
        'lh_parsorbitalis_thickness', 'rh_parsorbitalis_thickness', 'lh_parstriangularis_thickness', 'rh_parstriangularis_thickness', 
        'lh_pericalcarine_thickness', 'rh_pericalcarine_thickness', 'lh_postcentral_thickness', 'rh_postcentral_thickness', 
        'lh_posteriorcingulate_thickness', 'rh_posteriorcingulate_thickness', 'lh_precentral_thickness', 'rh_precentral_thickness', 
        'lh_precuneus_thickness', 'rh_precuneus_thickness', 'lh_rostralanteriorcingulate_thickness', 'rh_rostralanteriorcingulate_thickness', 
        'lh_rostralmiddlefrontal_thickness', 'rh_rostralmiddlefrontal_thickness', 'lh_superiorfrontal_thickness', 'rh_superiorfrontal_thickness', 
        'lh_superiorparietal_thickness', 'rh_superiorparietal_thickness', 'lh_superiortemporal_thickness', 'rh_superiortemporal_thickness', 
        'lh_supramarginal_thickness', 'rh_supramarginal_thickness', 'lh_transversetemporal_thickness', 'rh_transversetemporal_thickness']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    df = df.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

# %%
numerical_variables = ['age',
                       
        'lh_caudalanteriorcingulate_thickness', 'rh_caudalanteriorcingulate_thickness', 'lh_caudalmiddlefrontal_thickness',
        'rh_caudalmiddlefrontal_thickness', 'lh_cuneus_thickness', 'rh_cuneus_thickness', 'lh_entorhinal_thickness', 
        'rh_entorhinal_thickness', 'lh_fusiform_thickness', 'rh_fusiform_thickness', 'lh_inferiorparietal_thickness', 
        'rh_inferiorparietal_thickness', 'lh_inferiortemporal_thickness', 'rh_inferiortemporal_thickness', 'lh_insula_thickness', 
        'rh_insula_thickness', 'lh_isthmuscingulate_thickness', 'rh_isthmuscingulate_thickness', 'lh_lateraloccipital_thickness', 
        'rh_lateraloccipital_thickness', 'lh_lateralorbitofrontal_thickness', 'rh_lateralorbitofrontal_thickness', 
        'lh_lingual_thickness', 'rh_lingual_thickness', 'lh_medialorbitofrontal_thickness', 'rh_medialorbitofrontal_thickness', 
        'lh_middletemporal_thickness', 'rh_middletemporal_thickness', 'lh_paracentral_thickness', 'rh_paracentral_thickness', 
        'lh_parahippocampal_thickness', 'rh_parahippocampal_thickness', 'lh_parsopercularis_thickness', 'rh_parsopercularis_thickness', 
        'lh_parsorbitalis_thickness', 'rh_parsorbitalis_thickness', 'lh_parstriangularis_thickness', 'rh_parstriangularis_thickness', 
        'lh_pericalcarine_thickness', 'rh_pericalcarine_thickness', 'lh_postcentral_thickness', 'rh_postcentral_thickness', 
        'lh_posteriorcingulate_thickness', 'rh_posteriorcingulate_thickness', 'lh_precentral_thickness', 'rh_precentral_thickness', 
        'lh_precuneus_thickness', 'rh_precuneus_thickness', 'lh_rostralanteriorcingulate_thickness', 'rh_rostralanteriorcingulate_thickness', 
        'lh_rostralmiddlefrontal_thickness', 'rh_rostralmiddlefrontal_thickness', 'lh_superiorfrontal_thickness', 'rh_superiorfrontal_thickness', 
        'lh_superiorparietal_thickness', 'rh_superiorparietal_thickness', 'lh_superiortemporal_thickness', 'rh_superiortemporal_thickness', 
        'lh_supramarginal_thickness', 'rh_supramarginal_thickness', 'lh_transversetemporal_thickness', 'rh_transversetemporal_thickness']

categorical_variables = []

binary_variables = ['sex']

output_variables = ['20016-2.0']

input_variables = list(set(numerical_variables + categorical_variables + binary_variables) - set(output_variables))

# %%
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
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
        normalizer = RobustScaler()
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
        dir_name = f'/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/best_hyperparameters_CT/{output_variable}/split_{splt_idx}'
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


