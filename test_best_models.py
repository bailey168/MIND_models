# %%
import pandas as pd
import json
from sys import argv
import numpy as np
import xgboost as xgb
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import copy
import random

# %%
random.seed(42)
np.random.seed(42)

# %%
# Load the dataset
df = pd.read_csv('/Users/baileyng/MIND_data/ukb_FIS.csv', index_col=0)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0']

datafield_name = ['sex', 'age']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    df = df.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

# %%
numerical_variables = ['age',
                       
        'lh_bankssts', 'lh_caudalanteriorcingulate', 'lh_caudalmiddlefrontal',
        'lh_cuneus', 'lh_entorhinal', 'lh_fusiform', 'lh_inferiorparietal', 
        'lh_inferiortemporal', 'lh_isthmuscingulate', 'lh_lateraloccipital', 
        'lh_lateralorbitofrontal', 'lh_lingual', 'lh_medialorbitofrontal', 
        'lh_middletemporal', 'lh_parahippocampal', 'lh_paracentral', 
        'lh_parsopercularis', 'lh_parsorbitalis', 'lh_parstriangularis', 
        'lh_pericalcarine', 'lh_postcentral', 'lh_posteriorcingulate', 
        'lh_precentral', 'lh_precuneus', 'lh_rostralanteriorcingulate', 
        'lh_rostralmiddlefrontal', 'lh_superiorfrontal', 'lh_superiorparietal', 
        'lh_superiortemporal', 'lh_supramarginal', 'lh_frontalpole', 
        'lh_temporalpole', 'lh_transversetemporal', 'lh_insula', 
        'rh_bankssts', 'rh_caudalanteriorcingulate', 'rh_caudalmiddlefrontal', 
        'rh_cuneus', 'rh_entorhinal', 'rh_fusiform', 'rh_inferiorparietal', 
        'rh_inferiortemporal', 'rh_isthmuscingulate', 'rh_lateraloccipital', 
        'rh_lateralorbitofrontal', 'rh_lingual', 'rh_medialorbitofrontal', 
        'rh_middletemporal', 'rh_parahippocampal', 'rh_paracentral', 
        'rh_parsopercularis', 'rh_parsorbitalis', 'rh_parstriangularis', 
        'rh_pericalcarine', 'rh_postcentral', 'rh_posteriorcingulate', 
        'rh_precentral', 'rh_precuneus', 'rh_rostralanteriorcingulate', 
        'rh_rostralmiddlefrontal', 'rh_superiorfrontal', 'rh_superiorparietal', 
        'rh_superiortemporal', 'rh_supramarginal', 'rh_frontalpole', 
        'rh_temporalpole', 'rh_transversetemporal', 'rh_insula']

categorical_variables = []

binary_variables = ['sex']

# %%
def train_test_function(output_variable, level=1, sex='f'):
    hyperparameters = {
        'n_estimators': 0,
        'eta': 0,
        'max_depth': 0,
        'min_child_weight': 0,
        'subsample': 0,
        'gamma': 0,
        'colsample_bytree': 0,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'seed': 42
    }
    base_folder = '/Users/baileyng/MIND_data/best_hyperparameters'
    condition = '{}_{}_{}'.format(output_variable, level, sex)
    number_of_splits = 10
    y_test_conc = np.ndarray([])
    y_pred_conc = np.ndarray([])
    sv = np.ndarray([])
    sv_int = np.ndarray([])
    for split_idx in range(number_of_splits):
        split_folder = 'split_{}'.format(split_idx)
        data = np.load(os.path.join(base_folder, condition, split_folder, 'train_test_data.npz'),
                       allow_pickle=True)
        variables = data['column_names']
        X_trainval = pd.DataFrame(data=data['x_train'], columns=variables)
        y_trainval = pd.DataFrame(data=data['y_train'], columns=[output_variable])
        X_test = pd.DataFrame(data=data['x_test'], columns=variables)
        y_test = pd.DataFrame(data=data['y_test'], columns=[output_variable])
        #     print(X_trainval.head())

        numeric_vars = list(set(variables).intersection(numerical_variables))
        categorical_vars = list(set(variables).intersection(categorical_variables))
        binary_vars = list(set(variables).intersection(binary_variables))

        X_trainval[categorical_vars] = X_trainval[categorical_vars].astype('category')
        # X_trainval[binary_vars] = X_trainval[binary_vars].astype(int)
        for var in binary_vars:
            X_trainval[var] = pd.to_numeric(X_trainval[var], errors='coerce')
        X_test[categorical_vars] = X_test[categorical_vars].astype('category')
        # X_test[binary_vars] = X_test[binary_vars].astype(int)
        for var in binary_vars:
            X_test[var] = pd.to_numeric(X_test[var], errors='coerce')

        hyperparameters_file = open(os.path.join(base_folder, condition, split_folder, 'best_hyperparameters.json'))
        hyperparameters_best = json.load(hyperparameters_file)
        hyperparameters.update(hyperparameters_best)
        if np.unique(y_trainval).shape[0] >= 3:
            hyperparameters['eval_metric'] = 'rmse'
            hyperparameters['objective'] = 'reg:squarederror'
        else:
            hyperparameters['eval_metric'] = 'auc'
            hyperparameters['objective'] = 'binary:logistic'
        n_estimators = int(hyperparameters['n_estimators'])
        del hyperparameters['n_estimators']
        normalizer = RobustScaler()
        to_normalize = X_trainval[numeric_vars].values
        X_trainval[numeric_vars] = normalizer.fit_transform(X_trainval[numeric_vars])
        X_test[numeric_vars] = normalizer.transform(X_test[numeric_vars])

        traindata_xgb = xgb.DMatrix(X_trainval, y_trainval, enable_categorical=True)
        booster = xgb.train(hyperparameters, traindata_xgb, num_boost_round=n_estimators)

        testdata_xgb = xgb.DMatrix(X_test, y_test, enable_categorical=True)
        y_pred = booster.predict(testdata_xgb)
        shap_values = booster.predict(testdata_xgb, pred_contribs=True)
        shap_values_interaction = booster.predict(testdata_xgb, pred_interactions=True)

        #         sv=copy.deepcopy(shap_values)
        #         sv_int=copy.deepcopy(shap_values_interaction)
        #         print(sv.size)
        y_test_conc = np.vstack([y_test_conc, y_test.values]) if y_test_conc.size > 1 else y_test.values
        y_pred_conc = np.vstack([y_pred_conc, np.expand_dims(y_pred, 1)]) if y_pred_conc.size > 1 else np.expand_dims(
            y_pred, 1)
        sv = np.vstack([sv, copy.deepcopy(shap_values)]) if sv.size > 1 else copy.deepcopy(shap_values)
        sv_int = np.concatenate([sv_int, copy.deepcopy(shap_values_interaction)]) if sv_int.size > 1 else copy.deepcopy(
            shap_values_interaction)

        columns = X_test.columns

    return sv, sv_int, y_pred_conc, y_test_conc, columns

# %%
output_variables = ['20016-2.0']

binary_output_variables = []
continuous_output_variables = ['20016-2.0']

# %%
sexes = ['m', 'f']
parallel_coordinates = np.mgrid[0:2, 0:2, 0:11].reshape(3,-1)
instance_idx = int(argv[1])
# output_variables_1 = ['C-reactive protein',
#                       'C-reactive protein',
#                       'Pulse wave Arterial Stiffness index',
#                       'I48']
# lvls_1 = [1, 0, 0, 0]
# sexes_1 = ['m', 'f', 'm', 'm']

outvar = output_variables[parallel_coordinates[2, instance_idx]]
level = int(parallel_coordinates[1, instance_idx])
sx = sexes[parallel_coordinates[0, instance_idx]]
print(outvar, level, sx)

shap_values, shap_values_interaction, test_pred, test_true, colnames = train_test_function(outvar,
                                                                                                       level=level,
                                                                                                       sex=sx)
# save output
dir_name = '/external/rprshnas01/netdata_kcni/dflab/team/ma/bd_cvd/best_hyperparameters/{}_{}_{}'.format(outvar, level, sx)

colnames = np.array(list(colnames))
np.savez(os.path.join(dir_name, 'shap_values_output.npz'), test_pred=test_pred, test_true=test_true,
                     shap_values=shap_values, shap_values_interaction=shap_values_interaction,
                     column_names=colnames)

# %%



