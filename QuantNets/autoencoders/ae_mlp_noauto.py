# %% [markdown]
# # Set up

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import time
import os
import gc
import copy
from pathlib import Path
from tqdm.auto import tqdm
import optuna
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import sys
sys.path.append('/scratch/bng/cartbind/code/MIND_models/QuantNets/autoencoders')
from models import AgeGuidedAutoencoder, AgeGuidedLoss, MLPRegressor
from metrics import calc_r2_corr

# %%
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# # Configuration

# %%
base_dir = Path('/scratch/bng/cartbind/code/MIND_models')
data_dir = Path('/scratch/bng/cartbind/data/UKB_new_data/combined_data_no_outliers')
splits_dir = base_dir / 'scaling_law_splits'
region_dir = base_dir / 'region_names'

rename_df = pd.read_csv(region_dir / 'col_renames_dnanexus.csv')
rename_dict = dict(zip(rename_df['datafield_code'], rename_df['datafield_name']))

run_name = 'AE_MLP_new_may25'
# Switched weights_dir to params_dir since MLPs don't cleanly output coefficient vectors per feature
params_dir = base_dir / f'models_AE_MLP_dnanexus/{run_name}_params_scaling_law'
results_dir = base_dir / f'models_AE_MLP_dnanexus/{run_name}_scaling_law_results'
predictions_dir = base_dir / f'models_AE_MLP_dnanexus/{run_name}_predictions_scaling_law'
ae_curves_dir = base_dir / f'models_AE_MLP_dnanexus/{run_name}_ae_training_curves'

os.makedirs(params_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(ae_curves_dir, exist_ok=True)

# --- GLOBAL VARIABLES FOR EASY CONFIGURATION ---
MLP_N_TRIALS = 100                    # Number of trials per fold for Optuna
MLP_EPOCHS = 150                     # Epoches for training MLP
MLP_EARLY_STOP_PATIENCE = 15         # Early stop patience for MLP
AE_EARLY_STOP_PATIENCE = 15          # Early stop patience for AE

targets = {
    'GF': ('GF', 'p20016_i2'),
    'PAL': ('PAL', 'p20197_i2'),
    'DSST': ('DSST', 'p23324_i2'),
    'TMT': ('TMT', 'p6350_i2'),
}

data_configs = {
    'FC25': (region_dir / 'FC25_regions.txt', ['p31', 'p21003_i2', 'p54_i2', 'p25741_i2']),
    # 'FC100': (region_dir / 'FC100_regions.txt', ['p31', 'p21003_i2', 'p54_i2', 'p25741_i2']),
    'MIND': (region_dir / 'MIND_regions.txt', ['p31', 'p21003_i2', 'p54_i2']),   
}
sample_sizes = ['all']

ae_hyperparams = {
    'FC25': {
        '1_hide': {'hidden_dims': [128], 'latent_dim': 64, 'lr': 0.00508689, 'recon_weight': 0.93, 'ae_weight_decay': 5.463816, 'age_weight_decay': 0.0450122, 'age_predictor_hidden_dims': [32, 16, 8, 4], 'age_predictor_dropout': 0.25, 'epochs': 150},   # trial 286
        # '2_hide': {'hidden_dims': [128, 64], 'latent_dim': 32, 'lr': 1e-3, 'recon_weight': 0.5, 'ae_weight_decay': 1e-4, 'age_weight_decay': 1e-4, 'age_predictor_hidden_dims': [16, 8], 'age_predictor_dropout': 0.1, 'epochs': 150}
    },
    # 'FC100': {
    #     '1_hide': {'hidden_dims': [1024], 'latent_dim': 512, 'lr': 0.000534843, 'recon_weight': 0.88, 'ae_weight_decay': 1.481096, 'age_weight_decay': 0.000229702, 'age_predictor_hidden_dims': [4], 'age_predictor_dropout': 0.7, 'epochs': 150},   # trial 296
    #     # '2_hide': {'hidden_dims': [512], 'latent_dim': 256, 'lr': 0.001581248146663570, 'recon_weight': 0.8556561897363870, 'ae_weight_decay': 9.659428526695E-05, 'age_weight_decay': 1.56089378355989E-06, 'age_predictor_hidden_dims': [16], 'age_predictor_dropout': 0.5910979365703270, 'epochs': 150}
    # },
    'MIND': {
        '1_hide': {'hidden_dims': [1024], 'latent_dim': 512, 'lr': 0.0001293066, 'recon_weight': 0.99, 'ae_weight_decay': 1.56703, 'age_weight_decay': 0.0000839535, 'age_predictor_hidden_dims': [256, 32], 'age_predictor_dropout': 0.55, 'epochs': 150},   # trial 286
        # '2_hide': {'hidden_dims': [1024, 512], 'latent_dim': 256, 'lr': 1e-3, 'recon_weight': 0.5, 'ae_weight_decay': 1e-4, 'age_weight_decay': 1e-4, 'age_predictor_hidden_dims': [128, 64], 'age_predictor_dropout': 0.1, 'epochs': 150}
    }
}

# %% [markdown]
# # Training functions & MLP Optimization

# %%
def extract_features(model, loader, device):
    model.eval()
    latents, recons, age_preds = [], [], []
    with torch.inference_mode():
        for batch_x, batch_age in loader:
            batch_x = batch_x.to(device)
            x_hat, z, age_pred = model(batch_x)
            latents.append(z.cpu().float().numpy())
            recons.append(x_hat.cpu().float().numpy())
            age_preds.append(age_pred.cpu().float().numpy())
            
    return np.concatenate(latents, axis=0), np.concatenate(recons, axis=0), np.concatenate(age_preds, axis=0)

# %%
def train_ae_and_extract_latent(X_tr_br, X_vl_br, X_test_br, age_tr, age_vl, age_test, ae_params, early_stop_patience):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    X_scaler = StandardScaler()
    x_tr_scaled = X_scaler.fit_transform(X_tr_br.values)
    x_vl_scaled = X_scaler.transform(X_vl_br.values)
    x_test_scaled = X_scaler.transform(X_test_br.values)
    
    age_scaler = StandardScaler()
    age_tr_scaled = age_scaler.fit_transform(age_tr.values.reshape(-1, 1)).flatten()
    age_vl_scaled = age_scaler.transform(age_vl.values.reshape(-1, 1)).flatten()
    age_test_scaled = age_scaler.transform(age_test.values.reshape(-1, 1)).flatten()

    batch_size = 512
    train_loader = DataLoader(TensorDataset(torch.tensor(x_tr_scaled, dtype=torch.float32), torch.tensor(age_tr_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_vl_scaled, dtype=torch.float32), torch.tensor(age_vl_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    tr_extract_loader = DataLoader(TensorDataset(torch.tensor(x_tr_scaled, dtype=torch.float32), torch.tensor(age_tr_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    vl_extract_loader = DataLoader(TensorDataset(torch.tensor(x_vl_scaled, dtype=torch.float32), torch.tensor(age_vl_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_extract_loader = DataLoader(TensorDataset(torch.tensor(x_test_scaled, dtype=torch.float32), torch.tensor(age_test_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    try:
        model = AgeGuidedAutoencoder(
            input_dim=X_tr_br.shape[1], latent_dim=ae_params['latent_dim'], hidden_dims=ae_params['hidden_dims'],
            age_predictor_hidden_dims=ae_params['age_predictor_hidden_dims'], age_predictor_dropout=ae_params['age_predictor_dropout']
        ).to(device)

        model = torch.compile(model)
        
        criterion = AgeGuidedLoss(recon_weight=ae_params['recon_weight'], age_weight=1.0 - ae_params['recon_weight'])
        
        ae_decay, age_decay, no_decay = [], [], []
        for name, param in model.named_parameters():
            if 'bn' in name or 'bias' in name: no_decay.append(param)
            elif 'age_predictor' in name: age_decay.append(param)
            else: ae_decay.append(param)

        optimizer = torch.optim.AdamW([
            {'params': ae_decay, 'weight_decay': ae_params.get('ae_weight_decay', 1e-4)},
            {'params': age_decay, 'weight_decay': ae_params.get('age_weight_decay', 1e-4)},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=ae_params['lr'])

        num_epochs = ae_params['epochs']
        warmup_epochs = max(1, int(num_epochs * 0.05))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        decay_epochs = num_epochs - warmup_epochs
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
        curve_records = []
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_age in train_loader:
                batch_x, batch_age = batch_x.to(device), batch_age.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                x_hat, _, age_pred = model(batch_x)
                loss = criterion(batch_x, x_hat, batch_age, age_pred)
                    
                loss['total_loss'].backward()
                optimizer.step()
                    
                epoch_loss += loss['total_loss'].item() * batch_x.size(0)
                
            train_loss = epoch_loss / len(train_loader.dataset)
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.inference_mode():
                for batch_x, batch_age in val_loader:
                    batch_x, batch_age = batch_x.to(device), batch_age.to(device)
                    
                    x_hat, _, age_pred = model(batch_x)
                    loss = criterion(batch_x, x_hat, batch_age, age_pred)
                    
                    val_loss += loss['total_loss'].item() * batch_x.size(0)
                    
            val_loss = val_loss / len(val_loader.dataset)
            curve_records.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = copy.deepcopy(model._orig_mod.state_dict()) 
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stop_patience: break

        if best_model_state is not None:
            model._orig_mod.load_state_dict(best_model_state)

        z_tr, _, _ = extract_features(model, tr_extract_loader, device)
        z_vl, _, _ = extract_features(model, vl_extract_loader, device)
        z_test, x_hat_test, age_pred_test = extract_features(model, test_extract_loader, device)

        x_test_inv = X_scaler.inverse_transform(x_hat_test)
        age_pred_inv = age_scaler.inverse_transform(age_pred_test.reshape(-1, 1)).flatten()
        
        ae_recon_r2 = r2_score(X_test_br.values, x_test_inv)
        ae_recon_r2_corr = sum(calc_r2_corr(X_test_br.values[:, i], x_test_inv[:, i]) for i in range(x_test_inv.shape[1])) / x_test_inv.shape[1]
        ae_age_r2 = r2_score(age_test.values, age_pred_inv)
        ae_age_r2_corr = calc_r2_corr(age_test.values, age_pred_inv)

        return z_tr, z_vl, z_test, ae_recon_r2, ae_recon_r2_corr, ae_age_r2, ae_age_r2_corr, curve_records

    finally:
        try:
            del model, optimizer, scheduler, train_loader, val_loader, tr_extract_loader, vl_extract_loader, test_extract_loader
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
def analysis(X, y, df, data_name, target_name, arc_name, sample_size, brain_regions, demographic_vars, ae_params, n_splits=10):
    os.makedirs(params_dir / target_name, exist_ok=True)
    os.makedirs(predictions_dir / target_name, exist_ok=True)
    
    preds_path = predictions_dir / target_name / f'{run_name}_{arc_name}_preds_{data_name}_{target_name}_{sample_size}.csv'
    params_path = params_dir / target_name / f'{run_name}_{arc_name}_params_{data_name}_{target_name}_{sample_size}.csv'
    
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    outer_mae, outer_rmse, outer_r2, outer_r2_corr = [], [], [], []
    ae_recons, ae_recons_corr, ae_ages, ae_ages_corr = [], [], [], []
    params_records = []
    
    continuous_dvars = [c for c in demographic_vars if c not in ['sex', 'assessment_centre']]
    categorical_dvars = [c for c in demographic_vars if c in ['sex', 'assessment_centre']]

    print(f"\nEvaluating Base={data_name}, Target={target_name}, Arc={arc_name}, N={sample_size}")
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), total=n_splits, desc=f"CV Folds", leave=False), start=1):
        
        X_train_fold, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test = y.iloc[train_idx], y.iloc[test_idx]
        df_train_fold, df_test_fold = df.iloc[train_idx], df.iloc[test_idx]
        
        # 85/15 validation split on this fold's train set
        tr_sub_idx, vl_sub_idx = train_test_split(np.arange(len(X_train_fold)), test_size=0.15, random_state=seed)
        
        X_tr = X_train_fold.iloc[tr_sub_idx]
        X_vl = X_train_fold.iloc[vl_sub_idx]
        y_tr = y_train_fold.iloc[tr_sub_idx]
        y_vl = y_train_fold.iloc[vl_sub_idx]
        
        df_tr = df_train_fold.iloc[tr_sub_idx]
        df_vl = df_train_fold.iloc[vl_sub_idx]
        
        age_col = 'age' if 'age' in df.columns else 'p21003_i2'
        age_tr = df_tr[age_col]
        age_vl = df_vl[age_col]
        age_test = df_test_fold[age_col]
        
        # Train AE and return latents scoped appropriately for each dataset
        z_tr, z_vl, z_test, recon_r2, recon_r2_corr, age_r2, age_r2_corr, curve_df = train_ae_and_extract_latent(
            X_tr[brain_regions], X_vl[brain_regions], X_test[brain_regions], 
            age_tr, age_vl, age_test, ae_params, early_stop_patience=AE_EARLY_STOP_PATIENCE
        )
        
        ae_recons.append(recon_r2); ae_recons_corr.append(recon_r2_corr)
        ae_ages.append(age_r2); ae_ages_corr.append(age_r2_corr)
        pd.DataFrame(curve_df).to_csv(ae_curves_dir / f"{data_name}_{arc_name}_{target_name}_n{sample_size}_f{fold}_curve.csv", index=False)

        # Concatenate Demographics with Features
        z_cols = [f"z_{i}" for i in range(z_tr.shape[1])]
        X_mlp_tr_df = pd.concat([df_tr[demographic_vars].reset_index(drop=True), pd.DataFrame(z_tr, columns=z_cols)], axis=1)
        X_mlp_vl_df = pd.concat([df_vl[demographic_vars].reset_index(drop=True), pd.DataFrame(z_vl, columns=z_cols)], axis=1)
        X_mlp_test_df = pd.concat([df_test_fold[demographic_vars].reset_index(drop=True), pd.DataFrame(z_test, columns=z_cols)], axis=1)

        # Scale MLP Inputs 
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), continuous_dvars + z_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_dvars),
        ])
        
        X_mlp_tr_np = preprocessor.fit_transform(X_mlp_tr_df)
        X_mlp_vl_np = preprocessor.transform(X_mlp_vl_df)
        X_mlp_test_np = preprocessor.transform(X_mlp_test_df)
        
        # Scale Targets
        y_scaler = StandardScaler()
        y_tr_np = y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        y_vl_np = y_scaler.transform(y_vl.values.reshape(-1, 1)).flatten()

        X_mlp_tr_tensor = torch.tensor(X_mlp_tr_np, dtype=torch.float32).to(device)
        y_tr_tensor = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
        X_mlp_vl_tensor = torch.tensor(X_mlp_vl_np, dtype=torch.float32).to(device)
        y_vl_tensor = torch.tensor(y_vl_np, dtype=torch.float32).to(device)

        batch_size = 256
        mlp_train_loader = DataLoader(TensorDataset(X_mlp_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        mlp_val_loader = DataLoader(TensorDataset(X_mlp_vl_tensor, y_vl_tensor), batch_size=batch_size, shuffle=False)

        best_mlp_state = None
        best_mlp_val_loss = float('inf')

        def mlp_objective(trial):
            nonlocal best_mlp_state, best_mlp_val_loss

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e1, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.95, step=0.05)

            latent_dim = ae_params['latent_dim']
            mlp_pred_depth = trial.suggest_int("mlp_depth", 1, 4)
            mlp_hidden_dims = []
            
            current_upper_bound = int(np.log2(latent_dim)) - 1
            for i in range(mlp_pred_depth):
                min_possible = mlp_pred_depth - i
                lower_bound = max(1, min_possible)
                power = trial.suggest_int(f"mlp_dim_exp_l{i}", lower_bound, current_upper_bound)
                mlp_hidden_dims.append(2 ** power)
                current_upper_bound = power - 1

            model = MLPRegressor(input_dim=X_mlp_tr_np.shape[1], hidden_dims=mlp_hidden_dims, dropout_rate=dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Re-implementing original LR Scheduler inside Optuna Objective Loop 
            warmup_epochs = max(1, int(MLP_EPOCHS * 0.05))
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            decay_epochs = MLP_EPOCHS - warmup_epochs
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

            criterion = nn.MSELoss()

            trial_best_val_loss = float('inf')
            early_stop_counter = 0
            trial_best_model_state = None

            for epoch in range(MLP_EPOCHS):
                model.train()
                for bx, by in mlp_train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    
                    preds = model(bx)
                    loss = criterion(preds, by)
                    
                    loss.backward()
                    optimizer.step()
                        
                scheduler.step()
                
                model.eval()
                val_loss = 0
                with torch.inference_mode():
                    for bx, by in mlp_val_loader:
                        preds = model(bx)
                        loss = criterion(preds, by)
                        val_loss += loss.item() * bx.size(0)
                
                val_loss /= len(mlp_val_loader.dataset)
                
                if np.isnan(val_loss) or np.isinf(val_loss):
                    raise optuna.exceptions.TrialPruned()

                if val_loss < trial_best_val_loss:
                    trial_best_val_loss = val_loss
                    early_stop_counter = 0
                    trial_best_model_state = copy.deepcopy(model.state_dict())
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= MLP_EARLY_STOP_PATIENCE:
                    break

            if trial_best_val_loss < best_mlp_val_loss:
                best_mlp_val_loss = trial_best_val_loss
                best_mlp_state = copy.deepcopy(trial_best_model_state)

            return trial_best_val_loss

        # Launch 10-Fold Scoped Optuna Trials
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with tqdm(total=MLP_N_TRIALS, desc=f"Fold {fold} Optuna Trials", leave=False) as pbar:
            def tqdm_callback(study, trial):
                pbar.update(1)
                
            study.optimize(mlp_objective, n_trials=MLP_N_TRIALS, callbacks=[tqdm_callback])

        # Restore Top Model & Architecture
        best_params = study.best_params
        best_mlp_hidden_dims = []
        for i in range(best_params['mlp_depth']):
            best_mlp_hidden_dims.append(2 ** best_params[f"mlp_dim_exp_l{i}"])
            
        best_model = MLPRegressor(input_dim=X_mlp_tr_np.shape[1], hidden_dims=best_mlp_hidden_dims, dropout_rate=best_params['dropout']).to(device)
        best_model.load_state_dict(best_mlp_state)
        best_model = torch.compile(best_model)
        best_model.eval()

        # Evaluation Pipeline
        X_test_tensor = torch.tensor(X_mlp_test_np, dtype=torch.float32).to(device)
        with torch.inference_mode():
            test_preds_scaled = best_model(X_test_tensor).cpu().numpy()
                
        y_pred = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

        fold_df = pd.DataFrame({'fold': fold, 'eid': y_test.index, 'actual': y_test.values, 'predicted': y_pred})
        fold_df.to_csv(preds_path, mode='a', header=(fold==1), index=False)
        
        # Track Optuna params logic instead of weights
        params_df = pd.DataFrame([{'fold': fold, **study.best_params}])
        params_records.append(params_df)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        r2_corr = calc_r2_corr(y_test, y_pred)

        outer_mae.append(mae)
        outer_rmse.append(rmse)
        outer_r2.append(r2)
        outer_r2_corr.append(r2_corr)

        print(f'  Fold {fold:02d} • MAE={mae:.3f} • RMSE={rmse:.3f} • R²={r2:.3f} • R²(corr)={r2_corr:.3f} '
              f'• Optuna Loss={best_mlp_val_loss:.4f} '
              f'• AE Recon R²(corr)={recon_r2_corr:.3f} • AE Age R²(corr)={age_r2_corr:.3f}')

        # Garbage Collection per Fold
        del best_model, preprocessor, study
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    print(f'\n  Mean MAE             : {np.mean(outer_mae):.3f} ± {np.std(outer_mae):.3f}')
    print(f'  Mean RMSE            : {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}')
    print(f'  Mean R²              : {np.mean(outer_r2):.3f} ± {np.std(outer_r2):.3f}')
    print(f'  Mean R²(corr)        : {np.mean(outer_r2_corr):.3f} ± {np.std(outer_r2_corr):.3f}')
    print(f'  Mean AE Recon R²(cor): {np.mean(ae_recons_corr):.3f} ± {np.std(ae_recons_corr):.3f}')
    print(f'  Mean AE Age R²(corr) : {np.mean(ae_ages_corr):.3f} ± {np.std(ae_ages_corr):.3f}')

    pd.concat(params_records).to_csv(params_path, index=False)
    
    elapsed_time = time.time() - start_time

    return {
        'arc_name':                  arc_name,
        'mean_mae':                  np.mean(outer_mae),
        'mean_rmse':                 np.mean(outer_rmse),
        'mean_r2':                   np.mean(outer_r2),
        'std_r2':                    np.std(outer_r2),
        'mean_r2_corr':              np.mean(outer_r2_corr),
        'std_r2_corr':               np.std(outer_r2_corr),
        'ae_test_recon_r2':          np.mean(ae_recons),
        'std_ae_test_recon_r2':      np.std(ae_recons),
        'ae_test_recon_r2_corr':     np.mean(ae_recons_corr),
        'std_ae_test_recon_r2_corr': np.std(ae_recons_corr),
        'ae_test_age_r2':            np.mean(ae_ages),
        'std_ae_test_age_r2':        np.std(ae_ages),
        'ae_test_age_r2_corr':       np.mean(ae_ages_corr),
        'std_ae_test_age_r2_corr':   np.std(ae_ages_corr),
        'elapsed_time_sec':          elapsed_time
    }

# %% [markdown]
# # Scaling Law Loop

# %%
# Main scaling entry (Largely identical logic as previous notebook)
for target_name, (test_key, score_col) in targets.items():
    data_file = data_dir / f'combined_data_{test_key}_no_outliers.csv'
    df_full = pd.read_csv(data_file, index_col=0)
    
    df_full = df_full.rename(columns=rename_dict)
    if score_col in rename_dict:
        score_col = rename_dict[score_col]

    for data_name, (regions_file, demographic_vars) in data_configs.items():
        with open(regions_file, 'r') as f:
            brain_regions = [line.strip() for line in f]
            
        brain_regions = [rename_dict.get(br, br) for br in brain_regions]
        demographic_vars = [rename_dict.get(dv, dv) for dv in demographic_vars]
        all_vars = demographic_vars + brain_regions

        for arc_name in ['1_hide']:
            ae_params = ae_hyperparams[data_name][arc_name]

            for sample_size in sample_sizes:
                eid_file = splits_dir / target_name / (f'{target_name}_all_eids.txt' if sample_size == 'all' else f'{target_name}_eids_{sample_size}.txt')
                if not eid_file.exists(): continue
                    
                sample_eids = np.loadtxt(eid_file, dtype=int)
                df = df_full[df_full['eid'].isin(sample_eids)] if 'eid' in df_full.columns else df_full[df_full.index.isin(sample_eids)]

                X, y = df[all_vars], df[score_col]
                
                metrics = analysis(X, y, df, data_name, target_name, arc_name, sample_size, brain_regions, demographic_vars, ae_params)

                row_df = pd.DataFrame([{
                    'target_name': target_name, 'data_name': data_name, 
                    'arc_name': arc_name, 'sample_size': sample_size, 'actual_n': len(df), **metrics
                }])
                
                results_file = results_dir / f'scaling_law_results_AE_MLP_{target_name}.csv'
                row_df.to_csv(str(results_file), mode='a', header=not results_file.exists(), index=False)


