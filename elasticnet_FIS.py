# %%
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %%
# Load the dataset
df = pd.read_csv('/Users/baileyng/MIND_models/ukb_FIS.csv', index_col=0)

# %%
# Histogram of Fluid Intelligence Scores
sns.histplot(df['20016-2.0'])
plt.title('Distribution of Fluid Intelligence Scores')
plt.xlabel('Fluid Intelligence Score')
plt.ylabel('Frequency')
plt.show()

# %%
print(len(df))

# %% [markdown]
# ##### Using MIND to predict Fluid Intelligence Score

# %%
# Set X and y

X = df[['31-0.0', '21003-2.0',
        
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
        'rh_temporalpole', 'rh_transversetemporal', 'rh_insula']]

y = df['20016-2.0']

# %%
# Cross-validation set-up
outer_cv = KFold(n_splits=20, shuffle=True, random_state=42)

outer_mae = []
outer_rmse = []
outer_r2 = []
best_params_per_fold = []
nonzero_predictors = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner CV
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=np.linspace(0.3, 0.9, 7),
            alphas=np.logspace(-4, 2, 20),
            cv=20,
            max_iter=5000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    outer_mae.append(mae)
    outer_rmse.append(rmse)
    outer_r2.append(r2)

    # --- store best α & l1_ratio for this fold ---
    est = model.named_steps['elasticnetcv']
    best_params_per_fold.append(
        {'alpha': est.alpha_, 'l1_ratio': est.l1_ratio_}
    )

    # --- predictors that survived ---
    coefs = est.coef_
    surviving = [col for col, c in zip(X.columns, coefs) if c != 0]
    nonzero_predictors.append(surviving)

    print(f"Fold {fold:02d} • MAE={mae:.3f} • RMSE={rmse:.3f} • R²={r2:.3f} "
          f"• α={est.alpha_:.4g} • l1_ratio={est.l1_ratio_:.2f}")
    

# Aggregate results
print("\n=== 10-fold CV summary ===")
print(f"Mean MAE :  {np.mean(outer_mae):.3f}  ± {np.std(outer_mae):.3f}")
print(f"Mean RMSE:  {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}")
print(f"Mean R²  :  {np.mean(outer_r2):.3f}  ± {np.std(outer_r2):.3f}")



# %%
# View parameter choices & surviving variables
param_df = pd.DataFrame(best_params_per_fold)
print("\nBest α and l1_ratio per fold\n", param_df)

print("\nVariables that kept non-zero coefficients in ≥1 fold:")
print(sorted({v for fold_vars in nonzero_predictors for v in fold_vars}))

# %%
# Final refit on all data
final_model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(
        l1_ratio=np.linspace(0.3, 0.9, 7),
        alphas=np.logspace(-4, 2, 20),
        cv=20,
        max_iter=5000,
        random_state=42
    )
).fit(X, y)

print("\n=== Final model ===")
print(f"α  = {final_model.named_steps['elasticnetcv'].alpha_:.4g}")
print(f"l1 = {final_model.named_steps['elasticnetcv'].l1_ratio_:.2f}")


