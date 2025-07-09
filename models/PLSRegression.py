# %% [markdown]
# # Set up

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
# import nibabel.freesurfer.io as fsio
# from nilearn import datasets, plotting
import random
# from adjustText import adjust_text
# from matplotlib.patches import Ellipse

# %%
random.seed(42)
np.random.seed(42)

# %% [markdown]
# # PLS Analysis Function

# %%
def pct_var(pls: PLSRegression, X: np.ndarray, Y: np.ndarray):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    SSX = np.sum(Xc**2)
    SSY = np.sum(Yc**2)

    T = pls.x_scores_
    P = pls.x_loadings_
    Q = pls.y_loadings_

    pct_X = []
    pct_Y = []
    for k in range(1, pls.n_components+1):
        Tk = T[:, :k]
        Pk = P[:, :k]
        Qk = Q[:, :k]

        Xhat = Tk @ Pk.T
        Yhat = Tk @ Qk.T

        pct_X.append(100 * np.sum(Xhat**2) / SSX)
        pct_Y.append(100 * np.sum(Yhat**2) / SSY)

    return np.array(pct_X), np.array(pct_Y)

# %%
# def varname_to_region(var):
#     if var.startswith('lh_'):
#         return 'lh', var.replace('lh_', '').replace('_thickness', '')
#     elif var.startswith('rh_'):
#         return 'rh', var.replace('rh_', '').replace('_thickness', '').replace('.', '')
#     return None, None

# # 2. Set FreeSurfer fsaverage path
# fsaverage_path = '/Applications/freesurfer/subjects/fsaverage'

# # 3. Load annotation files
# lh_annot = fsio.read_annot(f'{fsaverage_path}/label/lh.aparc.annot')
# rh_annot = fsio.read_annot(f'{fsaverage_path}/label/rh.aparc.annot')
# lh_labels, lh_cmap, lh_names = lh_annot
# rh_labels, rh_cmap, rh_names = rh_annot
# lh_names = [name.decode('utf-8') for name in lh_names]
# rh_names = [name.decode('utf-8') for name in rh_names]

# %%
def pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10):
    preprocessor = ColumnTransformer(transformers=[
        # scale continuous features
        ('num', StandardScaler(), continuous_vars),
        # one-hot encode the assessment centre (drop one level to avoid collinearity)
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars),
    ])

    # Cross-validation set-up
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    outer_mae, outer_rmse, outer_r2 = [], [], []
    best_ncomps = []
    coefs_list = []
    vip_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV
        pipe = make_pipeline(
            preprocessor,
            PLSRegression()
        )

        param_grid = {
            'plsregression__n_components': list(range(1, 11))
        }

        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipe, 
            param_grid, 
            cv=inner_cv, 
            scoring='neg_mean_squared_error',  # or 'r2'
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        best_n = grid.best_params_['plsregression__n_components']
        best_ncomps.append(best_n)
        
        y_pred = grid.predict(X_test)

        # --- metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        outer_mae.append(mae)
        outer_rmse.append(rmse)
        outer_r2.append(r2)

        # --- extract loadings and compute VIP ---
        pls = grid.best_estimator_.named_steps['plsregression']
        coefs_list.append(pls.coef_.ravel())
        
        T = pls.x_scores_
        W = pls.x_weights_
        Q = pls.y_loadings_
        p, h = W.shape

        # S = np.sum((T ** 2) * (Q.T ** 2), axis=0)
        S = np.sum((T ** 2) * (Q ** 2), axis=0)
        total_S = np.sum(S)
        vip = np.zeros(p)
        for j in range(p):
            # weight_sq = [(W[j, k] ** 2) * S[k] for k in range(h)]
            # vip[j] = np.sqrt(p * np.sum(weight_sq) / total_S)
            vip[j] = np.sqrt(p * np.sum((W[j, :]**2) * S) / total_S)
        vip_list.append(vip)

        print(f'Fold {fold:02d} • n_comp={best_n:02d} • '
            f'MAE={mae:.3f} • RMSE={rmse:.3f} • R²={r2:.3f}')
        

    # Aggregate results
    print('\n=== 10-fold CV summary ===')
    print(f'n_components (mean ± std): '
        f'{np.mean(best_ncomps):.1f} ± {np.std(best_ncomps):.1f}')
    print(f'MAE :  {np.mean(outer_mae):.3f} ± {np.std(outer_mae):.3f}')
    print(f'RMSE:  {np.mean(outer_rmse):.3f} ± {np.std(outer_rmse):.3f}')
    print(f'R²  :  {np.mean(outer_r2):.3f} ± {np.std(outer_r2):.3f}')

    # # Get feature names after preprocessing
    # # First, fit the preprocessor to get the transformed feature names
    # preprocessor_fitted = preprocessor.fit(X)

    # # Get feature names for each transformer
    # num_features = continuous_vars
    # cat_features = list(preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_vars))

    # # Combine all feature names in the correct order
    # all_feature_names = num_features + cat_features


    # # Final refit on all data
    # final_pipe = make_pipeline(
    #     preprocessor,
    #     PLSRegression(
    #         n_components=int(np.round(np.mean(best_ncomps))),
    #     )
    # ).fit(X, y)

    # pls = final_pipe.named_steps['plsregression']
    # ct  = final_pipe.named_steps['columntransformer']




    # # 3) re-transform your original X to the exact matrix PLS saw
    # X_proc = ct.transform(X)            # this is a NumPy array, shape (n_samples, n_features')

    # # 4) compute the cumulative % variance explained
    # X_pctvar, Y_pctvar = pct_var(pls, X_proc, y)

    # print("Percentage of variance explained:")
    # print("Component\tX (Predictors)\tY (Response)")
    # print("-" * 40)
    # for i in range(len(X_pctvar)):
    #     print(f"PLS{i+1}\t\t{X_pctvar[i]:.2f}%\t\t{Y_pctvar[i]:.2f}%")

    # print(f"\nTotal variance explained:")
    # print(f"X: {X_pctvar[-1]:.2f}%")
    # print(f"Y: {Y_pctvar[-1]:.2f}%")

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # components = np.arange(1, len(X_pctvar) + 1)
    # plt.plot(components, X_pctvar, 'b-o', label='X (Predictors)', linewidth=2)
    # plt.plot(components, Y_pctvar, 'r-o', label='Y (Fluid Intelligence)', linewidth=2)
    # plt.xlabel('PLS Component')
    # plt.ylabel('Cumulative Variance Explained (%)')
    # plt.title('Percentage of Variance Explained by PLS Components')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.xticks(components)
    # plt.show()


    # T = pls.x_scores_

    # # Scatter plot of Fluid Intelligence Score vs PLS1 scores
    # plt.figure(figsize=(8, 6))
    # plt.scatter(T[:,0], y, alpha=0.6, s=12, color='blue')
    # plt.xlabel("PLS1 Scores")
    # plt.ylabel("Fluid Intelligence Score")
    # plt.title("Fluid Intelligence Score vs PLS1 Scores")
    # plt.show()

    # # Scatter plot of Fluid Intelligence Score vs PLS2 scores
    # plt.figure(figsize=(8, 6))
    # plt.scatter(T[:,1], y, alpha=0.6, s=12, color='red')
    # plt.xlabel("PLS2 Scores")
    # plt.ylabel("Fluid Intelligence Score")
    # plt.title("Fluid Intelligence Score vs PLS2 Scores")
    # plt.show()

    # W = pls.x_weights_
    # plt.figure(figsize=(12, 6))
    # plt.bar(all_feature_names, W[:,0])
    # plt.xticks(rotation=90)
    # plt.title("PLS1 Weights")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.bar(all_feature_names, W[:,1])
    # plt.xticks(rotation=90)
    # plt.title("PLS2 Weights")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.bar(all_feature_names, W[:,2])
    # plt.xticks(rotation=90)
    # plt.title("PLS3 Weights")
    # plt.tight_layout()
    # plt.show()


    # # # Calculate Z-scores for the weights
    # # W1_mean = np.mean(W[:, 0])
    # # W1_std = np.std(W[:, 0])
    # # W2_mean = np.mean(W[:, 1])
    # # W2_std = np.std(W[:, 1])

    # # plt.figure(figsize=(8, 10))
    # # plt.axhline(0, color='grey', lw=1)
    # # plt.axvline(0, color='grey', lw=1)

    # # # Draw arrows for each variable
    # # for i, var in enumerate(all_feature_names):
    # #     plt.arrow(0, 0, W[i, 0], W[i, 1], 
    # #             color='b', alpha=0.6, head_width=0.01, head_length=0.01, length_includes_head=True)

    # # # Label all variables, closer to the arrow tip
    # # texts = []
    # # for i, var in enumerate(all_feature_names):
    # #     texts.append(
    # #         plt.text(W[i, 0]*1.02, W[i, 1]*1.02, var, color='r', ha='center', va='center', 
    # #                 fontsize=8, clip_on=True)
    # #     )

    # # ellipse = Ellipse((0, 0), 
    # #                 width=6*W1_std, height=6*W2_std,  # 6 = 2*3 std devs (±3)
    # #                 fill=False, color='g', linestyle='--', linewidth=2, 
    # #                 label='Z-score = ±3 (elliptical)')
    # # plt.gca().add_patch(ellipse)

    # # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5), 
    # #             expand_points=(1.2, 1.2), expand_text=(1.2, 1.2), force_text=0.5, force_points=0.5)

    # # plt.xlabel("PLS1 Weight")
    # # plt.ylabel("PLS2 Weight")
    # # plt.title("PLS Weight Plot")
    # # plt.grid(True)
    # # plt.axis('equal')
    # # plt.tight_layout()
    # # plt.legend()
    # # plt.show()

    # # 1. Variable names and coefficients
    # vars_to_plot = np.array(all_feature_names)
    # means = pls.x_weights_[:,0]

    # # 4. Prepare vertex data arrays
    # lh_vertex_data = np.full_like(lh_labels, np.nan, dtype=float)
    # rh_vertex_data = np.full_like(rh_labels, np.nan, dtype=float)

    # for var, coef in zip(vars_to_plot, means):
    #     hemi, region = varname_to_region(var)
    #     if hemi == 'lh':
    #         idxs = [i for i, n in enumerate(lh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             lh_vertex_data[lh_labels == idx] = coef
    #     elif hemi == 'rh':
    #         idxs = [i for i, n in enumerate(rh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             rh_vertex_data[rh_labels == idx] = coef

    # # 5. Load inflated_pre surfaces
    # lh_inflated_pre_path = f'{fsaverage_path}/surf/lh.inflated_pre'
    # rh_inflated_pre_path = f'{fsaverage_path}/surf/rh.inflated_pre'
    # lh_coords, lh_faces = fsio.read_geometry(lh_inflated_pre_path)
    # rh_coords, rh_faces = fsio.read_geometry(rh_inflated_pre_path)
    # lh_mesh = (lh_coords, lh_faces)
    # rh_mesh = (rh_coords, rh_faces)

    # # 6. Load fsaverage sulcal maps for background
    # fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    # sulc_left = fsavg.sulc_left
    # sulc_right = fsavg.sulc_right

    # # 7. Plotting function
    # global_vmax = np.nanmax(np.abs(np.concatenate([lh_vertex_data, rh_vertex_data])))

    # def plot_three_views_custom(surf_mesh, vertex_data, bg_map, hemi, title):
    #     views = ['lateral', 'medial', 'dorsal']
    #     fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(40, 20))
    #     cmap = 'seismic'
    #     for i, view in enumerate(views):
    #         plotting.plot_surf_stat_map(
    #             surf_mesh, vertex_data, hemi=hemi, bg_map=bg_map,
    #             cmap=cmap, colorbar=False, vmax=global_vmax, vmin=-global_vmax, symmetric_cbar=True,
    #             view=view, axes=axes[i], title='', figure=fig, alpha=0.9
    #         )
    #         axes[i].set_title(view.capitalize(), fontsize=30)
    #         axes[i].axis('off')
    #     # Colorbar
    #     cbar_ax = fig.add_axes([0.33, 0.09, 0.34, 0.03])
    #     norm = plt.Normalize(vmin=-global_vmax, vmax=global_vmax)
    #     cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    #     cb.set_label('PLS1 Weights', fontsize=16)
    #     cb.ax.tick_params(labelsize=18)
    #     plt.suptitle(title, fontsize=38, y=0.96)
    #     plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    #     plt.show()

    # # 8. Plot with inflated_pre
    # plot_three_views_custom(lh_mesh, lh_vertex_data, sulc_left, 'left', '(Demographic + MIND)')
    # plot_three_views_custom(rh_mesh, rh_vertex_data, sulc_right, 'right', '(Demographic + MIND)')

    # # 1. Variable names and coefficients
    # vars_to_plot = np.array(all_feature_names)
    # means = pls.x_weights_[:,1]

    # # 4. Prepare vertex data arrays
    # lh_vertex_data = np.full_like(lh_labels, np.nan, dtype=float)
    # rh_vertex_data = np.full_like(rh_labels, np.nan, dtype=float)

    # for var, coef in zip(vars_to_plot, means):
    #     hemi, region = varname_to_region(var)
    #     if hemi == 'lh':
    #         idxs = [i for i, n in enumerate(lh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             lh_vertex_data[lh_labels == idx] = coef
    #     elif hemi == 'rh':
    #         idxs = [i for i, n in enumerate(rh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             rh_vertex_data[rh_labels == idx] = coef

    # # 5. Load inflated_pre surfaces
    # lh_inflated_pre_path = f'{fsaverage_path}/surf/lh.inflated_pre'
    # rh_inflated_pre_path = f'{fsaverage_path}/surf/rh.inflated_pre'
    # lh_coords, lh_faces = fsio.read_geometry(lh_inflated_pre_path)
    # rh_coords, rh_faces = fsio.read_geometry(rh_inflated_pre_path)
    # lh_mesh = (lh_coords, lh_faces)
    # rh_mesh = (rh_coords, rh_faces)

    # # 6. Load fsaverage sulcal maps for background
    # fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    # sulc_left = fsavg.sulc_left
    # sulc_right = fsavg.sulc_right

    # # 7. Plotting function
    # global_vmax = np.nanmax(np.abs(np.concatenate([lh_vertex_data, rh_vertex_data])))

    # def plot_three_views_custom(surf_mesh, vertex_data, bg_map, hemi, title):
    #     views = ['lateral', 'medial', 'dorsal']
    #     fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(40, 20))
    #     cmap = 'seismic'
    #     for i, view in enumerate(views):
    #         plotting.plot_surf_stat_map(
    #             surf_mesh, vertex_data, hemi=hemi, bg_map=bg_map,
    #             cmap=cmap, colorbar=False, vmax=global_vmax, vmin=-global_vmax, symmetric_cbar=True,
    #             view=view, axes=axes[i], title='', figure=fig, alpha=0.9
    #         )
    #         axes[i].set_title(view.capitalize(), fontsize=30)
    #         axes[i].axis('off')
    #     # Colorbar
    #     cbar_ax = fig.add_axes([0.33, 0.09, 0.34, 0.03])
    #     norm = plt.Normalize(vmin=-global_vmax, vmax=global_vmax)
    #     cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    #     cb.set_label('PLS2 Weights', fontsize=16)
    #     cb.ax.tick_params(labelsize=18)
    #     plt.suptitle(title, fontsize=38, y=0.96)
    #     plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    #     plt.show()

    # # 8. Plot with inflated_pre
    # plot_three_views_custom(lh_mesh, lh_vertex_data, sulc_left, 'left', '(Demographic + MIND)')
    # plot_three_views_custom(rh_mesh, rh_vertex_data, sulc_right, 'right', '(Demographic + MIND)')

    # # 1. Variable names and coefficients
    # vars_to_plot = np.array(all_feature_names)
    # means = pls.x_weights_[:,2]

    # # 4. Prepare vertex data arrays
    # lh_vertex_data = np.full_like(lh_labels, np.nan, dtype=float)
    # rh_vertex_data = np.full_like(rh_labels, np.nan, dtype=float)

    # for var, coef in zip(vars_to_plot, means):
    #     hemi, region = varname_to_region(var)
    #     if hemi == 'lh':
    #         idxs = [i for i, n in enumerate(lh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             lh_vertex_data[lh_labels == idx] = coef
    #     elif hemi == 'rh':
    #         idxs = [i for i, n in enumerate(rh_names) if region.lower() in n.lower()]
    #         for idx in idxs:
    #             rh_vertex_data[rh_labels == idx] = coef

    # # 5. Load inflated_pre surfaces
    # lh_inflated_pre_path = f'{fsaverage_path}/surf/lh.inflated_pre'
    # rh_inflated_pre_path = f'{fsaverage_path}/surf/rh.inflated_pre'
    # lh_coords, lh_faces = fsio.read_geometry(lh_inflated_pre_path)
    # rh_coords, rh_faces = fsio.read_geometry(rh_inflated_pre_path)
    # lh_mesh = (lh_coords, lh_faces)
    # rh_mesh = (rh_coords, rh_faces)

    # # 6. Load fsaverage sulcal maps for background
    # fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    # sulc_left = fsavg.sulc_left
    # sulc_right = fsavg.sulc_right

    # # 7. Plotting function
    # global_vmax = np.nanmax(np.abs(np.concatenate([lh_vertex_data, rh_vertex_data])))

    # def plot_three_views_custom(surf_mesh, vertex_data, bg_map, hemi, title):
    #     views = ['lateral', 'medial', 'dorsal']
    #     fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(40, 20))
    #     cmap = 'seismic'
    #     for i, view in enumerate(views):
    #         plotting.plot_surf_stat_map(
    #             surf_mesh, vertex_data, hemi=hemi, bg_map=bg_map,
    #             cmap=cmap, colorbar=False, vmax=global_vmax, vmin=-global_vmax, symmetric_cbar=True,
    #             view=view, axes=axes[i], title='', figure=fig, alpha=0.9
    #         )
    #         axes[i].set_title(view.capitalize(), fontsize=30)
    #         axes[i].axis('off')
    #     # Colorbar
    #     cbar_ax = fig.add_axes([0.33, 0.09, 0.34, 0.03])
    #     norm = plt.Normalize(vmin=-global_vmax, vmax=global_vmax)
    #     cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    #     cb.set_label('PLS3 Weights', fontsize=16)
    #     cb.ax.tick_params(labelsize=18)
    #     plt.suptitle(title, fontsize=38, y=0.96)
    #     plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    #     plt.show()

    # # 8. Plot with inflated_pre
    # plot_three_views_custom(lh_mesh, lh_vertex_data, sulc_left, 'left', '(Demographic + MIND)')
    # plot_three_views_custom(rh_mesh, rh_vertex_data, sulc_right, 'right', '(Demographic + MIND)')




# %% [markdown]
# # GF

# %%
# Load the dataset
df = pd.read_csv('/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/ukb_master_GF_no_outliers.csv', index_col=0)

# %%
# Histogram of Fluid Intelligence Scores
sns.histplot(df['20016-2.0'])
plt.title('Distribution of Fluid Intelligence Scores')
plt.xlabel('Fluid Intelligence Score')
plt.ylabel('Frequency')
plt.show()

# %%
print(len(df))
print(df.shape)

# %% [markdown]
# ## GF vs. MIND (avg)

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_avg_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['20016-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## GF vs. MIND

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['20016-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## GF vs. CT

# %%
X = df[['31-0.0', '21003-2.0', '54-2.0',
        
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']]

y = df['20016-2.0']

# %%
datafield_code = ['31-0.0', '21003-2.0', '54-2.0',
    
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']

datafield_name = ['sex', 'age', 'assessment_centre',
    
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
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## GF vs. FC

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/FC_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['20016-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

datafield_name = ['sex', 'age', 'assessment_centre', 'head_motion']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# # PAL

# %%
# Load the dataset
df = pd.read_csv('/Users/baileyng/MIND_data/ukb_master_PAL_no_outliers.csv', index_col=0)

# %%
# Histogram of PAL Scores
sns.histplot(df['20197-2.0'])
plt.title('Distribution of PAL Scores')
plt.xlabel('PAL Score')
plt.ylabel('Frequency')
plt.show()

# %%
print(len(df))
print(df.shape)

# %% [markdown]
# ## PAL vs. MIND (avg)

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_avg_regions.txt', 'r') as f:
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
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## PAL vs. MIND

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_regions.txt', 'r') as f:
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
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## PAL vs. CT

# %%
X = df[['31-0.0', '21003-2.0', '54-2.0',
        
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']]

y = df['20197-2.0']

# %%
datafield_code = ['31-0.0', '21003-2.0', '54-2.0',
    
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']

datafield_name = ['sex', 'age', 'assessment_centre',
    
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
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## PAL vs. FC

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/FC_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['20197-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

datafield_name = ['sex', 'age', 'assessment_centre', 'head_motion']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# # DSST

# %%
# Load the dataset
df = pd.read_csv('/Users/baileyng/MIND_data/ukb_master_DSST_no_outliers.csv', index_col=0)

# %%
# Histogram of DSST Scores
sns.histplot(df['23324-2.0'])
plt.title('Distribution of DSST Scores')
plt.xlabel('DSST Score')
plt.ylabel('Frequency')
plt.show()

# %%
print(len(df))
print(df.shape)

# %% [markdown]
# ## DSST vs. MIND (avg)

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_avg_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['23324-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## DSST vs. MIND

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['23324-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## DSST vs. CT

# %%
X = df[['31-0.0', '21003-2.0', '54-2.0',
        
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']]

y = df['23324-2.0']

# %%
datafield_code = ['31-0.0', '21003-2.0', '54-2.0',
    
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']

datafield_name = ['sex', 'age', 'assessment_centre',
    
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
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## DSST vs. FC

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/FC_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['23324-2.0']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

datafield_name = ['sex', 'age', 'assessment_centre', 'head_motion']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# # TMT

# %%
# Load the dataset
df = pd.read_csv('/Users/baileyng/MIND_data/ukb_master_TMT_no_outliers.csv', index_col=0)

# %%
# Histogram of TMT Scores
sns.histplot(df['trailmaking_score'])
plt.title('Distribution of TMT Scores')
plt.xlabel('TMT Score')
plt.ylabel('Frequency')
plt.show()

# %%
print(len(df))
print(df.shape)

# %% [markdown]
# ## TMT vs. MIND (avg)

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_avg_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['trailmaking_score']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## TMT vs. MIND

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/MIND_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['trailmaking_score']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0']

datafield_name = ['sex', 'age', 'assessment_centre']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## TMT vs. CT

# %%
X = df[['31-0.0', '21003-2.0', '54-2.0',
        
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']]

y = df['trailmaking_score']

# %%
datafield_code = ['31-0.0', '21003-2.0', '54-2.0',
    
        '27174-2.0', '27267-2.0', '27175-2.0', '27268-2.0', '27176-2.0', '27269-2.0', '27177-2.0',
        '27270-2.0', '27178-2.0', '27271-2.0', '27179-2.0', '27272-2.0', '27180-2.0', '27273-2.0',
        '27204-2.0', '27297-2.0', '27181-2.0', '27274-2.0', '27182-2.0', '27275-2.0', '27183-2.0',
        '27276-2.0', '27184-2.0', '27277-2.0', '27185-2.0', '27278-2.0', '27186-2.0', '27279-2.0',
        '27188-2.0', '27281-2.0', '27187-2.0', '27280-2.0', '27189-2.0', '27282-2.0', '27190-2.0',
        '27283-2.0', '27191-2.0', '27284-2.0', '27192-2.0', '27285-2.0', '27193-2.0', '27286-2.0',
        '27194-2.0', '27287-2.0', '27195-2.0', '27288-2.0', '27196-2.0', '27289-2.0', '27197-2.0',
        '27290-2.0', '27198-2.0', '27291-2.0', '27199-2.0', '27292-2.0', '27200-2.0', '27293-2.0',
        '27201-2.0', '27294-2.0', '27202-2.0', '27295-2.0', '27203-2.0', '27296-2.0']

datafield_name = ['sex', 'age', 'assessment_centre',
    
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
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)

# %% [markdown]
# ## TMT vs. FC

# %%
# Set X and y
with open('/Users/baileyng/MIND_models/region_names/FC_regions.txt', 'r') as f:
    brain_regions = [line.strip() for line in f.readlines()]

# Define demographic/clinical features
demographic_vars = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

# Combine demographic features with brain region features
all_vars = demographic_vars + brain_regions

X = df[all_vars]
y = df['trailmaking_score']

print(X.shape)
print(y.shape)

# %%
# rename columns
datafield_code = ['31-0.0', '21003-2.0', '54-2.0', '25741-2.0']

datafield_name = ['sex', 'age', 'assessment_centre', 'head_motion']

if len(datafield_code) == len(datafield_name):
    rename_dict = dict(zip(datafield_code, datafield_name))
    X = X.rename(columns=rename_dict)
    print("Columns renamed successfully.")
else:
    print("Error: The number of datafield codes does not match the number of datafield names.")

categorical_vars = ['sex', 'assessment_centre']
continuous_vars  = [c for c in X.columns if c not in categorical_vars]

# %%
pls_analysis(X, y, continuous_vars, categorical_vars, n_splits=10)


