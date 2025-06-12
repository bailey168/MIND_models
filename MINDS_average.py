# %%
import os
import glob
import pandas as pd
import numpy as np

base_dir = '/Users/danielagiansante/cartbind_avg_results'
sites = ['camh', 'uhn', 'ubc']
dfs = []

for site in sites:
    site_dir = os.path.join(base_dir, site)
    csv_files = glob.glob(os.path.join(site_dir, '*.csv'))
    for csv_file in csv_files:
        # Read with row names as index, skip header if present
        df = pd.read_csv(csv_file, index_col=0)
        # Force numeric, in case any values are strings
        df = df.apply(pd.to_numeric, errors='coerce')
        dfs.append(df)

# Concatenate along axis=1 (columns), then take mean across columns for each row
all_data = pd.concat(dfs, axis=1)
mean_series = all_data.mean(axis=1, skipna=True)

# Optionally, to keep as a DataFrame with same structure (region names as index):
mean_df = mean_series.to_frame(name='mean')

# Save with row names
mean_df.to_csv(os.path.join(base_dir, 'mean_matrix.csv'))

print('Mean matrix with row names saved as mean_matrix_with_names.csv in', base_dir)

# %%
print("Number of subject CSV files averaged:", len(dfs))

# %%
import pandas as pd

mean_df = pd.read_csv('/Users/danielagiansante/cartbind_avg_results/mean_matrix.csv', index_col=0)
mean_df  

# %%
import seaborn as sns
import matplotlib.pyplot as plt

mean_df = pd.read_csv('/Users/danielagiansante/cartbind_avg_results/mean_matrix.csv', index_col=0)
plt.figure(figsize=(12, 10))
sns.heatmap(mean_df, cmap='viridis', annot=True)
plt.title('Mean MINDs Matrix')
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
from nilearn import datasets, plotting
from matplotlib import cm
from nilearn import surface


# STEP 1: Load mean matrix CSV (regions as index, one 'mean' column)
mean_df = pd.read_csv('/Users/danielagiansante/cartbind_avg_results/mean_matrix.csv', index_col=0)

# STEP 2: Set fsaverage path
fsaverage_path = '/System/Volumes/Data/Users/danielagiansante/Desktop/Installations/freesurfer/subjects/fsaverage'

# STEP 3: Load annotation files (aparc parcellation)
lh_annot = fsio.read_annot(f'{fsaverage_path}/label/lh.aparc.annot')
rh_annot = fsio.read_annot(f'{fsaverage_path}/label/rh.aparc.annot')
lh_labels, lh_cmap, lh_names = lh_annot
rh_labels, rh_cmap, rh_names = rh_annot
lh_names = [name.decode('utf-8') for name in lh_names]
rh_names = [name.decode('utf-8') for name in rh_names]

# STEP 4: Prepare vertex data arrays
lh_vertex_data = np.full_like(lh_labels, np.nan, dtype=float)
rh_vertex_data = np.full_like(rh_labels, np.nan, dtype=float)

# Helper to match region names
def region_from_var(var):
    if var.startswith('lh_'):
        return 'lh', var.replace('lh_', '')
    elif var.startswith('rh_'):
        return 'rh', var.replace('rh_', '')
    return None, None

# Map mean values to surface vertices
for var, value in mean_df['mean'].items():
    hemi, region = region_from_var(var)
    if hemi == 'lh':
        idxs = [i for i, n in enumerate(lh_names) if region.lower() == n.lower()]
        for idx in idxs:
            lh_vertex_data[lh_labels == idx] = value
    elif hemi == 'rh':
        idxs = [i for i, n in enumerate(rh_names) if region.lower() == n.lower()]
        for idx in idxs:
            rh_vertex_data[rh_labels == idx] = value

# STEP 5: Load fsaverage surfaces and sulcal maps
fsavg = datasets.fetch_surf_fsaverage('fsaverage')
lh_mesh = surface.load_surf_mesh(fsavg.infl_left)
rh_mesh = surface.load_surf_mesh(fsavg.infl_right)

sulc_left = fsavg.sulc_left
sulc_right = fsavg.sulc_right

# STEP 6: Plotting function (slightly simplified)
def plot_three_views_custom(surf_mesh, vertex_data, bg_map, hemi, title):
    views = ['lateral', 'medial', 'dorsal']
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(24, 10))
    vmax = np.nanmax(vertex_data)
    vmin = np.nanmin(vertex_data)
    cmap = 'viridis'
    for i, view in enumerate(views):
        plotting.plot_surf_stat_map(
            surf_mesh, vertex_data, hemi=hemi, bg_map=bg_map,
            cmap=cmap, colorbar=False, vmax=vmax, vmin=vmin, 
            view=view, axes=axes[i], title='', figure=fig
        )
        axes[i].set_title(view.capitalize(), fontsize=18)
        axes[i].axis('off')
    # Colorbar
    cbar_ax = fig.add_axes([0.33, 0.09, 0.34, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label('Mean MINDs Value', fontsize=14)
    cb.ax.tick_params(labelsize=14)
    plt.suptitle(title, fontsize=24, y=0.96)
    plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    plt.show()

# STEP 7: Plot for both hemispheres
plot_three_views_custom(lh_mesh, lh_vertex_data, sulc_left, 'left', 'Mean MINDs - Left Hemisphere')
plot_three_views_custom(rh_mesh, rh_vertex_data, sulc_right, 'right', 'Mean MINDs - Right Hemisphere')

# %%
# PREDICTIVE MODELLING WITH THE AVERAGE MATRIX

# %%
#Step 1: getting the dataset ready 
    #merging all the matrices together 

import pandas as pd
import glob
import os

root_dir = '/Users/danielagiansante/cartbind_avg_results'
csv_files = glob.glob(os.path.join(root_dir, '*', '*_aparc_MIND_matrix.csv'))

data = []
subject_ids = []
sites = []

for file in csv_files:
    # Extract subject ID from filename (e.g. sub-098 from sub-098_aparc_MIND_matrix.csv)
    subj_id = os.path.basename(file).split('_')[0]
    # Extract site from the directory
    site = os.path.basename(os.path.dirname(file))
    df = pd.read_csv(file, sep=None, engine='python')
    df = df.set_index(df.columns[0])
    # Use 'average' column if present, otherwise first column
    if "average" in df.columns:
        row = df["average"].to_list()
    else:
        row = df.iloc[:,0].to_list()
    data.append(row)
    subject_ids.append(subj_id)
    sites.append(site)

# Use the index from the first file as column names
region_names = pd.read_csv(csv_files[0], sep=None, engine='python').iloc[:,0].to_list()

# Create combined DataFrame
combined_df = pd.DataFrame(data, columns=region_names)
combined_df.insert(0, 'site', sites)
combined_df.insert(0, 'subject', subject_ids)

# Save to CSV
combined_df.to_csv('combined_aparc_matrix_with_site.csv', index=False)

# %%
import pandas as pd

dataset = pd.read_csv('/Users/danielagiansante/CARTBIND/UTF-82025_04_Zhukovsky_CARTBIND_summary_unblinded.csv')

# 1. Extract subject number and site to match combined_df format
def extract_subj_site(row):
    subj_id = row.split('_')[-1]  # e.g., '002'
    site_raw = row.split('_')[1].lower()  # e.g., 'CAM'/'UHN'/'UBC'
    site_map = {'cam': 'camh', 'uhn': 'uhn', 'ubc': 'ubc'}
    site = site_map.get(site_raw, site_raw)
    return pd.Series([f"sub-{int(subj_id):03d}", site])

dataset[['subject', 'site']] = dataset['Subject_ID'].apply(extract_subj_site)

# 2. Create response column
dataset['response'] = dataset[['hrsd_resp_t30', 'hrsd_resp_t25', 'hrsd_resp_t20',
                              'hrsd_resp_t15', 'hrsd_resp_t10']].bfill(axis=1).iloc[:, 0]

# Merge 
merged_df = combined_df.merge(dataset[['subject', 'site', 'response']], on=['subject', 'site'], how='left')

# Save or check your merged dataframe
merged_df.to_csv('combined_aparc_matrix_with_response.csv', index=False)
print(merged_df[['subject', 'site', 'response']].head())


# %%
merged_df

# %%
print(f"Original number of subjects: {len(merged_df)}")

# Drop rows with any missing values 
clean_df = merged_df.dropna(subset=['response'])

print(f"Number of subjects after dropping missing response: {len(clean_df)}")

# Note to self: For the predictive modelling the dataset name is called 'clean_df'

# %%
#Setting up for logistic regression 

import pandas as pd
import numpy as np
import re

# Step 1: Define initial predictors
X = clean_df[['lh_bankssts', 'lh_caudalanteriorcingulate',
       'lh_caudalmiddlefrontal', 'lh_cuneus', 'lh_entorhinal', 'lh_fusiform',
       'lh_inferiorparietal', 'lh_inferiortemporal', 'lh_isthmuscingulate',
       'lh_lateraloccipital', 'lh_lateralorbitofrontal', 'lh_lingual',
       'lh_medialorbitofrontal', 'lh_middletemporal', 'lh_parahippocampal',
       'lh_paracentral', 'lh_parsopercularis', 'lh_parsorbitalis',
       'lh_parstriangularis', 'lh_pericalcarine', 'lh_postcentral',
       'lh_posteriorcingulate', 'lh_precentral', 'lh_precuneus',
       'lh_rostralanteriorcingulate', 'lh_rostralmiddlefrontal',
       'lh_superiorfrontal', 'lh_superiorparietal', 'lh_superiortemporal',
       'lh_supramarginal', 'lh_frontalpole', 'lh_temporalpole',
       'lh_transversetemporal', 'lh_insula', 'rh_bankssts',
       'rh_caudalanteriorcingulate', 'rh_caudalmiddlefrontal', 'rh_cuneus',
       'rh_entorhinal', 'rh_fusiform', 'rh_inferiorparietal',
       'rh_inferiortemporal', 'rh_isthmuscingulate', 'rh_lateraloccipital',
       'rh_lateralorbitofrontal', 'rh_lingual', 'rh_medialorbitofrontal',
       'rh_middletemporal', 'rh_parahippocampal', 'rh_paracentral',
       'rh_parsopercularis', 'rh_parsorbitalis', 'rh_parstriangularis',
       'rh_pericalcarine', 'rh_postcentral', 'rh_posteriorcingulate',
       'rh_precentral', 'rh_precuneus', 'rh_rostralanteriorcingulate',
       'rh_rostralmiddlefrontal', 'rh_superiorfrontal', 'rh_superiorparietal',
       'rh_superiortemporal', 'rh_supramarginal', 'rh_frontalpole',
       'rh_temporalpole', 'rh_transversetemporal', 'rh_insula']]

y = clean_df['response']


# %%
#Running logistic regression
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

all_y_true_average = []
all_y_prob_average = []

thresholds = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
results = {t: {'cm': np.zeros((2, 2), dtype=int), 'sens': [], 'spec': [], 'acc': []} for t in thresholds}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

surviving_vars_per_fold = []
coefs_list = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.6],
            Cs=[0.001, 0.01, 0.1, 1, 10, 100],
            max_iter=8000,
            class_weight='balanced',
            random_state=42,
            cv=10
        )
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    all_y_true_average.extend(y_test)
    all_y_prob_average.extend(y_prob)

    # Get coefficients from logisticregressioncv step
    coefs = model.named_steps['logisticregressioncv'].coef_.ravel()
    coefs_list.append(coefs)
    surviving = [name for name, coef in zip(X.columns, coefs) if coef != 0]
    surviving_vars_per_fold.append(surviving)

    #Confusion matrix and metrics calculation for each threshold
    for threshold in thresholds:
        y_pred_custom = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_custom)
        results[threshold]['cm'] += cm
        results[threshold]['acc'].append(accuracy_score(y_test, y_pred_custom))
        results[threshold]['sens'].append(recall_score(y_test, y_pred_custom))
        TN, FP = cm[0]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        results[threshold]['spec'].append(specificity)

# Print average metrics for each threshold
for threshold in thresholds:
    print(f"\n=== Threshold: {threshold} ===")
    print("Combined Confusion Matrix:\n", results[threshold]['cm'])
    print("Mean Accuracy:", np.mean(results[threshold]['acc']))
    print("Mean Sensitivity:", np.mean(results[threshold]['sens']))
    print("Mean Specificity:", np.mean(results[threshold]['spec']))

# Calculate AUC across all folds
auc_average = roc_auc_score(all_y_true_average, all_y_prob_average)
print("MINDS average AUC:", auc_average)

#Plotting AUC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr_average, tpr_average, _ = roc_curve(all_y_true_average, all_y_prob_average)
plt.figure(figsize=(6, 4))
plt.plot(fpr_average, tpr_average, label=f"AUC = {auc_average:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MINDS average AUC")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %%
best_c = model.named_steps['logisticregressioncv'].C_[0]
best_l1 = model.named_steps['logisticregressioncv'].l1_ratio_
print(f"Best C for this fold: {best_c}")
print(f"Best l1_ratio for this fold: {best_l1}")

# %%
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# 1. Print summary of variable survival across folds
flat_survivors = [item for sublist in surviving_vars_per_fold for item in sublist]
survivor_counts = Counter(flat_survivors)
print("\nNumber of folds each variable survived:")
for var, count in survivor_counts.items():
    print(f"{var}: {count} / {skf.get_n_splits()} folds")

print("\nVariables surviving in >50% of folds:")
for var, count in survivor_counts.items():
    if count > skf.get_n_splits() / 2:
        print(var)

print("\nVariables surviving in ALL folds:")
for var, count in survivor_counts.items():
    if count == skf.get_n_splits():
        print(var)

# 2. Aggregate  
coefs_array = np.array(coefs_list)  # shape: (n_folds, n_features)
mean_coefs = coefs_array.mean(axis=0)
std_coefs = coefs_array.std(axis=0, ddof=1)
n_folds = coefs_array.shape[0]
ci95 = 1.96 * std_coefs / np.sqrt(n_folds)  # 95% confidence interval

# 3. Identify surviving (nonzero) variables and plot
surviving_mask = np.any(coefs_array != 0, axis=0)
surviving_vars = np.array(X.columns)[surviving_mask]

print("\nSurviving variables (non-zero in at least one fold):")
print(surviving_vars.tolist())
print("Mean coefficients for surviving variables:")
print(mean_coefs[surviving_mask])

print("95% CI for each surviving variable:")
for name, mean, ci in zip(surviving_vars, mean_coefs[surviving_mask], ci95[surviving_mask]):
    print(f"{name}: {mean:.3f} ± {ci:.3f}")

# 4. Plot (only surviving variables for clarity)
if len(surviving_vars) == 0:
    print("No surviving variables to plot.")
else:
    mean_surv = mean_coefs[surviving_mask]
    ci95_surv = ci95[surviving_mask]
    sorted_idx = np.argsort(np.abs(mean_surv))[::-1]
    plt.figure(figsize=(10, 20))
    plt.barh(
        y=surviving_vars[sorted_idx],
        width=mean_surv[sorted_idx],
        xerr=ci95_surv[sorted_idx],
        capsize=6,
        ecolor="black",
        linewidth=2
    )
    plt.xlabel("Mean Coefficient Value Across Folds (±95% CI)")
    plt.title("MINDS average Surviving Variables")
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()
    plt.show()

# %%
# After calculating mean_coefs and ci95

# For surviving variables only
means = mean_coefs[surviving_mask]
cis = ci95[surviving_mask]
names = surviving_vars

# CI does not cross zero if lower bound and upper bound have the same sign (and are not zero)
lower = means - cis
upper = means + cis
no_cross_zero_mask = (lower > 0) | (upper < 0)

survivors_no_cross_zero_average = names[no_cross_zero_mask]

print("Number of variables whose 95% CI does NOT cross zero:", no_cross_zero_mask.sum())
print("\nSurviving variables whose 95% CI does NOT cross 0:")
for name, mean, lo, hi in zip(names[no_cross_zero_mask], means[no_cross_zero_mask], lower[no_cross_zero_mask], upper[no_cross_zero_mask]):
    print(f"{name}: mean={mean:.3f}, 95% CI= [{lo:.3f}, {hi:.3f}]")

# %%
#Re Running Logistic Regression with MINDS average surviving variables where CI dont cross 0
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Prepare data using only strict surviving variables 
X_average = X[survivors_no_cross_zero_average]

#  Re-run cross-validated logistic regression 
all_y_truestrict_average = []
all_y_probstrict_average = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X_average, y):
    X_train, X_test = X_average.iloc[train_idx], X_average.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.6],
            Cs=[0.001, 0.01, 0.1, 1, 10, 100],
            max_iter=8000,
            class_weight='balanced',
            random_state=42,
            cv=10
        )
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    all_y_truestrict_average.extend(y_test)
    all_y_probstrict_average.extend(y_prob)

aucstrict_average = roc_auc_score(all_y_truestrict_average, all_y_probstrict_average)
print(f"\nAUC for MINDS average strict surviving variables: {aucstrict_average:.3f}")

# Plot ROC Curve 
fprstrict_average, tprstrict_average, _ = roc_curve(all_y_truestrict_average, all_y_probstrict_average)
plt.figure(figsize=(7,7))
plt.plot(fprstrict_average, tprstrict_average, label=f'AUC = {aucstrict_average:.3f}')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'AUC for MINDS average Surviving Variables')
plt.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
from nilearn import datasets, plotting
from matplotlib import cm

# 1. Variable names and coefficients

means = mean_coefs[surviving_mask]
cis = ci95[surviving_mask]
names = surviving_vars

lower = means - cis
upper = means + cis
no_cross_zero_mask = (lower > 0) | (upper < 0)

survivors_no_cross_zero = names[no_cross_zero_mask]

# 2. Set FreeSurfer fsaverage path
fsaverage_path = '/System/Volumes/Data/Users/danielagiansante/Desktop/Installations/freesurfer/subjects/fsaverage'

# 3. Load annotation files
lh_annot = fsio.read_annot(f'{fsaverage_path}/label/lh.aparc.annot')
rh_annot = fsio.read_annot(f'{fsaverage_path}/label/rh.aparc.annot')
lh_labels, lh_cmap, lh_names = lh_annot
rh_labels, rh_cmap, rh_names = rh_annot
lh_names = [name.decode('utf-8') for name in lh_names]
rh_names = [name.decode('utf-8') for name in rh_names]

# 4. Prepare vertex data arrays
lh_vertex_data = np.full_like(lh_labels, np.nan, dtype=float)
rh_vertex_data = np.full_like(rh_labels, np.nan, dtype=float)

def varname_to_region(var):
    if var.startswith('lh_'):
        return 'lh', var.replace('lh_', '').replace('_thickness', '')
    elif var.startswith('rh_'):
        return 'rh', var.replace('rh_', '').replace('_thickness', '').replace('.', '')
    return None, None

for var, coef in zip(survivors_no_cross_zero, means):
    hemi, region = varname_to_region(var)
    if hemi == 'lh':
        idxs = [i for i, n in enumerate(lh_names) if region.lower() in n.lower()]
        for idx in idxs:
            lh_vertex_data[lh_labels == idx] = coef
    elif hemi == 'rh':
        idxs = [i for i, n in enumerate(rh_names) if region.lower() in n.lower()]
        for idx in idxs:
            rh_vertex_data[rh_labels == idx] = coef

# 5. Load inflated_pre surfaces
import nibabel.freesurfer.io as fsio
lh_inflated_pre_path = f'{fsaverage_path}/surf/lh.inflated_pre'
rh_inflated_pre_path = f'{fsaverage_path}/surf/rh.inflated_pre'
lh_coords, lh_faces = fsio.read_geometry(lh_inflated_pre_path)
rh_coords, rh_faces = fsio.read_geometry(rh_inflated_pre_path)
lh_mesh = (lh_coords, lh_faces)
rh_mesh = (rh_coords, rh_faces)

# 6. Load fsaverage sulcal maps for background
fsavg = datasets.fetch_surf_fsaverage('fsaverage')
sulc_left = fsavg.sulc_left
sulc_right = fsavg.sulc_right

# 7. Plotting function
def plot_three_views_custom(surf_mesh, vertex_data, bg_map, hemi, title):
    views = ['lateral', 'medial', 'dorsal']
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(40, 20))
    vmax = np.nanmax(np.abs(vertex_data))
    cmap = 'seismic'
    for i, view in enumerate(views):
        plotting.plot_surf_stat_map(
            surf_mesh, vertex_data, hemi=hemi, bg_map=bg_map,
            cmap=cmap, colorbar=False, vmax=vmax, vmin=-vmax, symmetric_cbar=True,
            view=view, axes=axes[i], title='', figure=fig, alpha=0.9
        )
        axes[i].set_title(view.capitalize(), fontsize=30)
        axes[i].axis('off')
    # Colorbar
    cbar_ax = fig.add_axes([0.33, 0.09, 0.34, 0.03])
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label('Mean Coefficient Values', fontsize=16)
    cb.ax.tick_params(labelsize=18)
    plt.suptitle(title, fontsize=38, y=0.96)
    plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    plt.show()

# 8. Plot with inflated_pre
plot_three_views_custom(lh_mesh, lh_vertex_data, sulc_left, 'left', 'Left Hemisphere (Demographic + Cortical Thickness)')
plot_three_views_custom(rh_mesh, rh_vertex_data, sulc_right, 'right', 'Right Hemisphere (Demographic + Cortical Thickness)')

# %%



