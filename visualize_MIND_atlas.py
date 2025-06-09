import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
from nilearn import datasets, plotting
from matplotlib import cm
from nilearn import surface

mean_df = pd.read_csv('/Users/baileyng/MIND_data/fmriprep_cartbind_cmh_freesurfer/average/aparc_average.csv', index_col=0)

# plt.figure(figsize=(12, 10))
# sns.heatmap(mean_df, cmap='viridis', annot=True)
# plt.title('Mean MINDs Matrix')
# plt.show()

# STEP 3: Load annotation files (aparc parcellation)
lh_annot = fsio.read_annot('/Applications/freesurfer/subjects/fsaverage/label/lh.aparc.annot')
rh_annot = fsio.read_annot('/Applications/freesurfer/subjects/fsaverage/label/rh.aparc.annot')
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
for var, value in mean_df['average'].items():
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