# MIND_models

Repository of code and resources for processing, modelling, and analysis of MIND and other brain MRI modalities, with a focus on predictive machine learning (ElasticNet regression, PLS regression, XGBoost) and deep learning (GNNs) modelling. Main analyses are performed on UK Biobank dataset.

## Data Wrangling + MRI Processing
1. **`MIND_processing`**: Handling and preprocessing Morphometric INverse Divergence (MIND) structural connectivity matrices
  - Collects UK Biobank data from CAMH cluster
  - Runs MIND processing for each participant in dataset using DK or HCP atlas
  - Uses freesurfer cortical thickness, volume, surface area, curvature, and sulcal depth variables
2. **`other_dataset_processing`**: Preprocessing scripts for other datasets (not UK Biobank), unrelated to main project
  - OASIS, NEO, 3D datasets


## Machine Learning Models
All models are trained to predict performance on Fluid Intelligence, Paired Associate Learning, Digit Symbol Substitution Test, using Alphanumeric Trail Making Test using demographic data, MIND, functional connectivity, and cortical thickness. Models are optimized using nested 10-fold cross validation and grid search. Model weights and SHAP scores are analyzed for interpretability
1. **`matlab_plsregression`**: MATLAB scripts for performing Partial Least Squares (PLS) regression; includes permutation testing and bootstrapping (models trained on all predictors as well as models trained on neuroimaging data with demographic variables regressed out)
2. **`models_plsregression`**: Python scripts and notebooks for PLS regression modelling and analysis
3. **`models_xgboost`**: Scripts and notebooks for XGBoost modelling and analysis
4. **`models_elasticnet`**: Scripts and notebooks for Elastic Net regression modelling and analysis (models trained on all predictors as well as models trained on neuroimaging data with demographic variables regressed out)

## Graph Neural Networks
- **`QuantNets`**: Scripts and config files for building and experimenting various GNN architectures (GCNs, GATs, Quantized GCNs)
  - Preprocessing scripts for converting connectivity matrices into graphs (PyTorch Geometric), standardization, sparsification, and dataset splitting
  - Architecture configurations of different GNN types and different methods of injecting demographic data into models
  - Configurations of model hyperparameters and sizing
  - Experimental setup to train multiple model configurations and track evaluation results


## Utils and Micellaneous
1. **`miscellaneous`**: Miscellaneous scripts and resources
2. **`region_names`**: UK Biobank variable names and brain region names
3. **`visualization`**: Visualization of data and modelling results
4. **`sbatch_scripts`**: SLURM batch scripts for running computational jobs on clusters







---
