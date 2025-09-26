# MIND_models

Repository of code and resources for processing, modelling, and analysis of MIND and other brain MRI modalities, with a focus on predictive machine learning (ElasticNet regression, PLS regression, XGBoost) and deep learning (GNNs) modelling. Main analyses are performed on UK Biobank dataset.

## Main Folders

- **`MIND_processing`**: Handling and preprocessing MIND structural connectivity matrices
- **`QuantNets`**: Scripts and config files for building and experimenting various GNN architetures (GCNs, GATs, Quantized GCNs)
- **`matlab_plsregression`**: MATLAB scripts for performing Partial Least Squares (PLS) regression; includes permutation testing and bootstrapping (models trained on all predictors as well as models trained on neuroimaging data with demographic variables regressed out)
- **`miscellaneous`**: Miscellaneous scripts and resources
- **`models_elasticnet`**: Scripts and notebooks for Elastic Net regression modelling and analysis (models trained on all predictors as well as models trained on neuroimaging data with demographic variables regressed out)
- **`models_plsregression`**: Scripts and notebooks for PLS regression modelling and analysis
- **`models_xgboost`**: Scripts and notebooks for XGBoost modelling and analysis
- **`other_dataset_processing`**: Preprocessing scripts for other datasets (not UK Biobank)
- **`region_names`**: UK Biobank variable and brain region names
- **`sbatch_scripts`**: SLURM batch scripts for running computational jobs on clusters
- **`visualization`**: Visualization of data and modelling results

---
