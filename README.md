# Cognitive performance prediction using ML and DL models trained on Brain MRI data

Repository of code and resources for processing, modelling, and analysis of MIND and other brain MRI modalities, with a focus on predictive machine learning (ElasticNet regression, PLS regression, XGBoost) and deep learning (GNNs) modelling. Main analyses are performed on UK Biobank dataset.
<img width="8181" height="4583" alt="MScAC Aria Poster Figure Final" src="https://github.com/user-attachments/assets/7a0ed337-8f44-4ba3-9fb2-0b2cc0ccc673" />



## Data Wrangling + MRI Processing
1. **`MIND_processing`**: Handling and preprocessing Morphometric INverse Divergence (MIND) structural connectivity matrices
  - Collects UK Biobank data from CAMH cluster
  - Runs MIND processing for each participant in dataset using DK or HCP atlas
  - Uses freesurfer cortical thickness, volume, surface area, curvature, and sulcal depth variables
2. **`other_dataset_processing`**: Preprocessing scripts for other datasets (not UK Biobank), unrelated to main project
  - OASIS, NEO, 3D datasets


## Machine Learning Models
All models are trained to predict performance on Fluid Intelligence, Paired Associate Learning, Digit Symbol Substitution Test, and Alphanumeric Trail Making Test using demographic data, MIND, functional connectivity, and cortical thickness. Models are optimized using nested 10-fold cross-validation and grid search. To assess the role of demographic variables, models are trained both on the full predictor set and on neuroimaging predictors with demographic effects regressed out. Model weights and SHAP scores are analyzed for interpretability
1. **`matlab_plsregression`**: MATLAB scripts for **Partial Least Squares (PLS)** regression, including permutation testing and bootstrapping
2. **`models_plsregression`**: Python scripts and notebooks for **PLS regression** modelling and analysis
3. **`models_xgboost`**: Scripts and notebooks for **XGBoost** modelling and analysis
4. **`models_elasticnet`**: Scripts and notebooks for **Elastic Net regression** modelling and analysis

## Graph Neural Networks
- **`QuantNets`**: Scripts and config files for building and experimenting with various GNN architectures **(Graph Convolution Network (GCN), Graph Attention Networks (GAT), Quantized Graph Convolutional Network (QGRN))**
  - Preprocessing scripts for converting connectivity matrices into graphs (PyTorch Geometric), standardization, sparsification, and dataset splitting
  - Architecture configurations of different GNN types and different methods of injecting demographic data into models
  - Configurations of model hyperparameters and sizing (# of layers, dropout, weight decay, learning rate, learning rate schedulers, activation functions, pooling layers, normalization layers, embedding dimensions, hidden dimensions)
  - Experimental setup to train multiple model configurations and track evaluation results


## Utils and Miscellaneous
1. **`miscellaneous`**: Miscellaneous scripts and resources
2. **`region_names`**: UK Biobank variable names and brain region names
3. **`visualization`**: Visualization of data and modelling results
4. **`sbatch_scripts`**: SLURM batch scripts for running computational jobs on clusters

## Next Steps
1. Polish and implement Quantized Graph Convolutional architectures
2. Preprocess connectivity matrices with Fast Fourier Transform + inverse covariance matrix to sparsify
3. Pipeline to evaluate GNNs on different cognitive measures






---
