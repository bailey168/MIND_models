#!/bin/bash
#SBATCH --job-name=XGBoost_FIS_param_opt
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_param_opt_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_param_opt_%J.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 XGBoost_FIS.py