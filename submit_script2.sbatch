#!/bin/bash
#SBATCH --job-name=XGBoost_FIS_param_opt_CT
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_param_opt_CT_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_param_opt_CT_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 XGBoost_FIS_CT.py