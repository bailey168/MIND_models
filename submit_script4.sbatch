#!/bin/bash
#SBATCH --job-name=XGBoost_FIS_MIND_reg_06-30
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_MIND_reg_06-30_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_FIS_MIND_reg_06-30_%J.err
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=12
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models/XGBoost_FIS_MIND_reg.py