#!/bin/bash
#SBATCH --job-name=XGBoost_PAL_reg_07-31
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_reg_logs/XGBoost_PAL_reg_07-31_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_reg_logs/XGBoost_PAL_reg_07-31_%J.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_xgboost/XGBoost_PAL_reg2.py