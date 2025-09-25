#!/bin/bash
#SBATCH --job-name=XGBoost_PAL_reg_07-29
#SBATCH --output=/scratch/bng/cartbind/data/logs/xgboost_reg_logs/XGBoost_PAL_reg_07-29_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/xgboost_reg_logs/XGBoost_PAL_reg_07-29_%J.err
#SBATCH --time=32:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH --partition=low-moby

module load python/3.8.1
module load python/3.8.1

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/models_xgboost/XGBoost_PAL_reg.py