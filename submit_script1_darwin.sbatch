#!/bin/bash
#SBATCH --job-name=XGBoost_GF_CT_reg_07-20
#SBATCH --output=/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_GF_CT_reg_07-20_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/xgboost_logs/XGBoost_GF_CT_reg_07-20_%J.err
#SBATCH --time=32:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

module load python/3.8.1
module load python/3.8.1

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/models/XGBoost_GF_CT_reg.py