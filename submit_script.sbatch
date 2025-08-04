#!/bin/bash
#SBATCH --job-name=test_best_models_stats_scc_08-04
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_reg_logs/test_best_models_stats_scc_08-04_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_reg_logs/test_best_models_stats_scc_08-04_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/test_best_models_stats_scc.py