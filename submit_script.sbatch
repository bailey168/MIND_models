#!/bin/bash
#SBATCH --job-name=extract_tabular_07-02
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/extract_tabular_07-02_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/extract_tabular_07-02_%J.err
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=2
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/extract_tabular.py