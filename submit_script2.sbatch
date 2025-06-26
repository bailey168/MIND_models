#!/bin/bash
#SBATCH --job-name=extract_tabular_with_site2
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/extract_tabular_with_site2_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/xgboost_logs/extract_tabular_with_site2_%J.err
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 extract_tabular2.py