#!/bin/bash
#SBATCH --job-name=aggregate_aparc_full
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/matrix_logs/aggregate_aparc_full_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/matrix_logs/aggregate_aparc_full_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/aggregate/aggregate_aparc_full.py