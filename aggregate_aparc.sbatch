#!/bin/bash
#SBATCH --job-name=aggregate_aparc
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/aggregate_logs/aggregate_aparc_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/aggregate_logs/aggregate_aparc_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

python3 aggregate_aparc.py