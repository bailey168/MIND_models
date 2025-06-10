#!/bin/bash
#SBATCH --job-name=extract_tabular2
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/extract_tabular2_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/extract_tabular2_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

python3 extract_tabular.py