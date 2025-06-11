#!/bin/bash
#SBATCH --job-name=clean_csv
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=0

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

python3 clean_csv.py