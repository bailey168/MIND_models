#!/bin/bash
#SBATCH --job-clean_csv_07-04
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_07-04_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_07-04_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=4
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/clean_csv.py