#!/bin/bash
#SBATCH --job-name=clean_csv_with_site
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_with_site_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_with_site_%J.err
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 clean_csv.py