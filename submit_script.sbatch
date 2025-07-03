#!/bin/bash
#SBATCH --job-name=clean_csv_07-02
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_07-02_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/extract_logs/clean_csv_07-02_%J.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=4
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/clean_csv.py