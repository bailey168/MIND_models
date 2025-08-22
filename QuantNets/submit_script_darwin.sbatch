#!/bin/bash
#SBATCH --job-name=create_dataset_GF
#SBATCH --output=/scratch/bng/cartbind/data/logs/graph_logs/create_dataset_GF_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/graph_logs/create_dataset_GF_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/QuantNets/create_dataset.py