#!/bin/bash
#SBATCH --job-name=training_ids_08_29
#SBATCH --output=/scratch/bng/cartbind/data/logs/graph_logs/training_ids_08_29_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/graph_logs/training_ids_08_29_%J.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/QuantNets/training_ids.py