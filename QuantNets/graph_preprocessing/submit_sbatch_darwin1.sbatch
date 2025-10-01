#!/bin/bash
#SBATCH --job-name=convert_graph_PAL15_09_30
#SBATCH --output=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_PAL15_09_30_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_PAL15_09_30_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 -u /scratch/bng/cartbind/code/MIND_models/QuantNets/graph_preprocessing/pipeline.py --dataset PAL --keep-percent 0.15