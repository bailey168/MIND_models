#!/bin/bash
#SBATCH --job-name=elasticnet_GF_07-15
#SBATCH --output=/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-15_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-15_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=3
#SBATCH --array=0
#SBATCH --partition=low-moby

module load python/3.8.1
module load python/3.8.1

source /scratch/bng/cartbind/data/darwin/bin/activate

python3 /scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_GF.py