#!/bin/bash
#SBATCH --job-name=elasticnet_GF_07-15
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-15_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_GF_07-15_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=2
#SBATCH --array=0

module load Python/3.8
module load Python/3.8

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/venv/bin/activate

python3 /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_GF.py