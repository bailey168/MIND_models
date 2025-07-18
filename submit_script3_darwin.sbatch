#!/bin/bash
#SBATCH --job-name=elasticnet_DSST_07-18
#SBATCH --output=/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_DSST_07-18_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/elasticnet_logs/elasticnet_DSST_07-18_%J.err
#SBATCH --time=32:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=12
#SBATCH --array=0
#SBATCH --partition=low-moby

module load python/3.8.1
module load python/3.8.1

source /scratch/bng/cartbind/data/darwin/bin/activate

echo "Starting elastic net analysis script..."
python3 -u /scratch/bng/cartbind/code/MIND_models/models_elasticnet/elasticnet_DSST.py