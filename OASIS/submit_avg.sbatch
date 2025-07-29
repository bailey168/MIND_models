#!/bin/bash
#SBATCH --job-name=MIND_avg_processing
#SBATCH --output=/scratch/p/pz249/baileyng/logs/MIND_avg_logs/MIND_avg_processing_%J.out
#SBATCH --error=/scratch/p/pz249/baileyng/logs/MIND_avg_logs/MIND_avg_processing_%J.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0

module load python/3.8.5
source /home/p/pz249/baileyng/mind_env/bin/activate

python3 /home/p/pz249/baileyng/MIND_models/OASIS/avg_matrix_68.py
