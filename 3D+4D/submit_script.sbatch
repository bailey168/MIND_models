#!/bin/bash
#SBATCH --job-name=MIND_3D+4D_check_size
#SBATCH --output=/scratch/p/pz249/baileyng/logs/MIND_avg_logs/MIND_3D+4D_check_size_%J.out
#SBATCH --error=/scratch/p/pz249/baileyng/logs/MIND_avg_logs/MIND_3D+4D_check_size_%J.err
#SBATCH --time=00:60:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0

module load python/3.8.5
source /home/p/pz249/baileyng/mind_env/bin/activate

python3 /home/p/pz249/baileyng/MIND_models/3D+4D/check_size_68.py
