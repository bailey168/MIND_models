#!/bin/bash
#SBATCH --job-name=MIND_processing
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_logs/MIND_processing_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_logs/MIND_processing_%J.err
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=600M
#SBATCH --cpus-per-task=1
#SBATCH --array=0-999

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

PATHS_FILE=/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/subject_paths/paths_part_00
OUTPUT_DIR_APARC=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc
OUTPUT_DIR_HCP=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/HCP-MMP

mapfile -s $SLURM_ARRAY_TASK_ID -t -n1 SUBJECT < "$PATHS_FILE"
read subj_dir tag <<< "${SUBJECT[0]}"

python3 run_MIND_script.py --subj_dir "$subj_dir" --output_dir "$OUTPUT_DIR_APARC" --parcellation "aparc"

if [[ "$tag" == "HCP-MMP" ]]; then
    python3 run_MIND_script.py --subj_dir "$subj_dir" --output_dir "$OUTPUT_DIR_HCP" --parcellation "HCP-MMP"
fi