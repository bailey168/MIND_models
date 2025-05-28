#!/bin/bash
#SBATCH --job-name=MIND_processing
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/MIND_processing_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/MIND_processing_%J.err
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=600M
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5

module load Python/3.8
source /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/venv/bin/activate

DATA_DIR=/external/rprshnas01/external_data/uk_biobank/imaging/brain/nifti/t1_surface/data/
OUTPUT_DIR=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/MIND_results/
PARCELLATION=aparc

mapfile -s $SLURM_ARRAY_TASK_ID \
	-t -n1 SUBJECT < <(
    printf '%s\n' ${DATA_DIR}/*20263_2_0
)

python3 run_MIND_script.py --subj_dir "${SUBJECT[0]}" --output_dir "$OUTPUT_DIR" --parcellation "$PARCELLATION"