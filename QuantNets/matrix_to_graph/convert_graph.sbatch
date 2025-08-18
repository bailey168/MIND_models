#!/bin/bash
#SBATCH --job-name=convert_graph
#SBATCH --output=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.out
#SBATCH --error=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=36
#SBATCH --array=0

module load PYTHON/3.6
module load PYTHON/3.6

source /external/rprshnas01/tigrlab/scratch/bng/cartbind/data/darwin/bin/activate

INPUT_DIR=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/FC_matrices/DSST
OUTPUT_DIR=/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/FC_graphs/DSST

mkdir -p "$OUTPUT_DIR"

for csv_file in "$INPUT_DIR"/*.csv; do
    basename=$(basename "$csv_file" .csv)
    prefix="${basename:0:17}"
    output_file="$OUTPUT_DIR/${prefix}_fc_graph.pt"

    python3 -u /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/QuantNets/matrix_to_graph/matrix_to_graph.py \
        -i "$csv_file" \
        -o "$output_file" \
        -p 0.5 \
        --self-loops

    if [[ $? -eq 0 ]]; then
        echo "Successfully converted: $basename"
    else
        echo "Failed to convert: $basename"
    fi
done

echo "All files processed"
    
