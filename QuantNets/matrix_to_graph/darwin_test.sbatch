#!/bin/bash
#SBATCH --job-name=convert_graph
#SBATCH --output=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.out
#SBATCH --error=/scratch/bng/cartbind/data/logs/graph_logs/convert_graph_%J.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH --partition=low-moby

start_time=$(date +%s)

source /scratch/bng/cartbind/data/darwin/bin/activate

INPUT_DIR=/scratch/bng/cartbind/data/FC_matrices/DSST
OUTPUT_DIR=/scratch/bng/cartbind/data/FC_graphs/DSST

mkdir -p "$OUTPUT_DIR"

for csv_file in "$INPUT_DIR"/*.csv; do
    basename=$(basename "$csv_file" .csv)
    prefix="${basename:0:17}"
    output_file="$OUTPUT_DIR/${prefix}_fc_graph.pt"

    python3 -u /scratch/bng/cartbind/code/MIND_models/QuantNets/matrix_to_graph/matrix_to_graph.py \
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

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "All files processed"
echo "Total runtime: $runtime seconds"    