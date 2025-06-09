#!/bin/bash

SOURCE_DIR="/external/rprshnas01/tigrlab/scratch/bng/cartbind/data/test_results/aparc_avg_unique"
TXT_FILE="/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/aparc_base_id.txt"

> "$TXT_FILE"

for file in "$SOURCE_DIR"/*; do
    filename=$(basename "$file")
    id="${filename:0:7}"
    echo "$id" >> "$TXT_FILE"
done

echo "IDs piped successfully!"