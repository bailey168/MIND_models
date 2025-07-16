#!/bin/bash

SOURCE_DIR="/external/rprshnas01/external_data/uk_biobank/imaging/brain/correlation/rFMRI_par_corr_matrix_25"
TXT_FILE="/external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/NEO/FC_base_id.txt"

> "$TXT_FILE"

for file in "$SOURCE_DIR"/*; do
    filename=$(basename "$file")
    id="${filename:0:7}"
    echo "$id" >> "$TXT_FILE"
done

echo "IDs piped successfully!"