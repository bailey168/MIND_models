#!/bin/bash

LOC_FILE="$1"
CSV_DIR="$2"
OUTPUT_FILE="$3"

while read -r path; do
    path=${path%/}
    id=$(basename "$path")

    csv_file="${CSV_DIR}/${id}_HCP-MMP_MIND_matrix.csv"
    if [[ ! -f "$csv_file" ]]; then
        echo "$path" >> "$OUTPUT_FILE"
    fi

done < "$LOC_FILE"

echo "Done! Unprocessed paths written to '$OUTPUT_FILE'."