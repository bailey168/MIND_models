#!/bin/bash

# Usage: ./check_annot_files_stat.sh /path/to/data_dir annot_filename
# Example: ./check_annot_files_stat.sh /data lh.HCP-MMP.annot

DATA_DIR="$1"
ANNOT_FILE="$2"

if [ -z "$DATA_DIR" ] || [ -z "$ANNOT_FILE" ]; then
    echo "❌ Usage: $0 /path/to/data_dir annot_filename"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ '$DATA_DIR' is not a valid directory"
    exit 1
fi

echo "🔍 Checking directories under $DATA_DIR for label/$ANNOT_FILE ..."

found_count=0
missing_count=0
counter=0

# Use globbing instead of find
for dir in "$DATA_DIR"/*20263_2_0; do
    # Skip non-directories
    [ -d "$dir" ] || continue

    ((counter++))
    if (( counter % 1000 == 0 )); then
        echo "Checked $counter directories so far..."
    fi

    label_path="$dir/label/$ANNOT_FILE"
    if [ -f "$label_path" ]; then
        ((found_count++))
    else
        ((missing_count++))
    fi
done

# Summary
echo "----------------------------------------"
echo "📝 File checked: label/$ANNOT_FILE"
echo "✅ Found in:     $found_count directories"
echo "❌ Missing in:   $missing_count directories"
echo "----------------------------------------"

if [ "$missing_count" -eq 0 ]; then
    echo "✅ All label directories contain $ANNOT_FILE"
    exit 0
else
    echo "❌ Some label directories are missing $ANNOT_FILE"
    exit 2
fi

