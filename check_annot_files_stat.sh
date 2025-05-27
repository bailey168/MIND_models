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

# Output file
RESULTS_FILE="results.txt"
> "$RESULTS_FILE"  # clear if exists

echo "🔍 Checking directories under $DATA_DIR for label/$ANNOT_FILE using stat ..."

found_count=0
missing_count=0

counter=0
while IFS= read -r dir; do
    ((counter++))
    if (( counter % 100 == 0 )); then
        echo "Checked $counter directories so far..."
    fi

    label_path="$dir/label/$ANNOT_FILE"
    if stat "$label_path" >/dev/null 2>&1; then
        echo "FOUND $dir" >> "$RESULTS_FILE"
        ((found_count++))
    else
        echo "MISSING $label_path" >> "$RESULTS_FILE"
        ((missing_count++))
    fi
done < <(find "$DATA_DIR" -type d -name "*20263_2_0" | head -n 1000)

# Print missing results
grep "^MISSING" "$RESULTS_FILE"

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

