#!/bin/bash

# Usage: ./check_annot_files.sh /path/to/data_dir annot_filename
# Example: ./check_annot_files.sh /data lh.HCP-MMP.annot

DATA_DIR="$1"
ANNOT_FILE="$2"

if [ -z "$DATA_DIR" ] || [ -z "$ANNOT_FILE" ]; then
    echo "❌ Usage: $0 /path/to/data_dir annot_filename"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ '$DATA_DIR' is not a directory"
    exit 1
fi

# Counters
found_count=0
missing_count=0

# Use a temporary file to collect missing paths
tmpfile=$(mktemp)

# Use find to locate all dirs ending in 20263_2_0 and loop through them
while IFS= read -r dir; do
    label_path="$dir/label/$ANNOT_FILE"
    if [ -f "$label_path" ]; then
        ((found_count++))
    else
        echo "Missing: $label_path" >> "$tmpfile"
        ((missing_count++))
    fi
done < <(find "$DATA_DIR" -type d -name "*20263_2_0")

# Output results
cat "$tmpfile"
rm "$tmpfile"

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

