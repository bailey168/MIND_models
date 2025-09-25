#!/bin/bash

for dir in /external/rprshnas01/external_data/uk_biobank/imaging/brain/nifti/t1_surface/data/*20263_2_0/; do
    if [ -f "$dir/label/lh.HCP-MMP.annot" ]; then
        printf "%s\n" "$dir"
    fi
done > /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/paths_HCP-MMP.loc
