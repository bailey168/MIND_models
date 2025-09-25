#!/bin/bash

printf "%s\n" /external/rprshnas01/external_data/uk_biobank/imaging/brain/nifti/t1_surface/data/*20263_[23]_0/ | while read -r dir; do
    [ -d "$dir" ] && printf "%s\n" "$dir"
done > /external/rprshnas01/tigrlab/scratch/bng/cartbind/code/MIND_models/paths_aparc.loc
