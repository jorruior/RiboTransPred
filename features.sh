#!/bin/bash
#Author: Jorge Ruiz-Orera
#This script prepares the main RNA-seq and Ribo-seq data for the model.

# Activate the environment
source ~/.bashrc
mamba activate translatomer

METADATA=$1
REGION_LEN=$2
NBINS=$3
ASSEMBLY=$4


echo "Parameters:"
echo "METADATA: $METADATA"
echo "REGION_LEN: $REGION_LEN"
echo "NBINS: $NBINS"
echo "ASSEMBLY: $ASSEMBLY"

# Check if metadata file exists and is readable
if [ ! -f "$METADATA" ]; then
    echo "Error: Metadata file '$METADATA' not found!"
    exit 1
elif [ ! -r "$METADATA" ]; then
    echo "Error: Metadata file '$METADATA' is not readable!"
    exit 1
fi

# Process each line
while IFS=$'\t' read -r celltype study || [ -n "$celltype" ]; do
    # Skip empty lines or lines starting with #
    [[ -z "$celltype" || "$celltype" =~ ^# ]] && continue

    # Check for existence of exon_regions_output.txt, otherwise create it
    if [ ! -f "data/$ASSEMBLY/exon_regions_output.txt" ]; then
        make_region="yes"
    else
        make_region="no"
    fi

    echo "Processing: celltype=$celltype, study=$study, make_region=$make_region"

    # First command
    python3 scripts/features.py --assembly "$ASSEMBLY" \
        --celltype "$celltype" \
        --study "$study" \
        --region_len "$REGION_LEN" \
        --nBins "$NBINS" \
        --make_region "$make_region"

    # Check if previous command succeeded
    if [ $? -ne 0 ]; then
        echo "Error in first command for celltype=$celltype, study=$study"
        continue
    fi

done < "$METADATA"