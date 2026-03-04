#!/bin/bash
# Author: Jorge Ruiz-Orera
# This script extract RNA-seq and Ribo-seq coverage from features, and also extracts transcript sequences adapted for training

set -e  # Exit on error
set -o pipefail  # Catch pipe errors

conda activate translatomer

# Configuration
REGION_LEN=6000
RNACUTOFF=5

# Loop over NBINS values
for NBINS in 1000 3000 6000 500; do
    echo "Extracting features with NBINS=${NBINS}"
    python3 scripts/extract_cov_features.py $REGION_LEN $NBINS $RNACUTOFF
done
