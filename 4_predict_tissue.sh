#!/bin/bash

# Author: Jorge Ruiz-Orera
# Predicts tissue-conditioned RiboTransPred (FiLM)

conda activate ribotranspred

REGION_LEN=6000
MODELDIR=$1
SPECIES=$2
TISSUE=$3
MODELNAME="PosTransModelTCNFiLMRef"
NBINS=3000
BIOTYPE="protein_coding"

# Check if model checkpoint exists
if [ ! -f "$MODELDIR" ]; then
    echo "ERROR: Model checkpoint not found: $MODELDIR"
    exit 1
fi

mkdir -p predictions_tissues

OUTPUT_DIR="predictions_tissues/${SPECIES}_${TISSUE}_${MODELNAME}_${BIOTYPE}_${NBINS}"

python scripts/predict_tissues.py \
        --checkpoint $MODELDIR \
        --model_type $MODELNAME \
        --species $SPECIES \
        --tissue $TISSUE \
        --tracks_dir tracks \
        --output_dir $OUTPUT_DIR \
        --region_len $REGION_LEN \
        --nBins $NBINS \
        --biotype $BIOTYPE
