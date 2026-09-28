#!/bin/bash
#SBATCH --job-name=predict_tissues
#SBATCH --output=logs/pred_tissue_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --export=ALL

# Author: Jorge Ruiz-Orera
# Predicts tissue + condition dual-FiLM RiboTransPred

# ============ ENVIRONMENT ============
source ~/.bashrc
mamba activate ribotranspred

# Usage: sbatch 4_predict_tissue.sh CHECKPOINT MODEL SPECIES TISSUE CONDITION NBINS
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 CHECKPOINT MODEL SPECIES TISSUE CONDITION NBINS" >&2
    exit 1
fi
REGION_LEN=4500
MODELDIR=$1
MODELNAME=$2
SPECIES=$3
TISSUE=$4
CONDITION=$5
NBINS=$6
MUTATE=${7:-}
BIOTYPE="protein_coding"

# Check if model checkpoint exists
if [ ! -f "$MODELDIR" ]; then
    echo "ERROR: Model checkpoint not found: $MODELDIR"
    exit 1
fi

mkdir -p predictions_tissues

OUTPUT_DIR="predictions_tissues/${SPECIES}_${TISSUE}_${CONDITION}_${MODELNAME}_${BIOTYPE}_${NBINS}"

EXTRA_ARGS=()
if [ "$NBINS" -eq "$REGION_LEN" ]; then
    EXTRA_ARGS+=(--psites)
fi
if [ -n "$MUTATE" ]; then
    EXTRA_ARGS+=(--mutate "$MUTATE" --orfs additional/all_orfs.txt)
else
    EXTRA_ARGS+=(--no_attribution)
fi

python scripts/predict_tissues.py \
        --checkpoint  "$MODELDIR" \
        --model_type  "$MODELNAME" \
        --species     "$SPECIES" \
        --tissue      "$TISSUE" \
        --condition   "$CONDITION" \
        --tracks_dir  tracks \
        --output_dir  "$OUTPUT_DIR" \
        --region_len  "$REGION_LEN" \
        --nBins       "$NBINS" \
        --biotype     "$BIOTYPE" \
        "${EXTRA_ARGS[@]}"
