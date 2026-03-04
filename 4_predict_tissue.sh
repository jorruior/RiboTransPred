#!/bin/bash
#SBATCH --job-name=predict_tissues
#SBATCH --output=logs/pred_tissue_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --export=ALL

# Author: Jorge Ruiz-Orera
# Predicts tissue-conditioned RiboTransPred (FiLM)

# ============ ENVIRONMENT ============
source ~/.bashrc
mamba activate ribotranspred


REGION_LEN=6000
MODELDIR=$1
MODELNAME=$2
SPECIES=$3
TISSUE=$4
NBINS=$5
BIOTYPE="protein_coding"

# Check if model checkpoint exists
if [ ! -f "$MODELDIR" ]; then
    echo "ERROR: Model checkpoint not found: $MODELDIR"
    exit 1
fi

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
        --biotype $BIOTYPE \
        --orfs additional/all_orfs.txt