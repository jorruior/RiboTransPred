#!/bin/bash
# Author: Jorge Ruiz-Orera
# Trains tissue-conditioned RiboTransPred (FiLM)

conda activate ribotranspred

# ============ TRAINING CONFIGURATION ============
TRACKS=$1
REGION_LEN=6000
NBINS=3000
BIOTYPE="protein_coding"
MODEL="PosTransModelTCNFiLMRef"
SEED=4

BATCH_SIZE=4
MAX_EPOCHS=80
DROPOUT=0.3
LR=1e-5
WEIGHT_DECAY=0.0005
WARMUP=2000
GRAD_ACCUM=3
GRAD_CLIP=0.5
TISSUE_EMB_DIM=64

echo "Training Configuration (Tissue FiLM):"
echo "REGION_LEN=$REGION_LEN"
echo "NBINS=$NBINS"
echo "BIOTYPE=$BIOTYPE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "DROPOUT=$DROPOUT"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "WARMUP=$WARMUP"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "GRAD_CLIP=$GRAD_CLIP"
echo "TISSUE_EMB_DIM=$TISSUE_EMB_DIM"

mkdir -p results_tissues logs

python scripts/train_tissues.py \
        --region_len    $REGION_LEN \
        --nBins         $NBINS \
        --tracks        $TRACKS \
        --tracks_dir    tracks \
        --model         $MODEL \
        --save_path     results_tissues \
        --biotype       $BIOTYPE \
        --batch-size    $BATCH_SIZE \
        --max-epochs    $MAX_EPOCHS \
        --dropout       $DROPOUT \
        --learning_rate $LR \
        --weight_decay  $WEIGHT_DECAY \
        --warmup_steps  $WARMUP \
        --grad_accum    $GRAD_ACCUM \
        --grad_clip     $GRAD_CLIP \
        --tissue_emb_dim $TISSUE_EMB_DIM \
        --seed $SEED

echo "Running test..."
python -u scripts/train_tissues.py \
            --region_len    $REGION_LEN \
            --nBins         $NBINS \
            --tracks        $TRACKS \
            --tracks_dir    tracks \
            --model         $MODEL \
            --save_path     results_tissues \
            --biotype       $BIOTYPE \
            --batch-size    $BATCH_SIZE \
            --max-epochs    $MAX_EPOCHS \
            --dropout       $DROPOUT \
            --learning_rate $LR \
            --weight_decay  $WEIGHT_DECAY \
            --warmup_steps  $WARMUP \
            --grad_accum    $GRAD_ACCUM \
            --grad_clip     $GRAD_CLIP \
            --tissue_emb_dim $TISSUE_EMB_DIM \
            --checkpoint    "$BEST_CKPT" \
            --test \
            --seed $SEED

echo "Job finished at: $(date)"
