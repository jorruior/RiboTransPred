#!/bin/bash
# Author: Jorge Ruiz-Orera
# This script trains the model for predicting Ribo-seq data. It is recommended to run in multiple gpus due to the high computing cost of PosTransModel.


### CONFIGURATION BLOCK ###
LENGTH="6000"
NBINS="1000"
LINEAGE="primate"
METADATA="heart"
MODEL="PosTransModel"

BATCHES="32"
WORKERS="8"
NGPU="6"
EPOCHS="50"
PATIENCE="6"
#Other parameters can be modified inside the train.py script

### SCRIPT ###
# Training and validation
python3 scripts/train.py \
    --nBins "$NBINS" \
    --region_len "$LENGTH" \
    --save_path "results/model_${LINEAGE}_${MODEL}" \
    --data "data" \
    --assembly "$LINEAGE" \
    --dataset "${METADATA}.txt" \
    --model-type "$MODEL" \
    --patience "$PATIENCE" \
    --max-epochs "$EPOCHS" \
    --save-top-n "$EPOCHS" \
    --num-gpu "$NGPU" \
    --batch-size "$BATCHES" \
    --num-workers "$WORKERS" \
    --similar

# Test in unseen chromosomes
python3 scripts/train.py \
    --nBins "$NBINS" \
    --region_len "$LENGTH" \
    --save_path "results/model_${LINEAGE}_${MODEL}" \
    --data "data" \
    --assembly "$LINEAGE" \
    --dataset "${METADATA}.txt" \
    --model-type "$MODEL" \
    --similar \
    --test

