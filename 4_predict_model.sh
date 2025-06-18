#!/bin/bash
# Author: Jorge Ruiz-Orera
# This script predicts levels of translation of individual transcripts or full transcriptomes using the previously generated model. Two pre-built models are available: "PosTransModelRiboPos" (used in the manuscript) "PosTransModel" (lower computational demand)


### CONFIGURATION BLOCK ###
LENGTH="6000"
NBINS="1000"
NPROCESSES="1" #Transcripts can be parallelized
LINEAGE="primate"
TISSUE="heart" #Tissue to predict based on RNA-seq data
SPECIES="human" #Species to predict based on RNA-seq data
MODEL="PosTransModel" #Name of the built model
TRANSCRIPT="ENST00000268661" #specify one transcript or "all" to predict all transcripts 

### SCRIPT ###
python3 scripts/prediction.py \
	--checkpoint "results/model_${LINEAGE}_${MODEL}/models/last.ckpt" \
	--gene "$TRANSCRIPT" \
	--data "data/${LINEAGE}/${TISSUE}/${SPECIES}" \
	--assembly "$LINEAGE" \
	--tissue "$TISSUE" \
	--species "$SPECIES" \
	--region-len "$LENGTH" \
	--nBins "$NBINS" \
	--model-type "$MODEL" \
	--num_processes "$NPROCESSES"
