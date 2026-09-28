#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --time=96:00:00
#SBATCH --mem=600G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Author: Jorge Ruiz-Orera
# This script extract RNA-seq and Ribo-seq coverage from features, and also extracts transcript sequences adapted for training

source ~/.bashrc
mamba activate ribotranspred

# Configuration
REGION_LEN=4500
NBINS=1500
RNACUTOFF=5

python3 scripts/extract_cov_features.py $REGION_LEN $NBINS $RNACUTOFF
