# RiboDeepPred (v0.1)

[![GitHub license](https://img.shields.io/github/license/jorruior/RiboDeepPred)](https://github.com/jorruior/RiboDeepPred/LICENSE.md)

**RiboDeepPred** is a deep neural network designed to predict Ribo-seq signal profiles using DNA sequence and RNA-seq coverage as input. The repository also includes a collection of uORFs used in the original publication.

RiboDeepPred integrates code adapted from [Translatomer](https://github.com/xiongxslab/translatomer) by xiongxslab, and is released under the MIT License.

---

## Overview

This repository provides scripts to:

1. Parse genome and transcriptome data for given species.
2. Normalize and convert RNA-seq and Ribo-seq data from BAM format.
3. Train a deep learning model on the processed data.
4. Predict Ribo-seq data using sequence and RNA-seq as input.

---

## Citation

If you use this repository or any part of the codebase in your work, please cite:

> X> *(Replace this with the appropriate citation once the publication is available.)*

---

## Prerequisites

To get started, clone this repository and create the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ribodeeppred
```

FlashAttention-2 is not available via conda and must be installed separately.
You can install it via pip (ensure you have a supported CUDA version, e.g., 11.8 or 12.1):

```bash
pip install flash-attn --no-build-isolation
```

---

## 1. Parsing Genome and Transcriptome Data

Before training the model, genomic and transcriptomic data must be parsed and preprocessed. To do so, run:

```bash
bash 1_prepare_genomes.sh
```

📌 **Note:** Edit the `### CONFIGURATION BLOCK ###` and `### INPUT FILES ###` sections of the script to specify the species and input files relevant to your use case.

---

## 2. Preparing RNA-seq and Ribo-seq Data

To convert RNA-seq and Ribo-seq BAM files into a suitable format for model training, run:

```bash
bash 2_prepare_data.sh
```

📌 **Note:** Modify the configuration and input sections of the script accordingly. By default, the script uses test data stored in the `samples/` directory.

---

## 3. Training the Model

Once the data is prepared, you can train the model using:

```bash
bash 3_train_model.sh
```

📌 **Notes:**

- Edit the `### CONFIGURATION BLOCK ###` to tailor the training to your dataset.
- The model will train using all samples defined in the metadata generated in step 2. You may modify this metadata to train on a specific subset of species or samples.
- Training is computationally intensive. We recommend using multiple GPUs if available.

---

## 4. Prediction of Ribo-seq values

Once the model has finished training, you can use RNA-seq and sequence data to predict Ribo-seq normalized log-values for a specific transcript or for all transcripts in the transcriptome. To run predictions, use:

```bash
bash 4_predict_model.sh
```

📌 **Notes:**

- Be sure to edit the ### CONFIGURATION BLOCK ### in the script to match your dataset and target transcript(s).
- For whole-transcriptome predictions, we recommend parallelizing the job using multiple processes to speed up computation.

### Available Models

- ``PosTransModelRiboPos`` (default): High accuracy, higher computational cost.
- ``PosTransModel``: Less resource-intensive with competitive performance.

**Outputs:**

- ``transcripts.out`` : Tabulated file with information about Ribo-seq prediction and correlation with observed values.
- ``predictedribo.bedgraph``: BEDGraph file with the predicted normalized log-values of Ribo-seq for each bin.
- ``attribution_scores.bedgraph``: BEDGraph file with the atribution scores of each nucleotide position in each bin.
- ``inputribo.bedgraph``: BEDGraph file with the initial observed normalized log-values of Ribo-seq for each bin. Do not use if Ribo-seq data was not available for the RNA-seq sample used for prediction.
- ``inputrna.bedgraph``: BEDGraph file with the predicted normalized log-values of Ribo-seq for each bin.
    
---

## uORFs and CDSs used in the original study

The `uorfs/` directory includes the list of upstream open reading frames and protein-coding sequences used in the original study.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
