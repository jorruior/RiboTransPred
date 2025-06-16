# RiboDeepPred

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

### Available Models

- ``PosTransModelRiboPos`` (default): High accuracy, higher computational cost.
- ``PosTransModel``: Less resource-intensive with competitive performance.

---

## 4. Using the generated model to predict Ribo-seq values

---

## uORFs

The `uORFs/` directory includes the list of upstream open reading frames used in the original study.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
