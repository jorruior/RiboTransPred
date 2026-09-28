# RiboTransPred

RiboTransPred trains a neural network to predict Ribo-seq coverage along spliced transcripts from nucleotide sequence and matched RNA-seq coverage. This guide covers reference preparation, training-data preparation, tissue- and condition-aware training with `PosTransModelTCNFiLMRef`, and prediction of Ribo-seq coverage from sequence and RNA-seq.

## Overview

1. Prepare matching genome and transcript annotations.
2. Normalize coverage and extract transcript features.
3. Train a model conditioned on tissue and experimental condition.
4. Predict transcript Ribo-seq coverage from sequence and RNA-seq.

## Installation

Use Linux with Bash and a CUDA-capable NVIDIA GPU for training. The supplied job scripts use SLURM. Run commands from the repository root.

```bash
mamba env create -f environment.yml
mamba activate ribotranspred
mkdir -p genomes coordinates tracks logs results_tissues
```

`environment.yml` includes PyTorch with CUDA 12.1, Lightning, the Python preprocessing and plotting dependencies, deepTools, samtools and gffread. Select a compatible PyTorch/CUDA build if your machine requires a different CUDA runtime. The environment name used by the supplied scripts is `ribotranspred`.

Before submitting jobs, adapt the SLURM partition, GPU type, memory, time limit and network interfaces to your cluster. The training launcher requests two nodes with eight GPUs per node. For one GPU, use the direct Python command below.

## 1. Prepare genomes and annotations

For every species, obtain a genome FASTA and matching Ensembl-style GTF from the same assembly and annotation release. The GTF must contain exon and CDS annotations, transcript identifiers, and `transcript_biotype` attributes for protein-coding selection. Chromosome names must agree with the aligned BAM files.

Use the species label from your sample manifest in the filenames:

```text
genomes/<species>.fa
coordinates/<species>.gtf
genomes/<species>.transcripts.fa
```

For example, after downloading and decompressing mouse references:

```bash
cp /path/to/mouse_genome.fa genomes/mouse.fa
cp /path/to/mouse_annotation.gtf coordinates/mouse.gtf
samtools faidx genomes/mouse.fa
gffread coordinates/mouse.gtf -g genomes/mouse.fa \
  -w genomes/mouse.transcripts.fa
```

Alternatively, provide the matching transcript FASTA from the annotation provider. Transcript FASTA identifiers must match GTF transcript identifiers. Repeat for each species. Reference files are local inputs and are excluded from Git.

## 2. Prepare coverage and transcript features

### Sample manifest

Copy the example and replace its paths and labels:

```bash
cp tracks.example.txt tracks.txt
```

Use five whitespace-separated fields, without a header; lines starting with `#` are comments:

```text
/path/to/mouse_heart_rna.bam mouse heart adult training
/path/to/mouse_heart_ribo.bam mouse heart adult training
/path/to/mouse_brain_rna.bam mouse brain adult test
/path/to/mouse_brain_ribo.bam mouse brain adult test
```

The fields are BAM path, species, tissue, condition and dataset (`training` or `test`). Provide coordinate-sorted, indexed RNA-seq and Ribo-seq BAMs for each sample. Basenames must identify the assay with `rna` or `ribo`. Each species/tissue/condition combination should identify one matched RNA/Ribo pair; pool replicates before this step if needed. Tissue and condition labels also select learned embeddings that condition the model through FiLM layers. Use consistent labels across samples.

### Generate coverage

```bash
sbatch 1_prepare_data.sh tracks.txt
```

The script produces per-base RNA, Ribo and P-site BigWigs under `tracks/<species>/<tissue>_<condition>/`. Coverage is RPKM-normalized. Ribo footprints are restricted to lengths 28-30, and P-site tracks use a fixed offset of 12. Review these settings for your library preparation.

Wait for this job to finish successfully before extracting features.

### Extract features

```bash
sbatch 2_extract_features.sh
```

The default configuration is 4,500 nt, 1,500 output bins and a raw RNA coverage cutoff of 5. To run extraction directly:

```bash
python scripts/extract_cov_features.py 4500 1500 5
```

Extraction discovers BigWigs under `tracks/` and requires matching references for each species found. It splices exons in transcript orientation, trims long transcripts at the 3-prime end, and pads short transcripts with Ns. It generates transcript sequences, CDS masks, RNA features, Ribo targets and RNA eligibility masks. Coverage is transformed with `log(coverage + 0.0001)`; 1,500 bins correspond to 3 nt each.

Transcripts with more than 90% of bases below the raw RNA cutoff are excluded through the eligibility mask. This filter is computed on the spliced transcript before truncation. Use the same transcript length and bin count for extraction and training. Keep each feature set together with the coordinate tables and references used to create it. Use a separate working directory when preparing an incompatible feature configuration; existing coordinate tables are reused.

Generated files include:

| Location or suffix | Content |
| --- | --- |
| `coordinates/<species>_coordinates_v2.txt` | Transcript sequence, annotation and CDS mask |
| `*_rnaseq_final_v2.pt` | RNA features, including log-transformed variants |
| `*_riboseq_final_v2.pt` | Ribo targets, including log-transformed variants |
| `*_eligible_v2.npy` | Transcript eligibility mask |

Training converts feature tensors to NumPy files for memory-mapped loading.

## 3. Train the tissue-conditioned model

`PosTransModelTCNFiLMRef` predicts coverage from five nucleotide channels (A, T, C, G and N), one RNA-coverage channel, and learned tissue and condition embeddings. Dual FiLM layers apply context-dependent scaling and shifting to intermediate features. Parallel causal convolutions with kernels 3, 6 and 25 feed dilated residual blocks. The backbone dilation schedule extends to cover the input window; the TCN uses position-local normalization and dropout. Predictions are pooled into bins, and a smaller refinement TCN produces a gated additive correction.

### Train on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tissues.py \
  --tracks tracks.txt --tracks_dir tracks --save_path results_tissues \
  --model-type PosTransModelTCNFiLMRef \
  --region_len 4500 --nBins 1500 --biotype protein_coding \
  --batch-size 2 --max-epochs 80 --dropout 0.33 \
  --learning_rate 0.0002 --weight_decay 0.0037 \
  --tissue_emb_dim 32 --cond_emb_dim 32 \
  --warmup_steps 2000 --grad_accum 3 --grad_clip 0.5 --seed 4
```

The command sets training parameters explicitly. Use `python scripts/train_tissues.py --help` for all options and their Python defaults. Keep the same arguments when resuming or testing a run.

### Train with SLURM

```bash
sbatch 3_train_model_tissue.sh tracks.txt 4500 1500 PosTransModelTCNFiLMRef 4
```

Positional arguments are manifest, transcript length, output-bin count, model and seed. This launcher requests two nodes with eight A40 GPUs per node. Its settings include batch size 2 per GPU, dropout 0.334, learning rate 0.0002, weight decay 0.0037 and 32-dimensional tissue/condition embeddings. Edit resource requests and environment/network settings for your cluster. The launcher trains the model and then runs testing with the selected checkpoint.

Training uses AdamW, a cosine learning-rate schedule, gradient accumulation, gradient clipping and bfloat16 mixed precision. Effective batch size is batch size per GPU multiplied by GPU count and gradient accumulation. Warmup is capped at `max(100, total_optimizer_steps // 10)`.

The objective combines weighted MSE with a per-transcript Pearson-correlation penalty. `--zero_w` defaults to 0.1 and reduces the MSE weight of zero-coverage target bins; `--pcc_loss_w` defaults to 0.2. Padding is masked. Reported epoch PCC pools valid bins across transcripts and GPUs. Epoch loss accumulates its components globally instead of averaging batch losses.

Validation runs every two epochs. Early stopping defaults to eight validation checks without improvement. `--monitor val/loss_epoch` selects the lowest validation loss; `--monitor val/pcc_epoch` selects the highest validation PCC. The default monitor is validation loss.

### Chromosome split and testing

By default, only manifest entries marked `training` contribute to fitting and validation. Chromosomes 16, 1 and X are reserved for testing when available; the code uses fallback chromosomes if needed. Validation chromosomes are selected using `--seed`, with species-specific fallbacks, and the remaining chromosomes form the training partition. Separate manifest entries marked `test` are evaluated only in test mode.

To test, repeat the training command with identical model, data and run settings and add `--test`. It loads `best.ckpt` from that run's output directory. Alternatively, add `--test --checkpoint /path/to/best.ckpt` to select an explicit checkpoint. The trainer constructs the model from command-line arguments, so architecture, embeddings and sequence-control flags must match the checkpoint.

### Homology-separated training and validation

Use `--homology /path/to/clusters.tsv` to split eligible homology clusters into 70% training and 30% validation, using `--seed`. The tab-separated input must contain a header and one transcript per row:

```text
cluster_id	transcript_id
cluster_000001	ENST00000511072
cluster_000001	ENSMUST00000087557
```

All members of a cluster stay in the same partition across species, tissues and conditions. Transcript IDs must exactly match the coordinate tables. Unmapped transcripts are excluded. Existing biotype, RNA-eligibility and usable-sequence filters still apply. The ratio is by eligible cluster count, rounded to whole clusters, not transcript count. At least two eligible clusters are required.

This mode creates no test partition, ignores test-tagged tracks, and cannot be combined with `--test` or the chromosome-debugging option `--trial`. Cluster assignments are saved in `homology_split.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tissues.py \
  --tracks tracks.txt --model-type PosTransModelTCNFiLMRef \
  --region_len 4500 --nBins 1500 --seed 4 \
  --homology /path/to/clusters.tsv
```

This command runs training and validation only. Provide your own mapping file; homology inputs are excluded from Git. Add the desired training hyperparameters as in the one-GPU example above.

### RNA-only control

Add `--nosequence` to train with RNA coverage and tissue/condition context, without nucleotide features. Use the same split and hyperparameters as your baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tissues.py \
  --tracks tracks.txt --model-type PosTransModelTCNFiLMRef \
  --region_len 4500 --nBins 1500 --seed 4 --nosequence
```

### Training outputs and resumption

Each run writes a configuration-specific directory under `results_tissues/` containing:

- `best.ckpt`, selected by the validation monitor, validation-ranked epoch checkpoints, and `last.ckpt`, the latest saved training state.
- `tissue_vocab.json` and `cond_vocab.json`, written when training finishes; vocabularies are also stored in checkpoints.
- CSV logs and `training_history.csv`, with training and validation loss/PCC by epoch.
- `training_validation_loss.pdf` and `training_validation_pcc.pdf`, vector plots written when training finishes, including early stopping.
- `homology_split.json` when using homology splitting.

Plots label epochs starting at 1; the history CSV stores zero-based epoch indices. Training points appear every epoch and validation points only on evaluated epochs. The plotted loss is the combined objective, not MSE alone. Sequence controls and homology mappings receive distinct output-directory suffixes.

To resume training, repeat its command and add `--checkpoint /path/to/last.ckpt`. Preserve the original inputs, arguments and data split. Retain logs from earlier training segments separately: the final history/plots describe epochs recorded in the current process.

Keep checkpoints, sample manifests, homology mappings and feature metadata for reproducibility. Generated data, results and local analysis folders are excluded from Git.

## 4. Predict Ribo-seq coverage

Use a trained checkpoint to predict normalized Ribo-seq coverage for a species, tissue and condition. Sequence and RNA-seq are the model inputs; measured Ribo-seq is optional and is used only for comparison. Predictions cover the first 4,500 nt in transcript orientation, with 1,500 bins of 3 nt each.

### Required inputs

- A tissue-conditioned checkpoint, preferably `best.ckpt` from the selected training run.
- `coordinates/<species>_coordinates_v2.txt`, containing transcript sequences and annotation.
- `coordinates/<species>.bed`, containing the exon mapping generated during feature extraction.
- RNA features at `tracks/<species>/<tissue>_<condition>/<species>_<tissue>_<condition>_rna_4500_log_rnaseq_final_v2.npy`.

RNA feature rows must match the complete coordinate table in the same order, including transcripts that are later filtered by biotype. Use the same normalization and feature extraction as for training. Tissue and condition labels should match the checkpoint vocabularies for predictions in a trained context.

For a new RNA-only sample, prepare the references as in step 1 and supply an RNA-seq row to step 2's coverage preparation, for example:

```text
/path/to/mouse_heart_rna.bam mouse heart adult test
```

Run coverage preparation and feature extraction as described above. The preparation tools can process RNA without a paired Ribo-seq track for prediction. Feature extraction writes `.pt` tensors; training converts them to `.npy`, but a new prediction-only sample needs that conversion explicitly. For the example sample:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import torch

path = Path("tracks/mouse/heart_adult/mouse_heart_adult_rna_4500_log_rnaseq_final_v2.pt")
output = path.with_suffix(".npy")
if not output.exists():
    values = torch.load(path, map_location="cpu", weights_only=True)
    np.save(output, values.numpy())
PY
```

Use species, tissue and condition names appropriate to your sample throughout. If measured Ribo-seq is available, its matching log-feature `.npy` file enables comparison statistics and an observed coverage track. If it is absent, prediction continues with a warning and observed-Ribo comparison statistics are unavailable.

### Run prediction with SLURM

Run from the repository root after preparing the inputs:

```bash
mkdir -p logs
sbatch 4_predict_tissue.sh \
  /path/to/best.ckpt PosTransModelTCNFiLMRef mouse heart adult 1500
```

The six arguments are checkpoint, model, species, tissue, condition and number of output bins. This launcher requests one A40 GPU and predicts protein-coding transcripts with attribution calculation disabled. Review its SLURM resources and environment activation before submission. Input length is 4,500 nt; use features and binning matching the trained checkpoint.

For direct execution:

```bash
python scripts/predict_tissues.py \
  --checkpoint /path/to/best.ckpt \
  --model_type PosTransModelTCNFiLMRef \
  --species mouse --tissue heart --condition adult \
  --tracks_dir tracks --coordinates coordinates \
  --region_len 4500 --nBins 1500 --biotype protein_coding \
  --output_dir predictions_tissues/mouse_heart_adult \
  --no_attribution
```

The prediction script reads model type, tissue/condition vocabularies and embedding dimensions from checkpoint metadata. Sequence length and output bin count must still be supplied consistently. Dropout is disabled during prediction. A checkpoint trained with `--nosequence` uses RNA and tissue/condition context without nucleotide features.

### Prediction outputs

The SLURM launcher writes to `predictions_tissues/<species>_<tissue>_<condition>_<model>_protein_coding_<bins>/`:

| File | Content |
| --- | --- |
| `predictions.bedgraph` | Predicted normalized Ribo-seq coverage projected onto genomic exon coordinates |
| `rnaseq.bedgraph` | Input RNA-seq coverage projected onto genomic coordinates |
| `riboseq.bedgraph` | Observed Ribo-seq coverage, when its features are available |
| `transcript_stats.tsv` | Transcript-level summaries and correlations with observed Ribo-seq when available |

Coverage values are inverse-transformed to the normalized non-log scale by default; they are not raw read counts. The direct Python option `--output_raw_log` writes predictions on the model's log scale instead.

The `.bedgraph` files are extended seven-column tables: chromosome, start, end, value, strand, transcript ID and bin index. Coordinates are zero-based, half-open. Multiple isoforms can contribute overlapping intervals, so these files require an explicit isoform-selection or aggregation step before conversion to a conventional genome-browser BigWig.

The script skips transcripts already recorded in `transcript_stats.tsv` when rerunning into the same output directory. Use a new output directory for a different checkpoint or changed inputs. Prediction outputs and input data remain excluded from Git; the launcher and required Python modules are included.
