#!/usr/bin/env python3
"""
RiboTransPred — Tissue + Condition dual-FiLM Ribo-seq prediction
=================================================================
Predicts Ribo-seq profiles from RNA-seq + DNA sequence, conditioned on
both tissue identity and experimental condition via dual FiLM layers.

Author: Jorge Ruiz-Orera
"""

import argparse
import csv
import hashlib
import random
import datetime
import gc
import json
import logging
import shutil
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

_RANK       = int(os.environ.get("SLURM_PROCID",  os.environ.get("RANK", 0)))
_LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
_WORLD_SIZE = int(os.environ.get("SLURM_NTASKS",  os.environ.get("WORLD_SIZE", 1)))
_NNODES     = int(os.environ.get("SLURM_NNODES",  1))


def _dbg(msg):
    sys.stderr.write(f"[rank {_RANK}] {msg}\n")
    sys.stderr.flush()


if _RANK != 0:
    import builtins
    builtins._original_print = builtins.print
    builtins.print = lambda *a, **k: None
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")

_NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[_v] = str(max(1, _NCPU))

import numpy as np
import pandas as pd
from sequence_controls import training_sequences
from training_curves import record_epoch, save_training_curves
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from torchmetrics import Metric
from training_metrics_v2 import (ProfileMetrics, masked_profile_loss, load_rna_eligibility,
                                 EvaluationDataset, DistributedEvaluationSampler)

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
import pytorch_lightning.callbacks as plc
from transformers import get_cosine_schedule_with_warmup

import model.models2 as models

# ═══════════════════════════════════════════════════════════════════════════
# § 2  File conversion
# ═══════════════════════════════════════════════════════════════════════════

def _pt_to_npy(pt_path):
    npy_path = pt_path[:-3] + ".npy"
    if os.path.exists(npy_path):
        return npy_path
    if not os.path.exists(pt_path):
        return None

    lock_path = npy_path + ".lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        deadline = time.time() + 300
        while time.time() < deadline:
            if os.path.exists(npy_path):
                return npy_path
            time.sleep(2)
        _dbg(f"WARNING: timed out waiting for {npy_path}")
        return None

    try:
        if os.path.exists(npy_path):
            return npy_path
        t = torch.load(pt_path, map_location="cpu", weights_only=True)
        tmp = f"{npy_path[:-4]}.tmp.{os.getpid()}.npy"
        np.save(tmp, t.numpy())
        del t; gc.collect()
        os.replace(tmp, npy_path)
        print(f"  converted {os.path.basename(pt_path)}")
        return npy_path
    except Exception as e:
        _dbg(f"WARNING: cannot convert {pt_path}: {e}")
        tmp = f"{npy_path[:-4]}.tmp.{os.getpid()}.npy"
        try: os.remove(tmp)
        except OSError: pass
        return None
    finally:
        try: os.remove(lock_path)
        except OSError: pass


def convert_all_tracks(training_tracks, args):
    """
    Convert .pt track files to .npy on each node independently.
    Uses a per-node filesystem sentinel — no dist primitives needed.
    """
    job_id  = os.environ.get("SLURM_JOB_ID", "0")
    node_id = os.environ.get("SLURM_NODEID", "0")
    done_file = f"_conv_done_{job_id}_node{node_id}"

    if _LOCAL_RANK == 0:
        print(f"\n  Converting .pt -> .npy on node {node_id} ...")
        for t in training_tracks:
            sp, tc = t["species"], t["tissue_condition"]
            bn = f"{sp}_{tc}"
            td = args.tracks_dir
            tag = "ribo.psites" if args.psites else "ribo"
            inp_pt = f"{td}/{sp}/{tc}/{bn}_rna_{args.region_len}_log_rnaseq_final_v2.pt"
            out_pt = (f"{td}/{sp}/{tc}/{bn}_{tag}_"
                      f"{args.region_len}_{args.nBins}_log_riboseq_final_v2.pt")
            if not os.path.exists(inp_pt[:-3] + ".npy") and os.path.exists(inp_pt):
                _pt_to_npy(inp_pt)
            if not os.path.exists(out_pt[:-3] + ".npy") and os.path.exists(out_pt):
                _pt_to_npy(out_pt)
        print("  Conversions done.")
        open(done_file, "w").close()
    else:
        deadline = time.time() + 300
        while not os.path.exists(done_file):
            if time.time() > deadline:
                _dbg(f"WARNING: timed out waiting for conversion sentinel {done_file}")
                break
            time.sleep(2)

    if _LOCAL_RANK == int(os.environ.get("SLURM_NTASKS_PER_NODE", 8)) - 1:
        try:
            time.sleep(5)
            os.remove(done_file)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Coordinates + chromosome splitting
# ═══════════════════════════════════════════════════════════════════════════

def _npy_paths(tracks_dir, species, tissue_condition, region_len, nbins, psites):
    bn  = f"{species}_{tissue_condition}"
    d   = f"{tracks_dir}/{species}/{tissue_condition}"
    inp = f"{d}/{bn}_rna_{region_len}_log_rnaseq_final_v2.npy"
    tag = "ribo.psites" if psites else "ribo"
    out = f"{d}/{bn}_{tag}_{region_len}_{nbins}_log_riboseq_final_v2.npy"
    return inp, out


def load_coords(species):
    """Load ALL transcripts to keep mmap row alignment."""
    path = f"coordinates/{species}_coordinates_v2.txt"
    if not os.path.exists(path):
        _dbg(f"ERROR: coordinate file not found: {path}")
        return None
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            rows.append({
                "chr": p[0], "start": int(p[1]), "end": int(p[2]),
                "id": p[3], "biotype": p[5], "sequence": p[7],
                "region": p[8] if len(p) > 8 else None,
                "species": species,
            })
    return pd.DataFrame(rows) if rows else None


def _norm_chr(c):
    return c[3:] if c.lower().startswith("chr") else c


def split_chromosomes(chroms, trial=None, fixed_val=None):
    chroms = [c for c in chroms if len(c) <= 7 and "MT" not in c.upper()]
    normed = {c: _norm_chr(c) for c in chroms}

    if trial is not None:
        parts = [p.strip() for p in str(trial).split(",")]
        if len(parts) == 3:
            tr = [c for c in chroms if normed[c] == _norm_chr(parts[0])]
            vl = [c for c in chroms if normed[c] == _norm_chr(parts[1])]
            te = [c for c in chroms if normed[c] == _norm_chr(parts[2])]
            return tr, vl, te
        else:
            hits = [c for c in chroms if normed[c] == _norm_chr(parts[0])]
            return ([hits[0]], [hits[0]], [hits[0]]) if hits else ([], [], [])

    test = []
    for want in ("16", "1", "X"):
        for c in chroms:
            if normed[c] == want and c not in test:
                test.append(c); break
    remaining = [c for c in chroms if c not in test]
    test += remaining[:max(0, 3 - len(test))]
    remaining = [c for c in chroms if c not in test]

    if fixed_val:
        fv = set(fixed_val)
        val = [c for c in remaining if normed[c] in fv]
        remaining = [c for c in remaining if c not in val]
        if len(val) < 3:
            val += remaining[:3 - len(val)]
            remaining = [c for c in remaining if c not in val]
    else:
        val, remaining = remaining[:3], remaining[3:]
    return remaining, val, test


def load_homology(path):
    """Read exact transcript IDs from cluster_id/transcript_id TSV."""
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["cluster_id", "transcript_id"]:
            raise ValueError("Homology TSV must have columns: cluster_id, transcript_id")
        for line, row in enumerate(reader, start=2):
            cluster = (row.get("cluster_id") or "").strip()
            transcript = (row.get("transcript_id") or "").strip()
            if None in row or not cluster or not transcript:
                raise ValueError(f"Invalid homology row {line}: expected cluster_id and transcript_id")
            previous = mapping.setdefault(transcript, cluster)
            if previous != cluster:
                raise ValueError(f"Transcript {transcript} belongs to multiple homology clusters")
    if not mapping:
        raise ValueError("Homology file contains no transcripts")
    return mapping


def split_homology_clusters(clusters, seed):
    """Split eligible clusters globally; all ranks use the same sorted input."""
    clusters = sorted(set(clusters))
    if len(clusters) < 2:
        raise ValueError("Homology training requires at least two eligible clusters")
    random.Random(seed).shuffle(clusters)
    n_train = max(1, min(len(clusters) - 1, int(0.7 * len(clusters))))
    return set(clusters[:n_train]), set(clusters[n_train:])


def _biotype_mask(df_all, biotype):
    if biotype == "protein_coding":
        return (df_all["biotype"] == "protein_coding").to_numpy(copy=True)
    elif biotype == "non_coding":
        return df_all["biotype"].isin(["lncRNA", "lincRNA"]).to_numpy(copy=True)
    return np.ones(len(df_all), dtype=bool)


def compute_shared_val_chromosomes(training_tracks, args, biotype):
    job_id    = os.environ.get("SLURM_JOB_ID", "0")
    chrom_file = f"_val_chroms_tissue_{job_id}.json"

    if _RANK == 0:
        per_species = {}
        for t in training_tracks:
            df = load_coords(t["species"])
            if df is not None:
                per_species.setdefault(t["species"], set()).update(
                    _norm_chr(c) for c in df["chr"].unique())
        test_chroms = {"16", "1", "X"}
        shared = []
        for n in range(1, 23):
            s = str(n)
            if s in test_chroms: continue
            if all(s in v for v in per_species.values()):
                shared.append(s)

        import random
        rng = random.Random(args.seed)
        candidates = sorted(rng.sample(shared, 3)) if len(shared) >= 3 else shared
        tmp = f"{chrom_file}.tmp.{os.getpid()}"
        with open(tmp, "w") as f: json.dump(candidates, f)
        os.replace(tmp, chrom_file)
        print(f"  Validation chromosomes: {candidates}")

    deadline = time.time() + 120
    while not os.path.exists(chrom_file):
        if time.time() > deadline: return ["1", "2", "3"]
        time.sleep(1)
    with open(chrom_file) as f:
        result = json.load(f)

    if _RANK == 0:
        try: time.sleep(5); os.remove(chrom_file)
        except OSError: pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
# § 4  Dataset — returns tissue_id AND cond_id
# ═══════════════════════════════════════════════════════════════════════════

_OHE_TABLE = np.zeros((256, 5), dtype=np.float32)
for _i, _b in enumerate("ATCGN"):  _OHE_TABLE[ord(_b), _i] = 1.0
for _i, _b in enumerate("atcgn"):  _OHE_TABLE[ord(_b), _i] = 1.0

_BASES    = np.array(list("ATCG"))
_N_UPPER  = np.uint8(ord("N"))
_N_LOWER  = np.uint8(ord("n"))


def _valid_sequence_mask(sequences, index_array, seq_len):
    valid = np.zeros(len(index_array), dtype=bool)
    for k, idx in enumerate(index_array):
        seq = sequences[int(idx)]
        if not seq: continue
        s = seq[:seq_len]
        codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        if np.any((codes != _N_UPPER) & (codes != _N_LOWER)):
            valid[k] = True
    return valid


def _encode_seq_fast(seq, seq_len):
    s    = seq[:seq_len]
    codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    ohe  = _OHE_TABLE[codes].copy()
    if len(ohe) < seq_len:
        ohe = np.concatenate([ohe, np.zeros((seq_len - len(ohe), 5), dtype=np.float32)])
    return ohe


def _randomize_seq(seq, seq_len):
    s    = seq[:seq_len]
    arr  = np.array(list(s))
    non_n = (arr != 'N') & (arr != 'n')
    n_replace = non_n.sum()
    if n_replace > 0:
        arr[non_n] = _BASES[np.random.randint(0, 4, size=n_replace)]
    return ''.join(arr)


def _make_mask_fast(seq, seq_len, pool_k):
    s     = seq[:seq_len]
    codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    valid = ((codes != ord("N")) & (codes != ord("n"))).astype(np.float32)
    raw   = np.zeros(seq_len, dtype=np.float32)
    raw[:len(valid)] = valid
    if pool_k > 1:
        L   = (seq_len // pool_k) * pool_k
        raw = raw[:L].reshape(-1, pool_k).mean(axis=1)
    return raw


class TissueTranscriptDataset(Dataset):
    """Mmap-backed dataset returning tissue_id, cond_id, and track_id."""

    def __init__(self, sequences, index_array, inp_mmap, out_mmap,
                 tissue_id, cond_id, seq_len, nbins,
                 nosequence=False, track_id=0):
        self.sequences  = sequences
        self.inp_mmap   = inp_mmap
        self.out_mmap   = out_mmap
        self.tissue_id  = tissue_id
        self.cond_id    = cond_id
        self.track_id   = track_id
        self.seq_len    = seq_len
        self.nbins      = nbins
        self.nosequence = nosequence
        self.pool_k     = max(1, seq_len // nbins)

        valid_mask       = _valid_sequence_mask(sequences, index_array, seq_len)
        self.index_array = index_array[valid_mask]

    def __len__(self):
        return len(self.index_array)

    def __getitem__(self, i):
        row  = int(self.index_array[i])
        seq  = self.sequences[row]
        mask = _make_mask_fast(seq, self.seq_len, self.pool_k)

        rna = self.inp_mmap[row].astype(np.float32)
        if self.nosequence:
            features = rna[:, np.newaxis]
        else:
            ohe = _encode_seq_fast(seq, self.seq_len)
            features = np.concatenate([ohe, rna[:, np.newaxis]], axis=1)
        target   = self.out_mmap[row].astype(np.float32)

        return {
            "features":  torch.from_numpy(features),
            "target":    torch.from_numpy(target),
            "mask":      torch.from_numpy(mask),
            "tissue_id": torch.tensor(self.tissue_id, dtype=torch.long),
            "cond_id":   torch.tensor(self.cond_id,   dtype=torch.long),
            "track_id":  torch.tensor(self.track_id,  dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# § 5  DataModule — tissue + condition aware
# ═══════════════════════════════════════════════════════════════════════════

class TissueRiboDataModule(LightningDataModule):

    def __init__(self, args, training_tracks, val_chroms,
                 tissue_vocab, cond_vocab, test_tracks=None):
        super().__init__()
        self.args            = args
        self.training_tracks = training_tracks
        self.homology = load_homology(args.homology) if getattr(args, "homology", None) else None
        if self.homology is not None and (args.test or args.trial):
            raise ValueError("--homology cannot be combined with --test or --trial")
        self.test_tracks     = [] if self.homology is not None else (test_tracks or [])
        self.val_chroms_norm = set(val_chroms)
        self.tissue_vocab    = tissue_vocab
        self.cond_vocab      = cond_vocab
        self._train_ds    = None
        self._val_ds      = None
        self._test_ds     = None
        self._test_new_ds = None
        self._mmap_refs   = []
        self.track_names          = []
        self.test_new_track_names = []

    def setup(self, stage=None):
        if self._train_ds is not None:
            return

        if _RANK == 0:
            _dbg(f"setup() n_tracks={len(self.training_tracks)} "
                 f"n_tissues={len(self.tissue_vocab)} "
                 f"n_conditions={len(self.cond_vocab)}")

        # Wait for npy files
        deadline = time.time() + 300
        while time.time() < deadline:
            all_ready = True
            for t in self.training_tracks:
                inp_npy, out_npy = _npy_paths(
                    self.args.tracks_dir, t["species"], t["tissue_condition"],
                    self.args.region_len, self.args.nBins, self.args.psites)
                if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                    all_ready = False; break
            if all_ready: break
            time.sleep(2)

        train_parts, val_parts, test_parts = [], [], []
        loaded_tracks = []

        for track in self.training_tracks:
            sp  = track["species"]
            tc  = track["tissue_condition"]
            ti  = track["tissue"]
            cnd = track["condition"]
            tissue_id = self.tissue_vocab.get(ti,  len(self.tissue_vocab))
            cond_id   = self.cond_vocab.get(cnd,   len(self.cond_vocab))
            track_id  = len(self.track_names)
            self.track_names.append((sp, tc))

            inp_npy, out_npy = _npy_paths(
                self.args.tracks_dir, sp, tc,
                self.args.region_len, self.args.nBins, self.args.psites)
            if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                if _RANK == 0:
                    _dbg(f"SKIP {sp}/{tc}: missing npy")
                continue

            df_all = load_coords(sp)
            if df_all is None or df_all.empty:
                if _RANK == 0:
                    _dbg(f"SKIP {sp}/{tc}: no coordinates")
                continue

            inp_mm = np.load(inp_npy, mmap_mode="r")
            out_mm = np.load(out_npy, mmap_mode="r")
            self._mmap_refs.extend([inp_mm, out_mm])

            if inp_mm.shape[0] != len(df_all) or out_mm.shape[0] != len(df_all):
                if _RANK == 0:
                    _dbg(f"SKIP {sp}/{tc}: shape mismatch "
                         f"mmap={inp_mm.shape[0]} coords={len(df_all)}")
                continue

            bt_ok       = _biotype_mask(df_all, self.args.biotype)

            bt_ok &= load_rna_eligibility(inp_npy, len(df_all))
            sequences   = training_sequences(df_all, self.args)
            if self.homology is not None:
                # Include only eligible, mapped transcripts with usable sequence.
                bt_ok &= df_all["id"].isin(self.homology).to_numpy()
                candidates = np.flatnonzero(bt_ok)
                bt_ok[candidates] &= _valid_sequence_mask(sequences, candidates, self.args.region_len)
            loaded_tracks.append((df_all, sequences, bt_ok, inp_mm, out_mm,
                                  tissue_id, cond_id, track_id, sp, tc))

        if self.homology is not None:
            eligible_clusters = {
                self.homology[transcript]
                for df, _, mask, *_ in loaded_tracks
                for transcript in df.loc[mask, "id"]
            }
            self.train_clusters, self.val_clusters = split_homology_clusters(eligible_clusters, self.args.seed)
            if _RANK == 0:
                print(f"  Homology split: {len(self.train_clusters)} training clusters, "
                      f"{len(self.val_clusters)} validation clusters; testing disabled")
                split_record = dict(seed=self.args.seed, homology_file=os.path.abspath(self.args.homology),
                                    homology_sha256=hashlib.sha256(json.dumps(self.homology, sort_keys=True).encode()).hexdigest(),
                                    train_clusters=sorted(self.train_clusters),
                                    validation_clusters=sorted(self.val_clusters))
                if getattr(self.args, "save_dir", None):
                    path = os.path.join(self.args.save_dir, "homology_split.json")
                    if os.path.exists(path):
                        with open(path, encoding="utf-8") as handle:
                            previous = json.load(handle)
                        for key in ("seed", "homology_sha256", "train_clusters", "validation_clusters"):
                            if previous[key] != split_record[key]:
                                raise ValueError("Homology split changed; use a different --save_path")
                    else:
                        os.makedirs(self.args.save_dir, exist_ok=True)
                        tmp = f"{path}.tmp.{os.getpid()}"
                        with open(tmp, "w", encoding="utf-8") as handle:
                            json.dump(split_record, handle, indent=2)
                        os.replace(tmp, path)

        for (df_all, sequences, bt_ok, inp_mm, out_mm,
             tissue_id, cond_id, track_id, sp, tc) in loaded_tracks:
            if self.homology is not None:
                clusters = df_all["id"].map(self.homology)
                tr_idx = np.flatnonzero(bt_ok & clusters.isin(self.train_clusters).to_numpy())
                vl_idx = np.flatnonzero(bt_ok & clusters.isin(self.val_clusters).to_numpy())
                te_idx = np.array([], dtype=np.int64)
            else:
                norm_chroms = np.array([_norm_chr(c) for c in df_all["chr"].values])
                train_chr, val_chr, test_chr = split_chromosomes(
                    df_all["chr"].unique().tolist(), trial=self.args.trial,
                    fixed_val=list(self.val_chroms_norm))
                tr_set = {_norm_chr(c) for c in train_chr}
                vl_set = {_norm_chr(c) for c in val_chr}
                te_set = {_norm_chr(c) for c in test_chr}
                tr_idx = np.where(np.isin(norm_chroms, list(tr_set)) & bt_ok)[0]
                vl_idx = np.where(np.isin(norm_chroms, list(vl_set)) & bt_ok)[0]
                te_idx = np.where(np.isin(norm_chroms, list(te_set)) & bt_ok)[0]

            ds_kwargs = dict(
                sequences=sequences, inp_mmap=inp_mm, out_mmap=out_mm,
                tissue_id=tissue_id, cond_id=cond_id,
                seq_len=self.args.region_len, nbins=self.args.nBins,
                nosequence=self.args.nosequence, track_id=track_id)

            if len(tr_idx):
                train_parts.append(TissueTranscriptDataset(index_array=tr_idx, **ds_kwargs))
            if len(vl_idx):
                val_parts.append(TissueTranscriptDataset(index_array=vl_idx, **ds_kwargs))
            if len(te_idx):
                test_parts.append(TissueTranscriptDataset(index_array=te_idx, **ds_kwargs))

            n_tr = len(train_parts[-1]) if len(tr_idx) else 0
            n_vl = len(val_parts[-1])   if len(vl_idx) else 0
            n_te = len(test_parts[-1])  if len(te_idx) else 0
            if _RANK == 0:
                _dbg(f"LOADED {sp}/{tc} "
                     f"(tissue_id={tissue_id} cond_id={cond_id} track_id={track_id}): "
                     f"train={n_tr} val={n_vl} test={n_te}")

        self._train_ds = ConcatDataset(train_parts) if train_parts else None
        self._val_ds   = ConcatDataset(val_parts)   if val_parts   else None
        self._test_ds  = ConcatDataset(test_parts)  if test_parts  else None

        if self.homology is not None and (not self._train_ds or not self._val_ds):
            raise RuntimeError("Homology split requires nonempty training and validation datasets")
        if self._train_ds is None and not self.args.test:
            raise RuntimeError("No training data loaded.")

        total_tr = len(self._train_ds) if self._train_ds else 0
        total_vl = len(self._val_ds)   if self._val_ds   else 0
        total_te = len(self._test_ds)  if self._test_ds  else 0
        split_label = "homology mode: testing disabled" if self.homology is not None else "hold-out chroms"
        if _RANK == 0:
            _dbg(f"Total: {total_tr} train, {total_vl} val, {total_te} test ({split_label})")
        print(f"\n  Total dataset: {total_tr} train, {total_vl} val, {total_te} test ({split_label})")
        print(f"  Tissues: {len(self.tissue_vocab)}  |  Conditions: {len(self.cond_vocab)}")

        # ── Test-tagged tracks ──────────────────────────────────────────
        if self.args.test and self.test_tracks:
            test_new_parts = []
            for track in self.test_tracks:
                sp  = track["species"]
                tc  = track["tissue_condition"]
                ti  = track["tissue"]
                cnd = track["condition"]
                tissue_id = self.tissue_vocab.get(ti,  len(self.tissue_vocab))
                cond_id   = self.cond_vocab.get(cnd,   len(self.cond_vocab))
                test_new_track_id = len(self.test_new_track_names)
                self.test_new_track_names.append((sp, tc))

                inp_npy, out_npy = _npy_paths(
                    self.args.tracks_dir, sp, tc,
                    self.args.region_len, self.args.nBins, self.args.psites)
                if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                    if _RANK == 0:
                        _dbg(f"SKIP test track {sp}/{tc}: missing npy"); continue

                df_all = load_coords(sp)
                if df_all is None or df_all.empty:
                    if _RANK == 0:
                        _dbg(f"SKIP test track {sp}/{tc}: no coordinates"); continue

                inp_mm = np.load(inp_npy, mmap_mode="r")
                out_mm = np.load(out_npy, mmap_mode="r")
                self._mmap_refs.extend([inp_mm, out_mm])

                if inp_mm.shape[0] != len(df_all):
                    if _RANK == 0:
                        _dbg(f"SKIP test track {sp}/{tc}: shape mismatch"); continue

                bt_ok     = _biotype_mask(df_all, self.args.biotype)

                bt_ok &= load_rna_eligibility(inp_npy, len(df_all))
                sequences = training_sequences(df_all, self.args)
                all_idx   = np.where(bt_ok)[0]

                if len(all_idx):
                    ds = TissueTranscriptDataset(
                        sequences=sequences, index_array=all_idx,
                        inp_mmap=inp_mm, out_mmap=out_mm,
                        tissue_id=tissue_id, cond_id=cond_id,
                        seq_len=self.args.region_len, nbins=self.args.nBins,
                        nosequence=self.args.nosequence, track_id=test_new_track_id)
                    test_new_parts.append(ds)
                    if _RANK == 0:
                        _dbg(f"LOADED test track {sp}/{tc} (all chroms): n={len(ds)}")

            self._test_new_ds = ConcatDataset(test_new_parts) if test_new_parts else None
            if self._test_new_ds:
                print(f"  Test-tagged tracks: {len(self._test_new_ds)} samples "
                      f"from {len(self.test_new_track_names)} tracks\n")
        else:
            message = "disabled by --homology" if self.homology is not None else "not loaded (use --test to load)"
            print(f"  Test-tagged tracks: {message}\n")

    def _make_loader(self, dataset, shuffle, drop_last):
        sampler = None
        try:    world = self.trainer.world_size
        except: world = _WORLD_SIZE
        if world > 1:
            if shuffle:
                sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
            else:
                dataset = EvaluationDataset(dataset)
                sampler = DistributedEvaluationSampler(dataset)
            shuffle = False
        nw = self.args.num_workers
        return DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=shuffle,
            sampler=sampler, num_workers=nw, pin_memory=False,
            drop_last=drop_last, persistent_workers=(nw > 0),
            prefetch_factor=1 if nw > 0 else None, timeout=600)

    def train_dataloader(self):
        if self._train_ds is None: raise RuntimeError("setup() not called")
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self._val_ds is None or len(self._val_ds) == 0: return None
        return self._make_loader(self._val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self):
        dataloaders = []
        if self._test_ds     is not None and len(self._test_ds)     > 0:
            dataloaders.append(self._make_loader(self._test_ds,     shuffle=False, drop_last=False))
        if self._test_new_ds is not None and len(self._test_new_ds) > 0:
            dataloaders.append(self._make_loader(self._test_new_ds, shuffle=False, drop_last=False))
        return dataloaders if dataloaders else None


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Metrics and loss
# ═══════════════════════════════════════════════════════════════════════════

class StreamingPearsonR(Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ("sum_xy", "sum_x", "sum_y", "sum_x2", "sum_y2", "count"):
            self.add_state(name,
                           default=torch.tensor(0.0, dtype=torch.float64),
                           dist_reduce_fx="sum")

    def update(self, pred, target, mask):
        m = mask.reshape(-1).bool()
        p = (pred * mask).reshape(-1)[m].double()
        t = (target * mask).reshape(-1)[m].double()
        n = p.numel()
        if n < 2: return
        self.sum_xy += (p * t).sum()
        self.sum_x  += p.sum()
        self.sum_y  += t.sum()
        self.sum_x2 += (p * p).sum()
        self.sum_y2 += (t * t).sum()
        self.count  += n

    def compute(self):
        n   = self.count.clamp(min=2)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        vx  = self.sum_x2 - self.sum_x ** 2 / n
        vy  = self.sum_y2 - self.sum_y ** 2 / n
        return (cov / torch.sqrt((vx * vy).clamp(min=1e-16))).float()


def _per_sample_pcc_loss(pred, target, mask):
    pred   = pred.float()
    target = target.float()
    mask   = mask.float()
    p, t = pred * mask, target * mask
    n    = mask.sum(dim=1, keepdim=True).clamp(min=1)
    p_c  = (p - p.sum(dim=1, keepdim=True) / n) * mask
    t_c  = (t - t.sum(dim=1, keepdim=True) / n) * mask
    cov  = (p_c * t_c).sum(dim=1)
    denom = torch.sqrt(((p_c**2).sum(dim=1) * (t_c**2).sum(dim=1)).clamp(min=1e-8))
    r     = cov / denom
    valid = (n.squeeze(1) >= 10)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - r[valid].mean()


# ═══════════════════════════════════════════════════════════════════════════
# § 7  LightningModule — tissue + condition conditioned
# ═══════════════════════════════════════════════════════════════════════════

class TissueRiboModel(LightningModule):

    def __init__(self, args, tissue_vocab, cond_vocab):
        super().__init__()
        self.save_hyperparameters({
            "model_type":      args.model_type,
            "region_len":      args.region_len,
            "nBins":           args.nBins,
            "dropout":         args.dropout,
            "learning_rate":   args.learning_rate,
            "weight_decay":    args.weight_decay,
            "warmup_steps":    args.warmup_steps,
            "batch_size":      args.batch_size,
            "grad_accum":      args.grad_accum,
            "grad_clip":       args.grad_clip,
            "monitor":         args.monitor,
            "nosequence":      args.nosequence,
            "shuflledutr": args.shuflledutr,
            "shuffle_seed": args.seed if args.shuffle_seed is None else args.shuffle_seed,
            "sequence_input_mode": "rna_only" if args.nosequence else "sequence_and_rna",
            "homology":        getattr(args, "homology", None),
            "homology_sha256": getattr(args, "homology_sha256", None),
            "seed":            args.seed,
            "tissue_emb_dim":  args.tissue_emb_dim,
            "cond_emb_dim":    args.cond_emb_dim,
            "num_tissues":     len(tissue_vocab),
            "num_conditions":  len(cond_vocab),
            "tissue_vocab":    tissue_vocab,
            "cond_vocab":      cond_vocab,
            "zero_w":          args.zero_w,
            "pcc_loss_w":      args.pcc_loss_w,
        })
        self.args        = args
        self.tissue_vocab = tissue_vocab
        self.cond_vocab   = cond_vocab

        ModelCls = getattr(models, args.model_type)
        self.net = ModelCls(
            num_genomic_features=1,
            target_length=args.region_len,
            nbins=args.nBins,
            num_tissues=len(tissue_vocab),
            tissue_emb_dim=args.tissue_emb_dim,
            num_conditions=len(cond_vocab),
            cond_emb_dim=args.cond_emb_dim,
            dropout=args.dropout,
            seqno=args.nosequence,
        )

        self.train_pcc    = StreamingPearsonR()
        self.val_pcc      = StreamingPearsonR()
        self._tr_loss_sum = 0.0
        self._tr_loss_n   = 0
        self._vl_loss_sum = 0.0
        self._vl_loss_n   = 0
        self._history     = []

        self._test_per_track = {}
        self._track_names    = None
        self._test_phase     = "holdout"
        self._zero_w         = args.zero_w
        self._pcc_loss_w     = args.pcc_loss_w

    def forward(self, x, tissue_ids, cond_ids):
        return self.net(x, tissue_ids, cond_ids)

    def _compute_loss(self, pred, target, mask):
        return masked_profile_loss(pred, target, mask, self._zero_w, self._pcc_loss_w)

    def training_step(self, batch, batch_idx):
        pred   = self(batch["features"].float(), batch["tissue_id"], batch["cond_id"])
        target = batch["target"].float()
        mask   = batch["mask"].float()

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            _dbg(f"NaN/Inf in pred at step {self.global_step} — zeroing loss")
            if batch_idx % 50 == 0:
                self.log("train/loss_step", torch.tensor(0.0), prog_bar=True, sync_dist=False)
            return pred.sum() * 0.0

        loss = self._compute_loss(pred, target, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            _dbg(f"NaN/Inf loss at step {self.global_step} — zeroing loss")
            if batch_idx % 50 == 0:
                self.log("train/loss_step", torch.tensor(0.0), prog_bar=True, sync_dist=False)
            return pred.sum() * 0.0

        self._train_profile_metrics.update(pred.detach(), target, mask)
        if batch_idx % 50 == 0:
            self.log("train/loss_step", loss.detach(), prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["features"].float(), batch["tissue_id"], batch["cond_id"])
        self._val_profile_metrics.update(pred, batch["target"].float(), batch["mask"].float())

    def _get_test_track(self, tid):
        tid = int(tid)
        if tid not in self._test_per_track:
            self._test_per_track[tid] = {
                "loss_sum": 0.0, "loss_n": 0,
                "pcc": StreamingPearsonR().to(self.device),
            }
        return self._test_per_track[tid]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch["features"].float(), batch["tissue_id"], batch["cond_id"])
        target, mask, track_ids = batch["target"].float(), batch["mask"].float(), batch["track_id"]
        phase, _ = self._test_groups[dataloader_idx]
        for tid in track_ids.unique().tolist():
            selected = track_ids == tid
            self._test_metrics[(phase, int(tid))].update(pred[selected], target[selected], mask[selected])

    def on_test_epoch_end(self):
        for phase, names in self._test_groups:
            results = []
            overall = ProfileMetrics(self.device, self._zero_w, self._pcc_loss_w)
            for tid, (species, sample) in enumerate(names):
                metric = self._test_metrics[(phase, tid)]
                overall.state += metric.state
                scores = metric.compute()
                if scores["n_samples"] == 0:
                    continue
                label = f"{species}/{sample}"
                self.log(f"test_{phase}/loss_{label}", scores["loss"], sync_dist=False)
                self.log(f"test_{phase}/pcc_{label}", scores["pcc"], sync_dist=False)
                results.append({"species_tissue_condition": label, **scores})
                if self.global_rank == 0:
                    print(f"{phase} {label}: loss={scores['loss']:.5f} PCC={scores['pcc']:.4f} n={scores['n_samples']}")
            total = overall.compute()
            if self.global_rank == 0:
                print(f"{phase} overall: {total}")
                if results:
                    pd.DataFrame(results).to_csv(os.path.join(self.args.save_dir, f"test_results_{phase}.csv"), index=False)
        self._test_metrics = {}

    def _pcc_from_states(self, metric):
        n   = metric.count.clamp(min=2)
        cov = metric.sum_xy - metric.sum_x * metric.sum_y / n
        vx  = metric.sum_x2 - metric.sum_x ** 2 / n
        vy  = metric.sum_y2 - metric.sum_y ** 2 / n
        return (cov / torch.sqrt((vx * vy).clamp(min=1e-16))).item()

    def on_train_epoch_end(self):
        scores = self._train_profile_metrics.compute()
        self.log("train/loss_epoch", scores["loss"], sync_dist=False, prog_bar=True)
        self.log("train/pcc_epoch", scores["pcc"], sync_dist=False)
        if self.global_rank == 0:
            record_epoch(self._history, self.current_epoch, "train", scores)
            print(f"Epoch {self.current_epoch} train | loss={scores['loss']:.5f} PCC={scores['pcc']:.4f}")

    def on_validation_epoch_end(self):
        scores = self._val_profile_metrics.compute()
        self.log("val/loss_epoch", scores["loss"], sync_dist=False, prog_bar=True)
        self.log("val/pcc_epoch", scores["pcc"], sync_dist=False)
        if self.global_rank == 0:
            record_epoch(self._history, self.current_epoch, "val", scores)
            print(f"Epoch {self.current_epoch} val | loss={scores['loss']:.5f} PCC={scores['pcc']:.4f}")


    def on_train_end(self):
        self.net.set_mean_embedding()

        if self.global_rank == 0:
            save_training_curves(self._history, self.args.save_dir)

            vocab_path = os.path.join(self.args.save_dir, "tissue_vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(self.tissue_vocab, f, indent=2)
            print(f"  Tissue vocab -> {vocab_path}")

            cond_vocab_path = os.path.join(self.args.save_dir, "cond_vocab.json")
            with open(cond_vocab_path, "w") as f:
                json.dump(self.cond_vocab, f, indent=2)
            print(f"  Condition vocab -> {cond_vocab_path}")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        opt = torch.optim.AdamW([
            {"params": decay,    "weight_decay": self.args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.args.warmup_steps, max(100, total_steps // 10))
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_train_epoch_start(self):
        self._train_profile_metrics = ProfileMetrics(self.device, self._zero_w, self._pcc_loss_w)

    def on_validation_epoch_start(self):
        self._val_profile_metrics = ProfileMetrics(self.device, self._zero_w, self._pcc_loss_w)

    def on_test_epoch_start(self):
        dm = self.trainer.datamodule
        self._test_groups = []
        if dm._test_ds is not None and len(dm._test_ds) > 0:
            self._test_groups.append(("holdout", dm.track_names))
        if dm._test_new_ds is not None and len(dm._test_new_ds) > 0:
            self._test_groups.append(("new_samples", dm.test_new_track_names))
        # Identical metric keys on all GPUs, including tracks absent on one rank.
        self._test_metrics = {
            (phase, tid): ProfileMetrics(self.device, self._zero_w, self._pcc_loss_w)
            for phase, names in self._test_groups for tid in range(len(names))
        }


# ═══════════════════════════════════════════════════════════════════════════
# § 8  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("RiboTransPred - Tissue + Condition dual FiLM")
    p.add_argument("--seed",            type=int,   default=4)
    p.add_argument("--save_path",       default="results_tissues")
    p.add_argument("--tracks",          default="tracks.txt")
    p.add_argument("--tracks_dir",      default="tracks")
    p.add_argument("--model-type",      default="PosTransModelTCNFiLMRef",
                   choices=["PosTransModelTCNFiLM", "PosTransModelTCNFiLMRef",
                             "TransModelFiLM", "PosTransModelFiLM",
                             "PosTransModelFiLMRef"])
    p.add_argument("--region_len",      type=int,   default=4500)
    p.add_argument("--nBins",           type=int,   default=1500)
    p.add_argument("--biotype",         default="protein_coding",
                   choices=["protein_coding", "non_coding", "all"])
    p.add_argument("--psites",          action="store_true")
    p.add_argument("--homology", type=str, default=None, metavar="FILE",
                   help="Cluster/transcript TSV: split eligible clusters 70/30 for training/validation; exclude unmapped transcripts; no test")
    p.add_argument("--trial",           type=str,   default=None)
    p.add_argument("--nosequence",      action="store_true")
    p.add_argument("--shuflledutr", "--shuffledutr", dest="shuflledutr", action="store_true",
                    help="Shuffle 5-prime UTR sequence before annotated CDS; preserve RNA and CDS")
    p.add_argument("--shuffle-seed", dest="shuffle_seed", type=int, default=None)
    p.add_argument("--patience",        type=int,   default=8)
    p.add_argument("--max-epochs",      type=int,   default=80)
    p.add_argument("--save-top-n",      type=int,   default=5)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=None)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--weight_decay",    type=float, default=5e-4)
    p.add_argument("--warmup_steps",    type=int,   default=2000)
    p.add_argument("--grad_accum",      type=int,   default=3)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--tissue_emb_dim",  type=int,   default=64,
                   help="Dimension of tissue embedding for FiLM")
    p.add_argument("--cond_emb_dim",    type=int,   default=32,
                   help="Dimension of condition embedding for FiLM")
    p.add_argument("--zero_w",          type=float, default=0.1)
    p.add_argument("--pcc_loss_w",      type=float, default=0.2)
    p.add_argument("--monitor", default="val/loss_epoch",
                   choices=["val/loss_epoch", "val/pcc_epoch"])
    p.add_argument("--checkpoint",      type=str,   default=None)
    p.add_argument("--test",            action="store_true")
    args = p.parse_args()
    if args.homology and (args.test or args.trial is not None):
        p.error("--homology cannot be combined with --test or --trial")
    if args.nosequence and args.shuflledutr:
        p.error("--nosequence and --shuflledutr cannot be combined")
    if args.dropout is None:
        args.dropout = 0.33 if "TCN" in args.model_type else 0.3
    return args


# ═══════════════════════════════════════════════════════════════════════════
# § 9  Track parser + vocabularies
# ═══════════════════════════════════════════════════════════════════════════

def parse_tracks(path):
    """
    Parse tracks file.  Expected format (tab or space delimited):
      <bam_file> <species> <tissue> <condition> <dataset>

    Returns a list of dicts with keys:
      bam_file, species, tissue, condition, tissue_condition, dataset
    """
    tracks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 5:
                _dbg(f"WARNING: skipping malformed track line: {line}")
                continue
            bam, species, tissue, condition, dataset = parts[:5]
            if "ribo" in os.path.basename(bam).lower():
                tracks.append({
                    "bam_file":        bam,
                    "species":         species,
                    "tissue":          tissue,
                    "condition":       condition,
                    "tissue_condition": f"{tissue}_{condition}",
                    "dataset":         dataset,
                })
    return tracks


def build_tissue_vocab(tracks):
    """Deterministic tissue vocabulary (sorted alphabetically)."""
    tissues = sorted({t["tissue"] for t in tracks})
    return {tissue: idx for idx, tissue in enumerate(tissues)}


def build_cond_vocab(tracks):
    """Deterministic condition vocabulary (sorted alphabetically)."""
    conditions = sorted({t["condition"] for t in tracks})
    return {cond: idx for idx, cond in enumerate(conditions)}


# ═══════════════════════════════════════════════════════════════════════════
# § 10  Main
# ═══════════════════════════════════════════════════════════════════════════

class BestModelCheckpoint(plc.ModelCheckpoint):
    """Atomically publish a best.ckpt copy selected by validation performance."""
    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        if not trainer.is_global_zero or not self.best_model_path:
            return
        destination = os.path.join(self.dirpath, "best.ckpt")
        identity = (self.best_model_path, float(self.best_model_score))
        if getattr(self, "_published_best", None) == identity and os.path.isfile(destination):
            return
        temporary = destination + f".tmp.{os.getpid()}"
        try:
            shutil.copy2(self.best_model_path, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary): os.remove(temporary)
        self._published_best = identity
        print(f"  Best checkpoint: {destination} ({self.monitor}={identity[1]:.6f})")


def training_callbacks(args, monitor=None, save_top_k=None):
    monitor = monitor or args.monitor
    mode = "min" if monitor == "val/loss_epoch" else "max"
    checkpoint = BestModelCheckpoint(
        dirpath=args.save_dir, filename="epoch={epoch}-score={" + monitor + ":.6f}",
        monitor=monitor, mode=mode,
        save_top_k=args.save_top_n if save_top_k is None else save_top_k,
        save_last=True, auto_insert_metric_name=False)
    if checkpoint.save_top_k == 0:
        raise ValueError("save_top_n must retain at least one checkpoint for best.ckpt")
    return [plc.EarlyStopping(monitor=monitor, patience=args.patience, mode=mode,
                             verbose=(_RANK == 0)),
            plc.LearningRateMonitor(logging_interval="step"), checkpoint]


def resolve_checkpoint(checkpoint, save_dir, test=False):
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    if test:
        best = os.path.join(save_dir, "best.ckpt")
        if not os.path.isfile(best):
            raise FileNotFoundError(f"No best.ckpt found at {best}; supply --checkpoint with an existing validation-selected checkpoint")
        return best
    return None


def main():
    args = parse_args()
    if args.homology:
        mapping = load_homology(args.homology)
        args.homology_sha256 = hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    pl.seed_everything(args.seed, workers=True)

    all_tracks      = parse_tracks(args.tracks)
    training_tracks = [t for t in all_tracks if t["dataset"] == "training"]
    test_tracks     = [] if args.homology else [t for t in all_tracks if t["dataset"] == "test"]
    if not training_tracks:
        print("ERROR: No training tracks found in", args.tracks)
        sys.exit(1)

    tissue_vocab = build_tissue_vocab(training_tracks)
    cond_vocab   = build_cond_vocab(training_tracks)

    suffix = f"_homology{args.homology_sha256[:12]}" if args.homology else ""
    if args.trial:      suffix += f"_trial{args.trial}"
    if args.nosequence: suffix += "_nosequence"
    if args.shuflledutr: suffix += f"_shuflledutr_seed{args.seed if args.shuffle_seed is None else args.shuffle_seed}"
    save_dir = (
        f"{args.save_path}/tissue_film_{args.region_len}_{args.nBins}"
        f"_bs{args.batch_size}"
        f"_lr{args.learning_rate}_wd{args.weight_decay}"
        f"_ws{args.warmup_steps}_ga{args.grad_accum}"
        f"_gc{args.grad_clip}_dp{args.dropout}"
        f"_emb{args.tissue_emb_dim}_cemb{args.cond_emb_dim}"
        f"_nt{len(tissue_vocab)}_nc{len(cond_vocab)}"
        f"_{args.model_type}_seed{args.seed}{suffix}"
    )
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if _RANK == 0:
        print(f"\n{'='*60}")
        print(f"  RiboTransPred - Tissue + Condition dual FiLM  |  {args.model_type}")
        print(f"  nodes={_NNODES}  gpus/node={n_gpus}  world_size={_WORLD_SIZE}")
        print(f"  region_len={args.region_len}  nBins={args.nBins}")
        print(f"  batch={args.batch_size}  accum={args.grad_accum}  "
              f"eff_batch={args.batch_size * args.grad_accum * _WORLD_SIZE}")
        print(f"  tissue_emb_dim={args.tissue_emb_dim}  cond_emb_dim={args.cond_emb_dim}")
        print(f"  zero_w={args.zero_w}  pcc_loss_w={args.pcc_loss_w}")
        print(f"  tissues ({len(tissue_vocab)}):")
        for tname, tid in tissue_vocab.items():
            print(f"    {tid}: {tname}")
        print(f"  conditions ({len(cond_vocab)}):")
        for cname, cid in cond_vocab.items():
            print(f"    {cid}: {cname}")
        species = sorted({t['species'] for t in training_tracks})
        print(f"  species: {species}")
        print(f"  tracks:  {len(training_tracks)} training, {len(test_tracks)} test")
        print(f"  biotype: {args.biotype}")
        print(f"{'='*60}\n")

    convert_all_tracks(training_tracks + test_tracks, args)

    if args.homology:
        val_chroms = []
    elif args.trial:
        parts = [p.strip() for p in args.trial.split(",")]
        val_chroms = [parts[1]] if len(parts) == 3 else [parts[0]]
    else:
        val_chroms = compute_shared_val_chromosomes(
            training_tracks, args, args.biotype)
    if _RANK == 0 and not args.homology:
        print(f"  Validation chromosomes: {val_chroms}\n")

    dm    = TissueRiboDataModule(args, training_tracks, val_chroms,
                                 tissue_vocab, cond_vocab,
                                 test_tracks=test_tracks)
    model = TissueRiboModel(args, tissue_vocab, cond_vocab)
    ckpt = resolve_checkpoint(args.checkpoint, save_dir, test=args.test)
    if args.test and _RANK == 0:
        print(f"  Testing checkpoint: {ckpt}")
    callbacks = training_callbacks(args)
    if _RANK == 0:
        callbacks.append(plc.RichProgressBar())

    world_size = _NNODES * n_gpus
    if world_size > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
            process_group_backend="nccl",
            timeout=datetime.timedelta(seconds=7200),
            cluster_environment=SLURMEnvironment(auto_requeue=False))
    else:
        strategy = "auto"

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n_gpus, num_nodes=_NNODES, strategy=strategy,
        max_epochs=args.max_epochs, precision="bf16-mixed",
        gradient_clip_val=args.grad_clip, gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.grad_accum,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=f"{save_dir}/csv"),
        log_every_n_steps=50, num_sanity_val_steps=0,
        check_val_every_n_epoch=2, sync_batchnorm=False,
        enable_checkpointing=True,
        enable_progress_bar=(_RANK == 0),
        enable_model_summary=(_RANK == 0),
        deterministic=False)

    if not args.test:
        if _RANK == 0: print("=== Starting Training ===\n")
        trainer.fit(model, dm, ckpt_path=ckpt)
    else:
        if _RANK == 0: print("=== Test Mode ===\n")
        trainer.test(model, datamodule=dm, ckpt_path=ckpt)

    if _RANK == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
