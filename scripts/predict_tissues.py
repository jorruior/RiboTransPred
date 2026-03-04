#!/usr/bin/env python3
"""
RiboTransPred — Tissue-conditioned Ribo-seq prediction (FiLM)
=============================================================
Predicts Ribo-seq profiles from RNA-seq + DNA sequence, conditioned on
tissue identity via FiLM (Feature-wise Linear Modulation).

Author: Jorge Ruiz-Orera
"""


import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from scipy import stats

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import model.models2 as models


# § Utilities
# ═══════════════════════════════════════════════════════════════════════════

_OHE_TABLE = np.zeros((256, 5), dtype=np.float32)
for _i, _b in enumerate("ATCGN"):
	_OHE_TABLE[ord(_b), _i] = 1.0
for _i, _b in enumerate("atcgn"):
	_OHE_TABLE[ord(_b), _i] = 1.0

COMPLEMENT = str.maketrans("ATCGatcg", "TAGCtagc")
EPS = 0.0001
ZERO_THRESHOLD = 0.0001


def _encode_seq_fast(seq, seq_len):
	s = seq[:seq_len]
	codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
	ohe = _OHE_TABLE[codes].copy()
	if len(ohe) < seq_len:
		ohe = np.concatenate([ohe, np.zeros((seq_len - len(ohe), 5),
											dtype=np.float32)])
	return ohe


def reverse_complement(seq):
	return seq.translate(COMPLEMENT)[::-1]


def inverse_log_transform(log_values):
	clipped = np.clip(log_values, -14, 14)
	transformed = np.exp(clipped) - EPS
	transformed[transformed <= ZERO_THRESHOLD] = 0.0
	return transformed


def load_coords(species):
	path = f"coordinates/{species}_coordinates.txt"
	if not os.path.exists(path):
		print(f"ERROR: coordinate file not found: {path}")
		return None
	rows = []
	with open(path) as f:
		for line in f:
			p = line.rstrip("\n").split("\t")
			if len(p) < 8:
				continue
			region_annot = p[8] if len(p) > 8 else ""
			rows.append({
				"chr": p[0], "start": int(p[1]), "end": int(p[2]),
				"id": p[3], "length": int(p[4]), "biotype": p[5],
				"region_len": int(p[6]), "sequence": p[7], "region": region_annot,
			})
	return pd.DataFrame(rows) if rows else None


def load_bed(species):
	path = f"coordinates/{species}.bed"
	if not os.path.exists(path):
		print(f"ERROR: BED file not found: {path}")
		return None
	rows = []
	with open(path) as f:
		for line in f:
			p = line.strip().split("\t")
			if len(p) < 6:
				continue
			rows.append({
				"chr": p[0], "start": int(p[1]), "end": int(p[2]),
				"strand": p[3], "id": p[4], "biotype": p[5],
			})
	return pd.DataFrame(rows) if rows else None


def load_mutations(mutate_file):
	mutations = defaultdict(list)
	with open(mutate_file) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split('\t')
			if len(parts) < 6:
				print(f"Warning: Skipping malformed line: {line}")
				continue
			mutations[parts[0]].append({
				'variant_id': parts[1], 'alt_allele': parts[2],
				'position': int(parts[3]), 'effect': parts[4],
				'classification': parts[5], 'original_line': line
			})
	return mutations


def load_orfs(orfs_file):
	"""
	Load ORF file.  Format (tab-separated):
		ORF_id   transcript_id   ORF_sequence
	Returns dict: transcript_id -> list of {orf_id, orf_seq}
	"""
	orfs = defaultdict(list)
	n_total = 0
	with open(orfs_file) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split('\t')
			if len(parts) < 3:
				print(f"Warning: Skipping malformed ORF line: {line[:80]}")
				continue
			orfs[parts[1]].append({
				'orf_id': parts[0],
				'orf_seq': parts[2],
			})
			n_total += 1
	return orfs, n_total


def _npy_paths(tracks_dir, species, tissue, region_len, nbins, psites):
	bn = f"{species}_{tissue}"
	d = f"{tracks_dir}/{species}/{tissue}"
	inp_rna = f"{d}/{bn}_rna_{region_len}_log_rnaseq_final.npy"
	tag = "ribo.psites" if psites else "ribo"
	inp_ribo = f"{d}/{bn}_{tag}_{region_len}_{nbins}_log_riboseq_final.npy"
	return inp_rna, inp_ribo


def calculate_correlation(values1, values2, mask, use_log=False):
	if mask is not None:
		v1_masked = values1[mask]
		v2_masked = values2[mask]
	else:
		v1_masked = values1
		v2_masked = values2
	if use_log:
		v1_masked = np.log(v1_masked + EPS)
		v2_masked = np.log(v2_masked + EPS)
	valid_mask = np.isfinite(v1_masked) & np.isfinite(v2_masked)
	v1_clean = v1_masked[valid_mask]
	v2_clean = v2_masked[valid_mask]
	if len(v1_clean) < 2 or np.std(v1_clean) == 0 or np.std(v2_clean) == 0:
		return float('nan')
	correlation, _ = stats.pearsonr(v1_clean, v2_clean)
	return correlation


def _pool_mask_to_bins(mask, pool_k, nBins):
	"""Pool a per-base boolean mask to bin-level mask."""
	if pool_k > 1:
		n_cb = len(mask) // pool_k
		if n_cb > 0:
			mask_bins = mask[:n_cb * pool_k].reshape(-1, pool_k).mean(axis=1) > 0.5
			if n_cb < nBins:
				mask_bins = np.pad(mask_bins, (0, nBins - n_cb), constant_values=False)
		else:
			mask_bins = np.zeros(nBins, dtype=bool)
	else:
		mask_bins = mask[:nBins] if len(mask) >= nBins else \
				   np.pad(mask, (0, nBins - len(mask)), constant_values=False)
	return mask_bins


def _pool_to_bins(data_nonlog, pool_k, nBins):
	"""Pool per-base data to bin-level means."""
	if len(data_nonlog) > nBins:
		if pool_k > 1:
			n_cb = len(data_nonlog) // pool_k
			if n_cb > 0:
				bins = data_nonlog[:n_cb * pool_k].reshape(-1, pool_k).mean(axis=1)
				if n_cb < nBins:
					bins = np.pad(bins, (0, nBins - n_cb), constant_values=0.0)
			else:
				bins = np.zeros(nBins)
		else:
			bins = data_nonlog[:nBins] if len(data_nonlog) >= nBins else \
				   np.pad(data_nonlog, (0, nBins - len(data_nonlog)), constant_values=0.0)
	else:
		bins = data_nonlog
	return bins


def get_fiveutr_mask(region_annot, region_len):
	"""
	5'UTR mask: all positions before the first CDS position ('2').
	"""
	fiveutr_mask = np.zeros(region_len, dtype=bool)
	if not region_annot:
		return fiveutr_mask
	annot = region_annot[:region_len]
	for ri, rc in enumerate(annot):
		if rc == '2':
			if ri > 0:
				fiveutr_mask[:ri] = True
			break
	return fiveutr_mask


def get_seq_context(seq, position, alt_allele):
	"""
	Get 5-nt context around a mutation: 2 upstream + alt + 2 downstream.
	Uses N for positions beyond transcript boundaries.
	"""
	pos_0 = position - 1
	context = []
	for offset in [-2, -1, 0, 1, 2]:
		p = pos_0 + offset
		if p < 0 or p >= len(seq):
			context.append('N')
		elif offset == 0:
			context.append(alt_allele.upper())
		else:
			context.append(seq[p].upper())
	return ''.join(context)


def find_orf_length_from_start(seq, start_pos_0based):
	"""
	Starting from a start codon at start_pos_0based, count how many nucleotides
	are in-frame until a stop codon (TAG, TGA, TAA) is reached.
	Returns nt from start codon to stop end (inclusive), or -1 if no stop found.
	"""
	stop_codons = {'TAG', 'TGA', 'TAA'}
	pos = start_pos_0based + 3
	while pos + 2 < len(seq):
		codon = seq[pos:pos + 3].upper()
		if codon in stop_codons:
			return (pos + 3) - start_pos_0based
		pos += 3
	return -1


def get_atg_orf_length(seq, position):
	pos_0 = position - 1
	for start_offset in range(-2, 1):
		atg_start = pos_0 + start_offset
		if atg_start < 0 or atg_start + 2 >= len(seq):
			continue
		triplet = seq[atg_start:atg_start + 3].upper()
		if triplet == 'ATG':
			return find_orf_length_from_start(seq, atg_start)
	return 0


def get_ctg_orf_length(seq, position):
	pos_0 = position - 1
	for start_offset in range(-2, 1):
		ctg_start = pos_0 + start_offset
		if ctg_start < 0 or ctg_start + 2 >= len(seq):
			continue
		triplet = seq[ctg_start:ctg_start + 3].upper()
		if triplet == 'CTG':
			return find_orf_length_from_start(seq, ctg_start)
	return 0


def _find_codon_start(seq, position, codon_str):
	"""
	Find the 0-based start of `codon_str` (e.g. 'ATG' or 'CTG') in the 5-nt
	context around `position` (1-based).  Returns -1 if not found.
	"""
	pos_0 = position - 1
	for start_offset in range(-2, 1):
		cs = pos_0 + start_offset
		if cs < 0 or cs + 2 >= len(seq):
			continue
		if seq[cs:cs + 3].upper() == codon_str:
			return cs
	return -1


CODON_TABLE = {
	'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
	'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
	'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
	'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
	'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
	'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
	'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
	'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
	'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
	'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
	'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
	'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
	'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
	'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
	'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
	'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate_orf(seq, start_pos_0, orf_nt_len):
	"""
	Translate the ORF starting at start_pos_0 for orf_nt_len nucleotides.
	If orf_nt_len == -1 (no stop found), translate from start to end of seq.
	Returns amino acid string (stop codon excluded).
	"""
	if orf_nt_len == -1:
		end = len(seq)
	else:
		# orf_nt_len includes stop codon; exclude its 3 nt from translation
		end = start_pos_0 + orf_nt_len - 3
	aas = []
	pos = start_pos_0
	while pos + 2 < end and pos + 2 < len(seq):
		codon = seq[pos:pos + 3].upper()
		aa = CODON_TABLE.get(codon, 'X')
		if aa == '*':
			break
		aas.append(aa)
		pos += 3
	# If no stop found (orf_nt_len == -1), translate remaining codons to end
	if orf_nt_len == -1:
		pass  # already handled above
	return ''.join(aas)


# ORF disruption helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_orf_in_transcript(transcript_seq, orf_seq):
	"""
	Find the start position (0-based) of orf_seq in transcript_seq.
	Case-insensitive.  Returns -1 if not found.
	"""
	return transcript_seq.upper().find(orf_seq.upper())


def disrupt_orf_start(transcript_seq, orf_start_0, orf_len):
	"""
	Disrupt ORF start: convert first 3 nt of ORF to AAA, plus any in-frame
	ATG or CTG codons within the ORF body to AAA.
	Returns the modified full transcript sequence.
	"""
	seq_list = list(transcript_seq)
	orf_end = orf_start_0 + orf_len
	# Mutate start codon -> AAA
	for k in range(3):
		if orf_start_0 + k < len(seq_list):
			seq_list[orf_start_0 + k] = 'A'
	# Scan in-frame codons inside the ORF for ATG / CTG -> AAA
	pos = orf_start_0 + 3
	while pos + 2 < len(seq_list) and pos + 2 < orf_end:
		codon = ''.join(seq_list[pos:pos + 3]).upper()
		if codon in ('ATG', 'CTG'):
			for k in range(3):
				seq_list[pos + k] = 'A'
		pos += 3
	return ''.join(seq_list)


def disrupt_orf_stop(transcript_seq, orf_start_0, orf_len):
	"""
	Disrupt ORF stop: convert last 3 nt of the ORF to AAA.
	Returns the modified full transcript sequence.
	"""
	seq_list = list(transcript_seq)
	stop_start = orf_start_0 + orf_len - 3
	for k in range(3):
		if stop_start + k < len(seq_list):
			seq_list[stop_start + k] = 'A'
	return ''.join(seq_list)


# Transcript → Genomic coordinate mapping
# ═══════════════════════════════════════════════════════════════════════════

def build_transcript_to_genome_map(bed_df, transcript_id):
	exons = bed_df[bed_df["id"] == transcript_id].copy()
	if exons.empty:
		return None, None
	strand = exons["strand"].iloc[0]
	exons = exons.sort_values("start")
	genome_positions = []
	for _, row in exons.iterrows():
		for pos in range(row["start"], row["end"]):
			genome_positions.append(pos)
	if strand == "-":
		genome_positions = genome_positions[::-1]
	return genome_positions, strand


def map_bin_to_genomic_intervals(bin_start_tx, bin_end_tx, genome_map, strand):
	if bin_start_tx >= len(genome_map):
		return []
	bin_end_tx = min(bin_end_tx, len(genome_map))
	intervals = []
	current_gpos = genome_map[bin_start_tx]
	for tx_pos in range(bin_start_tx + 1, bin_end_tx):
		if genome_map[tx_pos] != genome_map[tx_pos - 1] + 1:
			if strand == "+":
				intervals.append((current_gpos, genome_map[tx_pos - 1] + 1))
			else:
				intervals.append((genome_map[tx_pos - 1], current_gpos + 1))
			current_gpos = genome_map[tx_pos]
	if strand == "+":
		intervals.append((current_gpos, genome_map[bin_end_tx - 1] + 1))
	else:
		intervals.append((genome_map[bin_end_tx - 1], current_gpos + 1))
	return intervals


# Integrated Gradients (tissue-aware)
# ═══════════════════════════════════════════════════════════════════════════

def integrated_gradients(model, features_tensor, tissue_id_tensor,
						 cds_mask_bins, device, n_steps=20):
	features = features_tensor.clone().to(device)
	tissue_ids = tissue_id_tensor.to(device)
	baseline = features.clone()
	baseline[:, :, :5] = 0.0
	cds_mask_bins_t = torch.tensor(cds_mask_bins, dtype=torch.float32, device=device)
	grads_accumulated = torch.zeros_like(features[:, :, :5])
	for step in range(n_steps + 1):
		alpha = step / n_steps
		interp = baseline.clone()
		interp[:, :, :5] = baseline[:, :, :5] + alpha * (features[:, :, :5] - baseline[:, :, :5])
		interp.requires_grad_(True)
		pred = model(interp, tissue_ids)
		target = (pred * cds_mask_bins_t.unsqueeze(0)).sum()
		target.backward()
		grads_accumulated += interp.grad[:, :, :5].detach()
		interp.requires_grad_(False)
	avg_grads = grads_accumulated / (n_steps + 1)
	diff = (features[:, :, :5] - baseline[:, :, :5]).detach()
	ig = (avg_grads * diff).squeeze(0)
	return ig.sum(dim=1).cpu().numpy()


# Mutation helpers
# ═══════════════════════════════════════════════════════════════════════════

def apply_mutation(seq, position, alt_allele, strand="+"):
	pos_0based = position - 1
	if pos_0based >= len(seq):
		print(f"Warning: Position {position} exceeds sequence length {len(seq)}")
		return seq
	if strand == "-":
		alt_allele_fwd = reverse_complement(alt_allele)
		print(f"  Minus strand variant: {alt_allele} (reverse) -> {alt_allele_fwd} (forward)")
	else:
		alt_allele_fwd = alt_allele
	seq_list = list(seq)
	seq_list[pos_0based] = alt_allele_fwd.upper()
	return ''.join(seq_list)


def predict_single_transcript(net, seq, rna_data, region_len, tissue_id_tensor,
							  device, output_raw_log=False):
	ohe = _encode_seq_fast(seq, region_len)
	features = np.concatenate([ohe, rna_data[:, np.newaxis]], axis=1)
	features_t = torch.from_numpy(features).unsqueeze(0).to(device)
	tissue_ids = tissue_id_tensor.unsqueeze(0).to(device)
	with torch.no_grad():
		pred_log = net(features_t, tissue_ids).squeeze(0).cpu().numpy()
	return pred_log if output_raw_log else inverse_log_transform(pred_log)


#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
	p = argparse.ArgumentParser("RiboTransPred — Tissue Predict & Attribute")
	p.add_argument("--checkpoint",  required=True)
	p.add_argument("--species",     required=True)
	p.add_argument("--tissue",      required=True,
				   help="Tissue name (must match tissue_vocab in checkpoint, "
						"or will use mean embedding for unseen tissues)")
	p.add_argument("--tracks_dir",  default="tracks")
	p.add_argument("--output_dir",  default="predictions")
	p.add_argument("--region_len",  type=int, default=6000)
	p.add_argument("--nBins",       type=int, default=1000)
	p.add_argument("--model_type",  default="PosTransModelTCNFiLMRef",
				   choices=["PosTransModelTCNFiLM", "PosTransModelTCNFiLMRef",
							"TransModelFiLM"])
	p.add_argument("--dropout",     type=float, default=0.3)
	p.add_argument("--tissue_emb_dim", type=int, default=64)
	p.add_argument("--biotype",     default="protein_coding",
				   choices=["protein_coding", "non_coding", "all"])
	p.add_argument("--psites",      action="store_true")
	p.add_argument("--batch_size",  type=int, default=16)
	p.add_argument("--ig_steps",    type=int, default=20)
	p.add_argument("--no_attribution", action="store_true")
	p.add_argument("--gpu",         type=int, default=0)
	p.add_argument("--output_raw_log", action="store_true")
	p.add_argument("--mutate",      type=str, default=None)
	p.add_argument("--mutate_only", action="store_true")
	p.add_argument("--orfs",        type=str, default=None,
				   help="ORF file for start/stop disruption analysis "
						"(format: orf_id<tab>transcript_id<tab>orf_sequence)")
	p.add_argument("--zero_threshold", type=float, default=0.0001)
	p.add_argument("--write_rnaseq", action="store_true", default=True)
	p.add_argument("--write_riboseq", action="store_true", default=True)
	return p.parse_args()


def main():
	args = parse_args()

	global ZERO_THRESHOLD
	ZERO_THRESHOLD = args.zero_threshold

	do_mutations = args.mutate is not None
	do_orfs = args.orfs is not None
	do_variant_analysis = do_mutations or do_orfs

	# ── Device ────────────────────────────────────────────────────────
	if args.gpu >= 0 and torch.cuda.is_available():
		device = torch.device(f"cuda:{args.gpu}")
		print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
	else:
		device = torch.device("cpu")
		print("Using CPU")

	# ── Load checkpoint and extract tissue vocab ──────────────────────
	print(f"Loading checkpoint: {args.checkpoint}")
	ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

	hparams = ckpt.get("hyper_parameters", {})
	tissue_vocab = hparams.get("tissue_vocab", None)
	num_tissues = hparams.get("num_tissues", None)
	tissue_emb_dim = hparams.get("tissue_emb_dim", args.tissue_emb_dim)

	if tissue_vocab is None:
		ckpt_dir = os.path.dirname(args.checkpoint)
		vocab_path = os.path.join(ckpt_dir, "tissue_vocab.json")
		if os.path.exists(vocab_path):
			with open(vocab_path) as f:
				tissue_vocab = json.load(f)
			num_tissues = len(tissue_vocab)
			print(f"  Loaded tissue_vocab from {vocab_path}")

	if tissue_vocab is None:
		print("ERROR: Cannot find tissue_vocab in checkpoint or tissue_vocab.json")
		sys.exit(1)

	if num_tissues is None:
		num_tissues = len(tissue_vocab)

	print(f"  Tissue vocabulary ({num_tissues} tissues):")
	for tname, tid in sorted(tissue_vocab.items(), key=lambda x: x[1]):
		print(f"    {tid}: {tname}")

	# ── Resolve tissue_id ─────────────────────────────────────────────
	if args.tissue in tissue_vocab:
		tissue_id = tissue_vocab[args.tissue]
		print(f"\n  Tissue '{args.tissue}' -> id {tissue_id} (known)")
	else:
		tissue_id = num_tissues
		print(f"\n  Tissue '{args.tissue}' not in training vocab -> "
			  f"using mean embedding (id {tissue_id})")

	tissue_id_tensor = torch.tensor(tissue_id, dtype=torch.long)

	# ── Load model ────────────────────────────────────────────────────
	ckpt_model_type = hparams.get("model_type", args.model_type)
	if ckpt_model_type != args.model_type:
		print(f"  Note: using model_type from checkpoint: {ckpt_model_type}")
		args.model_type = ckpt_model_type

	print(f"Loading model: {args.model_type}")
	ModelCls = getattr(models, args.model_type)
	net = ModelCls(
		num_genomic_features=1, target_length=args.region_len,
		nbins=args.nBins, num_tissues=num_tissues,
		tissue_emb_dim=tissue_emb_dim,
	)

	state = ckpt["state_dict"]
	net_state = {k[4:]: v for k, v in state.items() if k.startswith("net.")}
	net.load_state_dict(net_state)
	net.to(device)
	net.eval()
	print(f"  Model loaded ({sum(p.numel() for p in net.parameters()):,} params)")

	# ── Load mutations if specified ───────────────────────────────────
	mutations = None
	if do_mutations:
		print(f"\nLoading mutations from: {args.mutate}")
		mutations = load_mutations(args.mutate)
		print(f"  Found mutations for {len(mutations)} transcripts")

	# ── Load ORFs if specified ────────────────────────────────────────
	orfs = None
	if do_orfs:
		print(f"\nLoading ORFs from: {args.orfs}")
		orfs, n_total_orfs = load_orfs(args.orfs)
		print(f"  Found {n_total_orfs} ORFs across {len(orfs)} transcripts")

	# ── Set up output files ───────────────────────────────────────────
	os.makedirs(args.output_dir, exist_ok=True)

	stats_path = os.path.join(args.output_dir, "transcript_stats.tsv")
	stats_file = open(stats_path, "wt")
	stats_file.write(
		"transcript_id\tregion_type\tregion_length\t"
		"rna_region_mean\tribo_region_mean\twt_region_mean\t"
		"corr_pred_obs_nonlog\tcorr_pred_obs_log\t"
		"corr_rna_ribo_nonlog\tcorr_rna_ribo_log\n"
	)

	# mutations.tsv is generated when --mutate or --orfs is used
	mut_file = None
	pred_mut_file = None
	fasta_mut_file = None
	orfs_fasta_file = None
	if do_variant_analysis:
		mut_path = os.path.join(args.output_dir, "mutations.tsv")
		mut_file = open(mut_path, "wt")
		mut_file.write(
			"variant_id\ttranscript_id\ttranscript_len\tposition\tref_allele\talt_allele\t"
			"effect\tclassification\t"
			"region_type\tregion_length\trna_region_mean\tribo_region_mean\twt_region_mean\tmut_region_mean\t"
			"log2_fold_change\tabs_diff\t"
			"corr_pred_obs_nonlog\tcorr_pred_obs_log\t"
			"corr_rna_ribo_nonlog\tcorr_rna_ribo_log\t"
			"fiveutr_length\twt_fiveutr_mean\tmut_fiveutr_mean\t"
			"corr_pred_obs_fiveutr_nonlog\tcorr_pred_obs_fiveutr_log\t"
			"corr_rna_ribo_fiveutr_nonlog\tcorr_rna_ribo_fiveutr_log\t"
			"seq_context\tatg_orf_length\tctg_orf_length\n"
		)

	if do_mutations:
		mutations_dir = os.path.join(args.output_dir, "mutations")
		os.makedirs(mutations_dir, exist_ok=True)

		pred_mut_path = os.path.join(mutations_dir, "predictions.bedgraph")
		pred_mut_file = open(pred_mut_path, "wt")
		pred_mut_file.write("#chr\tstart\tend\tvariant_id\tbin_idx\tstrand\tpredicted_ribo\tlog2_fc\n")

		fasta_mut_path = os.path.join(mutations_dir, "sequences.fasta")
		fasta_mut_file = open(fasta_mut_path, "wt")

		orfs_fasta_path = os.path.join(mutations_dir, "orfs.fasta")
		orfs_fasta_file = open(orfs_fasta_path, "wt")

	# ── Load data ─────────────────────────────────────────────────────
	print(f"\nLoading coordinates for {args.species}...")
	df = load_coords(args.species)
	if df is None or df.empty:
		print("ERROR: No coordinates found"); sys.exit(1)

	print(f"Loading BED exon map for {args.species}...")
	bed_df = load_bed(args.species)
	if bed_df is None:
		print("ERROR: No BED file found"); sys.exit(1)

	if args.biotype == "protein_coding":
		mask = df["biotype"] == "protein_coding"
	elif args.biotype == "non_coding":
		mask = df["biotype"] != "protein_coding"
	else:
		mask = pd.Series(True, index=df.index)

	df_working = df.copy()
	df_working['orig_idx'] = df.index.values
	df_working = df_working[mask].copy()

	if args.mutate_only and do_variant_analysis:
		tx_with_variants = set()
		if mutations:
			tx_with_variants.update(mutations.keys())
		if orfs:
			tx_with_variants.update(orfs.keys())
		df_working = df_working[df_working["id"].isin(tx_with_variants)].reset_index(drop=True)
		print(f"  Focusing on {len(df_working)} transcripts with mutations/ORFs")
	else:
		df_working = df_working.reset_index(drop=True)
		print(f"  Transcripts: {len(df_working)} ({args.biotype})")

	df_filt = df_working.drop('orig_idx', axis=1)
	orig_indices = df_working['orig_idx'].values

	inp_rna_npy, inp_ribo_npy = _npy_paths(args.tracks_dir, args.species,
											args.tissue, args.region_len,
											args.nBins, args.psites)

	if not os.path.exists(inp_rna_npy):
		print(f"ERROR: RNA-seq npy not found: {inp_rna_npy}"); sys.exit(1)
	inp_rna_mm = np.load(inp_rna_npy, mmap_mode="r")
	print(f"  RNA-seq npy: {inp_rna_npy} ({inp_rna_mm.shape})")

	inp_ribo_mm = None
	if args.write_riboseq:
		if not os.path.exists(inp_ribo_npy):
			print(f"WARNING: Observed Ribo-seq npy not found: {inp_ribo_npy}")
		else:
			inp_ribo_mm = np.load(inp_ribo_npy, mmap_mode="r")
			print(f"  Observed Ribo-seq npy: {inp_ribo_npy} ({inp_ribo_mm.shape})")

	pool_k = max(1, args.region_len // args.nBins)

	# ── Bedgraph output files ─────────────────────────────────────────
	if not (args.mutate_only and df_filt.empty):
		pred_file = open(os.path.join(args.output_dir, "predictions.bedgraph"), "wt")
		rnaseq_file = open(os.path.join(args.output_dir, "rnaseq.bedgraph"), "wt") if args.write_rnaseq else None
		riboseq_file = open(os.path.join(args.output_dir, "riboseq.bedgraph"), "wt") if (args.write_riboseq and inp_ribo_mm is not None) else None
		attr_file = open(os.path.join(args.output_dir, "attributions.bedgraph"), "wt") if not args.no_attribution else None
	else:
		pred_file = rnaseq_file = riboseq_file = attr_file = None

	# ── Prediction loop ───────────────────────────────────────────────
	print(f"\nProcessing {len(df_filt)} transcripts (tissue={args.tissue}, "
		  f"id={tissue_id})...")
	t0 = time.time()
	n_done = 0
	n_skipped = 0
	n_orfs_processed = 0
	n_orfs_not_found = 0

	for i, (df_idx, row) in enumerate(df_filt.iterrows()):
		transcript_id = row["id"]
		seq = row["sequence"]
		region_annot = row.get("region", "")
		orig_idx = orig_indices[i]

		if not seq or all(c in "Nn" for c in seq[:args.region_len]):
			n_skipped += 1
			continue

		rna_data = inp_rna_mm[orig_idx].astype(np.float32)
		rna_data_nonlog = inverse_log_transform(rna_data)

		# Always load observed Ribo-seq when available
		ribo_data_nonlog = None
		if inp_ribo_mm is not None:
			ribo_data = inp_ribo_mm[orig_idx].astype(np.float32)
			ribo_data_nonlog = inverse_log_transform(ribo_data)

		genome_map, strand = build_transcript_to_genome_map(bed_df, transcript_id)
		if genome_map is None:
			n_skipped += 1
			continue

		chrom = row["chr"]
		transcript_len = len(genome_map)

		wt_pred = predict_single_transcript(
			net, seq, rna_data, args.region_len, tissue_id_tensor,
			device, args.output_raw_log
		)

		# ── Write RNA-seq bedgraph ────────────────────────────────
		if rnaseq_file is not None:
			rna_bins_out = _pool_to_bins(rna_data_nonlog, pool_k, args.nBins)
			for bin_idx in range(args.nBins):
				tx_start = bin_idx * pool_k
				tx_end = min(tx_start + pool_k, transcript_len)
				if tx_start >= transcript_len: break
				for g_start, g_end in map_bin_to_genomic_intervals(tx_start, tx_end, genome_map, strand):
					val = rna_bins_out[bin_idx] if bin_idx < len(rna_bins_out) else 0.0
					rnaseq_file.write(f"{chrom}\t{g_start}\t{g_end}\t{val:.6f}\t{strand}\t{transcript_id}\t{bin_idx}\n")

		# ── Write observed Ribo-seq bedgraph ──────────────────────
		if riboseq_file is not None and ribo_data_nonlog is not None:
			ribo_bins_out = _pool_to_bins(ribo_data_nonlog, pool_k, args.nBins)
			for bin_idx in range(args.nBins):
				tx_start = bin_idx * pool_k
				tx_end = min(tx_start + pool_k, transcript_len)
				if tx_start >= transcript_len: break
				for g_start, g_end in map_bin_to_genomic_intervals(tx_start, tx_end, genome_map, strand):
					val = ribo_bins_out[bin_idx] if bin_idx < len(ribo_bins_out) else 0.0
					riboseq_file.write(f"{chrom}\t{g_start}\t{g_end}\t{val:.6f}\t{strand}\t{transcript_id}\t{bin_idx}\n")

		# ── Write wildtype predictions bedgraph ───────────────────
		if pred_file is not None:
			for bin_idx in range(args.nBins):
				tx_start = bin_idx * pool_k
				tx_end = min(tx_start + pool_k, transcript_len)
				if tx_start >= transcript_len: break
				for g_start, g_end in map_bin_to_genomic_intervals(tx_start, tx_end, genome_map, strand):
					pred_file.write(f"{chrom}\t{g_start}\t{g_end}\t{wt_pred[bin_idx]:.6f}\t{strand}\t{transcript_id}\t{bin_idx}\n")

		# ── Attribution ───────────────────────────────────────────
		if attr_file is not None and region_annot:
			annot = region_annot[:args.region_len]
			if len(annot) < args.region_len:
				annot = annot + "0" * (args.region_len - len(annot))
			annot_arr = np.array([c == "2" for c in annot], dtype=np.float32)
			if pool_k > 1:
				L = (args.region_len // pool_k) * pool_k
				cds_mask_bins = annot_arr[:L].reshape(-1, pool_k).mean(axis=1) > 0.5
			else:
				cds_mask_bins = annot_arr > 0.5
			if cds_mask_bins.any():
				ohe = _encode_seq_fast(seq, args.region_len)
				features = np.concatenate([ohe, rna_data[:, np.newaxis]], axis=1)
				features_t = torch.from_numpy(features).unsqueeze(0).to(device)
				tid_t = tissue_id_tensor.unsqueeze(0).to(device)
				attributions = integrated_gradients(net, features_t, tid_t, cds_mask_bins, device, n_steps=args.ig_steps)
				for bp in range(min(args.region_len, transcript_len)):
					gpos = genome_map[bp]
					attr_file.write(f"{chrom}\t{gpos}\t{gpos + 1}\t{attributions[bp]:.6f}\t{strand}\t{transcript_id}\n")

		# ── Region statistics ───────────────────
		cds_mask = np.array([c == '2' for c in region_annot[:args.region_len]])
		has_cds = np.any(cds_mask)
		region_type = "CDS" if has_cds else "transcript"
		region_mask = cds_mask if has_cds else np.ones(args.region_len, dtype=bool)
		region_length = int(np.sum(region_mask))
		region_mask_bins = _pool_mask_to_bins(region_mask, pool_k, args.nBins)

		rna_bins = _pool_to_bins(rna_data_nonlog, pool_k, args.nBins)
		rna_region_mean = np.mean(rna_bins[region_mask_bins]) if any(region_mask_bins) else 0.0
		wt_region_mean = np.mean(wt_pred[region_mask_bins]) if any(region_mask_bins) else 0.0

		ribo_bins = _pool_to_bins(ribo_data_nonlog, pool_k, args.nBins) if ribo_data_nonlog is not None else None
		ribo_region_mean = np.mean(ribo_bins[region_mask_bins]) if (ribo_bins is not None and any(region_mask_bins)) else float('nan')

		# CDS correlations
		if ribo_bins is not None and any(region_mask_bins):
			corr_pred_obs_nonlog = calculate_correlation(wt_pred, ribo_bins, region_mask_bins, use_log=False)
			corr_pred_obs_log = calculate_correlation(wt_pred, ribo_bins, region_mask_bins, use_log=True)
			corr_rna_ribo_nonlog = calculate_correlation(rna_bins, ribo_bins, region_mask_bins, use_log=False)
			corr_rna_ribo_log = calculate_correlation(rna_bins, ribo_bins, region_mask_bins, use_log=True)
		else:
			corr_pred_obs_nonlog = corr_pred_obs_log = float('nan')
			corr_rna_ribo_nonlog = corr_rna_ribo_log = float('nan')

		# 5'UTR correlations
		fiveutr_mask = get_fiveutr_mask(region_annot, args.region_len)
		fiveutr_length = int(np.sum(fiveutr_mask))
		fiveutr_mask_bins = _pool_mask_to_bins(fiveutr_mask, pool_k, args.nBins)

		if ribo_bins is not None and any(fiveutr_mask_bins):
			corr_pred_obs_fiveutr_nonlog = calculate_correlation(wt_pred, ribo_bins, fiveutr_mask_bins, use_log=False)
			corr_pred_obs_fiveutr_log = calculate_correlation(wt_pred, ribo_bins, fiveutr_mask_bins, use_log=True)
			corr_rna_ribo_fiveutr_nonlog = calculate_correlation(rna_bins, ribo_bins, fiveutr_mask_bins, use_log=False)
			corr_rna_ribo_fiveutr_log = calculate_correlation(rna_bins, ribo_bins, fiveutr_mask_bins, use_log=True)
		else:
			corr_pred_obs_fiveutr_nonlog = corr_pred_obs_fiveutr_log = float('nan')
			corr_rna_ribo_fiveutr_nonlog = corr_rna_ribo_fiveutr_log = float('nan')

		# ── Transcript_stats.tsv ─────────────────────
		stats_file.write(
			f"{transcript_id}\t{region_type}\t{region_length}\t"
			f"{rna_region_mean:.6f}\t{ribo_region_mean:.6f}\t{wt_region_mean:.6f}\t"
			f"{corr_pred_obs_nonlog:.6f}\t{corr_pred_obs_log:.6f}\t"
			f"{corr_rna_ribo_nonlog:.6f}\t{corr_rna_ribo_log:.6f}\n"
		)

		# ── Helper: write one variant row to mutations.tsv ────────
		def _write_mut_row(variant_id, position, ref_allele, alt_allele_fwd,
						   effect, classification, mut_pred_vals,
						   mut_seq_for_context=None):
			"""Write a single row to mutations.tsv.  Returns log2_fc."""
			mut_region_mean = np.mean(mut_pred_vals[region_mask_bins]) if any(region_mask_bins) else 0.0
			eps_fc = 1e-10
			l2fc = np.log2((mut_region_mean + eps_fc) / (wt_region_mean + eps_fc)) if wt_region_mean > 0 else 0.0
			abs_diff = mut_region_mean - wt_region_mean

			wt_fiveutr_mean = np.mean(wt_pred[fiveutr_mask_bins]) if any(fiveutr_mask_bins) else 0.0
			mut_fiveutr_mean = np.mean(mut_pred_vals[fiveutr_mask_bins]) if any(fiveutr_mask_bins) else 0.0

			ctx_seq = mut_seq_for_context if mut_seq_for_context is not None else seq
			if position > 0:
				seq_context = get_seq_context(ctx_seq, position, alt_allele_fwd)
				atg_orf_len = get_atg_orf_length(ctx_seq, position)
				ctg_orf_len = get_ctg_orf_length(ctx_seq, position)
			else:
				seq_context = "NA"
				atg_orf_len = 0
				ctg_orf_len = 0

			mut_file.write(
				f"{variant_id}\t{transcript_id}\t{transcript_len}\t"
				f"{position}\t{ref_allele}\t{alt_allele_fwd}\t"
				f"{effect}\t{classification}\t"
				f"{region_type}\t{region_length}\t{rna_region_mean:.6f}\t"
				f"{ribo_region_mean:.6f}\t"
				f"{wt_region_mean:.6f}\t{mut_region_mean:.6f}\t"
				f"{l2fc:.6f}\t{abs_diff:.6f}\t"
				f"{corr_pred_obs_nonlog:.6f}\t{corr_pred_obs_log:.6f}\t"
				f"{corr_rna_ribo_nonlog:.6f}\t{corr_rna_ribo_log:.6f}\t"
				f"{fiveutr_length}\t"
				f"{wt_fiveutr_mean:.6f}\t{mut_fiveutr_mean:.6f}\t"
				f"{corr_pred_obs_fiveutr_nonlog:.6f}\t{corr_pred_obs_fiveutr_log:.6f}\t"
				f"{corr_rna_ribo_fiveutr_nonlog:.6f}\t{corr_rna_ribo_fiveutr_log:.6f}\t"
				f"{seq_context}\t{atg_orf_len}\t{ctg_orf_len}\n"
			)
			return l2fc, atg_orf_len, ctg_orf_len

		# ── Process point mutations (--mutate) ────────────────────
		if mutations and transcript_id in mutations:
			for mut in mutations[transcript_id]:
				position = mut['position']
				alt_allele = mut['alt_allele']
				ref_allele = seq[position - 1] if position <= len(seq) else "N"

				mut_seq = apply_mutation(seq, position, alt_allele, strand)
				alt_allele_fwd = reverse_complement(alt_allele) if strand == "-" else alt_allele

				mut_pred = predict_single_transcript(
					net, mut_seq, rna_data, args.region_len, tissue_id_tensor,
					device, args.output_raw_log
				)

				variant_id = f"{mut['variant_id']}--{alt_allele_fwd}--{position}--{mut['effect']}--{mut['classification']}"

				log2_fc, atg_orf_len, ctg_orf_len = _write_mut_row(
					variant_id, position, ref_allele, alt_allele_fwd,
					mut['effect'], mut['classification'], mut_pred,
					mut_seq_for_context=mut_seq
				)

				# Per-bin bedgraph for mutant
				if pred_mut_file is not None:
					for bin_idx in range(args.nBins):
						tx_start = bin_idx * pool_k
						tx_end = min(tx_start + pool_k, transcript_len)
						if tx_start >= transcript_len: break
						for g_start, g_end in map_bin_to_genomic_intervals(tx_start, tx_end, genome_map, strand):
							pred_mut_file.write(f"{chrom}\t{g_start}\t{g_end}\t{variant_id}\t{bin_idx}\t{strand}\t{mut_pred[bin_idx]:.6f}\t{log2_fc:.6f}\n")

				if fasta_mut_file is not None:
					fasta_mut_file.write(f">{variant_id} {log2_fc:.6f}\n")
					fasta_mut_file.write(mut_seq[:args.region_len].replace("N", "") + "\n")

				# Write translated ORF sequences if ATG or CTG ORF found
				if orfs_fasta_file is not None:
					if atg_orf_len > 0 or atg_orf_len == -1:
						atg_start = _find_codon_start(mut_seq, position, 'ATG')
						if atg_start >= 0:
							aa_seq = translate_orf(mut_seq, atg_start, atg_orf_len)
							if aa_seq:
								orfs_fasta_file.write(f">{variant_id}|ATG|pos={atg_start + 1}|nt_len={atg_orf_len} {log2_fc:.6f}\n")
								orfs_fasta_file.write(aa_seq + "\n")
					if ctg_orf_len > 0 or ctg_orf_len == -1:
						ctg_start = _find_codon_start(mut_seq, position, 'CTG')
						if ctg_start >= 0:
							aa_seq = translate_orf(mut_seq, ctg_start, ctg_orf_len)
							if aa_seq:
								orfs_fasta_file.write(f">{variant_id}|CTG|pos={ctg_start + 1}|nt_len={ctg_orf_len} {log2_fc:.6f}\n")
								orfs_fasta_file.write(aa_seq + "\n")

		# ── Process ORF disruptions (--orfs) ──────────────────────
		if orfs and transcript_id in orfs:
			for orf_entry in orfs[transcript_id]:
				orf_id = orf_entry['orf_id']
				orf_seq = orf_entry['orf_seq']

				orf_pos = find_orf_in_transcript(seq, orf_seq)
				if orf_pos < 0:
					n_orfs_not_found += 1
					continue
				orf_len = len(orf_seq)

				# Skip if ORF starts beyond the model's input window
				if orf_pos >= args.region_len:
					continue

				# (a) Disrupt start: first 3 nt → AAA + in-frame ATG/CTG → AAA
				start_mut_seq = disrupt_orf_start(seq, orf_pos, orf_len)
				start_pred = predict_single_transcript(
					net, start_mut_seq, rna_data, args.region_len,
					tissue_id_tensor, device, args.output_raw_log
				)
				start_variant_id = f"{orf_id}--start"
				ref_start = seq[orf_pos:orf_pos + 3] if orf_pos + 3 <= len(seq) else "NNN"
				_write_mut_row(
					start_variant_id,
					orf_pos + 1,            # 1-based position of ORF start
					ref_start, "AAA",
					"orf_start_disruption", "orf_analysis",
					start_pred,
					mut_seq_for_context=start_mut_seq
				)

				# (b) Disrupt stop: last 3 nt → AAA
				stop_mut_seq = disrupt_orf_stop(seq, orf_pos, orf_len)
				stop_pred = predict_single_transcript(
					net, stop_mut_seq, rna_data, args.region_len,
					tissue_id_tensor, device, args.output_raw_log
				)
				stop_variant_id = f"{orf_id}--stop"
				stop_pos_0 = orf_pos + orf_len - 3
				ref_stop = seq[stop_pos_0:stop_pos_0 + 3] if stop_pos_0 + 3 <= len(seq) else "NNN"
				_write_mut_row(
					stop_variant_id,
					stop_pos_0 + 1,         # 1-based position of stop codon
					ref_stop, "AAA",
					"orf_stop_disruption", "orf_analysis",
					stop_pred,
					mut_seq_for_context=stop_mut_seq
				)

				n_orfs_processed += 1

		n_done += 1
		if (n_done % 500) == 0:
			elapsed = time.time() - t0
			rate = n_done / elapsed if elapsed > 0 else 0
			eta = (len(df_filt) - n_done) / rate if rate > 0 else 0
			print(f"  {n_done}/{len(df_filt)} transcripts "
				  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

	# ── Cleanup ───────────────────────────────────────────────────────
	for f in [pred_file, rnaseq_file, riboseq_file, attr_file]:
		if f is not None: f.close()
	stats_file.close()
	if mut_file is not None:
		mut_file.close()
	if pred_mut_file is not None:
		pred_mut_file.close()
	if fasta_mut_file is not None:
		fasta_mut_file.close()
	if orfs_fasta_file is not None:
		orfs_fasta_file.close()

	elapsed = time.time() - t0
	print(f"\nDone! {n_done} transcripts processed, {n_skipped} skipped")
	print(f"  Time: {elapsed:.1f}s")
	print(f"  Tissue: {args.tissue} (id={tissue_id})")
	print(f"  Transcript stats: {stats_path}")

	if do_mutations:
		print(f"  Mutation results: {mut_path}")
		print(f"  Mutant predictions: {pred_mut_path}")
		print(f"  Mutant sequences: {fasta_mut_path}")
		print(f"  Translated ORFs: {orfs_fasta_path}")
	if do_orfs:
		print(f"  ORF disruption results: {mut_path}")
		print(f"  ORFs processed: {n_orfs_processed} "
			  f"({n_orfs_not_found} not found in transcript)")
	if not args.output_raw_log:
		print(f"  Note: Predictions are inverse-transformed from log (exp(x) - {EPS}) with values <= {ZERO_THRESHOLD} rounded to zero")


if __name__ == "__main__":

	main()

