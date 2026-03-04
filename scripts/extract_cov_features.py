#!/usr/bin/env python3
"""
Author: Jorge Ruiz Orera
Description: Batch processing script to generate .pt files from all .bw files in tracks directory.
"""

import os
import sys
import glob
import pandas as pd
import pyBigWig
import torch
import numpy as np
from kipoiseq import Interval
import pyfaidx
from collections import defaultdict
from pathlib import Path

# Configuration
REGION_LEN = int(sys.argv[1])
NBINS = int(sys.argv[2])
RNACUTOFF = float(sys.argv[3])
TRACKS_DIR = "tracks"
COORDS_DIR = "coordinates"
GENOME_DIR = "genomes"

def read_fasta(file_path):
	"""Read FASTA file into dictionary."""
	sequences = {}
	with open(file_path, 'r') as file:
		header = None
		sequence = []
		for line in file:
			line = line.strip()
			if line.startswith('>'):
				if header:
					sequences[header] = ''.join(sequence)
					sequence = []
				header = line[1:]
			else:
				sequence.append(line)
		if header:
			sequences[header] = ''.join(sequence)
	return sequences

def reverse_complement(dna):
	"""Get reverse complement of DNA sequence."""
	complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
				  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
	reversed_dna = dna[::-1]
	reverse_complement_dna = ''.join(complement[base] for base in reversed_dna)
	return reverse_complement_dna

def extract_fasta_sequences(fasta_file, seq_len, intervals, strand, chromosome_length):
	"""Extract sequences from FASTA file."""
	sequence = ''
	for interval in intervals:
		trimmed_interval = Interval(interval.chrom,
									max(interval.start+1, 0),
									min(interval.end+1, chromosome_length))

		seq = str(pyfaidx.Fasta(fasta_file).get_seq(str(trimmed_interval.chrom),
													trimmed_interval.start,
													trimmed_interval.end-1).seq.upper())
		
		pad_upstream = 'N' * max(-interval.start, 0)
		pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
		sequence += pad_upstream + seq + pad_downstream

	total_len = len(sequence)

	if strand == "+":
		if len(sequence) > seq_len:
			sequence = sequence[:seq_len]
		elif len(sequence) < seq_len:
			sequence += 'N' * (seq_len - len(sequence))
	elif strand == "-":
		if len(sequence) > seq_len:
			sequence = sequence[-seq_len:]
		elif len(sequence) < seq_len:
			sequence = 'N' * (seq_len - len(sequence)) + sequence

	if strand == '-':
		sequence = reverse_complement(sequence)

	return sequence, total_len

def get_species_tissue_from_path(bw_path):
	"""Extract species and tissue from .bw file path."""
	# Remove tracks/ prefix and .bw suffix
	rel_path = bw_path.replace(f"{TRACKS_DIR}/", "").replace(".bw", "")
	parts = rel_path.split("/")
	
	if len(parts) >= 2:
		species = parts[0]
		tissue = parts[1]
		# Remove species_tissue_ prefix from tissue if present
		if tissue.startswith(f"{species}_"):
			tissue = tissue.replace(f"{species}_", "", 1)
		# Remove _rna or _ribo suffix
		for suffix in ["_rna", "_ribo", ".psites"]:
			if tissue.endswith(suffix):
				tissue = tissue[:-len(suffix)]
		return species, tissue
	return None, None

def extract_chromosome_lengths(fasta_file):
	"""Extract chromosome lengths from FASTA file."""
	chromosome_lengths = {}
	with pyfaidx.Fasta(fasta_file) as fasta:
		for chrom in fasta.keys():
			chromosome_lengths[chrom] = len(fasta[chrom])
	return chromosome_lengths

def parse_cds_positions(gtf_file):
	"""
	Parse CDS positions from GTF file.
	Returns three dicts:
	- transcript_cds: {transcript_id: [(start, end), ...]}
	- transcript_start_codons: {transcript_id: [(start, end), ...]}
	- transcript_stop_codons: {transcript_id: [(start, end), ...]}
	"""
	print(f"  Parsing CDS, start_codon, and stop_codon positions from GTF file: {gtf_file}")
	
	transcript_cds = defaultdict(list)
	transcript_start_codons = defaultdict(list)
	transcript_stop_codons = defaultdict(list)
	
	with open(gtf_file, 'r') as f:
		for line in f:
			if line.startswith('#'):
				continue
				
			parts = line.strip().split('\t')
			if len(parts) < 9:
				continue
				
			feature = parts[2]
			start = int(parts[3])
			end = int(parts[4]) + 1
			attributes = parts[8]
			
			# Parse attributes
			attrs_dict = {}
			for attr in attributes.split(';'):
				attr = attr.strip()
				if ' ' in attr:
					key, value = attr.split(' ', 1)
					attrs_dict[key] = value.strip('"')
			
			# Get transcript_id
			transcript_id = attrs_dict.get('transcript_id')
			if not transcript_id:
				continue
			
			# Store based on feature type
			if feature == "CDS":
				transcript_cds[transcript_id].append((start, end))
			elif feature == "start_codon":
				transcript_start_codons[transcript_id].append((start, end))
			elif feature == "stop_codon":
				transcript_stop_codons[transcript_id].append((start, end))
	
	# Sort CDS regions by position for each transcript
	for transcript_id in transcript_cds:
		transcript_cds[transcript_id].sort(key=lambda x: x[0])
	
	# Sort start/stop codons
	for transcript_id in transcript_start_codons:
		transcript_start_codons[transcript_id].sort(key=lambda x: x[0])
	for transcript_id in transcript_stop_codons:
		transcript_stop_codons[transcript_id].sort(key=lambda x: x[0])
	
	print(f"    Found CDS for {len(transcript_cds)} transcripts")
	print(f"    Found start_codon for {len(transcript_start_codons)} transcripts")
	print(f"    Found stop_codon for {len(transcript_stop_codons)} transcripts")
	
	return transcript_cds, transcript_start_codons, transcript_stop_codons

def parse_gtf_to_bed(gtf_file, output_bed_file):
	"""
	Parse Ensembl GTF file to create a BED file with exon regions.
	GTF format: seqname source feature start end score strand frame attributes
	We'll extract exons for protein-coding transcripts.
	"""
	print(f"  Parsing GTF file: {gtf_file}")
	
	# Store exons by transcript
	transcript_exons = defaultdict(list)
	
	with open(gtf_file, 'r') as f:
		for line in f:
			if line.startswith('#'):
				continue
				
			parts = line.strip().split('\t')
			if len(parts) < 9:
				continue
				
			seqname = parts[0]
			feature = parts[2]
			start = int(parts[3]) - 1
			end = int(parts[4])
			strand = parts[6]
			attributes = parts[8]
			
			# Only process exons
			if feature != "exon":
				continue
			
			# Parse attributes
			attrs_dict = {}
			for attr in attributes.split(';'):
				attr = attr.strip()
				if ' ' in attr:
					key, value = attr.split(' ', 1)
					attrs_dict[key] = value.strip('"')
			
			# Get transcript_id
			transcript_id = attrs_dict.get('transcript_id')
			if not transcript_id:
				continue
			
			# Get transcript_biotype
			transcript_biotype = attrs_dict.get('transcript_biotype', 'unknown')
			
			# Store exon with transcript info
			transcript_exons[transcript_id].append({
				'chr': seqname,
				'start': start,
				'end': end,
				'strand': strand,
				'transcript_biotype': transcript_biotype
			})
	
	# Write BED file
	with open(output_bed_file, 'w') as bed_f:
		for transcript_id, exons in transcript_exons.items():
			# Sort exons by position
			exons.sort(key=lambda x: x['start'])
			
			# Get transcript info from first exon
			first_exon = exons[0]
			chrom = first_exon['chr']
			strand = first_exon['strand']
			transcript_biotype = first_exon['transcript_biotype']
			
			# Write each exon as separate line with same transcript_id
			for exon in exons:
				bed_f.write(f"{chrom}\t{exon['start']}\t{exon['end']}\t{strand}\t{transcript_id}\t{transcript_biotype}\n")
	
	print(f"  Created BED file with {len(transcript_exons)} transcripts")
	return output_bed_file

def create_cds_vector(transcript_id, all_coords, cds_dict, start_codon_dict, stop_codon_dict, 
					 chromosome_length, strand="+"):
	"""
	Create CDS vector for a transcript.
	0 = no CDS, 2 = CDS (including stop codon)
	Only mark CDS if transcript has BOTH start_codon and stop_codon annotations.
	Vector is trimmed to maximum 6000 nucleotides.
	"""
	# Check if transcript has both start and stop codon annotations
	has_start_codon = transcript_id in start_codon_dict and len(start_codon_dict[transcript_id]) > 0
	has_stop_codon = transcript_id in stop_codon_dict and len(stop_codon_dict[transcript_id]) > 0
	
	# If missing either start or stop codon, return all zeros
	if not (has_start_codon and has_stop_codon):
		if strand == "-" and all_coords:
			# For - strand, return reversed vector of zeros
			return "0" * min(len(all_coords), 6000)
		return "0" * min(len(all_coords), 6000) if all_coords else ""
	
	# Check if transcript has CDS regions
	if transcript_id not in cds_dict or not all_coords:
		if strand == "-" and all_coords:
			# For - strand, return reversed vector of zeros
			return "0" * min(len(all_coords), 6000)
		return "0" * min(len(all_coords), 6000) if all_coords else ""
	
	cds_regions = cds_dict[transcript_id]
	
	# Create a vector of zeros (trimmed to max 6000)
	vector_length = min(len(all_coords), 6000)
	cds_vector = [0] * vector_length
	
	# Only consider coordinates up to 6000
	limited_coords = all_coords[:vector_length]
	
	# Mark CDS positions
	for cds_start, cds_end in cds_regions:
		# Find positions that overlap with this CDS region
		for i, coord in enumerate(limited_coords):
			if cds_start <= coord < cds_end:
				cds_vector[i] = 2
	
	if strand == "+":
		# For + strand: find stop codon positions from GTF
		# Get stop codon region
		stop_codons = stop_codon_dict.get(transcript_id, [])
		for stop_start, stop_end in stop_codons:
			# Mark stop codon positions as CDS (2)
			for i, coord in enumerate(limited_coords):
				if stop_start <= coord < stop_end:
					cds_vector[i] = 2
					
	elif strand == "-":
		# For - strand: find stop codon positions from GTF
		# Get stop codon region
		stop_codons = stop_codon_dict.get(transcript_id, [])
		for stop_start, stop_end in stop_codons:
			# Mark stop codon positions as CDS (2)
			for i, coord in enumerate(limited_coords):
				if stop_start <= coord < stop_end:
					cds_vector[i] = 2
	
	# Convert to string representation
	cds_string = ''.join(str(x) for x in cds_vector)
	
	# Reverse the string for - strand to match transcript orientation
	if strand == "-":
		cds_string = cds_string[::-1]
	
	return cds_string

def generate_coordinates(species, bw_paths):
	"""Generate coordinates file for a species if not already done."""
	coords_file = os.path.join(COORDS_DIR, f"{species}_coordinates.txt")
	
	# Skip if coordinates already exist
	if os.path.exists(coords_file):
		print(f"  Coordinates file already exists for {species}")
		return coords_file
	
	print(f"  Generating coordinates file for {species}")
	
	# Find required files
	gtf_file = os.path.join(COORDS_DIR, f"{species}.gtf")
	fasta_file = os.path.join(GENOME_DIR, f"{species}.fa")
	transcripts_file = os.path.join(GENOME_DIR, f"{species}.transcripts.fa")
	
	# Check if required files exist
	for f in [gtf_file, fasta_file]:
		if not os.path.exists(f):
			print(f"  ERROR: Required file not found: {f}")
			return None
	
	# Parse GTF to create BED file
	bed_file = os.path.join(COORDS_DIR, f"{species}.bed")
	parse_gtf_to_bed(gtf_file, bed_file)
	
	# Parse CDS and codon positions from GTF
	cds_dict, start_codon_dict, stop_codon_dict = parse_cds_positions(gtf_file)
	
	# Extract chromosome lengths from FASTA
	print(f"  Extracting chromosome lengths from {fasta_file}")
	chromosome_lengths = extract_chromosome_lengths(fasta_file)
	
	# Save chromosome lengths to file
	chrom_lengths_file = os.path.join(COORDS_DIR, f"{species}_chrom_lengths.txt")
	with open(chrom_lengths_file, 'w') as f:
		for chrom, length in chromosome_lengths.items():
			f.write(f"{chrom}\t{length}\n")
	
	# Read regions from BED file
	regions = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end', 'strand', 'id', 'transcript_biotype'], dtype={'chr': str})
	
	# Read transcripts if available
	if os.path.exists(transcripts_file):
		sequences_dict = read_fasta(transcripts_file)
	else:
		sequences_dict = {}
		print(f"  Warning: transcripts.fa not found for {species}, extracting from genome")
	
	# Group regions by transcript ID
	grouped_regions = regions.groupby('id', sort=False)
	
	# Count transcripts with complete CDS information
	complete_cds_count = 0
	incomplete_cds_count = 0
	
	# Generate coordinates
	with open(coords_file, 'w') as out_f:
		for transcript_id, group in grouped_regions:
			intervals_gene = []
			all_coords = []
			
			# Get transcript biotype and strand from first region
			transcript_biotype = group.iloc[0]['transcript_biotype']
			strand = group.iloc[0]['strand']  # Get strand information
			
			for _, region in group.iterrows():
				chrom = region['chr']
				start = region['start'] + 1
				end = region['end']
				chromosome_length = int(chromosome_lengths.get(chrom, 1e9))
				
				interval = Interval(chrom, start, end)
				intervals_gene.append(interval)
				
				for pos in range(max(start, 0), min((end+1), chromosome_length)):
					all_coords.append(pos)
			
			# Get CDS vector with strand information (max 6000 nucleotides)
			if all_coords:
				cds_vector = create_cds_vector(transcript_id, all_coords, cds_dict, 
											  start_codon_dict, stop_codon_dict,
											  chromosome_lengths.get(chrom, 1e9), strand)
				# Count transcripts with complete CDS info
				if "2" in cds_vector:
					complete_cds_count += 1
				else:
					incomplete_cds_count += 1
			else:
				cds_vector = ""
				incomplete_cds_count += 1
				
			# Get sequence
			if transcript_id in sequences_dict:
				sequence = sequences_dict[transcript_id]
				total_len = len(sequence)
				if len(sequence) > REGION_LEN:
					sequence = sequence[:REGION_LEN]
				elif len(sequence) < REGION_LEN:
					sequence += 'N' * (REGION_LEN - len(sequence))
			else:
				if intervals_gene:
					chrom = intervals_gene[0].chrom
					chromosome_length = int(chromosome_lengths.get(chrom, 1e9))
					sequence, total_len = extract_fasta_sequences(
						fasta_file, REGION_LEN, intervals_gene, 
						strand, chromosome_length
					)
				else:
					sequence = 'N' * REGION_LEN
					total_len = 0
			
			gt_len = len(all_coords) if all_coords else 0
			chrom = region['chr'] if not intervals_gene else intervals_gene[0].chrom
			
			# Write to coordinates file with CDS vector as additional column
			out_f.write(f"{chrom}\t{min(all_coords) if all_coords else 0}\t"
					   f"{max(all_coords) if all_coords else 0}\t"
					   f"{transcript_id}\t{total_len}\t{transcript_biotype}\t"
					   f"{len(sequence)}\t{sequence}\t{cds_vector}\n")
	
	print(f"  Generated coordinates file with {len(grouped_regions)} transcripts")
	print(f"    Transcripts with complete CDS (start+stop codon): {complete_cds_count}")
	print(f"    Transcripts without complete CDS: {incomplete_cds_count}")
	return coords_file

def process_rnaseq(bw_file, species, tissue, bed_file, fasta_file, chrom_lengths_file):
	"""Process RNA-seq .bw file."""
	print(f"  Processing RNA-seq: {os.path.basename(bw_file)}")
	
	# Read chromosome lengths
	chromosome_lengths = {}
	with open(chrom_lengths_file, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) >= 2:
				chromosome_lengths[parts[0]] = int(parts[1])
	
	# Read regions from BED file
	regions = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end', 'strand', 'id', 'transcript_biotype'], dtype={'chr': str})
	
	# Open BigWig file
	bw = pyBigWig.open(bw_file)
	
	# Group regions by transcript ID
	grouped_regions = regions.groupby('id', sort=False)
	
	target = []
	excluded_transcripts = []
	
	for transcript_id, group in grouped_regions:
		group_target = []
		
		for _, region in group.iterrows():
			chrom = region['chr']
			start = region['start'] + 1
			end = region['end']
			strand = region['strand']
			chromosome_length = int(chromosome_lengths.get(chrom, 1e9))
			
			interval = Interval(chrom, start, end)
			trimmed_interval = Interval(interval.chrom,
										max(interval.start, 0),
										min(interval.end+1, chromosome_length))
			
			if chrom in bw.chroms():
				signals = np.array(bw.values(chrom, trimmed_interval.start, 
											trimmed_interval.end)).astype(np.float32).tolist()
			else:
				signals = np.zeros(trimmed_interval.end - trimmed_interval.start, 
								  dtype=np.float32).tolist()
			
			pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
			pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
			tmp = pad_upstream + signals + pad_downstream
			arr = np.array(tmp).astype(np.float32)
			group_target.append(arr)
		
		# Combine regions
		if group_target:
			group_target = np.concatenate(group_target, axis=0)
		else:
			group_target = np.array([], dtype=np.float32)
		
		group_target = np.array(group_target).astype(np.float32)
		
		# Reverse if negative strand
		if strand == "-":
			group_target = np.flip(group_target)
		
		# Filter by expression
		if group_target.size > 0 and np.mean(group_target < RNACUTOFF) > 0.7:
			group_target = np.zeros_like(group_target)
			excluded_transcripts.append(transcript_id)
		
		# Resize to REGION_LEN
		if group_target.shape[0] > REGION_LEN:
			group_target = group_target[:REGION_LEN]
		elif group_target.shape[0] < REGION_LEN:
			pad_length = REGION_LEN - group_target.shape[0]
			group_target = np.pad(group_target, (0, pad_length), mode='constant', constant_values=0)
		
		target.append(group_target)
	
	# Convert to numpy array
	target = np.array(target).astype(np.float32)
	target = np.nan_to_num(target, 0)
	
	# Save files
	base_name = os.path.basename(bw_file).replace('.bw', '')
	output_dir = os.path.dirname(bw_file)
	
	# Non-log version
	nolog_file = os.path.join(output_dir, f"{base_name}_{REGION_LEN}_rnaseq_final.pt")
	torch.save(torch.Tensor(target), nolog_file)
	
	# Log version
	log_file = os.path.join(output_dir, f"{base_name}_{REGION_LEN}_log_rnaseq_final.pt")
	target_log = np.log(target + 0.0001)
	torch.save(torch.Tensor(target_log), log_file)
	
	bw.close()
	
	print(f"    Generated: {os.path.basename(nolog_file)}")
	print(f"    Generated: {os.path.basename(log_file)}")
	print(f"    Excluded {len(excluded_transcripts)} transcripts due to low expression")
	
	return excluded_transcripts

def process_riboseq(bw_file, species, tissue, bed_file, fasta_file, chrom_lengths_file, excluded_transcripts):
	"""Process Ribo-seq or P-sites .bw file."""
	is_psites = "psites" in bw_file
	seq_type = "P-sites" if is_psites else "Ribo-seq"
	print(f"  Processing {seq_type}: {os.path.basename(bw_file)}")
	
	# Read chromosome lengths
	chromosome_lengths = {}
	with open(chrom_lengths_file, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) >= 2:
				chromosome_lengths[parts[0]] = int(parts[1])
	
	# Read regions from BED file
	regions = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end', 'strand', 'id', 'transcript_biotype'], dtype={'chr': str}) 
	
	# Open BigWig file
	bw = pyBigWig.open(bw_file)
	
	# Group regions by transcript ID
	grouped_regions = regions.groupby('id', sort=False)
	
	target = []
	
	for transcript_id, group in grouped_regions:
		group_target = []
		
		for _, region in group.iterrows():
			chrom = region['chr']
			start = region['start'] + 1
			end = region['end']
			strand = region['strand']
			chromosome_length = int(chromosome_lengths.get(chrom, 1e9))
			
			interval = Interval(chrom, start, end)
			trimmed_interval = Interval(interval.chrom,
										max(interval.start, 0),
										min(interval.end+1, chromosome_length))
			
			if chrom in bw.chroms():
				signals = np.array(bw.values(chrom, trimmed_interval.start,
											trimmed_interval.end)).astype(np.float32).tolist()
			else:
				signals = np.zeros(trimmed_interval.end - trimmed_interval.start,
								  dtype=np.float32).tolist()
			
			pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
			pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
			tmp = pad_upstream + signals + pad_downstream
			arr = np.array(tmp).astype(np.float32)
			group_target.append(arr)
		
		# Combine regions
		if group_target:
			group_target = np.concatenate(group_target, axis=0)
		else:
			group_target = np.array([], dtype=np.float32)
		
		group_target = np.array(group_target).astype(np.float32)
		
		# Reverse if negative strand
		if strand == "-":
			group_target = np.flip(group_target)
		
		# Resize to REGION_LEN
		if group_target.shape[0] > REGION_LEN:
			group_target = group_target[:REGION_LEN]
		elif group_target.shape[0] < REGION_LEN:
			pad_length = REGION_LEN - group_target.shape[0]
			group_target = np.pad(group_target, (0, pad_length), mode='constant', constant_values=0)
		
		# Bin the data
		reshaped_arr = group_target.reshape(-1, int(REGION_LEN / NBINS))
		averages = np.mean(reshaped_arr, axis=1)
		
		# Set to zero if excluded from RNA-seq
		if transcript_id in excluded_transcripts:
			averages = np.zeros_like(averages)
		
		target.append(averages)
	
	# Convert to numpy array
	target = np.array(target).astype(np.float32)
	target = np.nan_to_num(target, 0)
	
	# Save files
	base_name = os.path.basename(bw_file).replace('.bw', '')
	output_dir = os.path.dirname(bw_file)
	
	# Non-log version
	nolog_file = os.path.join(output_dir, f"{base_name}_{REGION_LEN}_{NBINS}_riboseq_final.pt")
	torch.save(torch.Tensor(target), nolog_file)
	
	# Log version
	log_file = os.path.join(output_dir, f"{base_name}_{REGION_LEN}_{NBINS}_log_riboseq_final.pt")
	target_log = np.log(target + 0.0001)
	torch.save(torch.Tensor(target_log), log_file)
	
	bw.close()
	
	print(f"    Generated: {os.path.basename(nolog_file)}")
	print(f"    Generated: {os.path.basename(log_file)}")
	
	return True

def main():
	"""Main function to process all .bw files."""
	print("=" * 60)
	print("Batch Processing of .bw files to .pt files")
	print("=" * 60)
	
	# Create coordinates directory
	os.makedirs(COORDS_DIR, exist_ok=True)
	
	# Find all .bw files
	bw_files = sorted(glob.glob(os.path.join(TRACKS_DIR, "**", "*.bw"), recursive=True))
	
	if not bw_files:
		print(f"ERROR: No .bw files found in {TRACKS_DIR}")
		return
	
	print(f"Found {len(bw_files)} .bw files")
	print()
	
	# Track processing by species
	species_track = {}
	
	# First pass: Identify all species and generate coordinates
	print("Generating coordinates files...")
	for bw_file in bw_files:
		species, tissue = get_species_tissue_from_path(bw_file)
		if species and species not in species_track:
			species_track[species] = {
				'tissues': set(),
				'coords_generated': False
			}
			
			# Generate coordinates for this species
			coords_file = generate_coordinates(species, [bw_file])
			species_track[species]['coords_generated'] = (coords_file is not None)
		
		if species and tissue:
			species_track[species]['tissues'].add(tissue)
	
	print()
	
	# Second pass: Process files by species and tissue
	processed_count = 0
	error_count = 0
	
	# Organize files by species and tissue
	file_dict = defaultdict(lambda: defaultdict(list))
	
	for bw_file in bw_files:
		species, tissue = get_species_tissue_from_path(bw_file)
		if species and tissue:
			file_dict[species][tissue].append(bw_file)
	
	# Process each species
	for species, tissues in file_dict.items():
		print(f"Processing species: {species}")
		print(f"-" * 40)
		
		# Check if coordinates were generated
		if not species_track.get(species, {}).get('coords_generated', False):
			print(f"  ERROR: Coordinates not generated for {species}, skipping")
			error_count += len([f for t in tissues.values() for f in t])
			continue
		
		# Get required files for this species
		gtf_file = os.path.join(COORDS_DIR, f"{species}.gtf")
		fasta_file = os.path.join(GENOME_DIR, f"{species}.fa")
		bed_file = os.path.join(COORDS_DIR, f"{species}.bed")
		chrom_lengths_file = os.path.join(COORDS_DIR, f"{species}_chrom_lengths.txt")
		
		# Check if required files exist
		missing_files = []
		for f in [gtf_file, fasta_file, bed_file, chrom_lengths_file]:
			if not os.path.exists(f):
				missing_files.append(f)
		
		if missing_files:
			print(f"  ERROR: Missing required files: {missing_files}")
			error_count += len([f for t in tissues.values() for f in t])
			continue
		
		# Process each tissue
		for tissue, tissue_files in tissues.items():
			print(f"  Tissue: {tissue}")
			
			# Separate RNA-seq and Ribo-seq files
			rna_files = [f for f in tissue_files if "_rna.bw" in f]
			ribo_files = [f for f in tissue_files if "_ribo.bw" in f and "psites" not in f]
			psites_files = [f for f in tissue_files if "psites.bw" in f]
			
			# Process RNA-seq first to get excluded transcripts
			excluded_transcripts = []
			if rna_files:
				for rna_file in rna_files:
					try:
						excluded = process_rnaseq(rna_file, species, tissue, 
												 bed_file, fasta_file, chrom_lengths_file)
						excluded_transcripts.extend(excluded)
						processed_count += 1
					except Exception as e:
						print(f"    ERROR processing {os.path.basename(rna_file)}: {e}")
						error_count += 1
			
			# Process Ribo-seq files
			for ribo_file in ribo_files:
				try:
					process_riboseq(ribo_file, species, tissue, 
								   bed_file, fasta_file, chrom_lengths_file, excluded_transcripts)
					processed_count += 1
				except Exception as e:
					print(f"    ERROR processing {os.path.basename(ribo_file)}: {e}")
					error_count += 1
			
			# Process P-sites files
			for psites_file in psites_files:
				try:
					process_riboseq(psites_file, species, tissue,
								   bed_file, fasta_file, chrom_lengths_file, excluded_transcripts)
					processed_count += 1
				except Exception as e:
					print(f"    ERROR processing {os.path.basename(psites_file)}: {e}")
					error_count += 1
			
			print()
	
	# Summary
	print("=" * 60)
	print("PROCESSING SUMMARY")
	print("=" * 60)
	print(f"Total .bw files found: {len(bw_files)}")
	print(f"Successfully processed: {processed_count}")
	print(f"Errors: {error_count}")
	print()
	print(f"Files in {COORDS_DIR}/:")
	for species in species_track.keys():
		coords_file = os.path.join(COORDS_DIR, f"{species}_coordinates.txt")
		bed_file = os.path.join(COORDS_DIR, f"{species}.bed")
		chrom_file = os.path.join(COORDS_DIR, f"{species}_chrom_lengths.txt")
		
		for fname, fpath in [("coordinates", coords_file), ("BED", bed_file), ("chrom lengths", chrom_file)]:
			if os.path.exists(fpath):
				size = os.path.getsize(fpath) / (1024*1024)
				print(f"  {os.path.basename(fpath)} ({size:.1f} MB)")
	
	print()
	print("Output structure in tracks/:")
	print("  species/")
	print("  └── tissue/")
	print("      ├── species_tissue_rna.bw")
	print(f"      ├── species_tissue_rna_{REGION_LEN}_log_rnaseq_final.pt")
	print(f"      ├── species_tissue_rna_{REGION_LEN}_rnaseq_final.pt")
	print("      ├── species_tissue_ribo.bw")
	print(f"      ├── species_tissue_ribo_{REGION_LEN}_{NBINS}_log_riboseq_final.pt")
	print(f"      ├── species_tissue_ribo_{REGION_LEN}_{NBINS}_riboseq_final.pt")
	print("      ├── species_tissue_ribo.psites.bw")
	print(f"      ├── species_tissue_ribo.psites_{REGION_LEN}_{NBINS}_log_riboseq_final.pt")
	print(f"      └── species_tissue_ribo.psites_{REGION_LEN}_{NBINS}_riboseq_final.pt")
	print()
	print("Processing complete")

if __name__ == "__main__":
	main()
