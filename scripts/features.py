"""
Author: Jorge Ruiz Orera
Date: 2025-05-24
Description: Script to generate the input features needed for the transformer model.
"""

import os
import pathlib
import pandas as pd 
import argparse
import pyBigWig
import torch
import time
import math
import shutil
import numpy as np
from kipoiseq import Interval
from einops import rearrange
import pyfaidx
from collections import defaultdict
import random
import pyfaidx
from collections import defaultdict
import random


def read_fasta(file_path):
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
				header = line[1:]  # Remove the '>' character
			else:
				sequence.append(line)
		if header:  # Add the last sequence
			sequences[header] = ''.join(sequence)
	return sequences
	
def reverse_complement(dna):
	complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
				  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}

	# Reverse the DNA sequence
	reversed_dna = dna[::-1]

	# Complement each base in the reversed sequence
	reverse_complement_dna = ''.join(complement[base] for base in reversed_dna)

	return reverse_complement_dna


def extract_fasta_sequences(fasta_file, seq_len, intervals, strand, assembly, name, study, chromosome_length, id, **kwargs):
	sequence = ''
	for interval in intervals:

		trimmed_interval = Interval(interval.chrom,
										max(interval.start+1, 0),
										min(interval.end+1, chromosome_length))

		# pyfaidx wants a 1-based interval
		seq = str(pyfaidx.Fasta(fasta_file).get_seq(trimmed_interval.chrom,
													trimmed_interval.start,
													trimmed_interval.end-1).seq.upper())
		# Fill truncated values with N's
		pad_upstream = 'N' * max(-interval.start, 0)
		pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
		sequence += pad_upstream + seq + pad_downstream

	total_len = len(sequence) 

	# Trim or pad the concatenated sequence to match seq_len
	if strand == "+":
		if len(sequence) > seq_len:
			sequence = sequence[:seq_len]  # Trim to seq_len
		elif len(sequence) < seq_len:
			sequence += 'N' * (seq_len - len(sequence))  # Pad with N's downstream
	elif strand == "-":
		if len(sequence) > seq_len:
			sequence = sequence[-seq_len:]  # Trim the beginning to seq_len
		elif len(sequence) < seq_len:
			sequence = 'N' * (seq_len - len(sequence)) + sequence  # Pad with N's at the beginning

	#REVERSE COMPLEMENT
	if strand == '-':
		sequence = reverse_complement(sequence)

	return sequence,total_len


def generate_inputs(assembly, name, study, region_file, fasta_file, bw_file, tr_file, nBins, region_len, chromosome_lengths, make_region, rnacutoff):
	regions = pd.read_csv(region_file, sep='\t', names=['chr', 'start', 'end', 'strand','id'])

	rnaseq_file = f'data/{assembly}/{name}/{study}/{name}_{region_len}_log_rnaseq_final.pt'
	rnaseq_file_nolog = f'data/{assembly}/{name}/{study}/{name}_{region_len}_rnaseq_final.pt'

	if make_region == "yes":
		coords_names_file = f'data/{assembly}/exon_regions_output.txt'
		coords_names = open(coords_names_file, "w+")
	print(f'Feature path: {bw_file} \n Normalization status: log', flush=True)

	t0 = time.time()
	bw = pyBigWig.open(bw_file)
	print("Loaded bw", flush=True)

	sequences_dict = read_fasta(tr_file)
	print("Loaded sequences", flush=True)
	
	# Group regions by name
	grouped_regions = regions.groupby('id', sort=False)
	
	target = []
	excluded = []
	for id, group in grouped_regions:
		group_target = []
		all_coords = []
		intervals_gene = []
		for j, region in group.iterrows():
			chrom = region['chr']
			start = region['start']
			end = region['end']
			strand = region['strand']
			chromosome_length = int(chromosome_lengths[chrom])
			
			interval = Interval(chrom, start, end)
			trimmed_interval = Interval(interval.chrom,
										max(interval.start, 0),
										min(interval.end+1, chromosome_length))

			for pos in range(max(start, 0) , min((end+1), chromosome_length)):
				all_coords.append(pos)

			intervals_gene.append(interval)

			if chrom in bw.chroms():
				signals = np.array(bw.values(chrom, trimmed_interval.start, trimmed_interval.end)).astype(np.float32).tolist()
			else:
				signals = np.zeros(trimmed_interval.end - trimmed_interval.start, dtype=np.float32).tolist()
			
			pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
			pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
			tmp = pad_upstream + signals + pad_downstream
			arr = np.array(tmp).astype(np.float32)
			group_target.append(arr)
		
		# Combine all regions in the group into a single array
		group_target = np.concatenate(group_target, axis=0)
		gt_len = len(group_target)
		group_target = np.array(group_target).astype(np.float32)

		# Reverse the array if the strand is "-"
		if strand == "-":
			group_target = np.flip(group_target)
			all_coords = all_coords[::-1]

		#Filter by expression (before resizing)
		if np.mean(group_target < rnacutoff) > 0.7:
			group_target = np.zeros_like(group_target)
			excluded.append(id)

		# Resize group_target to region_len
		if group_target.shape[0] > region_len:
			group_target = group_target[:region_len]
		elif group_target.shape[0] < region_len:
			# Pad with zeros downstream until reaching region_len length
			pad_length = region_len - group_target.shape[0]
			group_target = np.pad(group_target, (0, pad_length), mode='constant', constant_values=0)

		#Only for coords, not needed for training
		reshaped_arr = group_target.reshape(-1, int(region_len / nBins))  # Reshape into bins: by default, three nucleotides
		averages = np.mean(reshaped_arr, axis=1)  # Calculate averages for each bin	

		target.append(group_target)
		
		if make_region == "yes":
			if tr_file == "":
				sequence, total_len = extract_fasta_sequences(fasta_file, region_len, intervals_gene, strand, assembly, name, study, chromosome_length, id)
			else:
				sequence = sequences_dict[id]
				total_len = len(sequence)
				if len(sequence) > region_len:
					sequence = sequence[:region_len]  # Trim to seq_len
				elif len(sequence) < region_len:
					sequence += 'N' * (region_len - len(sequence))  # Pad with N's downstream

			coords_names.write(chrom + "\t" + str(min(all_coords[0],all_coords[-1])) + "\t" + str(max(all_coords[0],all_coords[-1])) + "\t" + id + "\t" + str(total_len) + "\t" + str(gt_len) + "\t" + str(len(sequence)) +  "\t" + sequence + "\n")

	print("Scanned", flush=True)
	target = np.array(target).astype(np.float32)
	target = np.nan_to_num(target,0)
	target.tofile(rnaseq_file_nolog)
	target = np.log(target + 0.0001) # Apply a log transformation to the target values (log(x + 1)).
	target.tofile(rnaseq_file)
	print("Saved", flush=True)
	
	print(bw_file, time.time() - t0, flush=True)
	bw.close()
	if make_region == "yes":
		coords_names.close()

	print(f"Transcripts converted to zero due to low expression: {len(excluded)}")
	return excluded


def generate_outputs(assembly, name, study, region_file, fasta_file, bw_file, nBins, region_len, chromosome_lengths, ribocutoff, excluded):
	regions = pd.read_csv(region_file, sep='\t', names=['chr', 'start', 'end', 'strand', 'id'])

	riboseq_file = f'data/{assembly}/{name}/{study}/{name}_{region_len}_{nBins}_log_riboseq_final.pt'
	riboseq_file_nolog = f'data/{assembly}/{name}/{study}/{name}_{region_len}_{nBins}_riboseq_final.pt'

	print(f'Feature path: {bw_file} \n Normalization status: log', flush=True)

	t0 = time.time()
	bw = pyBigWig.open(bw_file)
	print("Loaded bw", flush=True)
	
	# Group regions
	grouped_regions = regions.groupby('id', sort=False)
	
	target = []
	for id, group in grouped_regions:
		group_target = []
		all_coords = []
		n = 0
		for j, region in group.iterrows():
			chrom = region['chr']
			start = region['start']
			end = region['end']
			strand = region['strand']
			chromosome_length = int(chromosome_lengths[chrom])
			
			interval = Interval(chrom, start, end)
			trimmed_interval = Interval(interval.chrom,
										max(interval.start, 0),
										min(interval.end + 1, chromosome_length))

			for pos in range(max(start, 0) , min((end+1), chromosome_length)):
				all_coords.append(pos)
			
			if chrom in bw.chroms():
				signals = np.array(bw.values(chrom, trimmed_interval.start, trimmed_interval.end)).astype(np.float32).tolist()
			else:
				signals = np.zeros(trimmed_interval.end - trimmed_interval.start, dtype=np.float32).tolist()
			
			pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
			pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
			tmp = pad_upstream + signals + pad_downstream
			
			arr = np.array(tmp).astype(np.float32)
			group_target.append(arr)
		
		# Combine all regions in the group into a single array
		group_target = np.concatenate(group_target, axis=0)
		group_target = np.array(group_target).astype(np.float32)

		# Reverse the array if the strand is "-"
		if strand == "-":
			group_target = np.flip(group_target)
			all_coords = all_coords[::-1]

		# Resize group_target to region_len
		if group_target.shape[0] > region_len:
			group_target = group_target[:region_len]
		elif group_target.shape[0] < region_len:
			# Pad with zeros downstream until reaching region_len length
			pad_length = region_len - group_target.shape[0]
			group_target = np.pad(group_target, (0, pad_length), mode='constant', constant_values=0)

		reshaped_arr = group_target.reshape(-1, int(region_len / nBins))  # Reshape into bins: by default, three nucleotides
		averages = np.mean(reshaped_arr, axis=1)  # Calculate averages for each bin

		if ribocutoff != 0: #Adjust ribo-seq noise to zero
			averages[averages < args.ribocutoff] = 0

		if id in excluded:
			averages = np.zeros_like(averages)
		
		target.append(averages)
	
	print("Scanned", flush=True)
	target = np.array(target).astype(np.float32)
	target = np.nan_to_num(target,0)
	torch.save(torch.Tensor(target), riboseq_file_nolog)
	target = np.log(target + 0.0001) # Apply a log transformation to the target values (log(x + 1)).
	torch.save(torch.Tensor(target), riboseq_file)
	print("Saved", flush=True)
	print(bw_file, time.time() - t0, flush=True)
	bw.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate features')
	parser.add_argument('--assembly', type=str, default='primate', help='genome reference')
	parser.add_argument('--celltype', type=str, default='heart', help='name of the organ')
	parser.add_argument('--study', type=str, default='human', help='name of the species')
	parser.add_argument('--region_len', type=int, default=6000, help='sequence length')
	parser.add_argument('--nBins', type=int, default=1000, help='number of bins')
	parser.add_argument('--make_region', type=str, default='yes', help='build an updated file with transcript region information, needed for training')
	parser.add_argument('--ribocutoff', type=int, default=0, help='cut-off to consider ribo-seq counts as non-translated (zero). Default = 0')
	parser.add_argument('--rnacutoff', type=int, default=5, help='cut-off to consider rna-seq counts as non-translated (zero). Default = 5')
	args = parser.parse_args()

	name = args.celltype
	study = args.study
	assembly = args.assembly

	print(name, flush=True)
	rna = f'data/{args.assembly}/{args.celltype}/{args.study}/rnaseq.normalized.bw'
	riboseq = f'data/{args.assembly}/{args.celltype}/{args.study}/riboseq.normalized.bw'

	region_file = f'data/{args.assembly}/exon_region.bed'
	transcripts = f'data/{args.assembly}/transcripts.fa'
		
	fasta_file = f'data/{args.assembly}/{args.assembly}.fa'
	chrom_file = f'data/{args.assembly}/{args.assembly}_chrom_lengths.txt'

	with open(chrom_file, "r") as file:
		chromosome_lengths = {line.split()[0]: int(line.split()[1]) for line in file}

	excluded = generate_inputs(assembly, name, study, region_file, fasta_file, rna, transcripts, args.nBins, args.region_len, chromosome_lengths, args.make_region, args.rnacutoff)
	generate_outputs(assembly, name, study, region_file, fasta_file, riboseq, args.nBins, args.region_len, chromosome_lengths, args.ribocutoff, excluded)
