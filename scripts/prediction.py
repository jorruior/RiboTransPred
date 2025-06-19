#!/usr/bin/env python3
import sys
import torch
import argparse
import numpy as np
import os
import pandas as pd
import pyBigWig
import multiprocessing
from multiprocessing import Pool, Manager, Lock
import time
import warnings
from scipy.stats import pearsonr, ConstantInputWarning
from kipoiseq import Interval
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn as nn
from torchmetrics import Metric
import math
sys.path.append("../scripts")
import model.models as models

# Suppress warnings
warnings.filterwarnings("ignore", category=ConstantInputWarning)

class MeanPearsonCorrCoefPerChannel(Metric):
	is_differentiable = False
	full_state_update = False
	higher_is_better = True
	
	def __init__(self, n_channels, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("product", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("true", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("true_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("pred", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("pred_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("count", default=torch.zeros(n_channels), dist_reduce_fx="sum")

	def update(self, preds, target):
		assert preds.shape == target.shape
		self.product += torch.sum(preds * target, dim=0)
		self.true += torch.sum(target, dim=0)
		self.true_squared += torch.sum(torch.square(target), dim=0)
		self.pred += torch.sum(preds, dim=0)
		self.pred_squared += torch.sum(torch.square(preds), dim=0)
		self.count += torch.sum(torch.ones_like(target), dim=0)

	def compute(self):
		if self.count != 0:
			true_mean = self.true / self.count
		else:
			true_mean = 0
		pred_mean = self.pred / self.count
		covariance = (self.product - true_mean * self.pred - pred_mean * self.true 
					 + self.count * true_mean * pred_mean)
		true_var = self.true_squared - self.count * torch.square(true_mean)
		pred_var = self.pred_squared - self.count * torch.square(pred_mean)
		tp_var = torch.sqrt(true_var + 1e-8) * torch.sqrt(pred_var + 1e-8)
		return covariance / tp_var

class TrainModule(LightningModule):
	def __init__(self, nbins=10000, sequence_length=30000, model_name='TransModel'):
		super().__init__()
		self.save_hyperparameters()
		self.model = self._get_model()
		self.criterion = nn.MSELoss()
		self.pcc = MeanPearsonCorrCoefPerChannel(1)
		
	def forward(self, x):
		return self.model(x)
	
	def _get_model(self):
		ModelClass = getattr(models, self.hparams.model_name)
		return ModelClass(
			num_genomic_features=1,
			mid_hidden=512,
			target_length=self.hparams.sequence_length,
			nbins=self.hparams.nbins
		)
	
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		return optimizer
	
	def configure_precision(self):
		return {'precision': 16}  # Use fp16 precision

def one_hot_encode(sequence):
	mapping = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4}
	seq_emb = np.zeros((len(sequence), 5))
	for i, ch in enumerate(sequence):
		seq_emb[i, mapping.get(ch, 4)] = 1
	return seq_emb.astype(np.float32)
	
def calculate_cor(d1, d2, d3, binsize):
	binsize = int(binsize)
	def bin_average(data, binsize):
		data = np.asarray(data)
		n = len(data)
		binned = []
		for i in range(0, n, binsize):
			binned.append(np.mean(data[i:i+binsize]))
		return np.array(binned)

	# Bin the data
	d1_binned = bin_average(d1, binsize)
	d2_binned = bin_average(d2, binsize)
	d3_binned = bin_average(d3, binsize)

	# Ensure equal lengths
	min_len = min(len(d1_binned), len(d2_binned), len(d3_binned))
	d1_binned = d1_binned[:min_len]
	d2_binned = d2_binned[:min_len]
	d3_binned = d3_binned[:min_len]

	# Calculate regular correlations
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=ConstantInputWarning)
		try:
			cor_d1_d3, _ = pearsonr(d1_binned, d3_binned)
		except:
			cor_d1_d3 = np.nan

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=ConstantInputWarning)
		try:
			cor_d1_d2, _ = pearsonr(d1_binned, d2_binned)
		except:
			cor_d1_d2 = np.nan

	# Calculate log10 correlations (with small offset)
	offset = 0.0001
	d1_log = np.log10(d1_binned + offset)
	d2_log = np.log10(d2_binned + offset)
	d3_log = np.log10(d3_binned + offset)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=ConstantInputWarning)
		try:
			log_cor_d1_d3, _ = pearsonr(d1_log, d3_log)
		except:
			log_cor_d1_d3 = np.nan

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=ConstantInputWarning)
		try:
			log_cor_d1_d2, _ = pearsonr(d1_log, d2_log)
		except:
			log_cor_d1_d2 = np.nan

	return cor_d1_d3, cor_d1_d2, log_cor_d1_d3, log_cor_d1_d2

def generate_inputs(regions, bw_file, bw_file_ribo, region_len, chromosome_lengths, gene_name):
	regions = pd.read_csv(regions, sep='\t', names=['chr','start','end','strand','id'])
	regions = regions[regions['id'] == gene_name.split("--")[0]]
	
	bw = pyBigWig.open(bw_file)
	if bw_file_ribo != "none":
		bw_ribo = pyBigWig.open(bw_file_ribo)
	
	group_target = []
	group_target2 = []
	all_coords = []
	intervals_gene = []
	
	for _, region in regions.iterrows():
		chrom = region['chr']
		start = region['start']
		end = region['end']
		strand = region['strand']
		chromosome_length = int(chromosome_lengths[chrom])
		
		interval = Interval(chrom, start, end)
		trimmed_interval = Interval(interval.chrom,
								  max(interval.start, 0),
								  min(interval.end+1, chromosome_length))

		# Collect coordinates
		for pos in range(max(start, 0), min((end+1), chromosome_length)):
			all_coords.append(pos)
			
		intervals_gene.append(interval)

		# Process RNA signals
		if chrom in bw.chroms():
			signals = np.array(bw.values(chrom, trimmed_interval.start, trimmed_interval.end), 
							 dtype=np.float32).tolist()
		else:
			signals = np.zeros(trimmed_interval.end - trimmed_interval.start, 
							 dtype=np.float32).tolist()
		
		pad_upstream = np.array([0] * max(-interval.start, 0), dtype=np.float32).tolist()
		pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0), 
								dtype=np.float32).tolist()
		arr = np.array(pad_upstream + signals + pad_downstream, dtype=np.float32)
		group_target.append(arr)

		# Process Ribo signals if available
		if bw_file_ribo != "none":
			if chrom in bw_ribo.chroms():
				signals = np.array(bw_ribo.values(chrom, trimmed_interval.start, trimmed_interval.end),
								 dtype=np.float32).tolist()
			else:
				signals = np.zeros(trimmed_interval.end - trimmed_interval.start,
								 dtype=np.float32).tolist()
			
			arr = np.array(pad_upstream + signals + pad_downstream, dtype=np.float32)
			group_target2.append(arr)
	
	# Combine all regions
	group_target = np.concatenate(group_target, axis=0).astype(np.float32)
	if bw_file_ribo != "none":
		group_target2 = np.concatenate(group_target2, axis=0).astype(np.float32)

	# Handle strand orientation
	if strand == "-":
		group_target = np.flip(group_target)
		all_coords = all_coords[::-1]
		if bw_file_ribo != "none":
			group_target2 = np.flip(group_target2)

	# Ensure correct length
	if group_target.shape[0] > region_len:
		group_target = group_target[:region_len]
		all_coords = all_coords[:region_len]
		if bw_file_ribo != "none":
			group_target2 = group_target2[:region_len]
	elif group_target.shape[0] < region_len:
		pad_length = region_len - group_target.shape[0]
		group_target = np.pad(group_target, (0, pad_length), mode='constant')
		if bw_file_ribo != "none":
			group_target2 = np.pad(group_target2, (0, pad_length), mode='constant')

	# Prepare final outputs
	target = np.array([group_target]).astype(np.float32)
	target = np.nan_to_num(target, 0)
	averages = np.mean(target, axis=1)
	target = np.log(target + 0.0001)
	
	if bw_file_ribo != "none":
		target2 = np.array([group_target2]).astype(np.float32)
		target2 = np.nan_to_num(target2, 0)
		target2 = np.log(target2 + 0.0001)
		bw_ribo.close()
	
	bw.close()
	
	return (target, target2 if bw_file_ribo != "none" else None, 
			all_coords, strand, averages)

def process_interval(args):
	torch.cuda.init()
	torch.cuda.empty_cache()
	index, target_interval, kwargs = args
	try:
		# Initialize CUDA for this process with fp16 support
		device = torch.device(kwargs['device'])
		torch.cuda.set_device(device)
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision('high')  # Enable TensorFloat-32
		
		# Load fresh model instance for this process with fp16
		model = TrainModule.load_from_checkpoint(
			kwargs['checkpoint_path'],
			nbins=kwargs['nbins'],
			sequence_length=kwargs['sequence_length'],
			model_name=kwargs['model_type'].replace("Cds","").replace("cds","")
		).to(device).half()  # Convert to fp16
		model.eval()
		
		# Extract parameters
		gene_name = kwargs['gene_name']
		exon_interval = kwargs['exon_interval']
		rna_bw_file = kwargs['rna_bw_file']
		ribo_bw_file = kwargs['ribo_bw_file']
		sequence_length = kwargs['sequence_length']
		chromosome_lengths = kwargs['chromosome_lengths']
		nbins = kwargs['nbins']
		assembly = kwargs['assembly']
		species = kwargs['species']
		model_type = kwargs['model_type']
		tissue = kwargs['tissue']
		folder_path = kwargs['folder_path']
		
		id = str(target_interval['id'])
		if not species in target_interval['chr']:
			return False

		try:
			ref_seq = str(target_interval['sequence'])
			if len(ref_seq) > sequence_length:
				ref_seq = ref_seq[:sequence_length]
			elif len(ref_seq) < sequence_length:
				ref_seq += 'N' * (sequence_length - len(ref_seq))
			
			# Generate inputs
			t, t2, all_coords, strand, averages_rna = generate_inputs(
				exon_interval, rna_bw_file, ribo_bw_file, 
				sequence_length, chromosome_lengths, id
			)

			# Skip if expression too low
			if averages_rna < 1:
				with kwargs['trans_lock']:
					with open(f"{folder_path}/transcripts.out", "a") as out_trans:
						out_trans.write(f"{id}\t{assembly}\t{species}\t{model_type}\t{averages_rna[0]}\tnan\tnan\tnan\tnan\ttotal\texcluded\n")
				return False

			# Prepare model inputs - convert to fp16
			ref_emb = torch.Tensor(one_hot_encode(ref_seq)).to(device).half()
			epi = torch.Tensor(t[0]).unsqueeze(1).to(device).half()
			reference_input = torch.cat([ref_emb, epi], dim=1).unsqueeze(0)

			# Get predictions
			with torch.cuda.amp.autocast():  # Enable automatic mixed precision
				pred = model(reference_input)[0].detach().cpu().numpy()
			
			raw_scale_predictions = pred[:math.ceil((len(ref_seq.rstrip("N")))/(sequence_length/nbins))]

			# Compute integrated gradients
			input_tensor = reference_input.clone()
			ig_scores = integrated_gradients(model, input_tensor)
			ig_scores = ig_scores[:int(len(ref_seq.rstrip("N")))]

			# Prepare nucleotide-level predictions
			num_predictions = pred.shape[0]
			bin_size = sequence_length / num_predictions
			raw_scale_predictions_nt = [num for num in raw_scale_predictions for _ in range(int(bin_size))]

			# Write attribution scores
			with kwargs['attrib_lock']:
				with open(f"{folder_path}/attribution_scores.bedgraph", "a") as out_attrib:
					for i, coord in enumerate(all_coords):
						try:
							if strand == "+":
								out_attrib.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord}\t{coord+1}\t{ig_scores[i]}\n")
							elif strand == "-":
								out_attrib.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord-1}\t{coord}\t{ig_scores[i]}\n")
						except:
							pass

			# Transform predictions
			raw_scale_predictions = np.clip(raw_scale_predictions, -20, 20)  # Prevent overflow
			raw_scale_predictions = np.exp(raw_scale_predictions) - 0.0001
			raw_scale_predictions = [x if x > 1 else 0 for x in raw_scale_predictions]
			raw_scale_predictions_nt = np.exp(raw_scale_predictions_nt) - 0.0001
			raw_scale_predictions_nt = [x if x > 1 else 0 for x in raw_scale_predictions_nt]
			raw_scale_predictions_nt = raw_scale_predictions_nt[:int(len(all_coords))]
			t[0] = np.clip(t[0], -20, 20)
			t[0] = np.exp(t[0]) - 0.0001
			t[0] = [x if x > 1 else 0 for x in t[0]]


			if ribo_bw_file != "none":
				t2[0] = np.clip(t2[0], -20, 20)
				t2[0] = np.exp(t2[0]) - 0.0001
				t2[0] = [x if x > 1 else 0 for x in t2[0]]

			# Write predictions
			with kwargs['pred_lock']:
				with open(f"{folder_path}/predictedribo.bedgraph", "a") as out:
					n = 0
					for i, coord in enumerate(all_coords):
						value = raw_scale_predictions_nt[n]
						# Replace infinite or extremely large values
						if not np.isfinite(value) or value > 1e6:
							value = 1e6
						if strand == "+":
							out.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord}\t{coord+1}\t{value}\t{strand}\t{id}\n")
						elif strand == "-":
							out.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord-1}\t{coord}\t{value}\t{strand}\t{id}\n")
						n += 1

			# Write RNA inputs
			with kwargs['rna_lock']:
				with open(f"{folder_path}/inputrna.bedgraph", "a") as out2:
					n = 0
					for i, coord in enumerate(all_coords):
						if strand == "+":
							out2.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord}\t{coord+1}\t{t[0][n]}\t{strand}\t{id}\n")
						elif strand == "-":
							out2.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord-1}\t{coord}\t{t[0][n]}\t{strand}\t{id}\n")
						n += 1

			# Write Ribo inputs if available
			if ribo_bw_file != "none":
				with kwargs['ribo_lock']:
					with open(f"{folder_path}/inputribo.bedgraph", "a") as out3:
						n = 0
						for i, coord in enumerate(all_coords):
							if strand == "+":
								out3.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord}\t{coord+1}\t{t2[0][n]}\t{strand}\t{id}\n")
							elif strand == "-":
								out3.write(f"{target_interval['chr'].replace('chr','').split('_')[0]}\t{coord-1}\t{coord}\t{t2[0][n]}\t{strand}\t{id}\n")
							n += 1

				correlation, correlation2, lcorrelation, lcorrelation2 = calculate_cor(t2[0][:int(len(all_coords))] , t[0][:int(len(all_coords))] , raw_scale_predictions_nt, bin_size)	

				averages_ribo_pred = np.mean(np.array(raw_scale_predictions_nt))
				averages_ribo = np.mean(t2[0][:int(len(all_coords))])

				l = len(raw_scale_predictions_nt)

				with kwargs['trans_lock']:
					with open(f"{folder_path}/transcripts.out", "a") as out_trans:
						out_trans.write(f"{id}\t{assembly}\t{species}\t{model_type}\t{averages_rna[0]}\t{averages_ribo}\t{averages_ribo_pred}\t{lcorrelation}\t{l}\ttotal\tincluded\n")	

			else:
				pass

			return True
	
		except Exception as e:
			print(f"Error processing interval {id}: {str(e)}")
			return False
			
	except Exception as e:
		print(f"Error initializing process: {str(e)}")
		return False
	finally:
		# Clean up CUDA resources
		if 'model' in locals():
			del model
		torch.cuda.empty_cache()

def integrated_gradients(model, input_tensor, baseline=None, steps=50):
	if baseline is None:
		baseline = torch.zeros_like(input_tensor)
	
	scaled_inputs = [baseline + (float(i)/steps)*(input_tensor-baseline) for i in range(0, steps+1)]
	
	gradients = []
	for scaled_input in scaled_inputs:
		scaled_input.requires_grad_(True)
		with torch.cuda.amp.autocast():  # Enable mixed precision for gradients
			output = model(scaled_input)
			output.backward(torch.ones_like(output))
		gradients.append(scaled_input.grad.detach())
		scaled_input.grad = None
	
	avg_gradients = torch.mean(torch.stack(gradients), dim=0)
	integrated_grad = (input_tensor - baseline) * avg_gradients
	return integrated_grad.sum(dim=-1).squeeze().cpu().numpy()


def run_parallel_processing(tasks, num_processes=None):
	success_count = 0
	pool = None
	try:
		processes = num_processes if num_processes else min(10, multiprocessing.cpu_count())  # Cap at 10
		print(f"Using {processes} processes")
		
		# IMPORTANT: Use maxtasksperchild to prevent hangs
		pool = Pool(processes=processes, maxtasksperchild=1)
		
		# Use get() with timeout for map_async
		results = pool.map_async(process_interval, tasks)
		
		# Wait with timeout (60 sec per task as safety)
		results = results.get(timeout=60*len(tasks))  
		success_count = sum(results)
		
	except multiprocessing.TimeoutError:
		print("Timeout occurred - terminating pool")
		if pool:
			pool.terminate()
		raise RuntimeError("Processing timed out")
		
	except Exception as e:
		print(f"Error during multiprocessing: {str(e)}")
		# Fallback to single process
		print("Attempting single process fallback...")
		success_count = 0
		for task in tasks:
			try:
				success_count += int(process_interval(task))
			except Exception as e:
				print(f"Error processing interval: {str(e)}")
				
	finally:
		if pool:
			pool.close()
		torch.cuda.empty_cache()
		print("All workers finished")
	return success_count


def parse_arguments():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description='RiboDeepPred Prediction')
	
	# Required arguments
	parser.add_argument('--checkpoint', dest='checkpoint', required=True,
					  help='Path to the model checkpoint')
	parser.add_argument('--data', dest='data_folder', required=True,
					  help='Path to the data folder')
	
	# Gene and model parameters
	parser.add_argument('--gene', dest='gene_name', default='all',
					  help='Gene name to predict or "all" for all genes')
	parser.add_argument('--assembly', dest='assembly', default='primate',
					  help='Genome assembly')
	parser.add_argument('--tissue', dest='tissue', required=True,
					  help='Tissue type')
	parser.add_argument('--species', dest='species', required=True,
					  help='Species name')
	parser.add_argument('--model-type', dest='model_type', default='TransModel',
					  help='Model type (TransModel or PosTransModel)')
	
	# Sequence parameters
	parser.add_argument('--region-len', dest='sequence_length', type=int, default=6000,
					  help='Sequence length')
	parser.add_argument('--nBins', dest='nbins', type=int, default=1000,
					  help='Number of bins')
	parser.add_argument('--num_processes', dest='n_processes', type=int, default=1000,
					  help='Number of processes')
	
	return parser.parse_args()


def main():
	args = parse_arguments()

	print(f"Gene to predict: {args.gene_name}")
	start_time = time.time()

	# Setup device with fp16 support
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	torch.backends.cudnn.benchmark = True
	torch.set_float32_matmul_precision('high')
	
	# Prepare file paths
	rna_bw_file = os.path.join(args.data_folder, 'rnaseq.normalized.bw')
	if not os.path.exists(os.path.join(args.data_folder, 'riboseq.normalized.bw')): #In case Ribo-seq data is unavailable
		ribo_bw_file = "none"
	else:
		ribo_bw_file = os.path.join(args.data_folder, 'riboseq.normalized.bw')

	# Load intervals
	species = args.species
	exon_interval = f"data/{args.assembly}/exon_region.bed"
	gene_interval = f"data/{args.assembly}/exon_regions_output.txt"
	chrom_file = f'data/{args.assembly}/{args.assembly}_chrom_lengths.txt'

	# Load chromosome lengths
	with open(chrom_file, "r") as file:
		chromosome_lengths = {line.split()[0]: int(line.split()[1]) for line in file}

	# Load gene intervals
	data = []
	with open(gene_interval, 'r') as file:
		for line in file:
			parts = line.strip().split('\t')
			try:
				data.append({
					'chr': parts[0],
					'start': parts[1],
					'end': parts[2],
					'id': parts[3],
					'length': parts[4],
					'length2': parts[5],
					'length_norm': parts[6],
					'sequence': parts[7]
				})
			except:
				continue
	intervals = pd.DataFrame(data)
	
	if args.gene_name != "all":
		if isinstance(args.gene_name, list):
			intervals = intervals[intervals['id'].isin(args.gene_name)]
		else:
			intervals = intervals[intervals['id'] == args.gene_name]
			
	# Create output directory
	folder_name = f"{args.gene_name}_{args.assembly}_{species}_{args.sequence_length}_{args.nbins}_{args.model_type}"
	folder_path = os.path.join("predictions", folder_name)
	os.makedirs(folder_path, exist_ok=True)

	# Initialize output files
	if not os.path.exists(f"{folder_path}/transcripts.out"):
		with open(f"{folder_path}/transcripts.out", "w") as f:
			f.write("id\tassembly\tspecies\tmodel_type\trna_density\tribo_density\tribo_density_pred\tcorrelationlog\tlength\tregion\tstatus\n")

	# Initialize locks
	manager = Manager()
	locks = {
		'attrib_lock': manager.Lock(),
		'trans_lock': manager.Lock(),
		'pred_lock': manager.Lock(),
		'rna_lock': manager.Lock(),
		'ribo_lock': manager.Lock() if ribo_bw_file != "none" else None
	}

	# Prepare arguments for multiprocessing
	kwargs = {
		'checkpoint_path': args.checkpoint,
		'gene_name': args.gene_name,
		'device': device,
		'exon_interval': exon_interval,
		'rna_bw_file': rna_bw_file,
		'ribo_bw_file': ribo_bw_file,
		'sequence_length': args.sequence_length,
		'chromosome_lengths': chromosome_lengths,
		'nbins': args.nbins,
		'assembly': args.assembly,
		'species': species,
		'model_type': args.model_type,
		'tissue': args.tissue,
		'folder_path': folder_path,
		**locks
	}

	# Process intervals
	tasks = []
	for index, row in intervals.iterrows():
		id = row['id']
		# Always add the original row first
		tasks.append((index, row, kwargs))
		
	success_count = run_parallel_processing(tasks, num_processes=args.n_processes)

	elapsed_time = time.time() - start_time
	mins, secs = divmod(elapsed_time, 60)
	print(f"Processing complete | Success: {success_count}/{len(intervals)} | Time: {int(mins)}m {int(secs)}s")

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn', force=True)
	main()
