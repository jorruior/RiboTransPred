"""
Author: Jorge Ruiz Orera
Date: 2025-05-24
Description: Script to train the transformer model with RNA-seq, Ribo-seq and sequence data.
"""

import sys
import torch
import argparse
import numpy as np
import os 
import json
import pandas as pd
import kipoiseq
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchmetrics import Metric

import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import DeviceStatsMonitor

from typing import Optional
sys.path.append("../scripts")
import model.models as models

import time
import random

from collections import defaultdict
import mmap
import datetime

class TensorLoader:
	def __init__(self, m: int, n: int):
		self.files = []
		self.m = m
		self.n = n

	def load(self, path: str):
		if os.path.getsize(path) != self.m * self.n * 4:
			raise Exception(f'file size is expected to be {self.m}*{self.n}*4')
		f = open(path, 'r+b')
		self.files.append(f)
		buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
		array = np.frombuffer(buffer, dtype=np.float32, count=self.m*self.n).reshape((self.m, self.n))
		return torch.tensor(array.copy(), dtype=torch.float32)

	def close(self):
		for f in self.files:
			f.close()

class MeanPearsonCorrCoefPerChannel(Metric):
	def __init__(self, n_channels: int, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
		self.n_channels = n_channels
		self.add_state("product", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("true", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("true_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("pred", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("pred_squared", default=torch.zeros(n_channels), dist_reduce_fx="sum")
		self.add_state("count", default=torch.zeros(n_channels), dist_reduce_fx="sum")

	def update(self, preds: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
		if mask is None:
			mask = torch.ones_like(target, dtype=torch.float32)
		
		preds = preds.float() * mask
		target = target.float() * mask

		self.product += (preds * target).sum(dim=tuple(range(preds.dim()-1)))
		self.true += target.sum(dim=tuple(range(preds.dim()-1)))
		self.true_squared += (target ** 2).sum(dim=tuple(range(preds.dim()-1)))
		self.pred += preds.sum(dim=tuple(range(preds.dim()-1)))
		self.pred_squared += (preds ** 2).sum(dim=tuple(range(preds.dim()-1)))
		self.count += mask.sum(dim=tuple(range(preds.dim()-1)))

	def compute(self):
		epsilon = 1e-8
		true_mean = self.true / (self.count + epsilon)
		pred_mean = self.pred / (self.count + epsilon)

		covariance = (self.product 
					  - true_mean * self.pred 
					  - pred_mean * self.true 
					  + self.count * true_mean * pred_mean)
		true_var = self.true_squared - self.count * (true_mean ** 2)
		pred_var = self.pred_squared - self.count * (pred_mean ** 2)
		
		correlation = covariance / (torch.sqrt(true_var + epsilon) * torch.sqrt(pred_var + epsilon))
		return correlation

class Dataset(torch.utils.data.Dataset):
	def __init__(self, regions, input_features, output_features, seq_len, target_length, cds_vectors=None):
		self.target_length = target_length
		self.seq_len = seq_len
		self.regions = regions
		self.input_features = input_features
		self.output_features = output_features
		self.cds_vectors = cds_vectors
		
		# Pre-compute valid indices (where mask is not all zeros)
		self.valid_indices = []
		for idx in range(len(regions)):
			loc_row = regions.iloc[idx]
			sequence = loc_row['sequence']
			if len(sequence) > self.seq_len:
				sequence = sequence[:self.seq_len]
			
			if self.cds_vectors is not None:
				transcript_id = loc_row['id']
				if transcript_id in self.cds_vectors:
					cds_vector = self.cds_vectors[transcript_id]
					if len(cds_vector) < self.seq_len:
						cds_vector = np.pad(cds_vector, (0, self.seq_len - len(cds_vector)), 'constant')
					elif len(cds_vector) > self.seq_len:
						cds_vector = cds_vector[:self.seq_len]
					mask = torch.tensor([1.0 if x in {2, 3} else 0.0 for x in cds_vector], dtype=torch.float32)
				else:
					mask = torch.ones(len(sequence), dtype=torch.float32)
			else:
				mask = torch.tensor([1.0 if base != 'N' else 0.0 for base in sequence], dtype=torch.float32)
			
			mask = F.avg_pool1d(mask.unsqueeze(0), kernel_size=round(self.seq_len / self.target_length)).squeeze(0)
			
			# Only keep indices where mask has at least one non-zero value
			if mask.sum() > 0:
				self.valid_indices.append(idx)
	
	@staticmethod
	def one_hot_encode(sequence):
		en_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
		en_seq = [en_dict[ch] for ch in sequence]
		np_seq = np.array(en_seq, dtype=int)
		seq_emb = np.zeros((len(np_seq), 5))
		seq_emb[np.arange(len(np_seq)), np_seq] = 1
		return seq_emb.astype(np.float32)

	def __len__(self):
		return len(self.valid_indices)

	def __getitem__(self, idx):
		# Use the pre-computed valid indices
		original_idx = self.valid_indices[idx]
		loc_row = self.regions.iloc[original_idx]
		sequence = loc_row['sequence']
		if len(sequence) > self.seq_len:
			sequence = sequence[:self.seq_len]
			
		sequence_one_hot = self.one_hot_encode(sequence)
		input_features = self.input_features[original_idx]
		output_features = self.output_features[original_idx]
		
		# Initialize mask (we know this will have at least one non-zero value)
		if self.cds_vectors is not None:
			transcript_id = loc_row['id']
			if transcript_id in self.cds_vectors:
				cds_vector = self.cds_vectors[transcript_id]
				if len(cds_vector) < self.seq_len:
					cds_vector = np.pad(cds_vector, (0, self.seq_len - len(cds_vector)), 'constant')
				elif len(cds_vector) > self.seq_len:
					cds_vector = cds_vector[:self.seq_len]
				mask = torch.tensor([1.0 if x in {2, 3} else 0.0 for x in cds_vector], dtype=torch.float32)
			else:
				mask = torch.ones(len(sequence), dtype=torch.float32)
		else:
			mask = torch.tensor([1.0 if base != 'N' else 0.0 for base in sequence], dtype=torch.float32)
		
		mask = F.avg_pool1d(mask.unsqueeze(0), kernel_size=round(self.seq_len / self.target_length)).squeeze(0)

		assert sequence_one_hot.shape[0] == input_features.shape[0], \
			f"Sequence length {sequence_one_hot.shape[0]} doesn't match features length {input_features.shape[0]}"
				
		return {
			'sequence': sequence_one_hot,
			'input_features': input_features,
			'output_features': output_features,
			'mask': mask,
			'chrom': loc_row['chr'],
			'start': loc_row['start'],
			'end': loc_row['end']
		}

class DataModule(LightningDataModule):
	def __init__(
		self,
		region_file: str = None,
		input_file: list = [],
		output_file: list = [],
		regions2: list = [],  
		metadata: list = [],
		seq_len: int = 6000,
		target_length: int = 1000,
		train_chrlist: list = [],
		val_chrlist: list = [],
		test_chrlist: list = [],
		batch_size: int = 32,
		eval_batch_size: int = None,
		num_workers: int = 4,
		pin_memory: bool = False,
		cds_vectors: dict = None,
		**kwargs
	):
		super().__init__()
		self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
		self.num_workers = min(num_workers, os.cpu_count()//2)  # Don't exceed available CPUs
		self.pin_memory = pin_memory
		self.regions = pd.read_csv(region_file, sep='\t', names=['chr', 'start', 'end', 'strand', 'id'])
		self.input_features = input_file
		self.output_features = output_file
		self.regions2 = regions2
		self.metadata = metadata
		self.cds_vectors = cds_vectors

		train_idx = self.regions2['chr'].isin(train_chrlist)
		val_idx = self.regions2['chr'].isin(val_chrlist)
		if len(test_chrlist) > 0:
			test_idx = self.regions2['chr'].isin(test_chrlist)

		train_dataset_list = []
		val_dataset_list = []
		test_dataset_list = []
		
		for i, element in enumerate(metadata):
			train_dataset = Dataset(
				self.regions2[train_idx], 
				self.input_features[i][train_idx], 
				self.output_features[i][train_idx], 
				seq_len, 
				target_length,
				cds_vectors=self.cds_vectors
			)
			val_dataset = Dataset(
				self.regions2[val_idx],
				self.input_features[i][val_idx],
				self.output_features[i][val_idx],
				seq_len,
				target_length,
				cds_vectors=self.cds_vectors
			)

			if len(test_chrlist) > 0:
				test_dataset = Dataset(
					self.regions2[test_idx],
					self.input_features[i][test_idx],
					self.output_features[i][test_idx],
					seq_len,
					target_length,
					cds_vectors=self.cds_vectors
				)
			
			train_dataset_list.append(train_dataset)
			val_dataset_list.append(val_dataset)
			if len(test_chrlist) > 0:
				test_dataset_list.append(test_dataset)
			
		self.train_dataset = ConcatDataset(train_dataset_list)
		self.val_dataset = ConcatDataset(val_dataset_list)
		if len(test_chrlist) > 0:
			self.test_dataset = ConcatDataset(test_dataset_list)  

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.eval_batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			persistent_workers=True,
			prefetch_factor=2,
			drop_last=True  # Important for DDP stability
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.eval_batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			persistent_workers=True,
			prefetch_factor=1
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.eval_batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			persistent_workers=True,
			prefetch_factor=1
		)

class TrainModule(LightningModule):
	def __init__(self, args):
		super().__init__()
		self.save_hyperparameters()
		self.model = self.get_model(args)
		self.args = args
		self.criterion = nn.MSELoss()
		self.pcc = MeanPearsonCorrCoefPerChannel(1)
		self.val_pcc = MeanPearsonCorrCoefPerChannel(1)
		self.test_pcc = MeanPearsonCorrCoefPerChannel(1)
		self.test_positions_count = 0

	def forward(self, x):
		return self.model(x)
	
	def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
		scheduler.step()
	
	def configure_optimizers(self):
		optimizer = optim.AdamW(
			self.parameters(), 
			lr=1e-5, 
			weight_decay=0.005,
			betas=(0.9, 0.999),
			eps=1e-8
		)
		
		# Calculate total steps
		if self.trainer.max_epochs > 0:
			total_steps = self.trainer.estimated_stepping_batches
			warmup_steps = min(2000, total_steps // 10)
		else:
			total_steps = 100000
			warmup_steps = 2000
		
		scheduler = get_cosine_schedule_with_warmup(
			optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=total_steps
		)
		
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "step",
				"frequency": 1,
			}
		}

	def _shared_step(self, batch, batch_idx, metric, prefix):
		seq = batch['sequence']
		epi = batch['input_features'].unsqueeze(-1)
		targets = batch['output_features'].float()
		mask = batch['mask']
		
		inputs = torch.cat([seq, epi], dim=2)
		pred = self(inputs)
		
		masked_pred = pred * mask
		masked_targets = targets * mask
		
		loss = (F.mse_loss(masked_pred, masked_targets, reduction='none') * mask).sum() / mask.sum()
		
		valid_indices = mask.bool()
		flat_pred = masked_pred[valid_indices]
		flat_targets = masked_targets[valid_indices]
		
		if len(flat_pred) > 1:
			metric.update(flat_pred, flat_targets)
			pcc = metric.compute()
			
			ranks1 = flat_pred.argsort().argsort().float()
			ranks2 = flat_targets.argsort().argsort().float()
			scc = self.pcc(ranks1, ranks2)
		else:
			pcc = torch.tensor(float('nan'), device=self.device)
			scc = torch.tensor(float('nan'), device=self.device)
		
		self.log_dict({
			f'loss/{prefix}_step': loss,
			f'pearson/{prefix}_step': pcc,
		}, batch_size=inputs.shape[0], prog_bar=(prefix == 'train'),sync_dist=True)
		
		return {'loss': loss, 'pcc': pcc, 'scc': scc}

	def training_step(self, batch, batch_idx):
		return self._shared_step(batch, batch_idx, self.pcc, 'train')

	def validation_step(self, batch, batch_idx):
		return self._shared_step(batch, batch_idx, self.val_pcc, 'val')

	def test_step(self, batch, batch_idx):
		result = self._shared_step(batch, batch_idx, self.test_pcc, 'test')
		mask = batch['mask']
		self.test_positions_count += mask.sum().item()
		return result
		
	def _shared_epoch_end(self, step_outputs):
		if not step_outputs:
			return {
				'loss': torch.tensor(float('nan')),
				'pcc': torch.tensor(float('nan')),
				'scc': torch.tensor(float('nan'))
			}
			
		losses = []
		pccs = []
		sccs = []
		
		for out in step_outputs:
			if out is not None:
				losses.append(out['loss'])
				pccs.append(out['pcc'])
				sccs.append(out['scc'])
		
		if not losses:
			return {
				'loss': torch.tensor(float('nan')),
				'pcc': torch.tensor(float('nan')),
				'scc': torch.tensor(float('nan'))
			}
			
		loss_tensor = torch.stack(losses)
		pcc_tensor = torch.stack([pcc for pcc in pccs if not torch.isnan(pcc)])
		scc_tensor = torch.stack([scc for scc in sccs if not torch.isnan(scc)])
		
		return {
			'loss': loss_tensor.mean(),
			'pcc': pcc_tensor.mean() if len(pcc_tensor) > 0 else torch.tensor(float('nan')),
			'scc': scc_tensor.mean() if len(scc_tensor) > 0 else torch.tensor(float('nan'))
		}

	def training_epoch_end(self, step_outputs):
		ret_metrics = self._shared_epoch_end(step_outputs)
		metrics = {
			'train_loss': ret_metrics['loss'], 
			'train_pcc': ret_metrics['pcc'], 
			'train_scc': ret_metrics['scc']
		}
		self.log_dict(metrics, on_epoch=True, sync_dist=True)
		self.pcc.reset()

	def validation_epoch_end(self, step_outputs):
		ret_metrics = self._shared_epoch_end(step_outputs)
		metrics = {
			'val_loss': ret_metrics['loss'], 
			'val_pcc': ret_metrics['pcc'], 
			'val_scc': ret_metrics['scc']
		}
		self.log_dict(metrics, on_epoch=True, sync_dist=True)
		self.val_pcc.reset()
		
	def test_epoch_end(self, step_outputs):
		ret_metrics = self._shared_epoch_end(step_outputs)
		metrics = {
			'test_loss': ret_metrics['loss'], 
			'test_pcc': ret_metrics['pcc'], 
			'test_scc': ret_metrics['scc'],
			'test_positions_count': self.test_positions_count
		}
		self.log_dict(metrics, on_epoch=True, sync_dist=True)
		self.test_pcc.reset()
		
		# Print the count (only on rank 0 to avoid duplicate output in DDP)
		if self.trainer.global_rank == 0:
			print(f"\nTotal positions evaluated during test: {self.test_positions_count}")
			print("="*80)
			print("Test Results Summary:")
			print(f"Test Loss: {ret_metrics['loss'].item():.4f}")
			print(f"Test PCC: {ret_metrics['pcc'].item():.4f}")
			print(f"Test SCC: {ret_metrics['scc'].item():.4f}")
			print("="*80)
		
		# Reset the counter for potential future test runs
		self.test_positions_count = 0
	
	def get_model(self, args):
		model_name = args.model_type
		num_genomic_features = 1
		ModelClass = getattr(models, model_name)
		model = ModelClass(
			num_genomic_features, 
			mid_hidden=512, 
			target_length=args.region_len, 
			nbins=args.nBins,
			n_heads=args.n_heads,
			dropout=args.dropout
		)
		return model
		
def distribute_chromosomes_fixed(chrlist, args):
	chrlist = [chrom for chrom in chrlist if len(chrom) <= 15]  # Remove non-classical
	
	# Separate by species
	species_chr = defaultdict(list)
	for chrom in chrlist:
		species = chrom.split('_')[-1]
		species_chr[species].append(chrom)
	
	# Initialize sets
	train_chrlist, val_chrlist, test_chrlist = [], [], []
	
	# Special test cases
	if "test_human_" in args.dataset and args.test:
		test_chrlist = [c for c in chrlist if c.endswith('_human')]
	elif "test_mouse_" in args.dataset and args.test:
		test_chrlist = [c for c in chrlist if c.endswith('_mouse')]
	elif "test_chicken_" in args.dataset and args.test:
		test_chrlist = [c for c in chrlist if c.endswith('_chicken')]
	elif args.dataset_assembly == "gorilla":
		test_chrlist = [c for c in chrlist if c.endswith('_gorilla')]
	else:
		# Standard fixed-number distribution
		for species, chromosomes in species_chr.items():
			random.shuffle(chromosomes)
			
			if args.similar:
				# Handle similar chromosomes
				similar_chrs = ["chr4", "chr5", "chr12"]
				test_chrlist.extend([c for c in chromosomes 
								   if any(s in c for s in similar_chrs)])
				chromosomes = [c for c in chromosomes 
							 if not any(s in c for s in similar_chrs)]
				
				val_chrs = ["chr13", "chr15", "chr19"]
				val_chrlist.extend([c for c in chromosomes 
								  if any(s in c for s in val_chrs)])
				chromosomes = [c for c in chromosomes 
							 if not any(s in c for s in val_chrs)]
			else:
				# Fixed number: 3 per species for test, 3 for val
				test_chrlist.extend(chromosomes[:3])
				val_chrlist.extend(chromosomes[3:6])
				chromosomes = chromosomes[6:]
			
			train_chrlist.extend(chromosomes)
	
	# Filter by study list
	def filter_by_study(chr_list):
		return [chrom for chrom in chr_list 
			   if chrom.split("_")[-1] in study_list]
	
	train_chrlist = filter_by_study(train_chrlist)
	val_chrlist = filter_by_study(val_chrlist)
	test_chrlist = filter_by_study(test_chrlist)
	
	return train_chrlist, val_chrlist, test_chrlist



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='RiboDeepPred')
	
	parser.add_argument('--seed', dest='run_seed', default=4, type=int, help='Random seed for training')
	parser.add_argument('--save_path', dest='run_save_path', default='checkpoints', help='Path to the model checkpoint')   
	parser.add_argument('--data', dest='dataset_data', default='data', help='Path of training data', required=True)
	parser.add_argument('--assembly', dest='dataset_assembly', default='primate', help='Genome assembly for training data')
	parser.add_argument('--dataset', dest='dataset', default='ribomap.txt', help='Multi input for data training')
	parser.add_argument('--model-type', dest='model_type', default='TransModel', help='Transformer: PosTransModel or TransModel')
	parser.add_argument('--region_len', type=int, default=6000, help='sequence length')
	parser.add_argument('--nBins', type=int, default=1000, help='number of bins')
	parser.add_argument('--similar', action='store_true', help='If enabled, isolate similar homologous primate chromosomes for validation and test')
	parser.add_argument('--test', action='store_true', help='Enable testing mode based on hold-out chromosomes')

	# Training Parameters
	parser.add_argument('--patience', dest='trainer_patience', default=8, type=int, help='Epoches before early stopping')
	parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=50, type=int, help='Max epochs')
	parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=50, type=int, help='Top n models to save')
	parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1, type=int, help='Number of GPUs to use')
	parser.add_argument('--cds_loss', dest='cds', help='Calculate MSE loss only on CDS sequences (Include vectors with 2s and 3s indicating CDS positions)')
	
	# Dataloader Parameters
	parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, type=int, help='Batch size')
	parser.add_argument('--num-workers', dest='dataloader_num_workers', default=2, type=int, help='Dataloader workers')
	parser.add_argument('--nheads', dest='n_heads', default=6, type=int, help='Number of attention heads')
	parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate')	
	parser.add_argument('--checkpoint', type=str, default=None)

	args = parser.parse_args()

	# Set up DDP environment
	if args.trainer_num_gpu > 1:
		os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
		os.environ['NCCL_SOCKET_NTHREADS'] = '4'
		os.environ['NCCL_DEBUG'] = 'INFO'
		os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
		os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
		os.environ['NCCL_TIMEOUT'] = '12000'
	
	# Initialize logger
	logger = pl.loggers.CSVLogger(save_dir=f'{args.run_save_path}/csv')
	
	# Callbacks
	callbacks_list = [
		callbacks.EarlyStopping(
			monitor='val_loss',
			min_delta=0.00,
			patience=args.trainer_patience,
			verbose=True,
			mode="min"
		),
		callbacks.ModelCheckpoint(
			dirpath=f'{args.run_save_path}/models',
			save_top_k=args.trainer_save_top_n,
			monitor='val_loss',
			mode='min',
			save_last=True
		),
		callbacks.LearningRateMonitor(logging_interval='step'),
		callbacks.RichProgressBar()
	]

	callbacks_list.append(DeviceStatsMonitor())

	# Configure precision
	precision = 'bf16'
	unused_par = False
	if "2step" in args.model_type:
		unused_par = True
	
	# Initialize trainer
	trainer = Trainer(
		accelerator='gpu',
		devices=args.trainer_num_gpu,
		num_nodes=1,
		strategy=(
			DDPStrategy(
				find_unused_parameters=unused_par,
				gradient_as_bucket_view=True,
				static_graph=False, ######
				process_group_backend='nccl',
				timeout=datetime.timedelta(seconds=14000)
			) if args.trainer_num_gpu > 1 else None
		),
		max_epochs=args.trainer_max_epochs,
		callbacks=callbacks_list,
		precision=precision,
		gradient_clip_val=0.5,
		num_sanity_val_steps=0,
		logger=logger,
		accumulate_grad_batches=2,
		log_every_n_steps=50,
		enable_checkpointing=True,
		enable_progress_bar=True,
		enable_model_summary=True,
		deterministic=False
	)


	# Print config only on rank 0
	if trainer.global_rank == 0:
		print("\n=== Training Configuration ===")
		print(f"PyTorch Version: {torch.__version__}")
		print(f"CUDA Available: {torch.cuda.is_available()}")
		print(f"CUDA Version: {torch.version.cuda}")
		print(f"Number of GPUs: {torch.cuda.device_count()}")
		print(f"Precision: {precision}-bit")
		print(f"Batch Size: {args.dataloader_batch_size}")
		print(f"Max Epochs: {args.trainer_max_epochs}\n")

	# Load data
	seq_len = args.region_len
	target_length = args.nBins
	
	region_file = f'data/{args.dataset_assembly}/exon_region.bed'
	
	with open(region_file, 'r') as file:
		unique_regions = set(line.split()[4] for line in file if line.strip("\t").rstrip("\n"))
		region_count = len(unique_regions)
	
	tensor_loader = TensorLoader(region_count, seq_len)

	input_data = []
	output_data = []
	data = []
	celltype_list = []
	study_list = []
	
	with open(f'data/{args.dataset_assembly}/{args.dataset}', 'r') as file:
		meta_data = [line.strip('\n') for line in file]
	if trainer.global_rank == 0:
		print("\n=== Metadata ===")
		print(meta_data)
	
	for i, element in enumerate(meta_data):
		celltype, study = element.split('\t')
		celltype_list.append(celltype)
		study_list.append(study)
		output_data.append(torch.load(
			f'data/{args.dataset_assembly}/{celltype}/{study}/{celltype}_{seq_len}_{target_length}_log_riboseq_final.pt'
		))
		input_data.append(tensor_loader.load(
			f'data/{args.dataset_assembly}/{celltype}/{study}/{celltype}_{seq_len}_log_rnaseq_final.pt'
		))

	with open(f'data/{args.dataset_assembly}/exon_regions_output.txt', 'r') as file:
		for line in file:
			parts = line.strip().split('\t')
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

	regions2 = pd.DataFrame(data)
	if trainer.global_rank == 0:
		print(f"TRANSCRIPT COUNT IN GENERATED FILE: {regions2.shape}")

	cds_vectors = None
	if args.cds:
		cds_vectors = {}
		with open(args.cds, 'r') as file:
			for line in file:
				parts = line.strip().split('\t')
				transcript_id = parts[0]
				vector = [int(x) for x in parts[1]]
				cds_vectors[transcript_id] = vector

	pl.seed_everything(args.run_seed, workers=True)
	
	# Distribute chromosomes
	with open(f'data/{args.dataset_assembly}_chrom_map.txt', 'r') as file:
		chrlist = [line.strip('\n').split("\t")[1] for line in file]
	train_chrlist, val_chrlist, test_chrlist = distribute_chromosomes_fixed(chrlist, args)
	if trainer.global_rank == 0:
		print("\n=== Chromosome Distribution ===")
		print(f"Training: {len(train_chrlist)} chromosomes")
		print(f"Validation: {len(val_chrlist)} chromosomes")
		print(f"Test: {len(test_chrlist)} chromosomes")
		print("="*30)
	
	dataset = DataModule(
		region_file=region_file,
		input_file=input_data, 
		output_file=output_data,
		regions2=regions2,
		metadata=meta_data,
		seq_len=seq_len,
		target_length=target_length,
		train_chrlist=train_chrlist,
		val_chrlist=val_chrlist,
		test_chrlist=test_chrlist,
		batch_size=args.dataloader_batch_size,
		num_workers=args.dataloader_num_workers,
		cds_vectors=cds_vectors
	)

	# Initialize model
	pl_module = TrainModule(args)
	
	# Training flow
	if args.checkpoint:
		if trainer.global_rank == 0:
			print(f"\nResuming training from checkpoint: {args.checkpoint}")
		trainer.fit(pl_module, dataset, ckpt_path=args.checkpoint)
	elif not args.test:
		if trainer.global_rank == 0:
			print("\nStarting new training run")
		trainer.fit(pl_module, dataset)
	else:
		if trainer.global_rank == 0 and args.test:
			print("\nRunning final test evaluation")
			trainer.test(pl_module, dataset, ckpt_path=f'{args.run_save_path}/models/last.ckpt')
	
	# Cleanup
	tensor_loader.close()
	if trainer.global_rank == 0:
		print("\nTraining completed successfully!")
