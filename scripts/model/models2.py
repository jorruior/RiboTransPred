from multiprocessing import context
import torch
import math
import torch.nn as nn
import model.blocks as blocks
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
import flash_attn
		
class CausalConv1d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
		super().__init__()
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.conv = nn.Conv1d(in_channels, out_channels,
							  kernel_size=kernel_size,
							  dilation=dilation, padding=0)

	def forward(self, x):
		pad = self.dilation * (self.kernel_size - 1)
		x = F.pad(x, (pad, 0))
		return self.conv(x)


class FiLMLayer(nn.Module):
	"""
	Feature-wise Linear Modulation.
	Takes a tissue embedding and produces (gamma, beta) to scale/shift
	feature maps: output = gamma * x + beta

	Initialised to identity (gamma=1, beta=0) so an untrained FiLM
	layer is a no-op, preserving the base model's behaviour.
	"""

	def __init__(self, tissue_emb_dim, num_channels):
		super().__init__()
		self.fc = nn.Linear(tissue_emb_dim, num_channels * 2)
		nn.init.zeros_(self.fc.weight)
		nn.init.zeros_(self.fc.bias)
		with torch.no_grad():
			self.fc.bias[:num_channels] = 1.0   # gamma = 1

	def forward(self, x, tissue_emb):
		"""
		x:          (B, C, L)  feature maps
		tissue_emb: (B, D)     tissue embedding
		Returns:    (B, C, L)  modulated feature maps
		"""
		params = self.fc(tissue_emb)           # (B, 2C)
		gamma, beta = params.chunk(2, dim=1)   # each (B, C)
		return gamma.unsqueeze(2) * x + beta.unsqueeze(2)


class CausalResidualBlockFiLM(nn.Module):
	"""Causal dilated residual block with FiLM after norm2."""

	def __init__(self, in_ch, out_ch, dilation, tissue_emb_dim):
		super().__init__()
		self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)
		self.norm1 = nn.GroupNorm(8, out_ch)
		self.relu  = nn.ReLU()
		self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
		self.norm2 = nn.GroupNorm(8, out_ch)
		self.residual = (nn.Conv1d(in_ch, out_ch, kernel_size=1)
						 if in_ch != out_ch else nn.Identity())
		self.film = FiLMLayer(tissue_emb_dim, out_ch)

	def forward(self, x, tissue_emb):
		res = self.residual(x)
		x = self.relu(self.norm1(self.conv1(x)))
		x = self.norm2(self.conv2(x))
		x = self.film(x, tissue_emb)
		return self.relu(x + res)

class PosTransModelTCNFiLMRef(nn.Module):
	"""
	PosTransModelTCN_FiLM with two-pass iterative refinement.

	Pass 1: FiLM tissue-conditioned backbone (identical to PosTransModelTCN_FiLM)
	Pass 2: Lightweight FiLM-conditioned refinement network that sees
	        original input + upsampled pass-1 prediction

	Final output = pass1 + sigmoid(gate) * correction
	"""

	def __init__(self, num_genomic_features, target_length, nbins,
				 num_tissues, tissue_emb_dim=64, seqno=False, **kwargs):
		super().__init__()
		self.seqno = seqno
		self.num_tissues = num_tissues
		self.tissue_emb_dim = tissue_emb_dim
		self.target_length = target_length
		self.nbins = nbins

		input_channels = num_genomic_features if seqno else 5 + num_genomic_features

		# Tissue embedding: num_tissues known + 1 "mean/unknown" slot
		self.tissue_embedding = nn.Embedding(num_tissues + 1, tissue_emb_dim)

		# ═══════════════════════════════════════════════════════════════
		# Pass 1: Backbone (same as PosTransModelTCN_FiLM)
		# ═══════════════════════════════════════════════════════════════

		self.conv_k3  = CausalConv1d(input_channels, 64, kernel_size=3)
		self.conv_k6  = CausalConv1d(input_channels, 64, kernel_size=6)
		self.conv_k25 = CausalConv1d(input_channels, 64, kernel_size=25)
		self.conv_gn   = nn.GroupNorm(8, 192)
		self.conv_relu = nn.ReLU()
		self.film_conv = FiLMLayer(tissue_emb_dim, 192)

		self.tcn_blocks = nn.ModuleList([
			CausalResidualBlockFiLM(192, 256, dilation=1,   tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(256, 256, dilation=4,   tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(256, 384, dilation=16,  tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(384, 384, dilation=64,  tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(384, 384, dilation=128, tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(384, 384, dilation=256, tissue_emb_dim=tissue_emb_dim),
		])

		self.dec_conv1 = nn.Conv1d(384, 256, kernel_size=1)
		self.dec_norm  = nn.GroupNorm(8, 256)
		self.dec_relu  = nn.ReLU()
		self.dec_conv2 = nn.Conv1d(256, 1, kernel_size=1)
		self.film_dec  = FiLMLayer(tissue_emb_dim, 256)

		if target_length == nbins:
			self.bin_pool = nn.Identity()
			self.pool_k = 1
		else:
			self.pool_k = target_length // nbins
			self.bin_pool = nn.AvgPool1d(
				kernel_size=self.pool_k, stride=self.pool_k)

		# ═══════════════════════════════════════════════════════════════
		# Pass 2: Refinement network (lightweight, FiLM-conditioned)
		# ═══════════════════════════════════════════════════════════════
		refine_in = input_channels + 1

		self.ref_conv_k3  = CausalConv1d(refine_in, 32, kernel_size=3)
		self.ref_conv_k6  = CausalConv1d(refine_in, 32, kernel_size=6)
		self.ref_conv_k25 = CausalConv1d(refine_in, 32, kernel_size=25)
		self.ref_conv_gn   = nn.GroupNorm(8, 96)
		self.ref_conv_relu = nn.ReLU()
		self.ref_film_conv = FiLMLayer(tissue_emb_dim, 96)

		self.ref_tcn_blocks = nn.ModuleList([
			CausalResidualBlockFiLM(96,  128, dilation=1,   tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(128, 128, dilation=16,  tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(128, 128, dilation=64,  tissue_emb_dim=tissue_emb_dim),
			CausalResidualBlockFiLM(128, 128, dilation=256, tissue_emb_dim=tissue_emb_dim),
		])

		self.ref_dec_conv1 = nn.Conv1d(128, 64, kernel_size=1)
		self.ref_dec_norm  = nn.GroupNorm(8, 64)
		self.ref_dec_relu  = nn.ReLU()
		self.ref_dec_conv2 = nn.Conv1d(64, 1, kernel_size=1)
		self.ref_film_dec  = FiLMLayer(tissue_emb_dim, 64)

		self.refine_gate = nn.Parameter(torch.tensor(0.0))

	def set_mean_embedding(self):
		with torch.no_grad():
			mean = self.tissue_embedding.weight[:self.num_tissues].mean(dim=0)
			self.tissue_embedding.weight[self.num_tissues] = mean

	def _backbone_forward(self, x, t_emb):
		x1, x2, x3 = self.conv_k3(x), self.conv_k6(x), self.conv_k25(x)
		ml = min(x1.size(2), x2.size(2), x3.size(2))
		x = torch.cat([x1[:,:,:ml], x2[:,:,:ml], x3[:,:,:ml]], dim=1)
		x = self.conv_relu(self.conv_gn(x))
		x = self.film_conv(x, t_emb)

		for blk in self.tcn_blocks:
			x = blk(x, t_emb)

		x = self.dec_conv1(x)
		x = self.dec_norm(x)
		x = self.film_dec(x, t_emb)
		x = self.dec_relu(x)
		x = self.dec_conv2(x)
		x = self.bin_pool(x)
		return x.squeeze(1)

	def _refine_forward(self, x, t_emb):
		r1, r2, r3 = self.ref_conv_k3(x), self.ref_conv_k6(x), self.ref_conv_k25(x)
		ml = min(r1.size(2), r2.size(2), r3.size(2))
		r = torch.cat([r1[:,:,:ml], r2[:,:,:ml], r3[:,:,:ml]], dim=1)
		r = self.ref_conv_relu(self.ref_conv_gn(r))
		r = self.ref_film_conv(r, t_emb)

		for blk in self.ref_tcn_blocks:
			r = blk(r, t_emb)

		r = self.ref_dec_conv1(r)
		r = self.ref_dec_norm(r)
		r = self.ref_film_dec(r, t_emb)
		r = self.ref_dec_relu(r)
		r = self.ref_dec_conv2(r)
		r = self.bin_pool(r)
		return r.squeeze(1)

	def forward(self, x, tissue_ids):
		t_emb = self.tissue_embedding(tissue_ids)
		x_ch = x.permute(0, 2, 1).float()

		# Pass 1
		pred1 = self._backbone_forward(x_ch, t_emb)

		# Upsample pred1 back to seq_len
		pred1_up = pred1.unsqueeze(1)
		pred1_up = pred1_up.repeat_interleave(self.pool_k, dim=2)
		seq_len = x_ch.size(2)
		if pred1_up.size(2) > seq_len:
			pred1_up = pred1_up[:, :, :seq_len]
		elif pred1_up.size(2) < seq_len:
			pred1_up = F.pad(pred1_up, (0, seq_len - pred1_up.size(2)))

		# Pass 2
		x_refine = torch.cat([x_ch, pred1_up.detach()], dim=1)
		correction = self._refine_forward(x_refine, t_emb)

		gate = torch.sigmoid(self.refine_gate)
		return pred1 + gate * correction

if __name__ == '__main__':
	main()
