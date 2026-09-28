from multiprocessing import context
import torch
import math
import torch.nn as nn
import model.blocks as blocks
import torch.nn as nn
import torch.nn.functional as F
from model.causal_v2 import PositionwiseLayerNorm, context_dilations
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm
#from mamba_ssm import Mamba


# ═══════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════════════

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

	Takes an embedding vector and produces (gamma, beta) to scale/shift
	feature maps:  output = gamma * x + beta

	Initialised to identity (gamma=1, beta=0) so an untrained FiLM
	layer is a no-op, preserving the base model's behaviour.

	Can be used for tissue, condition, or any other categorical axis.
	"""

	def __init__(self, emb_dim, num_channels):
		super().__init__()
		self.fc = nn.Linear(emb_dim, num_channels * 2)
		nn.init.zeros_(self.fc.weight)
		nn.init.zeros_(self.fc.bias)
		with torch.no_grad():
			self.fc.bias[:num_channels] = 1.0   # gamma = 1

	def forward(self, x, emb):
		"""
		x:   (B, C, L)  feature maps
		emb: (B, D)     embedding vector
		Returns: (B, C, L) modulated feature maps
		"""
		params = self.fc(emb)                   # (B, 2C)
		gamma, beta = params.chunk(2, dim=1)    # each (B, C)
		return gamma.unsqueeze(2) * x + beta.unsqueeze(2)


class DualFiLMLayer(nn.Module):
	"""
	Apply tissue FiLM followed by condition FiLM sequentially.

	The two modulations are kept separate so that tissue and condition
	capture orthogonal axes of variation.  Order: tissue first, then
	condition (condition modulates the tissue-adjusted representation).
	"""

	def __init__(self, tissue_emb_dim, cond_emb_dim, num_channels):
		super().__init__()
		self.tissue_film = FiLMLayer(tissue_emb_dim, num_channels)
		self.cond_film   = FiLMLayer(cond_emb_dim,   num_channels)

	def forward(self, x, tissue_emb, cond_emb):
		x = self.tissue_film(x, tissue_emb)
		x = self.cond_film(x,   cond_emb)
		return x


class CausalResidualBlockDualFiLM(nn.Module):
	"""Causal dilated residual block with position-local normalization, dropout and dual FiLM."""

	def __init__(self, in_ch, out_ch, dilation, tissue_emb_dim, cond_emb_dim, dropout=0.33):
		super().__init__()
		self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)
		self.norm1 = PositionwiseLayerNorm(out_ch)
		self.relu  = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
		self.norm2 = PositionwiseLayerNorm(out_ch)
		self.residual = (nn.Conv1d(in_ch, out_ch, kernel_size=1)
						 if in_ch != out_ch else nn.Identity())
		self.film = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, out_ch)

	def forward(self, x, tissue_emb, cond_emb):
		res = self.residual(x)
		x = self.dropout(self.relu(self.norm1(self.conv1(x))))
		x = self.dropout(self.norm2(self.conv2(x)))
		x = self.film(x, tissue_emb, cond_emb)
		return self.relu(x + res)


class ResidualBlockDualFiLM(nn.Module):
	"""Non-causal dilated residual block with dual FiLM (tissue + condition)."""

	def __init__(self, in_ch, out_ch, dilation, tissue_emb_dim, cond_emb_dim):
		super().__init__()
		self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
							   dilation=dilation, padding=dilation)
		self.norm1 = nn.GroupNorm(8, out_ch)
		self.relu  = nn.ReLU()
		self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
		self.norm2 = nn.GroupNorm(8, out_ch)
		self.residual = (nn.Conv1d(in_ch, out_ch, kernel_size=1)
						 if in_ch != out_ch else nn.Identity())
		self.film = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, out_ch)

	def forward(self, x, tissue_emb, cond_emb):
		res = self.residual(x)
		x = self.relu(self.norm1(self.conv1(x)))
		x = self.norm2(self.conv2(x))
		x = self.film(x, tissue_emb, cond_emb)
		return self.relu(x + res)


class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding for transformer inputs."""

	def __init__(self, d_model, max_len=4500):
		super().__init__()
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2) *
			(-torch.log(torch.tensor(10000.0)) / d_model)
		)
		pe = torch.zeros(1, max_len, d_model)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════════════
# Model classes — dual FiLM (tissue + condition)
# ═══════════════════════════════════════════════════════════════════════════

class TransModelFiLM(nn.Module):
	"""
	TransModel with dual FiLM conditioning (tissue + condition).

	FiLM modulation applied at:
	  1. After the initial conv1 block
	  2. After the transformer encoder
	  3. In the decoder before the final linear projection
	"""

	def __init__(self, num_genomic_features, target_length=4500, nbins=1500,
				 num_tissues=1, tissue_emb_dim=64,
				 num_conditions=1, cond_emb_dim=32,
				 n_heads=8, dropout=0.3, seqno=False, **kwargs):
		super().__init__()
		self.seqno = seqno
		self.num_tissues    = num_tissues
		self.num_conditions = num_conditions
		self.tissue_emb_dim = tissue_emb_dim
		self.cond_emb_dim   = cond_emb_dim
		self.nbins = nbins

		input_channels = num_genomic_features if seqno else 5 + num_genomic_features

		self.tissue_embedding = nn.Embedding(num_tissues + 1,    tissue_emb_dim)
		self.cond_embedding   = nn.Embedding(num_conditions + 1, cond_emb_dim)

		mid_hidden = 512
		self.conv1 = nn.Sequential(
			nn.Conv1d(input_channels, mid_hidden, kernel_size=129, stride=64, padding=64),
			nn.BatchNorm1d(mid_hidden),
			nn.ReLU(),
		)
		self.conv1_out_len = (target_length + 2 * 64 - 129) // 64 + 1

		self.film_conv1 = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, mid_hidden)

		self.attn       = blocks.AttnModule(hidden=mid_hidden, record_attn=False, inpu_dim=mid_hidden)
		self.film_attn  = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, mid_hidden)

		self.conv2   = nn.Conv1d(mid_hidden, 1, kernel_size=3, stride=1, padding=1)
		self.linear1 = nn.Linear(in_features=self.conv1_out_len, out_features=nbins)
		self.film_dec = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 1)

		self.dropout = nn.Dropout(p=dropout)

	def set_mean_embedding(self):
		with torch.no_grad():
			mean_t = self.tissue_embedding.weight[:self.num_tissues].mean(dim=0)
			self.tissue_embedding.weight[self.num_tissues] = mean_t
			mean_c = self.cond_embedding.weight[:self.num_conditions].mean(dim=0)
			self.cond_embedding.weight[self.num_conditions] = mean_c

	def forward(self, x, tissue_ids, cond_ids):
		if self.seqno:
			x = x[..., -1:]
		t_emb = self.tissue_embedding(tissue_ids)
		c_emb = self.cond_embedding(cond_ids)
		x = x.permute(0, 2, 1).float()

		x = self.conv1(x)
		x = self.film_conv1(x, t_emb, c_emb)

		x = x.permute(0, 2, 1)
		x = self.attn(x)
		x = self.dropout(x)
		x = x.permute(0, 2, 1)
		x = self.film_attn(x, t_emb, c_emb)

		x = self.conv2(x)
		x = self.film_dec(x, t_emb, c_emb)
		x = self.dropout(x)
		x = x.squeeze(1)
		x = self.linear1(x)
		x = F.relu(x)
		return x


class PosTransModelTCNFiLM(nn.Module):
	"""
	PosTransModelTCN with dual FiLM conditioning (tissue + condition).

	Convolutional weights are shared across all groups.
	Separate FiLM layers capture tissue identity and experimental condition
	independently, modulating representations at every TCN block and the decoder.
	"""

	def __init__(self, num_genomic_features, target_length, nbins,
				 num_tissues, tissue_emb_dim=64,
				 num_conditions=1, cond_emb_dim=32,
				 seqno=False, dropout=0.33, **kwargs):
		super().__init__()
		self.seqno          = seqno
		self.num_tissues    = num_tissues
		self.num_conditions = num_conditions
		self.tissue_emb_dim = tissue_emb_dim
		self.cond_emb_dim   = cond_emb_dim

		input_channels = num_genomic_features if seqno else 5 + num_genomic_features

		self.tissue_embedding = nn.Embedding(num_tissues + 1,    tissue_emb_dim)
		self.cond_embedding   = nn.Embedding(num_conditions + 1, cond_emb_dim)

		self.conv_k3  = CausalConv1d(input_channels, 64, kernel_size=3)
		self.conv_k6  = CausalConv1d(input_channels, 64, kernel_size=6)
		self.conv_k25 = CausalConv1d(input_channels, 64, kernel_size=25)
		self.conv_gn   = PositionwiseLayerNorm(192)
		self.conv_relu = nn.ReLU()
		self.film_conv = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 192)

		dilations = context_dilations(target_length, (1, 4, 16, 64, 128, 256))
		self.receptive_field = 25 + 2 * sum(dilations)
		blocks_v2 = []
		in_ch = 192
		for index, dilation in enumerate(dilations):
			out_ch = 256 if index < 2 else 384
			blocks_v2.append(CausalResidualBlockDualFiLM(in_ch, out_ch, dilation, tissue_emb_dim, cond_emb_dim, dropout=dropout))
			in_ch = out_ch
		self.tcn_blocks = nn.ModuleList(blocks_v2)

		self.dec_conv1 = nn.Conv1d(384, 256, kernel_size=1)
		self.dec_norm  = PositionwiseLayerNorm(256)
		self.dec_relu  = nn.ReLU()
		self.dec_conv2 = nn.Conv1d(256, 1, kernel_size=1)
		self.film_dec  = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 256)

		if target_length == nbins:
			self.bin_pool = nn.Identity()
		else:
			stride = target_length // nbins
			self.bin_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

	def set_mean_embedding(self):
		with torch.no_grad():
			mean_t = self.tissue_embedding.weight[:self.num_tissues].mean(dim=0)
			self.tissue_embedding.weight[self.num_tissues] = mean_t
			mean_c = self.cond_embedding.weight[:self.num_conditions].mean(dim=0)
			self.cond_embedding.weight[self.num_conditions] = mean_c

	def forward(self, x, tissue_ids, cond_ids):
		if self.seqno:
			x = x[..., -1:]
		t_emb = self.tissue_embedding(tissue_ids)
		c_emb = self.cond_embedding(cond_ids)
		x = x.permute(0, 2, 1).float()

		x1, x2, x3 = self.conv_k3(x), self.conv_k6(x), self.conv_k25(x)
		ml = min(x1.size(2), x2.size(2), x3.size(2))
		x = torch.cat([x1[:,:,:ml], x2[:,:,:ml], x3[:,:,:ml]], dim=1)
		x = self.conv_relu(self.conv_gn(x))
		x = self.film_conv(x, t_emb, c_emb)

		for blk in self.tcn_blocks:
			x = blk(x, t_emb, c_emb)

		x = self.dec_conv1(x)
		x = self.dec_norm(x)
		x = self.film_dec(x, t_emb, c_emb)
		x = self.dec_relu(x)
		x = self.dec_conv2(x)

		x = self.bin_pool(x)
		return x.squeeze(1)


class PosTransModelFiLM(nn.Module):
	"""
	PosTransModel with dual FiLM conditioning (tissue + condition).

	FiLM applied at:
	  1. After the multi-kernel conv block
	  2. Inside each dilated residual block (after second GroupNorm)
	  3. After the transformer encoder
	  4. In the decoder (after first conv+GroupNorm, before final 1×1)
	"""

	def __init__(self, num_genomic_features, target_length, nbins,
				 num_tissues, tissue_emb_dim=64,
				 num_conditions=1, cond_emb_dim=32,
				 n_heads=6, dropout=0.3, seqno=False, **kwargs):
		super().__init__()
		self.seqno          = seqno
		self.num_tissues    = num_tissues
		self.num_conditions = num_conditions
		self.tissue_emb_dim = tissue_emb_dim
		self.cond_emb_dim   = cond_emb_dim

		input_channels = num_genomic_features if seqno else 5 + num_genomic_features

		self.tissue_embedding = nn.Embedding(num_tissues + 1,    tissue_emb_dim)
		self.cond_embedding   = nn.Embedding(num_conditions + 1, cond_emb_dim)

		self.conv_k3  = nn.Conv1d(input_channels, 64, kernel_size=3, padding='same')
		self.conv_k6  = nn.Conv1d(input_channels, 64, kernel_size=6, padding=3)
		self.conv_k25 = nn.Conv1d(input_channels, 64, kernel_size=25, padding=12)
		self.conv_gn   = nn.GroupNorm(8, 192)
		self.conv_relu = nn.ReLU()
		self.film_conv = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 192)

		self.dilated_blocks = nn.ModuleList([
			ResidualBlockDualFiLM(192, 256, dilation=1,   tissue_emb_dim=tissue_emb_dim, cond_emb_dim=cond_emb_dim),
			ResidualBlockDualFiLM(256, 256, dilation=4,   tissue_emb_dim=tissue_emb_dim, cond_emb_dim=cond_emb_dim),
			ResidualBlockDualFiLM(256, 384, dilation=16,  tissue_emb_dim=tissue_emb_dim, cond_emb_dim=cond_emb_dim),
			ResidualBlockDualFiLM(384, 384, dilation=64,  tissue_emb_dim=tissue_emb_dim, cond_emb_dim=cond_emb_dim),
			ResidualBlockDualFiLM(384, 384, dilation=128, tissue_emb_dim=tissue_emb_dim, cond_emb_dim=cond_emb_dim),
		])

		self.pos_enc = PositionalEncoding(d_model=384, max_len=target_length)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=384, nhead=n_heads, dim_feedforward=1024,
			dropout=dropout, batch_first=True, activation='gelu')
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
		self.film_attn = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 384)

		self.dec_conv1 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
		self.dec_norm  = nn.GroupNorm(8, 256)
		self.dec_relu  = nn.ReLU()
		self.film_dec  = DualFiLMLayer(tissue_emb_dim, cond_emb_dim, 256)
		self.dec_conv2 = nn.Conv1d(256, 1, kernel_size=1)

		if target_length == nbins:
			self.bin_pool = nn.Identity()
		else:
			stride = target_length // nbins
			self.bin_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

	def set_mean_embedding(self):
		with torch.no_grad():
			mean_t = self.tissue_embedding.weight[:self.num_tissues].mean(dim=0)
			self.tissue_embedding.weight[self.num_tissues] = mean_t
			mean_c = self.cond_embedding.weight[:self.num_conditions].mean(dim=0)
			self.cond_embedding.weight[self.num_conditions] = mean_c

	def forward(self, x, tissue_ids, cond_ids):
		if self.seqno:
			x = x[..., -1:]
		t_emb = self.tissue_embedding(tissue_ids)
		c_emb = self.cond_embedding(cond_ids)
		x = x.permute(0, 2, 1).float()

		x1 = self.conv_k3(x)
		x2 = self.conv_k6(x)
		x3 = self.conv_k25(x)
		ml = min(x1.size(2), x2.size(2), x3.size(2))
		x = torch.cat([x1[:,:,:ml], x2[:,:,:ml], x3[:,:,:ml]], dim=1)
		x = self.conv_relu(self.conv_gn(x))
		x = self.film_conv(x, t_emb, c_emb)

		for blk in self.dilated_blocks:
			x = blk(x, t_emb, c_emb)

		x = x.permute(0, 2, 1)
		x = self.pos_enc(x)
		x = self.transformer_encoder(x)
		x = x.permute(0, 2, 1)
		x = self.film_attn(x, t_emb, c_emb)

		x = self.dec_conv1(x)
		x = self.dec_norm(x)
		x = self.film_dec(x, t_emb, c_emb)
		x = self.dec_relu(x)
		x = self.dec_conv2(x)

		x = self.bin_pool(x)
		return x.squeeze(1)


class PosTransModelTCNFiLMRef(nn.Module):
	"""
	Two-pass iterative refinement with dual FiLM (tissue + condition).

	Pass 1: Backbone — full-window causal TCN, peak 256 channels, decoder 128 ch.
	Pass 2: Lightweight refinement — 3 TCN blocks, peak 96 channels.

	Final output = pass1 + sigmoid(gate) * correction
	"""

	def __init__(
		self,
		num_genomic_features,
		target_length,
		nbins,
		num_tissues,
		tissue_emb_dim=64,
		num_conditions=1,
		cond_emb_dim=32,
		dropout=0.33,
		seqno=False,
		**kwargs
	):
		super().__init__()

		self.seqno = seqno
		self.num_tissues = num_tissues
		self.num_conditions = num_conditions
		self.tissue_emb_dim = tissue_emb_dim
		self.cond_emb_dim = cond_emb_dim
		self.target_length = target_length
		self.nbins = nbins

		input_channels = (
			num_genomic_features
			if seqno
			else 5 + num_genomic_features
		)

		self.tissue_embedding = nn.Embedding(
			num_tissues + 1,
			tissue_emb_dim
		)

		self.cond_embedding = nn.Embedding(
			num_conditions + 1,
			cond_emb_dim
		)

		# ── Pass 1: Backbone (full-window dilation schedule, 192→256 channels) ───────────────

		self.conv_k3 = CausalConv1d(
			input_channels,
			64,
			kernel_size=3
		)

		self.conv_k6 = CausalConv1d(
			input_channels,
			64,
			kernel_size=6
		)

		self.conv_k25 = CausalConv1d(
			input_channels,
			64,
			kernel_size=25
		)

		self.conv_gn = PositionwiseLayerNorm(192)
		self.conv_relu = nn.ReLU()

		self.film_conv = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			192
		)

		dilations = context_dilations(target_length, (1, 4, 16, 64, 128, 256))
		self.receptive_field = 25 + 2 * sum(dilations)
		blocks_v2 = []
		in_ch = 192
		for index, dilation in enumerate(dilations):
			out_ch = 256 if index < 2 else 256
			blocks_v2.append(CausalResidualBlockDualFiLM(in_ch, out_ch, dilation, tissue_emb_dim, cond_emb_dim, dropout=dropout))
			in_ch = out_ch
		self.tcn_blocks = nn.ModuleList(blocks_v2)

		self.dec_conv1 = nn.Conv1d(
			256,
			128,
			kernel_size=1
		)

		self.dec_norm = PositionwiseLayerNorm(128)
		self.dec_relu = nn.ReLU()

		self.dec_conv2 = nn.Conv1d(
			128,
			1,
			kernel_size=1
		)

		self.film_dec = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			128
		)

		if target_length == nbins:
			self.bin_pool = nn.Identity()
			self.pool_k = 1
		else:
			self.pool_k = target_length // nbins

			self.bin_pool = nn.AvgPool1d(
				kernel_size=self.pool_k,
				stride=self.pool_k
			)

		# ── Pass 2: Refinement (3 blocks, 64→96 channels) ───────────────

		refine_in = input_channels + 1

		self.ref_conv_k3 = CausalConv1d(
			refine_in,
			32,
			kernel_size=3
		)

		self.ref_conv_k6 = CausalConv1d(
			refine_in,
			32,
			kernel_size=6
		)

		self.ref_conv_k25 = CausalConv1d(
			refine_in,
			32,
			kernel_size=25
		)

		self.ref_conv_gn = PositionwiseLayerNorm(96)
		self.ref_conv_relu = nn.ReLU()

		self.ref_film_conv = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			96
		)

		self.ref_tcn_blocks = nn.ModuleList([
			CausalResidualBlockDualFiLM(
				96,
				96,
				dilation=1,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim, dropout=dropout
			),

			CausalResidualBlockDualFiLM(
				96,
				96,
				dilation=16,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim, dropout=dropout
			),

			CausalResidualBlockDualFiLM(
				96,
				96,
				dilation=64,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim, dropout=dropout
			),
		])

		self.ref_dec_conv1 = nn.Conv1d(
			96,
			64,
			kernel_size=1
		)

		self.ref_dec_norm = PositionwiseLayerNorm(64)
		self.ref_dec_relu = nn.ReLU()

		self.ref_dec_conv2 = nn.Conv1d(
			64,
			1,
			kernel_size=1
		)

		self.ref_film_dec = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			64
		)

		self.refine_gate = nn.Parameter(
			torch.tensor(0.0)
		)

	def set_mean_embedding(self):

		with torch.no_grad():

			mean_t = self.tissue_embedding.weight[
				:self.num_tissues
			].mean(dim=0)

			self.tissue_embedding.weight[
				self.num_tissues
			] = mean_t

			mean_c = self.cond_embedding.weight[
				:self.num_conditions
			].mean(dim=0)

			self.cond_embedding.weight[
				self.num_conditions
			] = mean_c

	def _backbone_forward(self, x, t_emb, c_emb):

		x1 = self.conv_k3(x)
		x2 = self.conv_k6(x)
		x3 = self.conv_k25(x)

		ml = min(
			x1.size(2),
			x2.size(2),
			x3.size(2)
		)

		x = torch.cat([
			x1[:, :, :ml],
			x2[:, :, :ml],
			x3[:, :, :ml]
		], dim=1)

		x = self.conv_relu(
			self.conv_gn(x)
		)

		x = self.film_conv(
			x,
			t_emb,
			c_emb
		)

		for blk in self.tcn_blocks:
			x = blk(x, t_emb, c_emb)

		x = self.dec_conv1(x)
		x = self.dec_norm(x)

		x = self.film_dec(
			x,
			t_emb,
			c_emb
		)

		x = self.dec_relu(x)
		x = self.dec_conv2(x)
		x = self.bin_pool(x)

		return x.squeeze(1)

	def _refine_forward(self, x, t_emb, c_emb):

		r1 = self.ref_conv_k3(x)
		r2 = self.ref_conv_k6(x)
		r3 = self.ref_conv_k25(x)

		ml = min(
			r1.size(2),
			r2.size(2),
			r3.size(2)
		)

		r = torch.cat([
			r1[:, :, :ml],
			r2[:, :, :ml],
			r3[:, :, :ml]
		], dim=1)

		r = self.ref_conv_relu(
			self.ref_conv_gn(r)
		)

		r = self.ref_film_conv(
			r,
			t_emb,
			c_emb
		)

		for blk in self.ref_tcn_blocks:
			r = blk(r, t_emb, c_emb)

		r = self.ref_dec_conv1(r)
		r = self.ref_dec_norm(r)

		r = self.ref_film_dec(
			r,
			t_emb,
			c_emb
		)

		r = self.ref_dec_relu(r)
		r = self.ref_dec_conv2(r)
		r = self.bin_pool(r)

		return r.squeeze(1)

	def forward(self, x, tissue_ids, cond_ids):
		if self.seqno:
			x = x[..., -1:]

		t_emb = self.tissue_embedding(tissue_ids)
		c_emb = self.cond_embedding(cond_ids)

		x_ch = x.permute(0, 2, 1).float()

		pred1 = self._backbone_forward(
			x_ch,
			t_emb,
			c_emb
		)

		# Upsample pass-1 prediction back to seq_len for refinement input

		pred1_up = pred1.unsqueeze(1).repeat_interleave(
			self.pool_k,
			dim=2
		)

		seq_len = x_ch.size(2)

		if pred1_up.size(2) > seq_len:
			pred1_up = pred1_up[:, :, :seq_len]

		elif pred1_up.size(2) < seq_len:
			pred1_up = F.pad(
				pred1_up,
				(0, seq_len - pred1_up.size(2))
			)

		x_refine = torch.cat([
			x_ch,
			(pred1_up.detach() if self.training else pred1_up)
		], dim=1)

		correction = self._refine_forward(
			x_refine,
			t_emb,
			c_emb
		)

		gate = torch.sigmoid(self.refine_gate)

		return pred1 + gate * correction


class PosTransModelFiLMRef(nn.Module):
	"""
	PosTransModelFiLM (transformer backbone) with two-pass iterative
	refinement, dual FiLM (tissue + condition).

	Pass 1: Backbone — identical to PosTransModelFiLM: multi-kernel conv
	(192 ch) -> 5 non-causal dilated residual blocks (192->384 ch) ->
	transformer encoder (3 layers, d_model=384) -> decoder (256 ch).

	Pass 2: Lightweight refinement — multi-kernel conv (96 ch) -> 3
	non-causal dilated residual blocks (96 ch) -> decoder (64 ch).
	No transformer in pass 2 on purpose: the correction at this stage
	is local (fixing residual bias around the pass-1 prediction), and a
	second self-attention pass would erase most of the speed/memory
	benefit of doing this in two passes.

	Final output = pass1 + sigmoid(gate) * correction
	"""

	def __init__(
		self,
		num_genomic_features,
		target_length,
		nbins,
		num_tissues,
		tissue_emb_dim=64,
		num_conditions=1,
		cond_emb_dim=32,
		n_heads=6,
		dropout=0.3,
		seqno=False,
		**kwargs
	):
		super().__init__()

		self.seqno = seqno
		self.num_tissues = num_tissues
		self.num_conditions = num_conditions
		self.tissue_emb_dim = tissue_emb_dim
		self.cond_emb_dim = cond_emb_dim
		self.target_length = target_length
		self.nbins = nbins

		input_channels = (
			num_genomic_features
			if seqno
			else 5 + num_genomic_features
		)

		self.tissue_embedding = nn.Embedding(
			num_tissues + 1,
			tissue_emb_dim
		)

		self.cond_embedding = nn.Embedding(
			num_conditions + 1,
			cond_emb_dim
		)

		# ── Pass 1: Backbone (identical to PosTransModelFiLM) ───────────

		self.conv_k3 = nn.Conv1d(
			input_channels,
			64,
			kernel_size=3,
			padding='same'
		)

		self.conv_k6 = nn.Conv1d(
			input_channels,
			64,
			kernel_size=6,
			padding=3
		)

		self.conv_k25 = nn.Conv1d(
			input_channels,
			64,
			kernel_size=25,
			padding=12
		)

		self.conv_gn = nn.GroupNorm(8, 192)
		self.conv_relu = nn.ReLU()

		self.film_conv = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			192
		)

		self.dilated_blocks = nn.ModuleList([
			ResidualBlockDualFiLM(
				192,
				256,
				dilation=1,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				256,
				256,
				dilation=4,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				256,
				384,
				dilation=16,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				384,
				384,
				dilation=64,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				384,
				384,
				dilation=128,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),
		])

		self.pos_enc = PositionalEncoding(
			d_model=384,
			max_len=target_length
		)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=384,
			nhead=n_heads,
			dim_feedforward=1024,
			dropout=dropout,
			batch_first=True,
			activation='gelu'
		)

		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=3
		)

		self.film_attn = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			384
		)

		self.dec_conv1 = nn.Conv1d(
			384,
			256,
			kernel_size=3,
			padding=1
		)

		self.dec_norm = nn.GroupNorm(8, 256)
		self.dec_relu = nn.ReLU()

		self.film_dec = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			256
		)

		self.dec_conv2 = nn.Conv1d(
			256,
			1,
			kernel_size=1
		)

		if target_length == nbins:
			self.bin_pool = nn.Identity()
			self.pool_k = 1
		else:
			self.pool_k = target_length // nbins

			self.bin_pool = nn.AvgPool1d(
				kernel_size=self.pool_k,
				stride=self.pool_k
			)

		# ── Pass 2: Refinement (3 blocks, 96 ch, no transformer) ────────

		refine_in = input_channels + 1

		self.ref_conv_k3 = nn.Conv1d(
			refine_in,
			32,
			kernel_size=3,
			padding='same'
		)

		self.ref_conv_k6 = nn.Conv1d(
			refine_in,
			32,
			kernel_size=6,
			padding=3
		)

		self.ref_conv_k25 = nn.Conv1d(
			refine_in,
			32,
			kernel_size=25,
			padding=12
		)

		self.ref_conv_gn = nn.GroupNorm(8, 96)
		self.ref_conv_relu = nn.ReLU()

		self.ref_film_conv = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			96
		)

		self.ref_blocks = nn.ModuleList([
			ResidualBlockDualFiLM(
				96,
				96,
				dilation=1,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				96,
				96,
				dilation=16,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),

			ResidualBlockDualFiLM(
				96,
				96,
				dilation=64,
				tissue_emb_dim=tissue_emb_dim,
				cond_emb_dim=cond_emb_dim
			),
		])

		self.ref_dec_conv1 = nn.Conv1d(
			96,
			64,
			kernel_size=1
		)

		self.ref_dec_norm = nn.GroupNorm(8, 64)
		self.ref_dec_relu = nn.ReLU()

		self.ref_film_dec = DualFiLMLayer(
			tissue_emb_dim,
			cond_emb_dim,
			64
		)

		self.ref_dec_conv2 = nn.Conv1d(
			64,
			1,
			kernel_size=1
		)

		self.refine_gate = nn.Parameter(
			torch.tensor(0.0)
		)

	def set_mean_embedding(self):

		with torch.no_grad():

			mean_t = self.tissue_embedding.weight[
				:self.num_tissues
			].mean(dim=0)

			self.tissue_embedding.weight[
				self.num_tissues
			] = mean_t

			mean_c = self.cond_embedding.weight[
				:self.num_conditions
			].mean(dim=0)

			self.cond_embedding.weight[
				self.num_conditions
			] = mean_c

	def _backbone_forward(self, x, t_emb, c_emb):

		x1 = self.conv_k3(x)
		x2 = self.conv_k6(x)
		x3 = self.conv_k25(x)

		ml = min(
			x1.size(2),
			x2.size(2),
			x3.size(2)
		)

		x = torch.cat([
			x1[:, :, :ml],
			x2[:, :, :ml],
			x3[:, :, :ml]
		], dim=1)

		x = self.conv_relu(
			self.conv_gn(x)
		)

		x = self.film_conv(
			x,
			t_emb,
			c_emb
		)

		for blk in self.dilated_blocks:
			x = blk(x, t_emb, c_emb)

		x = x.permute(0, 2, 1)
		x = self.pos_enc(x)
		x = self.transformer_encoder(x)
		x = x.permute(0, 2, 1)

		x = self.film_attn(
			x,
			t_emb,
			c_emb
		)

		x = self.dec_conv1(x)
		x = self.dec_norm(x)

		x = self.film_dec(
			x,
			t_emb,
			c_emb
		)

		x = self.dec_relu(x)
		x = self.dec_conv2(x)
		x = self.bin_pool(x)

		return x.squeeze(1)

	def _refine_forward(self, x, t_emb, c_emb):

		r1 = self.ref_conv_k3(x)
		r2 = self.ref_conv_k6(x)
		r3 = self.ref_conv_k25(x)

		ml = min(
			r1.size(2),
			r2.size(2),
			r3.size(2)
		)

		r = torch.cat([
			r1[:, :, :ml],
			r2[:, :, :ml],
			r3[:, :, :ml]
		], dim=1)

		r = self.ref_conv_relu(
			self.ref_conv_gn(r)
		)

		r = self.ref_film_conv(
			r,
			t_emb,
			c_emb
		)

		for blk in self.ref_blocks:
			r = blk(r, t_emb, c_emb)

		r = self.ref_dec_conv1(r)
		r = self.ref_dec_norm(r)

		r = self.ref_film_dec(
			r,
			t_emb,
			c_emb
		)

		r = self.ref_dec_relu(r)
		r = self.ref_dec_conv2(r)
		r = self.bin_pool(r)

		return r.squeeze(1)

	def forward(self, x, tissue_ids, cond_ids):
		if self.seqno:
			x = x[..., -1:]

		t_emb = self.tissue_embedding(tissue_ids)
		c_emb = self.cond_embedding(cond_ids)

		x_ch = x.permute(0, 2, 1).float()

		pred1 = self._backbone_forward(
			x_ch,
			t_emb,
			c_emb
		)

		# Upsample pass-1 prediction back to seq_len for refinement input

		pred1_up = pred1.unsqueeze(1).repeat_interleave(
			self.pool_k,
			dim=2
		)

		seq_len = x_ch.size(2)

		if pred1_up.size(2) > seq_len:
			pred1_up = pred1_up[:, :, :seq_len]

		elif pred1_up.size(2) < seq_len:
			pred1_up = F.pad(
				pred1_up,
				(0, seq_len - pred1_up.size(2))
			)

		x_refine = torch.cat([
			x_ch,
			(pred1_up.detach() if self.training else pred1_up)
		], dim=1)

		correction = self._refine_forward(
			x_refine,
			t_emb,
			c_emb
		)

		gate = torch.sigmoid(self.refine_gate)

		return pred1 + gate * correction


if __name__ == '__main__':
	main()
