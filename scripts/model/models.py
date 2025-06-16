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


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=6000):
		super().__init__()
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe = torch.zeros(1, max_len, d_model)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)]
		return x


class PosTransModel(nn.Module):
	def __init__(self, num_genomic_features, mid_hidden, target_length, nbins, seqno=False):
		super().__init__()
		self.seqno = seqno
		input_channels = num_genomic_features if self.seqno else 5 + num_genomic_features

		# --- 1. Multi-Kernel Conv Block ---
		self.conv_k3 = nn.Conv1d(input_channels, 64, kernel_size=3, padding='same')   # codon-level
		self.conv_k6 = nn.Conv1d(input_channels, 64, kernel_size=6, padding=3)   # dicodon-level
		self.conv_k25 = nn.Conv1d(input_channels, 64, kernel_size=25, padding=12)  # long motifs
		self.conv_bn = nn.BatchNorm1d(64 * 3)
		self.conv_relu = nn.ReLU()

		# --- 2. Dilated Convolutions ---
		self.dilated_convs = nn.Sequential(
			self._make_residual_block(192, 256, dilation=1),
			self._make_residual_block(256, 256, dilation=4),
			self._make_residual_block(256, 384, dilation=16),
			self._make_residual_block(384, 384, dilation=64),
			self._make_residual_block(384, 384, dilation=128)
		)

		# --- 3. Transformer Encoder ---
		self.pos_enc = PositionalEncoding(d_model=384, max_len=target_length)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=384,
			nhead=6,
			dim_feedforward=1024,
			dropout=0.3,
			batch_first=True,
			activation='gelu'
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

		# --- 4. Decoder ---
		self.final_conv = nn.Sequential(
			nn.Conv1d(384, 256, kernel_size=3, padding=1),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 1, kernel_size=1)
		)

		# --- 5. Downsampling to nbins (e.g., 1000 if 6kb regions in 6bp bins) ---
		if target_length == nbins:
			self.bin_pool = nn.Identity()
		else:
			stride = target_length // nbins
			self.bin_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

	def _make_residual_block(self, in_channels, out_channels, dilation):
		return nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
			nn.Conv1d(out_channels, out_channels, kernel_size=1),
			nn.BatchNorm1d(out_channels)
		)

	def forward(self, x):
		x = x.permute(0, 2, 1).float()  # (batch, channels, seq_len)

		# Multi-kernel convolutions
		x1 = self.conv_k3(x)
		x2 = self.conv_k6(x)
		x3 = self.conv_k25(x)
		
		# Get minimum length across all three outputs
		min_len = min(x1.size(2), x2.size(2), x3.size(2))
		
		# Slice all tensors to the minimum length
		x1 = x1[:, :, :min_len]
		x2 = x2[:, :, :min_len]
		x3 = x3[:, :, :min_len]
		
		# Now concatenate (batch, 192, min_len)
		x = torch.cat([x1, x2, x3], dim=1)
		x = self.conv_bn(x)
		x = self.conv_relu(x)

		# Dilated convolution block
		x = self.dilated_convs(x)

		# Transformer block
		x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
		x = self.pos_enc(x)
		x = self.transformer_encoder(x)
		x = x.permute(0, 2, 1)  # (batch, channels, seq_len)

		# Decoder + pooling
		x = self.final_conv(x)
		x = self.bin_pool(x)

		return x.squeeze(1)  # (batch, nbins)


class TransModel(nn.Module):
	def __init__(self, num_genomic_features, mid_hidden, target_length=9000, nbins=180):
		super(TransModel, self).__init__()
		print('Initializing TransModel')
		# First convolution (aggressive downsampling)
		self.conv1 = nn.Sequential(
			nn.Conv1d(6, 512, kernel_size=129, stride=64, padding=64),  # Reduced stride from 128->64
			nn.BatchNorm1d(512),
			nn.ReLU(),
		)
		
		# Attention module (matches conv1 output channels)
		self.attn = blocks.AttnModule(hidden=mid_hidden, record_attn=False, inpu_dim=512)
		
		# Second convolution (channel reduction)
		self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # Changed to maintain 512 channels
		
		# Final prediction layer
		self.Linear1 = nn.Linear(in_features=512, out_features=nbins)
		
		self.record_attn = False
		self.dropout = nn.Dropout(p=0.1)

	def forward(self, x):
		# Input shape: (batch, 6, 9000)
		x = x.transpose(1, 2).contiguous().float()
		
		# Conv1: (batch, 512, L_out)
		# L_out = floor((9000 + 2*64 - 129)/64) + 1 = floor(8999/64) + 1 = 140 + 1 = 141
		x = self.conv1(x)
		
		# Prepare for attention: (batch, seq_len, features)
		x = x.transpose(1, 2).contiguous().float()  # (batch, 141, 512)
		
		# Attention processing
		if self.record_attn:
			x, attn_weights = self.attn(x)
		else:
			x = self.attn(x)
		
		x = self.dropout(x)
		
		# Prepare for conv2: (batch, 512, 141)
		x = x.transpose(1, 2).contiguous().float()
		
		# Conv2: maintains shape (batch, 512, 141)
		x = self.conv2(x)
		x = self.dropout(x)
		
		# Global average pooling before linear
		x = F.adaptive_avg_pool1d(x, 1)  # (batch, 512, 1)
		x = x.squeeze(-1)  # (batch, 512)
		
		# Final prediction
		x = self.Linear1(x)  # (batch, 180)
		x = F.relu(x)
		
		if self.record_attn:
			return x, attn_weights
		else:
			return x


class PosTransModelRiboPos(nn.Module):
	def __init__(self, num_genomic_features, mid_hidden, target_length, nbins, seqno=False):
		super().__init__()
		self.seqno = seqno
		input_channels = num_genomic_features if self.seqno else 5 + num_genomic_features

		# --- 1. Multi-Kernel Conv Block ---
		self.conv_k3 = nn.Conv1d(input_channels, 64, kernel_size=3, padding='same')
		self.conv_k6 = nn.Conv1d(input_channels, 64, kernel_size=6, padding=3)
		self.conv_k25 = nn.Conv1d(input_channels, 64, kernel_size=25, padding=12)
		self.conv_bn = nn.BatchNorm1d(64 * 3)
		self.conv_relu = nn.ReLU()

		# --- 2. Dilated Convolutions ---
		self.dilated_convs = nn.Sequential(
			self._make_residual_block(192, 256, dilation=1),
			self._make_residual_block(256, 256, dilation=4),
			self._make_residual_block(256, 384, dilation=16),
			self._make_residual_block(384, 384, dilation=64),
			self._make_residual_block(384, 384, dilation=128)
		)

		# --- 3. Causal Transformer Encoder ---
		self.pos_enc = PositionalEncoding(d_model=384, max_len=target_length)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=384,
			nhead=6,
			dim_feedforward=1024,
			dropout=0.3,
			batch_first=True,
			activation='gelu'
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

		# --- 4. Decoder ---
		self.final_conv = nn.Sequential(
			nn.Conv1d(384, 256, kernel_size=3, padding=1),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 1, kernel_size=1)
		)

		# --- 5. Downsampling ---
		if target_length == nbins:
			self.bin_pool = nn.Identity()
		else:
			stride = target_length // nbins
			self.bin_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

	def _make_residual_block(self, in_channels, out_channels, dilation):
		return nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
			nn.Conv1d(out_channels, out_channels, kernel_size=1),
			nn.BatchNorm1d(out_channels)
		)

	def forward(self, x):
		# --- Feature Extraction ---
		x = x.permute(0, 2, 1).float()
		
		# Multi-kernel convolutions
		x1 = self.conv_k3(x)
		x2 = self.conv_k6(x)
		x3 = self.conv_k25(x)
		min_len = min(x1.size(2), x2.size(2), x3.size(2))
		x = torch.cat([x1[:, :, :min_len], x2[:, :, :min_len], x3[:, :, :min_len]], dim=1)
		x = self.conv_bn(x)
		x = self.conv_relu(x)

		# Dilated convolutions
		x = self.dilated_convs(x)

		# --- Causal Transformer ---
		x = x.permute(0, 2, 1)
		x = self.pos_enc(x)
		
		# Create causal mask
		seq_len = x.size(1)
		mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
		
		x = self.transformer_encoder(x, mask=mask)
		x = x.permute(0, 2, 1)

		# --- Prediction ---
		x = self.final_conv(x)
		x = self.bin_pool(x)
		return x.squeeze(1)


class PosTransModel2step(nn.Module):
	def __init__(self, num_genomic_features, mid_hidden=384, target_length=1000, nbins=10, seqno=False):
		super().__init__()
		self.seqno = seqno
		input_channels = num_genomic_features if self.seqno else 5 + num_genomic_features
		self.target_length = target_length
		self.nbins = nbins
		
		# --- uORF Attention Module ---
		self.uorf_attention = nn.Sequential(
			nn.Conv1d(mid_hidden, mid_hidden, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.BatchNorm1d(mid_hidden),
			nn.Conv1d(mid_hidden, mid_hidden, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.BatchNorm1d(mid_hidden),
			nn.Conv1d(mid_hidden, 1, kernel_size=1),
			nn.Sigmoid()
		)
		
		# --- Positional Encoding ---
		self.pos_enc = PositionalEncoding(d_model=mid_hidden, max_len=target_length)
		
		# --- Multi-kernel Convolutions ---
		self.conv1 = nn.Sequential(
			nn.Conv1d(input_channels, mid_hidden, kernel_size=3, padding=1),
			nn.BatchNorm1d(mid_hidden),
			nn.ReLU()
		)
		self.conv3 = nn.Sequential(
			nn.Conv1d(input_channels, mid_hidden, kernel_size=9, padding=4),
			nn.BatchNorm1d(mid_hidden),
			nn.ReLU()
		)
		self.conv5 = nn.Sequential(
			nn.Conv1d(input_channels, mid_hidden, kernel_size=15, padding=7),
			nn.BatchNorm1d(mid_hidden),
			nn.ReLU()
		)
		self.conv_merge = nn.Sequential(
			nn.Conv1d(mid_hidden*3, mid_hidden, kernel_size=1),
			nn.BatchNorm1d(mid_hidden),
			nn.ReLU()
		)
		
		# --- Dilated Convolution Block ---
		self.dilated_convs = nn.ModuleList([
			nn.Sequential(
				nn.Conv1d(mid_hidden, mid_hidden, kernel_size=3, padding=2, dilation=2),
				nn.BatchNorm1d(mid_hidden),
				nn.ReLU()
			),
			nn.Sequential(
				nn.Conv1d(mid_hidden, mid_hidden, kernel_size=3, padding=4, dilation=4),
				nn.BatchNorm1d(mid_hidden),
				nn.ReLU()
			),
			nn.Sequential(
				nn.Conv1d(mid_hidden, mid_hidden, kernel_size=3, padding=8, dilation=8),
				nn.BatchNorm1d(mid_hidden),
				nn.ReLU()
			)
		])
		
		# --- Transformer Layers ---
		transformer_layer = nn.TransformerEncoderLayer(
			d_model=mid_hidden,
			nhead=8,  # 384/8 = 48 (divisible)
			dim_feedforward=1024,
			dropout=0.3,
			batch_first=True,
			activation='gelu'
		)
		self.uorf_effect_encoder = transformer_layer
		self.transformer_encoder = nn.TransformerEncoder(
			transformer_layer,
			num_layers=3
		)
		
		# --- Final Layers ---
		self.final_conv = nn.Sequential(
			nn.Conv1d(mid_hidden, mid_hidden//2, kernel_size=1),
			nn.BatchNorm1d(mid_hidden//2),
			nn.ReLU(),
			nn.Conv1d(mid_hidden//2, 1, kernel_size=1)
		)
		self.bin_pool = nn.AdaptiveAvgPool1d(nbins)

	def forward(self, x):
		# Ensure all paths use all parameters
		x = x.permute(0, 2, 1).float()
		
		# Multi-kernel convolutions
		x1 = self.conv1(x)
		x3 = self.conv3(x)
		x5 = self.conv5(x)
		x = torch.cat([x1, x3, x5], dim=1)
		x = self.conv_merge(x)
		
		# Dilated convolutions
		residual = x
		for conv in self.dilated_convs:
			x = conv(x)
		x = x + residual
		
		# Attention - ensure this always runs
		uorf_probs = self.uorf_attention(x)
		uorf_mask = (uorf_probs > 0.5).float()
		uorf_mask = uorf_mask.detach() + uorf_probs - uorf_probs.detach()
		
		# Transformer
		x_seq = x.permute(0, 2, 1)
		x_seq = self.pos_enc(x_seq)
		
		# Create mask
		seq_len = x_seq.size(1)
		device = x_seq.device
		attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'))
		attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
		
		# Apply uORF attention
		uorf_positions = (uorf_mask.squeeze(1) > 0.5)
		for b in range(x_seq.size(0)):
			for i in torch.where(uorf_positions[b])[0]:
				if i < seq_len - 1:
					attn_mask[i, i+1:] = 0.0
		
		# Ensure transformers are always used
		x_seq = self.uorf_effect_encoder(x_seq, src_mask=attn_mask)
		x_seq = self.transformer_encoder(x_seq, mask=attn_mask)
		
		# Final processing
		x = x_seq.permute(0, 2, 1)
		x = self.final_conv(x)
		x = self.bin_pool(x)
		
		return x.squeeze(1)

if __name__ == '__main__':
	main()
