"""Decoder-only language model using PyTorch TransformerEncoder building blocks."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class BuiltinLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_first: bool = False


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    if max_len <= 0 or d_model <= 0:
        raise ValueError('max_len and d_model must be positive')
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    angles = position * torch.exp(-math.log(10000.0) * even_indices / d_model).unsqueeze(0)
    encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding


class BuiltinDecoderLM(nn.Module):
    """Causal LM backed by nn.TransformerEncoderLayer/nn.TransformerEncoder."""
    def __init__(self, cfg: BuiltinLMConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads or cfg.max_seq_len < 1:
            raise ValueError('Invalid head division or sequence capacity')
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer('pos_emb',
                             sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
                             persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation='gelu', batch_first=True,
            norm_first=cfg.norm_first, bias=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        # Q1 uses bias-free attention projections but biased FFN/LayerNorm. The high-level
        # layer has one shared `bias` switch, so keep FFN/LN bias and freeze zero MHA bias.
        for block in self.encoder.layers:
            for bias in (block.self_attn.in_proj_bias, block.self_attn.out_proj.bias):
                if bias is not None:
                    nn.init.zeros_(bias)
                    bias.requires_grad_(False)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError('idx must have shape (batch, sequence)')
        batch, length = idx.shape
        if length < 1 or length > self.cfg.max_seq_len:
            raise ValueError(f'Sequence length {length} outside 1..{self.cfg.max_seq_len}')
        x = self.tok_emb(idx) + self.pos_emb[:length].to(dtype=self.tok_emb.weight.dtype).unsqueeze(0)
        # Float convention: 0 allowed, -inf blocked. Never rely on auto-causality hints.
        blocked = torch.ones(length, length, device=idx.device, dtype=torch.bool).triu(1)
        causal_mask = torch.zeros(length, length, device=idx.device, dtype=x.dtype)
        causal_mask = causal_mask.masked_fill(blocked, float('-inf'))
        x = self.encoder(x, mask=causal_mask)
        return self.head(self.ln_f(x))
