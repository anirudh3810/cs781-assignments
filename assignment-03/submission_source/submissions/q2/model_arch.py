"""Q2 decoder-only LM dispatching between standard MHA/GQA/MQA and Q2 MLA."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from attention import CausalSelfAttention, MultiHeadLatentAttention


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False
    attention_type: str = "standard"
    kv_latent_dim: int = None


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    if max_len <= 0 or d_model <= 0:
        raise ValueError("max_len and d_model must be positive")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * even_indices / d_model)
    angles = position * frequencies.unsqueeze(0)
    encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm_first = bool(cfg.norm_first)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        if cfg.attention_type == "mla":
            if cfg.pos_emb_type != "sinusoidal":
                raise ValueError("Q2 MLA requires sinusoidal positional encoding")
            self.attn = MultiHeadLatentAttention(
                cfg.d_model, cfg.n_heads, cfg.kv_latent_dim, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale,
            )
        elif cfg.attention_type == "standard":
            if cfg.n_kv_heads is None:
                raise ValueError("standard attention requires n_kv_heads")
            self.attn = CausalSelfAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale,
            )
        else:
            raise ValueError("attention_type must be 'standard' or 'mla'")
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class DecoderLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        if cfg.vocab_size <= 0 or cfg.n_layers <= 0:
            raise ValueError("vocab_size and n_layers must be positive")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            self.register_buffer(
                "pos_emb",
                sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
                persistent=False,
            )
        elif cfg.pos_emb_type != "rope":
            raise ValueError("pos_emb_type must be 'sinusoidal' or 'rope'")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        _, length = idx.shape
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        x = self.tok_emb(idx)
        if self.cfg.pos_emb_type == "sinusoidal":
            x = x + self.pos_emb[:length].to(dtype=x.dtype).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
