"""Manual MHA/GQA/MQA and Q2 joint-latent attention implementations."""
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Causal attention with independent query-head and key/value-head counts."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("d_model, n_heads, and n_kv_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_head = int(d_model) // self.n_heads
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        )
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.causal_mask.size(0)}"
            )
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        # Contiguous query-head groups share each KV head: [k0...k0, k1...k1, ...].
        if self.n_rep != 1:
            k = torch.repeat_interleave(k, self.n_rep, dim=1)
            v = torch.repeat_interleave(v, self.n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length].view(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = self.attn_dropout(weights)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))


class MultiHeadLatentAttention(nn.Module):
    """Q2 joint low-rank KV compression with direct, uncompressed queries."""
    def __init__(self, d_model: int, n_heads: int, kv_latent_dim: int,
                 max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not isinstance(kv_latent_dim, int) or not 0 < kv_latent_dim < d_model:
            raise ValueError("kv_latent_dim must be an integer strictly between 0 and d_model")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_heads = int(n_heads)
        self.d_head = int(d_model) // self.n_heads
        self.kv_latent_dim = int(kv_latent_dim)
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        )
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, self.kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(self.kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(self.kv_latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.causal_mask.size(0)}"
            )
        latent = self.kv_down_proj(x)  # computed once and shared by both reconstruction branches
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_up_proj(latent).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up_proj(latent).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length].view(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = self.attn_dropout(weights)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))
