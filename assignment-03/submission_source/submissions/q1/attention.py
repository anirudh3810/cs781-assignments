"""Multi-head causal self-attention, implemented from raw tensor operations."""
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.attention_scale = (1.0 / math.sqrt(self.d_head)
                                if attention_scale is None else float(attention_scale))
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.causal_mask.size(0)}"
            )
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length]
        scores = scores.masked_fill(~allowed.view(1, 1, length, length), float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))
