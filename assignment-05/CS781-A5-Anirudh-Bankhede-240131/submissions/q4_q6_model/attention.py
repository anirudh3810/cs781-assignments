"""Manual MHA/GQA/MQA causal attention with optional split-half RoPE."""
import math
import torch
import torch.nn as nn
from rope import build_rope_cache, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len,
                 dropout=0.0, use_rope=False, attention_scale=None):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("Model/head widths must be positive")
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("d_model must divide by n_heads, and n_heads by n_kv_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_heads, self.n_kv_heads = int(n_heads), int(n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_head = d_model // self.n_heads
        self.use_rope = bool(use_rope)
        if self.use_rope and self.d_head % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.attention_scale = (1.0 / math.sqrt(self.d_head)
                                if attention_scale is None else float(attention_scale))
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(
            max_seq_len, max_seq_len, dtype=torch.bool)), persistent=False)
        if self.use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Attention expects (batch, sequence, d_model)")
        batch, length, width = x.shape
        if length > self.causal_mask.shape[0]:
            raise ValueError("Sequence exceeds configured attention capacity")
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            cos, sin = self.rope_cos[:length], self.rope_sin[:length]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        # Each KV head is shared by a contiguous group of query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length].view(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = self.attn_dropout(weights)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.out_proj(output))
