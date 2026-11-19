"""Split-half rotary positional embeddings for the manual Q3 attention layer."""
import torch


def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0):
    """Return cos/sin caches of shape (T, d_head) for split-half channel pairs."""
    if max_seq_len <= 0 or d_head <= 0 or d_head % 2:
        raise ValueError("max_seq_len must be positive and d_head must be positive and even")
    if base <= 1:
        raise ValueError("RoPE base must be greater than one")
    pair_index = torch.arange(d_head // 2, dtype=torch.float32)
    inverse_frequency = base ** (-2.0 * pair_index / d_head)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequency)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair channel i with channel i+d_head/2, returning a 90-degree rotation."""
    if x.shape[-1] <= 0 or x.shape[-1] % 2:
        raise ValueError("RoPE requires a positive, even final dimension")
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (batch, heads, sequence, head_dim) without changing its shape."""
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape (batch, heads, sequence, head_dim)")
    expected = (x.shape[-2], x.shape[-1])
    if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
        raise ValueError(f"RoPE cache must have shape {expected}")
    cos = cos.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin
