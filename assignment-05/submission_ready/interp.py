"""interp.py — Q4-Q6: port and analyze YOUR A-5 Q1 DecoderLM.

Q4  port the DecoderLM into TransformerLens, verify logit parity, compare checkpoints.
Q5  attention-head analysis: previous-token and induction scores, single-head ablation.
Q6  residual stream: logit lens and an exact direct-logit-attribution decomposition.

Conventions: X is a batch of doubled passages (B, 2P+1); the model input is X[:, :-1]
(length 2P), and input position q predicts token X[:, q+1].
"""
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens import utils as tl_utils
from tl_analysis_support import install_analysis_support
from tl_analysis_support import logit_lens as _shared_logit_lens
from tl_analysis_support import decompose_correct_logit as _shared_decompose_correct_logit
from tl_analysis_support import summarize_dla as _shared_summarize_dla


# =========================================================================== Q4
def to_hooked(lm):
    """Copy the trained A3 decoder into a real TransformerLens HookedTransformer."""
    cfg = lm.cfg
    d_head = cfg.d_model // cfg.n_heads
    if cfg.attention_type != "standard" or cfg.kv_latent_dim is not None:
        raise ValueError("The Q1 handoff requires standard MHA/GQA/MQA attention")
    if cfg.pos_emb_type not in ("sinusoidal", "rope"):
        raise ValueError(f"Unsupported native position type: {cfg.pos_emb_type}")
    native_scale = lm.blocks[0].attn.attention_scale
    if any(abs(block.attn.attention_scale - native_scale) > 1e-12 for block in lm.blocks):
        raise ValueError("TransformerLens configuration requires one attention scale for all layers")
    tl_cfg = HookedTransformerConfig(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_key_value_heads=(
            cfg.n_kv_heads if cfg.n_kv_heads != cfg.n_heads else None),
        d_model=cfg.d_model, d_head=d_head, d_mlp=cfg.d_ff,
        d_vocab=cfg.vocab_size, d_vocab_out=cfg.vocab_size, n_ctx=cfg.max_seq_len,
        act_fn="gelu", normalization_type="LN", eps=lm.ln_f.eps,
        positional_embedding_type=("standard" if cfg.pos_emb_type == "sinusoidal" else "rotary"),
        rotary_dim=(d_head if cfg.pos_emb_type == "rope" else None),
        rotary_base=10000, rotary_adjacent_pairs=False,
        use_attn_scale=True, attn_scale=1.0 / native_scale,
        use_attn_result=True, default_prepend_bos=False,
        dtype=lm.tok_emb.weight.dtype, device=str(lm.tok_emb.weight.device),
    )
    hm = HookedTransformer(tl_cfg).eval()
    with torch.no_grad():
        def copy(dst, src, name):
            if tuple(dst.shape) != tuple(src.shape):
                raise ValueError(f"{name}: TransformerLens {tuple(dst.shape)} != native {tuple(src.shape)}")
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

        copy(hm.embed.W_E, lm.tok_emb.weight, "token embedding")
        if cfg.pos_emb_type == "sinusoidal":
            copy(hm.pos_embed.W_pos, lm.pos_emb, "sinusoidal position table")
        for layer, (native, hooked) in enumerate(zip(lm.blocks, hm.blocks)):
            attn, h_attn = native.attn, hooked.attn
            q = attn.q_proj.weight.reshape(cfg.n_heads, d_head, cfg.d_model).transpose(1, 2)
            k = attn.k_proj.weight.reshape(cfg.n_kv_heads, d_head, cfg.d_model).transpose(1, 2)
            v = attn.v_proj.weight.reshape(cfg.n_kv_heads, d_head, cfg.d_model).transpose(1, 2)
            o = attn.out_proj.weight.reshape(cfg.d_model, cfg.n_heads, d_head).permute(1, 2, 0)
            copy(h_attn.W_Q, q, f"L{layer}.W_Q")
            copy(h_attn._W_K if cfg.n_kv_heads != cfg.n_heads else h_attn.W_K,
                 k, f"L{layer}.W_K")
            copy(h_attn._W_V if cfg.n_kv_heads != cfg.n_heads else h_attn.W_V,
                 v, f"L{layer}.W_V")
            copy(h_attn.W_O, o, f"L{layer}.W_O")
            for name in ("b_Q", "b_O"):
                getattr(h_attn, name).zero_()
            for name in ("b_K", "b_V"):
                target = getattr(h_attn, "_" + name) if cfg.n_kv_heads != cfg.n_heads else getattr(h_attn, name)
                target.zero_()
            for ln_name in ("ln1", "ln2"):
                source_ln, target_ln = getattr(native, ln_name), getattr(hooked, ln_name)
                if source_ln.eps != tl_cfg.eps:
                    raise ValueError(f"L{layer}.{ln_name}: LayerNorm epsilon mismatch")
                copy(target_ln.w, source_ln.weight, f"L{layer}.{ln_name}.w")
                copy(target_ln.b, source_ln.bias, f"L{layer}.{ln_name}.b")
            feed = native.ffn.net
            copy(hooked.mlp.W_in, feed[0].weight.T, f"L{layer}.mlp.W_in")
            copy(hooked.mlp.b_in, feed[0].bias, f"L{layer}.mlp.b_in")
            copy(hooked.mlp.W_out, feed[3].weight.T, f"L{layer}.mlp.W_out")
            copy(hooked.mlp.b_out, feed[3].bias, f"L{layer}.mlp.b_out")
        copy(hm.ln_final.w, lm.ln_f.weight, "final LayerNorm weight")
        copy(hm.ln_final.b, lm.ln_f.bias, "final LayerNorm bias")
        copy(hm.unembed.W_U, lm.head.weight.T, "unembedding")
        hm.unembed.b_U.zero_()
    return install_analysis_support(hm, norm_first=cfg.norm_first).eval()


def logits_match(logits_a, logits_b, atol=1e-3) -> bool:
    """True iff the two logit tensors agree elementwise within atol."""
    return (tuple(logits_a.shape) == tuple(logits_b.shape) and
            bool(torch.isfinite(logits_a).all()) and bool(torch.isfinite(logits_b).all()) and
            bool(torch.allclose(logits_a, logits_b, atol=atol, rtol=0.0)))


@torch.no_grad()
def logits_and_cache(hm, X):
    """Run the hooked model on X[:, :-1] with run_with_cache; return (logits, cache)."""
    return hm.run_with_cache(X[:, :-1])


@torch.no_grad()
def attention_entropy(cache, n_layers: int) -> torch.Tensor:
    """Mean entropy (nats) of each head's attention rows, averaged over the batch and over
    query positions >= 1. Clamp probabilities at 1e-12 before the log. Returns (L, H)."""
    rows = []
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, 1:, :]
        entropy = -(pattern * pattern.clamp_min(1e-12).log()).sum(-1).mean((0, 2))
        rows.append(entropy)
    return torch.stack(rows)


# =========================================================================== Q5
@torch.no_grad()
def previous_token_scores(cache, n_layers: int) -> torch.Tensor:
    """Mean attention from query q to key q-1, over all q >= 1 and the batch. Returns (L, H)."""
    rows = []
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        length = pattern.shape[-1]
        q = torch.arange(1, length, device=pattern.device)
        rows.append(pattern[:, :, q, q - 1].mean((0, 2)))
    return torch.stack(rows)


@torch.no_grad()
def induction_scores(cache, n_layers: int, P: int) -> torch.Tensor:
    """Mean attention from each second-copy query q (input positions P+1 .. 2P-1) to key q-P
    — the token that FOLLOWED the current token's earlier occurrence. Returns (L, H)."""
    rows = []
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        if pattern.shape[-1] < 2 * P:
            raise ValueError("Induction scoring requires the complete model input length 2P")
        q = torch.arange(P + 1, 2 * P, device=pattern.device)
        rows.append(pattern[:, :, q, q - P].mean((0, 2)))
    return torch.stack(rows)


@torch.no_grad()
def ablate_head(hm, X, layer: int, head: int):
    """Zero-ablate one head's z activation (tl_utils.get_act_name("z", layer), shape
    (B, T, H, d_head)) with a forward hook and return the logits on X[:, :-1]."""
    def zero_selected(z, hook):
        updated = z.clone()
        updated[:, :, head, :] = 0
        return updated
    return hm.run_with_hooks(X[:, :-1], fwd_hooks=[(tl_utils.get_act_name("z", layer), zero_selected)])


@torch.no_grad()
def repeat_accuracy(logits, X, P: int) -> float:
    """GIVEN. Accuracy on the repeated-half targets (input positions P+1 .. 2P-1)."""
    rep = slice(P + 1, 2 * P)
    return (logits[:, rep].argmax(-1) == X[:, 1:][:, rep]).float().mean().item()


@torch.no_grad()
def shift_test(hm, X, P: int, k: int, layer: int, head: int, filler: torch.Tensor) -> dict:
    """Content or position? Insert k filler tokens between <SEP> and the second copy (dropping
    the last k tokens of the second copy so the length stays 2P+1):
        Xk = [p_0..p_{P-1}, SEP, f_0..f_{k-1}, p_0..p_{P-1-k}]
    For queries q = P+1+k .. 2P-1 (inputs that predict a copy token), the token to copy now sits
    at key q-P-k (CONTENT target) while the old fixed offset points at key q-P (OFFSET target).
    `filler` is a (B, k) tensor of token ids. Run hm on Xk[:, :-1] with a cache and return
      {"Xk": Xk, "copy_acc": accuracy on those queries,
       "attn_content": mean attention of (layer, head) to the content target,
       "attn_offset":  mean attention of (layer, head) to the offset target}.
    With k = 0 both targets coincide."""
    if not (0 <= k < P) or tuple(filler.shape) != (X.shape[0], k):
        raise ValueError("shift k and filler shape must satisfy 0 <= k < P and filler=(B,k)")
    Xk = torch.cat((X[:, :P + 1], filler, X[:, P + 1:2 * P + 1 - k]), dim=1)
    logits, cache = hm.run_with_cache(Xk[:, :-1])
    q = torch.arange(P + 1 + k, 2 * P, device=X.device)
    copy_acc = (logits[:, q].argmax(-1) == Xk[:, q + 1]).float().mean().item()
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]
    return {"Xk": Xk, "copy_acc": copy_acc,
            "attn_content": pattern[:, q, q - P - k].mean().item(),
            "attn_offset": pattern[:, q, q - P].mean().item()}


@torch.no_grad()
def head_ablation_effect(hm, X, P: int) -> list:
    """For every (layer, head) in order: zero-ablate it and record the held-out
    repeated-half accuracy and its drop from the intact model. Return a list of dicts
    {"layer", "head", "acc_after", "acc_drop"}."""
    intact = repeat_accuracy(hm(X[:, :-1]), X, P)
    rows = []
    for layer in range(hm.cfg.n_layers):
        for head in range(hm.cfg.n_heads):
            acc = repeat_accuracy(ablate_head(hm, X, layer, head), X, P)
            rows.append({"layer": layer, "head": head,
                         "acc_after": acc, "acc_drop": intact - acc})
    return rows


# =========================================================================== Q6
@torch.no_grad()
def logit_lens(hm, cache, X, P: int) -> list:
    """GIVEN. Use the shared helper to decode all residual-stream depths."""
    return _shared_logit_lens(hm, cache, X, P)


@torch.no_grad()
def decompose_correct_logit(hm, cache, X) -> dict:
    """GIVEN. Use the shared exact DLA implementation for the TransformerLens port API."""
    return _shared_decompose_correct_logit(hm, cache, X)


def summarize_dla(decomp: dict, P: int) -> dict:
    """GIVEN. Mean contribution of every component over repeated-half positions."""
    return _shared_summarize_dla(decomp, P)

# ---------------------------------------------------------------------- Q4
def block0_by_hand(model, tokens):
    """Compute the toy block directly from its parameters, without using its cache."""
    block = model.blocks[0]
    embed = F.embedding(tokens, model.embed.W_E)
    pos = model.pos_embed.W_pos[:tokens.shape[1]].unsqueeze(0)
    resid_pre = embed + pos
    def layer_norm(x, layer):
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = centered / (centered.square().mean(dim=-1, keepdim=True) + layer.eps).sqrt()
        return normalized, normalized * layer.w + layer.b
    norm1_raw, norm1 = layer_norm(resid_pre, block.ln1)
    attn = block.attn
    q = torch.einsum("btd,hdf->bthf", norm1, attn.W_Q) + attn.b_Q
    k = torch.einsum("btd,hdf->bthf", norm1, attn.W_K) + attn.b_K
    v = torch.einsum("btd,hdf->bthf", norm1, attn.W_V) + attn.b_V
    scores = torch.einsum("bthd,bshd->bhts", q, k) / attn.attn_scale
    length = tokens.shape[1]
    causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
    pattern = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    z = torch.einsum("bhts,bshd->bthd", pattern, v)
    attn_out = torch.einsum("bthd,hdm->btm", z, attn.W_O) + attn.b_O
    resid_mid = resid_pre + attn_out
    norm2_raw, norm2 = layer_norm(resid_mid, block.ln2)
    mlp = block.mlp
    hidden = F.gelu(norm2 @ mlp.W_in + mlp.b_in)
    mlp_out = hidden @ mlp.W_out + mlp.b_out
    resid_post = resid_mid + mlp_out
    return {"resid_pre": resid_pre, "ln1.hook_normalized": norm1_raw,
            "attn_out": attn_out, "resid_mid": resid_mid,
            "ln2.hook_normalized": norm2_raw, "mlp_out": mlp_out,
            "resid_post": resid_post}

# ---------------------------------------------------------------------- Q5
@torch.no_grad()
def mean_ablate_head(hm, X, layer, head, mean_z):
    """Replace the selected z vector with a fixed mean from a clean run."""
    def replace(z, hook):
        updated = z.clone()
        updated[:, :, head, :] = mean_z.to(device=z.device, dtype=z.dtype)
        return updated
    return hm.run_with_hooks(X[:, :-1], fwd_hooks=[(tl_utils.get_act_name("z", layer), replace)])

@torch.no_grad()
def patch_head_z(hm, corrupt_tokens, layer, head, clean_z):
    """Replace one corrupted z activation with the matching clean activation."""
    if clean_z.ndim != 3 or clean_z.shape[0] != corrupt_tokens.shape[0]:
        raise ValueError("clean_z must have shape (B,T,d_head)")
    if clean_z.shape[1] == corrupt_tokens.shape[1]:
        inputs = corrupt_tokens
    elif clean_z.shape[1] == corrupt_tokens.shape[1] - 1:
        inputs = corrupt_tokens[:, :-1]
    else:
        raise ValueError("clean_z length must equal full or next-token input length")
    def replace(z, hook):
        updated = z.clone()
        updated[:, :, head, :] = clean_z.to(device=z.device, dtype=z.dtype)
        return updated
    return hm.run_with_hooks(inputs, fwd_hooks=[(tl_utils.get_act_name("z", layer), replace)])

@torch.no_grad()
def head_patching_effect(hm, clean_tokens, corrupt_tokens, P):
    """Measure per-example clean-margin recovery after patching each head's z."""
    if clean_tokens.shape != corrupt_tokens.shape or clean_tokens.shape[0] < 2:
        raise ValueError("Patching requires at least two matched clean/corrupt examples")
    clean_logits, clean_cache = hm.run_with_cache(clean_tokens[:, :-1])
    corrupt_logits = hm(corrupt_tokens[:, :-1])
    q = torch.arange(P + 1, 2 * P, device=clean_tokens.device)
    clean_target = clean_tokens[:, q + 1]
    corrupt_target = corrupt_tokens[:, q + 1]
    def margin(logits):
        positions = logits[:, q]
        good = positions.gather(-1, clean_target.unsqueeze(-1)).squeeze(-1)
        bad = positions.gather(-1, corrupt_target.unsqueeze(-1)).squeeze(-1)
        return (good - bad).mean(dim=-1)
    clean_margin, corrupt_margin = margin(clean_logits), margin(corrupt_logits)
    denominator = clean_margin - corrupt_margin
    valid = denominator.abs() >= 1e-8
    recovery = clean_logits.new_full((hm.cfg.n_layers, hm.cfg.n_heads), float("nan"))
    if not bool(valid.any()):
        return recovery
    for layer in range(hm.cfg.n_layers):
        for head in range(hm.cfg.n_heads):
            z = clean_cache[tl_utils.get_act_name("z", layer)][:, :, head, :]
            patched = patch_head_z(hm, corrupt_tokens, layer, head, z)
            values = (margin(patched) - corrupt_margin)[valid] / denominator[valid]
            recovery[layer, head] = values.mean()
    return recovery

# ---------------------------------------------------------------------- Q6
def offset_score_profile(cache, layer, head):
    """Mean finite causal QK scores at each relative offset q-k."""
    scores = cache[f"blocks.{layer}.attn.hook_attn_scores"]
    if scores.ndim != 4 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("Expected attention scores with shape (batch, heads, T, T)")
    if not 0 <= head < scores.shape[1]:
        raise IndexError("head index out of range")
    T = scores.shape[-1]
    offsets = torch.arange(T, device=scores.device)
    means = []
    for offset in range(T):
        q = torch.arange(offset, T, device=scores.device)
        k = q - offset
        values = scores[:, head, q, k]
        finite = values[torch.isfinite(values)]
        means.append(finite.mean() if finite.numel() else values.new_tensor(float("nan")))
    return offsets, torch.stack(means)

# Static OV matrix analysis is optional; Q6 does not require students to implement it.
