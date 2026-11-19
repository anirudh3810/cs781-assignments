"""Validation helpers for student-written TransformerLens ports."""
from __future__ import annotations

from typing import Any

import torch


BASE_CFG_FIELDS = ("n_layers", "n_heads", "d_model", "d_vocab")


def _cache_key(cache: Any, key: str) -> bool:
    try:
        cache[key]
        return True
    except Exception:
        return False


def _activation(cache: Any, key: str):
    try:
        return cache[key]
    except Exception as exc:
        raise KeyError(f"Port cache is missing required activation {key!r}.") from exc


def validate_transformerlens_port(model, tokens: torch.Tensor, *, require_dla: bool = True):
    """Run a small forward/cache pass and return actionable port-contract errors.

    Required cache conventions:
      hook_embed: (B,T,D)
      blocks.{l}.attn.hook_pattern / hook_attn_scores: (B,H,T,T)
      blocks.{l}.attn.hook_z: (B,T,H,D_head)
      blocks.{l}.hook_resid_pre / hook_resid_mid / hook_resid_post: (B,T,D)
      blocks.{l}.hook_mlp_out: (B,T,D)

    The adapted model must be an actual TransformerLens HookedTransformer. The supplied
    ``tl_analysis_support`` module supplies architecture-independent readout/DLA helpers;
    students implement the weight mapping in ``to_hooked``.
    """
    errors = []
    try:
        from transformer_lens import HookedTransformer
        if not isinstance(model, HookedTransformer):
            errors.append("to_hooked(lm) must return a real transformer_lens.HookedTransformer.")
    except Exception as exc:
        return [f"Could not import the pinned TransformerLens HookedTransformer: {exc}"]
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return ["HookedTransformer.cfg is missing n_layers, n_heads, d_model, or d_vocab."]
    for field in BASE_CFG_FIELDS:
        if not hasattr(cfg, field):
            errors.append(f"HookedTransformer.cfg.{field} is required.")
    if errors:
        return errors
    for method in ("run_with_cache", "run_with_hooks"):
        if not callable(getattr(model, method, None)):
            errors.append(f"HookedTransformer.{method}(...) is required.")
    if not callable(model):
        errors.append("The HookedTransformer must be callable as model(tokens) -> logits.")
    if errors:
        return errors

    try:
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
    except Exception as exc:
        return [f"HookedTransformer.run_with_cache failed: {type(exc).__name__}: {exc}"]
    b, t = tokens.shape
    expected_logits = (b, t, int(cfg.d_vocab))
    if tuple(logits.shape) != expected_logits:
        errors.append(f"forward logits must have shape {expected_logits}; got {tuple(logits.shape)}.")
    for key, shape in (("hook_embed", (b, t, int(cfg.d_model))),):
        if not _cache_key(cache, key):
            errors.append(f"cache is missing {key!r}.")
        elif tuple(_activation(cache, key).shape) != shape:
            errors.append(f"cache[{key!r}] must have shape {shape}.")

    for layer in range(int(cfg.n_layers)):
        prefix = f"blocks.{layer}"
        required = {
            f"{prefix}.attn.hook_pattern": (b, int(cfg.n_heads), t, t),
            f"{prefix}.attn.hook_attn_scores": (b, int(cfg.n_heads), t, t),
            f"{prefix}.attn.hook_z": (b, t, int(cfg.n_heads), None),
            f"{prefix}.hook_resid_pre": (b, t, int(cfg.d_model)),
            f"{prefix}.hook_resid_mid": (b, t, int(cfg.d_model)),
            f"{prefix}.hook_resid_post": (b, t, int(cfg.d_model)),
            f"{prefix}.hook_mlp_out": (b, t, int(cfg.d_model)),
        }
        if layer == 0:
            required[f"{prefix}.ln1.hook_normalized"] = (b, t, int(cfg.d_model))
            required[f"{prefix}.ln2.hook_normalized"] = (b, t, int(cfg.d_model))
        for key, shape in required.items():
            if not _cache_key(cache, key):
                errors.append(f"cache is missing {key!r}.")
                continue
            actual = tuple(_activation(cache, key).shape)
            if len(actual) != len(shape) or any(want is not None and got != want
                                                for got, want in zip(actual, shape)):
                errors.append(f"cache[{key!r}] must have shape pattern {shape}; got {actual}.")
        head_writes = getattr(model, "head_writes", None)
        if callable(head_writes):
            try:
                value = head_writes(cache, layer)
                if tuple(value.shape) != (b, t, int(cfg.n_heads), int(cfg.d_model)):
                    errors.append(f"head_writes(cache,{layer}) must have shape (B,T,H,D_model).")
            except Exception as exc:
                errors.append(f"head_writes(cache,{layer}) failed: {type(exc).__name__}: {exc}")
        elif not callable(getattr(model, "component_writes", None)) and not (
                hasattr(model, "blocks") and
                hasattr(getattr(model.blocks[layer], "attn", None), "W_O")):
            errors.append(
                f"Layer {layer}: provide head_writes(cache, layer) for per-head residual writes."
            )

    if require_dla:
        generic = all(callable(getattr(model, name, None)) for name in (
            "component_writes", "project_component_to_true_logit", "readout_bias_to_true_logit",
            "residual_to_logits"))
        transformer_lens_style = all(hasattr(model, name) for name in ("ln_final", "W_U", "unembed"))
        if not generic and not transformer_lens_style:
            errors.append(
                "For logit lens/DLA, provide residual_to_logits, component_writes, "
                "project_component_to_true_logit, and readout_bias_to_true_logit methods."
            )
    return errors
