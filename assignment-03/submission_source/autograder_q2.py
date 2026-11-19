#!/usr/bin/env python3
"""
Autograder for Question 2: MHA / GQA / MQA / Q2-only Multi-Head Latent Attention.

Usage:
    python autograder_q2.py --student_dir /path/to/students/alice

Expects under --student_dir/submissions/q2: attention.py, model_arch.py, model.pt,
config.json, and metrics.json.
"""
import argparse
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common_grading import (
    GradeReport,
    load_submission_dir,
    load_student_module,
    check_param_budget,
    load_full_student_stack,
    check_causality,
    check_forbidden_implementations,
    metric_is_finite,
    metric_is_integer,
    run_grading_case,
    assert_tensors_close,
    make_test_attention,
    randomize_attention_projections,
    reference_causal_attention,
)


def check_q2_evidence(metrics, config, report):
    rows = metrics.get("per_config_results") if isinstance(metrics, dict) else None
    if not isinstance(rows, list):
        report.check("Q2 per_config_results is a list", False)
        return
    variants = []
    d, h, layers = (config.get(k) for k in ("d_model", "n_heads", "n_layers"))
    dims_ok = metric_is_integer(d) and h == 8 and type(h) is int and metric_is_integer(layers) and d % h == 0
    report.check("Q2 evidence has valid matched model dimensions with n_heads=8", dims_ok)
    for i, row in enumerate(rows):
        name = f"Q2 row {i}"
        if not isinstance(row, dict):
            report.check(name + " is a dictionary", False)
            continue
        variant = row.get("attention_type")
        valid_variant = isinstance(variant, str) and variant in ("mha", "gqa", "mqa", "mla")
        report.check(name + " declares mha/gqa/mqa/mla", valid_variant)
        if valid_variant:
            variants.append(variant)
        report.check(name + " contains both dimension fields",
                     "n_kv_heads" in row and "kv_latent_dim" in row)
        kv, latent = row.get("n_kv_heads"), row.get("kv_latent_dim")
        if valid_variant and variant == "mla":
            valid_dims = kv is None and metric_is_integer(latent) and dims_ok and latent < d
            report.check(name + " MLA has null n_kv_heads and 0 < kv_latent_dim < d_model", valid_dims)
            expected_kv = 3 * d * latent * layers if valid_dims else None
            expected_bytes = latent * layers * 4 if valid_dims else None
        elif valid_variant:
            valid_dims = type(kv) is int and kv == {"mha": 8, "gqa": 2, "mqa": 1}[variant] and latent is None
            report.check(name + " standard variant has matching head count and null latent width", valid_dims)
            expected_kv = 2 * d * kv * (d // h) * layers if dims_ok and valid_dims else None
            expected_bytes = 2 * kv * (d // h) * layers * 4 if dims_ok and valid_dims else None
        else:
            expected_kv = expected_bytes = None
        for key in ("held_out_ppl", "dialogue_ppl", "narration_ppl"):
            report.check(name + " " + key + " is finite and >= 1", metric_is_finite(row.get(key), 1))
        for key in ("params", "kv_projection_params", "kv_cache_bytes_per_token"):
            report.check(name + " " + key + " is a positive integer", metric_is_integer(row.get(key)))
        if expected_kv is not None:
            report.check(name + " KV-path weight count matches dimensions", row.get("kv_projection_params") == expected_kv)
            report.check(name + " FP32 cache bytes/token matches dimensions", row.get("kv_cache_bytes_per_token") == expected_bytes)
        if metric_is_integer(row.get("params")) and metric_is_integer(row.get("kv_projection_params")):
            report.check(name + " total params >= KV-path params", row["params"] >= row["kv_projection_params"])
            report.check(name + " respects 50M model budget", row["params"] <= 50_000_000)
    report.check("Q2 covers exactly one row for each of mha/gqa/mqa/mla",
                 len(rows) == 4 and sorted(variants) == ["gqa", "mha", "mla", "mqa"])


def check_q2_attention(attn_mod, report):
    """Independent MHA/GQA/MQA reference checks, without training."""
    for kv in (8, 2, 1):
        def run(kv=kv):
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(781 + kv)
                generator = torch.Generator().manual_seed(781 + kv)
                attention = make_test_attention(attn_mod.CausalSelfAttention, kv)
                randomize_attention_projections(attention, generator)
                for length in (1, 7, 16):
                    x = torch.randn(2, length, 64, generator=generator)
                    assert_tensors_close(
                        attention(x), reference_causal_attention(attention, x, kv)
                    )
            return "reference matched at sequence lengths 1, 7 and 16"
        run_grading_case(report, f"Q2 attention numerics, n_kv_heads={kv}", run)



def check_q2_mla(attn_mod, model_mod, report):
    """Independent effective-weight reference, causality, and model dispatch."""
    for latent_dim in (5, 16):
        def run(latent_dim=latent_dim):
            import math
            import torch.nn.functional as F
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1781 + latent_dim)
                gen = torch.Generator().manual_seed(1781 + latent_dim)
                for scale in (None, 0.37):
                    attention = attn_mod.MultiHeadLatentAttention(
                        d_model=64, n_heads=8, kv_latent_dim=latent_dim,
                        max_seq_len=16, dropout=0.0, attention_scale=scale).eval()
                    shapes = {"q_proj": (64, 64), "kv_down_proj": (latent_dim, 64),
                              "k_up_proj": (64, latent_dim), "v_up_proj": (64, latent_dim),
                              "out_proj": (64, 64)}
                    for name, shape in shapes.items():
                        layer = getattr(attention, name)
                        if not isinstance(layer, torch.nn.Linear) or tuple(layer.weight.shape) != shape or layer.bias is not None:
                            raise AssertionError(f"{name} must be bias-free Linear with weight shape {shape}")
                        with torch.no_grad():
                            layer.weight.copy_(torch.randn(shape, generator=gen) * 0.08)
                    # Count the shared compression matrix once, then both up-projections.
                    kv_count = sum(getattr(attention, name).weight.numel()
                                   for name in ("kv_down_proj", "k_up_proj", "v_up_proj"))
                    if kv_count != 3 * 64 * latent_dim:
                        raise AssertionError("incorrect MLA KV-path parameter count")
                    for length in (1, 7, 16):
                        x = torch.randn(2, length, 64, generator=gen)
                        # Independent reference folds compression into effective K/V weights.
                        # Do not call any student forward method or use its mask/scale.
                        q = F.linear(x, attention.q_proj.weight).view(2, length, 8, 8).transpose(1, 2)
                        k = F.linear(x, attention.k_up_proj.weight @ attention.kv_down_proj.weight).view(2, length, 8, 8).transpose(1, 2)
                        v = F.linear(x, attention.v_up_proj.weight @ attention.kv_down_proj.weight).view(2, length, 8, 8).transpose(1, 2)
                        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(8) if scale is None else scale)
                        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
                        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
                        y = (weights @ v).transpose(1, 2).reshape(2, length, 64)
                        expected = F.linear(y, attention.out_proj.weight)
                        actual = attention(x)
                        assert_tensors_close(actual, expected)
                        if length > 1:
                            split = length // 2
                            changed = x.clone()
                            changed[:, split:] = torch.randn(2, length-split, 64, generator=gen) * 3
                            assert_tensors_close(attention(changed)[:, :split], actual[:, :split])
                        # Gradients must follow both branches through the shared latent.
                        probe = torch.randn(expected.shape, generator=gen)
                        params = [getattr(attention, name).weight for name in shapes]
                        actual_grads = torch.autograd.grad((actual * probe).sum(), params, allow_unused=True)
                        expected_grads = torch.autograd.grad((expected * probe).sum(), params)
                        for name, got, want in zip(shapes, actual_grads, expected_grads):
                            if got is None or not torch.isfinite(got).all():
                                raise AssertionError(f"missing/nonfinite gradient for {name}")
                            assert_tensors_close(got, want)
            return "MLA outputs/gradients match reference; causal; lengths 1/7/16 and scale override checked"
        run_grading_case(report, f"Q2 MLA reference, latent_dim={latent_dim}", run)

    def model_dispatch():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2781)
            for norm_first in (False, True):
                cfg = model_mod.LMConfig(vocab_size=32, d_model=64, n_layers=2,
                    n_heads=8, n_kv_heads=None, d_ff=96, max_seq_len=16,
                    dropout=0.0, pos_emb_type="sinusoidal", norm_first=norm_first,
                    attention_type="mla", kv_latent_dim=13)
                model = model_mod.DecoderLM(cfg).eval()
                if not all(isinstance(block.attn, attn_mod.MultiHeadLatentAttention) for block in model.blocks):
                    raise AssertionError("LMConfig(attention_type='mla') must select MLA in every block")
                ids = torch.randint(0, cfg.vocab_size, (2, 9))
                with torch.no_grad():
                    logits = model(ids)
                    if logits.shape != (2, 9, cfg.vocab_size) or not torch.isfinite(logits).all():
                        raise AssertionError("MLA LM logits must have correct shape and be finite")
                    changed = ids.clone()
                    changed[:, 4:] = (changed[:, 4:] + 1) % cfg.vocab_size
                    assert_tensors_close(model(changed)[:, :4], logits[:, :4])
        return "MLA decoder integration works for pre-norm and post-norm"
    run_grading_case(report, "Q2 MLA model integration", model_dispatch)


def main(student_dir: str):
    report = GradeReport("Question 2")
    sub_dir = os.path.join(student_dir, "submissions", "q2")
    attn_py = os.path.join(sub_dir, "attention.py")
    model_py = os.path.join(sub_dir, "model_arch.py")

    if not os.path.exists(attn_py):
        report.check("attention.py present", False)
        return report.summary()
    report.check("attention.py present", True)

    if not check_forbidden_implementations(report, sub_dir, question=2):
        return report.summary()
    # --- structural check: CausalSelfAttention accepts n_kv_heads --------------------
    # attention.py may by now (post Question 3) import from rope.py, so load the
    # dependency chain in order rather than importing attention.py in isolation.
    try:
        stack = load_full_student_stack(sub_dir, need=("attention", "model_arch"))
        attn_mod = stack["attention"]
        sig = inspect.signature(attn_mod.CausalSelfAttention.__init__)
        has_kv = "n_kv_heads" in sig.parameters
        report.check("CausalSelfAttention.__init__ accepts n_kv_heads", has_kv,
                     f"params: {list(sig.parameters)}")
    except Exception as e:
        report.check("CausalSelfAttention.__init__ accepts n_kv_heads", False, repr(e))
        return report.summary()

    # --- functional check: MQA actually shares K/V across heads (parameter-count test)
    try:
        mha = attn_mod.CausalSelfAttention(d_model=64, n_heads=8, n_kv_heads=8, max_seq_len=16)
        mqa = attn_mod.CausalSelfAttention(d_model=64, n_heads=8, n_kv_heads=1, max_seq_len=16)
        mha_kv_params = sum(p.numel() for p in mha.k_proj.parameters()) + sum(p.numel() for p in mha.v_proj.parameters())
        mqa_kv_params = sum(p.numel() for p in mqa.k_proj.parameters()) + sum(p.numel() for p in mqa.v_proj.parameters())
        report.check("MQA (n_kv_heads=1) has strictly fewer K/V parameters than MHA",
                     mqa_kv_params < mha_kv_params,
                     f"MHA K/V params={mha_kv_params}, MQA K/V params={mqa_kv_params}")

        # forward pass shape check for a GQA config (non-trivial grouping)
        gqa = attn_mod.CausalSelfAttention(d_model=64, n_heads=8, n_kv_heads=2, max_seq_len=16)
        x = torch.randn(2, 10, 64)
        out = gqa(x)
        report.check("GQA (n_heads=8, n_kv_heads=2) forward pass produces correct shape",
                     out.shape == (2, 10, 64), f"got {tuple(out.shape)}")
    except Exception as e:
        report.check("MQA/GQA functional checks", False, repr(e))
        return report.summary()

    check_q2_attention(attn_mod, report)
    check_q2_mla(attn_mod, stack["model_arch"], report)

    # --- submission artifacts ---------------------------------------------------------
    try:
        config, metrics, model_path = load_submission_dir(sub_dir)
        report.check("submission artifacts load", True)
    except FileNotFoundError as e:
        report.check("submission artifacts load", False, str(e))
        return report.summary()

    report.check("config declares n_kv_heads < n_heads (used GQA or MQA, not plain MHA)",
                 config.get("n_kv_heads", config.get("n_heads", 1)) < config.get("n_heads", 1),
                 f"n_heads={config.get('n_heads')}, n_kv_heads={config.get('n_kv_heads')}")
    report.check("config declares n_heads == 8 and n_kv_heads == 2",
                 config.get("n_kv_heads",0)==2 and config.get("n_heads",0)== 8,
                 f"n_kv_heads={config.get('n_kv_heads',0)}, n_heads={config.get('n_heads',0)}")
    report.check("n_heads is divisible by n_kv_heads",
                 config.get("n_heads", 1) % max(config.get("n_kv_heads", 1), 1) == 0)
    report.check("Q2 primary checkpoint uses standard attention", config.get("attention_type") == "standard")
    report.check("Q2 primary checkpoint uses sinusoidal positions", config.get("pos_emb_type") == "sinusoidal")
    check_q2_evidence(metrics, config, report)

    # --- load and forward-pass the saved model ----------------------------------------
    try:
        model_mod = stack["model_arch"]
        if model_mod is None:
            raise FileNotFoundError("model_arch.py not found")
        cfg_fields = {k: config[k] for k in
                      ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                       "d_ff", "max_seq_len", "dropout", "pos_emb_type","attention_scale", "norm_first", "attention_type", "kv_latent_dim"] if k in config}
        cfg = model_mod.LMConfig(**cfg_fields)
        model = model_mod.DecoderLM(cfg)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        check_causality(model, cfg, report) # does the causal mask check remove it if not wanted.
        check_param_budget(model, report)
        dummy = torch.randint(0, cfg.vocab_size, (2, min(16, cfg.max_seq_len)))
        with torch.no_grad():
            logits = model(dummy)
        report.check("saved model forward pass shape correct",
                     logits.shape == (2, min(16, cfg.max_seq_len), cfg.vocab_size))
    except Exception as e:
        report.check("saved model loads and runs", False, repr(e))

    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    ok = main(args.student_dir)
    sys.exit(0 if ok else 1)
