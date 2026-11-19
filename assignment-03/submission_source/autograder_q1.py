#!/usr/bin/env python3
"""
Autograder for Question 1: from-scratch decoder-only Transformer (MHA + sinusoidal PE).

Usage:
    python autograder_q1.py --student_dir /path/to/students/alice

Expects `a1_tokenizer.py` at the student root (or its saved Q1 snapshot) and, under
--student_dir/submissions/q1: attention.py, model_arch.py, model.pt, config.json, and
metrics.json.

Runs in well under a minute on a laptop CPU: no retraining, just static source checks
plus a handful of forward passes on a <=50M-parameter model.
"""
import argparse
import math
import json
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common_grading import (
    GradeReport, load_submission_dir, load_student_module,
    ast_defines_class, check_param_budget, try_compute_perplexity,
    build_tokenizer_for_grading, get_shared_eval_set,
    load_full_student_stack, check_causality,
    check_forbidden_implementations,
    metric_is_finite, metric_is_integer, metric_is_curve,
)


def check_q1_metrics(metrics, report):
    if not isinstance(metrics, dict):
        report.check("Q1 metrics is a dictionary", False)
        return
    report.check("Q1 held_out_perplexity is finite and >= 1",
                 metric_is_finite(metrics.get("held_out_perplexity"), 1.0))
    report.check("Q1 train_loss_curve is nonempty, finite and nonnegative",
                 metric_is_curve(metrics.get("train_loss_curve")))
    accuracy = metrics.get("extrinsic_probe_accuracy")
    report.check("Q1 extrinsic_probe_accuracy is a finite fraction in [0, 1]",
                 metric_is_finite(accuracy, minimum=0) and accuracy <= 1.0)


def check_q1_scaling(metrics, config, report):
    rows = metrics.get("attention_scaling_results") if isinstance(metrics, dict) else None
    if not isinstance(rows, list):
        report.check("Q1 attention_scaling_results is a list", False)
        return
    d, h = config.get("d_model"), config.get("n_heads")
    if not (metric_is_integer(d) and metric_is_integer(h) and d % h == 0):
        report.check("Q1 scaling: valid d_model and n_heads", False)
        return
    standard = 1.0 / math.sqrt(d // h)
    covered = set()
    for i, row in enumerate(rows):
        name = f"Q1 scaling row {i}"
        if not isinstance(row, dict):
            report.check(name + " is a dictionary", False)
            continue
        report.check(name + " held_out_ppl is finite and >= 1",
                     metric_is_finite(row.get("held_out_ppl"), 1.0))
        report.check(name + " train_loss_curve is valid", metric_is_curve(row.get("train_loss_curve")))
        label = str(row.get("label", "")).strip().lower()
        label_kind = {"standard": "standard", "unscaled": "unscaled"}.get(label)
        kinds = set()
        if "attention_scale" in row:
            value = row["attention_scale"]
            valid = value is None or (metric_is_finite(value, minimum=0) and value > 0)
            report.check(name + " attention_scale is None or finite and positive", valid)
            if not valid:
                continue
            alpha = standard if value is None else value
            for kind, target in (("standard", standard), ("unscaled", 1.0)):
                if math.isclose(alpha, target, rel_tol=1e-6, abs_tol=1e-8):
                    kinds.add(kind)
            if label_kind:
                report.check(name + " label agrees with attention_scale", label_kind in kinds)
                kinds &= {label_kind}
        elif label_kind:
            # The question explicitly allows recording alpha OR a readable label.
            kinds = {label_kind}
        else:
            report.check(name + " identifies its scale", False)
        covered.update(kinds)
    report.check("Q1 has separate standard and unscaled experiment entries",
                 len(rows) >= 2 and covered >= {"standard", "unscaled"})


def main(student_dir: str):
    report = GradeReport("Question 1")
    sub_dir = os.path.join(student_dir, "submissions", "q1")
    attn_py = os.path.join(sub_dir, "attention.py")
    model_py = os.path.join(sub_dir, "model_arch.py")

    for name, path in [("attention.py present", attn_py), ("model_arch.py present", model_py)]:
        report.check(name, os.path.exists(path))
    if not (os.path.exists(attn_py) and os.path.exists(model_py)):
        return report.summary()

    # --- static checks: no nn.Transformer / nn.MultiheadAttention anywhere -----------
    if not check_forbidden_implementations(report, sub_dir, question=1):
        return report.summary()

    report.check("attention.py defines CausalSelfAttention", ast_defines_class(attn_py, "CausalSelfAttention"))
    report.check("model_arch.py defines DecoderLM", ast_defines_class(model_py, "DecoderLM"))
    report.check("model_arch.py defines LMConfig", ast_defines_class(model_py, "LMConfig"))

    # --- Q1.4 attention-scale hook -----------------------------------------------------
    try:
        attn_mod = load_student_module(attn_py, "student_attention_q1_scale")
        sig = inspect.signature(attn_mod.CausalSelfAttention.__init__)
        has_hook = "attention_scale" in sig.parameters
        report.check("CausalSelfAttention.__init__ exposes attention_scale hook (Q1.4)",
                     has_hook, f"params: {list(sig.parameters)}")
        if has_hook:
            d_model, n_heads = 64, 8
            standard = 1.0 / math.sqrt(d_model // n_heads)
            a_default = attn_mod.CausalSelfAttention(d_model, n_heads, 16, dropout=0.0)
            a_explicit = attn_mod.CausalSelfAttention(d_model, n_heads, 16, dropout=0.0,
                                                        attention_scale=standard)
            a_unscaled = attn_mod.CausalSelfAttention(d_model, n_heads, 16, dropout=0.0,
                                                       attention_scale=1.0)
            a_explicit.load_state_dict(a_default.state_dict())
            a_unscaled.load_state_dict(a_default.state_dict())
            a_default.eval(); 
            a_explicit.eval();
            a_unscaled.eval()
            x = torch.randn(2, 10, d_model)
            with torch.no_grad():
                y_default = a_default(x)
                y_explicit = a_explicit(x)
                y_unscaled = a_unscaled(x)
            report.check("attention_scale=None matches explicit 1/sqrt(d_head)",
                          torch.allclose(y_default, y_explicit, atol=1e-6, rtol=1e-5))
            report.check("changing attention_scale changes attention output (hook is functional)",
                         not torch.allclose(y_default, y_unscaled, atol=1e-6, rtol=1e-5))
    except Exception as e:
        report.check("Q1.4 attention_scale hook functional check", False, repr(e))

    # --- A1 tokenizer retraining hook --------------------------------------------------
    tokenizer_candidates = [
        os.path.join(student_dir, "a1_tokenizer.py"),
        os.path.join(sub_dir, "a1_tokenizer.py"),
    ]
    tokenizer_py = next((p for p in tokenizer_candidates if os.path.exists(p)), tokenizer_candidates[0])
    report.check("a1_tokenizer.py present for A1-tokenizer retraining", os.path.exists(tokenizer_py))
    if os.path.exists(tokenizer_py):
        try:
            tok_mod = load_student_module(tokenizer_py, "student_a1_tokenizer_q1")
            report.check(
                "a1_tokenizer.py exposes train_tokenizer(train_text, vocab_size)",
                callable(getattr(tok_mod, "train_tokenizer", None)),
            )
        except Exception as e:
            report.check("a1_tokenizer.py imports cleanly", False, repr(e))

    # --- load submission artifacts ----------------------------------------------------
    try:
        config, metrics, model_path = load_submission_dir(sub_dir)
        report.check("submission artifacts load (config.json, metrics.json, model.pt)", True)
    except FileNotFoundError as e:
        report.check("submission artifacts load (config.json, metrics.json, model.pt)", False, str(e))
        return report.summary()

    check_q1_metrics(metrics, report)
    check_q1_scaling(metrics, config, report)

    # --- instantiate model from saved config and load weights ------------------------
    try:
        stack = load_full_student_stack(sub_dir, need=("attention", "model_arch"))
        model_mod = stack["model_arch"]

        cfg_fields = {k: config[k] for k in
                      ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                       "d_ff", "max_seq_len", "dropout", "pos_emb_type", "attention_scale", "norm_first"] if k in config}
        cfg = model_mod.LMConfig(**cfg_fields)
        model = model_mod.DecoderLM(cfg)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        check_causality(model, cfg, report) # does the causal mask check remove it if not wanted.
        report.check("model instantiates from config.json and loads model.pt", True)
    except Exception as e:
        report.check("model instantiates from config.json and loads model.pt", False, repr(e))
        return report.summary()

    n_params = check_param_budget(model, report)
    report.check("config's reported param_count matches actual", abs(config.get("param_count", -1) - n_params) < 10)

    # --- forward pass shape check ------------------------------------------------------
    try:
        B, T = 2, min(16, cfg.max_seq_len)
        dummy = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits = model(dummy)
        shape_ok = logits.shape == (B, T, cfg.vocab_size)
        report.check("forward pass produces (B, T, vocab_size) logits", shape_ok, f"got {tuple(logits.shape)}")
    except Exception as e:
        report.check("forward pass produces (B, T, vocab_size) logits", False, repr(e))
        return report.summary()

    # --- perplexity sanity recheck (best-effort; skipped if no corpus locally) --------
    tok, used_student_tokenizer, tok_diagnostics = build_tokenizer_for_grading(
        student_dir, config.get("vocab_size", 8000)
    )
    if tok is not None:
        report.check(
            "A1 tokenizer hook retrains successfully and exposes encode/decode/vocabulary size",
            used_student_tokenizer,
            "; ".join(tok_diagnostics) if tok_diagnostics else "",
        )
        # Even if the required hook fails, continue numerical grading using the fallback.
        if not used_student_tokenizer:
            print("    [tokenizer fallback] Student A1 hook was not usable:")
            for item in tok_diagnostics:
                print(f"      - {item}")

        _, held_out, _ = get_shared_eval_set()
        recheck_ppl = try_compute_perplexity(model, tok, held_out[:10], max_len=cfg.max_seq_len)
        if recheck_ppl is not None:
            label = "student retrained A1 tokenizer" if used_student_tokenizer else "fallback BPE"
            print(f"    (independent sentence-level re-check ppl on {label}/subset: {recheck_ppl:.2f})")

    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    ok = main(args.student_dir)
    sys.exit(0 if ok else 1)
