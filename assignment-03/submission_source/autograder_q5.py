#!/usr/bin/env python3
"""
Autograder for Question 5: rebuilding with torch.nn.Transformer* building blocks.

Usage:
    python autograder_q5.py --student_dir /path/to/students/alice

Expects under --student_dir/submissions/q5: model_builtin.py and its dependency source
files, model.pt, config.json, and metrics.json.

Unlike Questions 1-4, this grader requires the built-in modules to be used (the
runtime module check).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common_grading import (
    GradeReport,
    load_submission_dir,
    check_param_budget,
    load_full_student_stack,
    check_causality,
    metric_is_finite,
    run_grading_case,
    assert_tensors_close,
    make_test_attention,
    randomize_attention_projections,
)


def check_builtin_usage(model, cfg, report):
    """Accept aliases; require execution and an observable contribution to logits."""
    import torch
    def run():
        types = (torch.nn.Transformer, torch.nn.TransformerEncoder,
                 torch.nn.TransformerDecoder, torch.nn.TransformerEncoderLayer,
                 torch.nn.TransformerDecoderLayer, torch.nn.MultiheadAttention)
        candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, types)]
        if not candidates:
            raise AssertionError("no instantiated built-in attention/Transformer module")
        # Hook outermost built-ins; fused execution may bypass child hooks.
        outer = [(name, module) for name, module in candidates
                 if not any(parent != name and (parent == "" or name.startswith(parent + "."))
                            for parent, _ in candidates)]
        modes = [(module, module.training) for module in model.modules()]
        handles, called = [], set()
        try:
            model.eval()
            for name, module in outer:
                def record(module, args, output, name=name):
                    called.add(name)
                handles.append(module.register_forward_hook(record))
            rng = torch.Generator().manual_seed(786)
            device = next(model.parameters()).device
            length = min(9, cfg.max_seq_len)
            ids = torch.randint(cfg.vocab_size, (2, length), generator=rng).to(device)
            with torch.no_grad():
                baseline = model(ids)
            for handle in handles:
                handle.remove()
            handles.clear()
            if not called:
                raise AssertionError("built-in modules exist but were not executed")
            if not isinstance(baseline, torch.Tensor) or not torch.isfinite(baseline).all().item():
                raise AssertionError("invalid baseline logits")

            def ablate(module, args, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

            for name, module in outer:
                if name in called:
                    handles.append(module.register_forward_hook(ablate))
            with torch.no_grad():
                altered = model(ids)
            if not isinstance(altered, torch.Tensor) or altered.shape != baseline.shape:
                raise AssertionError("invalid logits during built-in ablation")
            if not torch.isfinite(altered).all().item():
                raise AssertionError("nonfinite logits during built-in ablation")
            if torch.allclose(baseline, altered, atol=1e-7, rtol=1e-6):
                raise AssertionError("zeroing executed built-in outputs did not affect logits")
            return "executed and affected logits: " + ", ".join(sorted(called))
        finally:
            for handle in handles:
                handle.remove()
            for module, training in modes:
                module.training = training
    run_grading_case(report, "Q5 built-in modules are used in the model computation", run)

def check_q5_metric(metrics, report):
    """Validate the numerical results required by Q5.2 and Q5.4."""
    is_dict = isinstance(metrics, dict)
    report.check( "Q5 metrics is a dictionary", is_dict, "expected a JSON object")
    if not is_dict:
        return

    # Q5.2: reported numerical attention-equivalence result.
    diff = metrics.get("correctness_max_abs_diff")
    report.check( "Q5 correctness_max_abs_diff is finite and 0 <= diff < 1e-3",
        metric_is_finite(diff, minimum=0) and diff < 1e-3, f"reported={diff!r}")

    # Q5.4: both models must have actual measured perplexities.
    for key in ("builtin_held_out_ppl", "scratch_held_out_ppl"):
        value = metrics.get(key)
        report.check( f"Q5 {key} is finite and >= 1", metric_is_finite(value, minimum=1),f"reported={value!r}")

    # Q5.4/Q5.7: both models must have actual per-step timings,
    # expressed in seconds.
    for key in ( "builtin_seconds_per_step", "scratch_seconds_per_step"):
        value = metrics.get(key)
        report.check(f"Q5 {key} is finite and > 0", metric_is_finite(value, minimum=0) and value > 0, f"reported={value!r}")


def check_q5_equivalence(attn_mod, report):
    """Rerun Q5.2 against PyTorch using the saved Q1 scratch attention source."""
    import torch
    def run():
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(787)
            rng = torch.Generator().manual_seed(787)
            scratch = make_test_attention(attn_mod.CausalSelfAttention)
            randomize_attention_projections(scratch, rng)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                if getattr(scratch, name).bias is not None:
                    raise AssertionError("Q5.2 requires bias-free scratch projections")
            builtin = torch.nn.MultiheadAttention(64, 8, dropout=0.0, bias=False, batch_first=True).eval()
            builtin.in_proj_weight.copy_(torch.cat([scratch.q_proj.weight, scratch.k_proj.weight,
                                                   scratch.v_proj.weight], dim=0))
            builtin.out_proj.weight.copy_(scratch.out_proj.weight)
            worst = 0.0
            for length in (1, 7, 16):
                x = torch.randn(2, length, 64, generator=rng)
                blocked = torch.ones(length, length, dtype=torch.bool).triu(1)
                expected, _ = builtin(x, x, x, attn_mask=blocked, need_weights=False)
                actual = scratch(x)
                assert_tensors_close(actual, expected, atol=1e-3, rtol=0.0)
                worst = max(worst, (actual - expected).abs().max().item())
            if worst >= 1e-3:
                raise AssertionError(f"max abs diff={worst:.3e} is not < 1e-3")
        return f"independent max abs diff={worst:.3e}"
    run_grading_case(report, "Q5 scratch attention independently matches nn.MultiheadAttention", run)


def main(student_dir: str):
    report = GradeReport("Question 5")
    sub_dir = os.path.join(student_dir, "submissions", "q5")
    builtin_py = os.path.join(sub_dir, "model_builtin.py")

    if not os.path.exists(builtin_py):
        report.check("model_builtin.py present", False)
        return report.summary()
    report.check("model_builtin.py present", True)

    try:
        stack = load_full_student_stack(sub_dir, need=("model_builtin",))
        model_mod = stack["model_builtin"]
        report.check("model_builtin.py defines BuiltinDecoderLM", hasattr(model_mod, "BuiltinDecoderLM"))
        report.check("model_builtin.py defines BuiltinLMConfig", hasattr(model_mod, "BuiltinLMConfig"))
    except Exception as e:
        report.check("model_builtin.py imports cleanly "
                      "(common cause of failure here: importing sinusoidal_positional_encoding "
                      "from model_arch.py without model_arch.py -- and its own dependencies, "
                      "attention.py / rope.py -- being alongside it)", False, repr(e))
        return report.summary()

    # --- submission artifacts ---------------------------------------------------------
    try:
        config, metrics, model_path = load_submission_dir(sub_dir)
        report.check("submission artifacts load", True)
    except FileNotFoundError as e:
        report.check("submission artifacts load", False, str(e))
        return report.summary()
    
    check_q5_metric(metrics, report)

    # --- load and forward-pass the saved model ----------------------------------------
    try:
        cfg_fields = {k: config[k] for k in
                      ["vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "dropout", "norm_first"]
                      if k in config}
        cfg = model_mod.BuiltinLMConfig(**cfg_fields)
        model = model_mod.BuiltinDecoderLM(cfg)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        check_builtin_usage(model, cfg, report)
        check_causality(model, cfg, report) # does the causal mask check remove it if not wanted.
        check_param_budget(model, report)
        dummy = torch.randint(0, cfg.vocab_size, (2, min(16, cfg.max_seq_len)))
        with torch.no_grad():
            logits = model(dummy)
        report.check("saved built-in model forward pass shape correct",
                     logits.shape == (2, min(16, cfg.max_seq_len), cfg.vocab_size))
        
    except Exception as e:
        report.check("saved built-in model loads, runs, and is causal", False, repr(e))

    # Q5.2 explicitly compares the saved Q1 scratch implementation.
    q1_dir = os.path.join(student_dir, "submissions", "q1")
    try:
        q1_stack = load_full_student_stack(q1_dir, need=("attention",))
    except Exception as exc:
        report.check("Q5 independent equivalence: Q1 attention source loads", False, repr(exc))
    else:
        check_q5_equivalence(q1_stack["attention"], report)

    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    ok = main(args.student_dir)
    sys.exit(0 if ok else 1)
