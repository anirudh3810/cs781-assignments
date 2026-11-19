#!/usr/bin/env python3
"""
Autograder for Question 3: Rotary Positional Embeddings (RoPE) from scratch.

Usage:
    python autograder_q3.py --student_dir /path/to/students/alice

Expects under --student_dir/submissions/q3: rope.py, attention.py, model_arch.py,
model.pt, config.json, and metrics.json.

The core check here is a deterministic unit test run directly against the student's
own rope.py: the relative-position invariance property of RoPE. This requires no
trained model and no corpus, so it always runs, even on a bare-bones grading machine.
"""
import argparse
import math
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
    check_forbidden_implementations,
    metric_is_finite,
    check_metric_curve,
    metric_dict,
    run_grading_case,
    assert_tensors_close,
    make_test_attention,
    randomize_attention_projections,
    reference_causal_attention,
)


def check_q3_experiment_evidence(metrics, report):
    metrics = metric_dict(report, "Q3 metrics", metrics)

    for key in ("sinusoidal_held_out_ppl", "rope_held_out_ppl"):
        report.check( key, metric_is_finite(metrics.get(key), minimum=1), "expected finite PPL >= 1")

    for key in ("sinusoidal_loss_curve", "rope_loss_curve"):
        check_metric_curve(report, key, metrics.get(key))

    length = metric_dict( report, "length_extrapolation", metrics.get("length_extrapolation"))

    for scheme in ("sinusoidal", "rope"):
        values = metric_dict( report, f"length_extrapolation.{scheme}",length.get(scheme))

        for key in ("ppl_64", "ppl_128"):
            report.check( f"length_extrapolation.{scheme}.{key}", metric_is_finite(values.get(key), minimum=1), "expected finite PPL >= 1")

    freq = metric_dict( report, "frequency_ablation", metrics.get("frequency_ablation"))

    for key in ( "baseline_rope_ppl","freeze_slowest_half_ppl", "freeze_fastest_half_ppl"):
        report.check( f"frequency_ablation.{key}", metric_is_finite(freq.get(key), minimum=1), "expected finite PPL >= 1")

    for hit_key, ppl_key in (
        ("slow_hit", "freeze_slowest_half_ppl"),
        ("fast_hit", "freeze_fastest_half_ppl"),
    ):
        hit = freq.get(hit_key)
        ablated = freq.get(ppl_key)
        baseline = freq.get("baseline_rope_ppl")

        # Negative changes are valid: an ablation may improve measured PPL.
        ok = (
            metric_is_finite(hit) and metric_is_finite(ablated, minimum=1)
            and metric_is_finite(baseline, minimum=1)
            and math.isclose( hit, ablated - baseline, rel_tol=1e-4, abs_tol=1e-3)
        )

        report.check(f"frequency_ablation.{hit_key}", ok, f"must equal {ppl_key} - baseline_rope_ppl")


def _reference_rope_qk(q, k):
    """Independent base-10000, split-half RoPE; rotate Q and K only."""
    length, dim = q.shape[-2:]
    frequencies = 10000.0 ** (-torch.arange(0, dim, 2, dtype=q.dtype, device=q.device) / dim)
    angles = torch.arange(length, dtype=q.dtype, device=q.device)[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]

    def rotate(t):
        first, second = t.chunk(2, dim=-1)
        return t * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()

    return rotate(q), rotate(k)


def check_q3_rope_integration(attn_mod, report):
    """Matched weights, independent references for both use_rope settings."""
    for kv in (8, 2, 1):
        def run(kv=kv):
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(781 + kv)
                generator = torch.Generator().manual_seed(781 + kv)
                plain = make_test_attention(attn_mod.CausalSelfAttention, kv, require_rope=True)
                rotated = make_test_attention(attn_mod.CausalSelfAttention, kv, rope=True, require_rope=True)
                randomize_attention_projections(plain, generator)
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    getattr(rotated, name).load_state_dict(getattr(plain, name).state_dict())
                for length in (1, 7, 16):
                    x = torch.randn(2, length, 64, generator=generator)
                    y_plain, y_rope = plain(x), rotated(x)
                    assert_tensors_close(y_plain, reference_causal_attention(plain, x, kv))
                    assert_tensors_close(y_rope, reference_causal_attention(
                        rotated, x, kv, transform_qk=_reference_rope_qk
                    ))
                    if length > 1 and torch.allclose(y_plain, y_rope, atol=1e-6, rtol=1e-5):
                        raise AssertionError("use_rope has no effect with matched weights")
            return "both modes match their references at lengths 1, 7 and 16"
        run_grading_case(report, f"Q3 RoPE integration, n_kv_heads={kv}", run)


def main(student_dir: str):
    report = GradeReport("Question 3")
    sub_dir = os.path.join(student_dir, "submissions", "q3")
    rope_py = os.path.join(sub_dir, "rope.py")

    if not os.path.exists(rope_py):
        report.check("rope.py present", False)
        return report.summary()
    report.check("rope.py present", True)
    if not check_forbidden_implementations(report, sub_dir, question=3):
        return report.summary()

    # Load the full dependency chain (rope -> attention -> model_arch) in the correct
    # order, once, up front. This avoids the ModuleNotFoundError bugs that come from
    # importing attention.py in isolation when it does `from rope import ...`.
    try:
        stack = load_full_student_stack(sub_dir, need=("rope", "attention", "model_arch"))
        rope_mod = stack["rope"]
        attn_mod = stack["attention"]
        model_mod = stack["model_arch"]
        for fn_name in ["build_rope_cache", "rotate_half", "apply_rope"]:
            report.check(f"rope.py defines {fn_name}()", hasattr(rope_mod, fn_name))
    except Exception as e:
        report.check("rope.py / attention.py import cleanly", False, repr(e))
        return report.summary()

    # --- THE key unit test: relative-position invariance, no training required -------
    try:
        d_head = 16
        cos, sin = rope_mod.build_rope_cache(max_seq_len=300, d_head=d_head)
        report.check("build_rope_cache returns (cos, sin) each of shape (max_seq_len, d_head)",
                     tuple(cos.shape) == (300, d_head) and tuple(sin.shape) == (300, d_head),
                     f"cos.shape={tuple(cos.shape)}, sin.shape={tuple(sin.shape)}")

        def rotated_dot(q_vec, k_vec, m, n):
            q_rot = rope_mod.apply_rope(q_vec, cos[m:m + 1], sin[m:m + 1])
            k_rot = rope_mod.apply_rope(k_vec, cos[n:n + 1], sin[n:n + 1])
            return (q_rot * k_rot).sum().item()

        # Average the invariance check over several random vectors and several distinct
        # relative distances, rather than a single (potentially unlucky) sample -- a
        # single random vector pair can land near a coincidental near-tie between two
        # different distances, which made an earlier version of this check flaky.
        torch.manual_seed(0)
        n_trials = 8
        same_distance_diffs = []
        cross_distance_diffs = []
        for _ in range(n_trials):
            q_vec = torch.randn(1, 1, 1, d_head)
            k_vec = torch.randn(1, 1, 1, d_head)
            # same relative distance (3), far apart absolute positions
            d_close = rotated_dot(q_vec, k_vec, 5, 2)
            d_far = rotated_dot(q_vec, k_vec, 205, 202)
            same_distance_diffs.append(abs(d_close - d_far))
            # clearly different relative distance (3 vs. 43) using the SAME vectors and
            # the SAME starting position, so only the distance changes
            d_other_distance = rotated_dot(q_vec, k_vec, 45, 2)
            cross_distance_diffs.append(abs(d_close - d_other_distance))

        mean_same = sum(same_distance_diffs) / n_trials
        mean_cross = sum(cross_distance_diffs) / n_trials

        report.check(
            "RoPE relative-position invariance: dot(m,n) depends only on m-n "
            f"(mean |diff| over {n_trials} random vector pairs, same distance, different absolute positions)",
            mean_same < 1e-2,
            f"mean |diff| = {mean_same:.2e}",
        )
        report.check(
            "sanity: a clearly DIFFERENT relative distance (3 vs. 43) gives a "
            "meaningfully different dot product on average (catches a no-op / identity "
            "apply_rope, or one that collapses all distances together, which would "
            "otherwise trivially pass the check above)",
            mean_cross > 0.1,
            f"mean |diff| = {mean_cross:.2e}",
        )
    except Exception as e:
        report.check("RoPE unit test runs", False, repr(e))

    check_q3_rope_integration(attn_mod, report)

    # --- submission artifacts ---------------------------------------------------------
    try:
        config, metrics, model_path = load_submission_dir(sub_dir)
        report.check("submission artifacts load", True)
    except FileNotFoundError as e:
        report.check("submission artifacts load", False, str(e))
        return report.summary()

    report.check("config declares pos_emb_type == 'rope'", config.get("pos_emb_type") == "rope",
                 f"got {config.get('pos_emb_type')!r}")
    
    check_q3_experiment_evidence(metrics, report)

    try:
        if model_mod is None:
            raise FileNotFoundError("model_arch.py not found")
        cfg_fields = {k: config[k] for k in
                      ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                       "d_ff", "max_seq_len", "dropout", "pos_emb_type","attention_scale", "norm_first"] if k in config}
        cfg = model_mod.LMConfig(**cfg_fields)
        model = model_mod.DecoderLM(cfg)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        check_causality(model, cfg, report) # does the causal mask check remove it if not wanted.
        check_param_budget(model, report)
        dummy = torch.randint(0, cfg.vocab_size, (2, min(16, cfg.max_seq_len)))
        with torch.no_grad():
            logits = model(dummy)
        report.check("saved RoPE model forward pass shape correct",
                     logits.shape == (2, min(16, cfg.max_seq_len), cfg.vocab_size))
    except Exception as e:
        report.check("saved RoPE model loads and runs", False, repr(e))

    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    ok = main(args.student_dir)
    sys.exit(0 if ok else 1)
