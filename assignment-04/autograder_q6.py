#!/usr/bin/env python3
"""
Autograder for Question 6: knowledge distillation into DistilGPT-2.

Usage:
    python autograder_q6.py --student_dir /path/to/student [--skip_reload]

Expects submissions/q6/{q6_distill.py, weights.pt, metrics.json} plus q2_classifier.py.
The loss is compared with an independent reference implementation on random logits; the
reload check downloads distilbert/distilgpt2 (skip with --skip_reload).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, SubmissionDir, check_metrics_block, check_reload, is_finite,  # noqa: E402
                            run_grading_case)


def reference_loss(s, t, y, tau, lam):
    ce = F.cross_entropy(s, y)
    pt = F.softmax(t / tau, -1)
    kd = (pt * (F.log_softmax(t / tau, -1) - F.log_softmax(s / tau, -1))).sum(-1).mean() * tau ** 2
    return (1 - lam) * ce + lam * kd, ce, kd


def check_loss(m, report):
    def values():
        g = torch.Generator().manual_seed(21)
        worst = 0.0
        for tau in (1.0, 2.0, 4.0):
            for lam in (0.0, 0.5, 1.0):
                s, t = torch.randn(16, 2, generator=g) * 3, torch.randn(16, 2, generator=g) * 3
                y = torch.randint(0, 2, (16,), generator=g)
                total, parts = m.distillation_loss(s, t, y, tau=tau, kd_weight=lam)
                ref, ce, kd = reference_loss(s, t, y, tau, lam)
                worst = max(worst, abs(float(total) - float(ref)))
                assert abs(parts["ce"] - float(ce)) < 1e-5 and abs(parts["kd"] - float(kd)) < 1e-5, \
                    f"components differ at tau={tau}, lambda={lam}: {parts} vs ce={float(ce):.5f}, kd={float(kd):.5f}"
        assert worst < 1e-5, f"total loss differs from (1-λ)CE + λτ²KL(p_t||p_s) by {worst:.1e}"
        return f"9 (τ, λ) settings, max |Δ| {worst:.1e}"
    run_grading_case(report, "6.2 loss value: forward KL, τ² scaling, batch mean", values)

    def gradients():
        s = torch.randn(8, 2, requires_grad=True)
        t = torch.randn(8, 2, requires_grad=True)
        y = torch.randint(0, 2, (8,))
        total, _ = m.distillation_loss(s, t, y)
        total.backward()
        assert s.grad is not None and float(s.grad.abs().sum()) > 0, "no gradient to the student"
        assert t.grad is None, "teacher logits must be detached"
        big = torch.tensor([[1e4, -1e4], [-5e3, 5e3]], requires_grad=True)
        total, _ = m.distillation_loss(big, -big.detach(), torch.tensor([0, 1]), tau=2.0)
        assert torch.isfinite(total), "large logits produced a non-finite loss"
        same, parts = m.distillation_loss(s.detach(), s.detach(), y, tau=2.0, kd_weight=0.5)
        assert abs(parts["kd"]) < 1e-6, "KD term must vanish for identical logits"
        return "student-only gradients, finite for large logits, zero KD for identical logits"
    run_grading_case(report, "6.2 detachment and numerical stability", gradients)


def check_weights_and_metrics(weights, metrics, report):
    def run():
        for mode in ("ce", "kd"):
            entry = weights[mode]
            assert any(k.startswith("backbone.") for k in entry["state"]), f"{mode}: full student backbone expected"
            assert any(k.startswith("classifier.") for k in entry["state"]), f"{mode}: head missing"
            cfg = entry["config"]
            assert cfg.get("student") and cfg.get("max_length"), f"{mode}: config incomplete"
        assert weights["kd"]["config"].get("kd_weight", 0) > 0 and weights["ce"]["config"].get("kd_weight", 0) == 0
        n = sum(v.numel() for v in weights["kd"]["state"].values())
        assert n < 100_000_000, f"student has {n:,} parameters; expected DistilGPT-2 (~82M)"
        return "CE and KD students with configs"
    run_grading_case(report, "6.4 weights.pt students", run)
    check_metrics_block(report, "teacher test", metrics.get("teacher"))
    for mode in ("ce", "kd"):
        check_metrics_block(report, f"student {mode} test", metrics.get(mode, {}).get("test"))
    agr = metrics.get("agreement", {})
    report.check("6.3 teacher agreement + conditional accuracy recorded",
                 all(is_finite(agr.get(k), 0, 1) for k in ("ce", "kd")), f"got {agr}")
    cache = metrics.get("teacher_cache", {})
    report.check("6.1 teacher cache covers every training ID exactly once, no held-out IDs",
                 cache.get("all_train_ids_present") is True and cache.get("duplicates") == 0
                 and cache.get("held_out_ids_in_cache") == 0, f"got {cache}")


def main(student_dir, skip_reload=False):
    report = GradeReport("Question 6")
    sub = SubmissionDir(student_dir, "q6")
    try:
        mods = sub.load_modules("q2_classifier", "q6_distill")
        m = mods["q6_distill"]
        report.check("q6_distill.py (and q2_classifier.py) import", True)
    except Exception as exc:
        report.check("q6_distill.py (and q2_classifier.py) import", False, repr(exc))
        return report.summary()
    check_loss(m, report)
    try:
        metrics, weights = sub.metrics(), sub.weights()
    except Exception as exc:
        report.check("submission artifacts load", False, repr(exc))
        return report.summary()
    check_weights_and_metrics(weights, metrics, report)
    if skip_reload:
        report.skip("reload checks", "--skip_reload")
    else:
        for mode in ("ce", "kd"):
            check_reload(report, mode, lambda mm=mode: m.reload_q6(mm, root=sub.root), metrics["reload_check"])
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--skip_reload", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir, args.skip_reload) else 1)
