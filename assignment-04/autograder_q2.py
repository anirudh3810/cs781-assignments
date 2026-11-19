#!/usr/bin/env python3
"""
Autograder for Question 2: GPT-2 sentiment classifier (linear probe vs full fine-tuning).

Usage:
    python autograder_q2.py --student_dir /path/to/student [--skip_reload]

Expects submissions/q2/{q2_classifier.py, weights.pt, metrics.json}. Pooling/padding checks
use a random tiny GPT-2 (no download). The reload check downloads openai-community/gpt2
and reproduces the stored validation predictions (skip with --skip_reload).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, SubmissionDir, check_metrics_block, check_reload, is_finite,  # noqa: E402
                            run_grading_case, tiny_gpt2)


def check_pooling(m, report):
    def run():
        h = torch.arange(2 * 5 * 3, dtype=torch.float).view(2, 5, 3)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        out = m.last_nonpad(h, mask)
        assert torch.equal(out, torch.stack([h[0, 2], h[1, 4]])), "wrong pooled positions"
        try:
            m.last_nonpad(h, torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]))
        except Exception:
            return "last visible position; all-padding row rejected"
        raise AssertionError("all-padding row was not rejected")
    run_grading_case(report, "2.2 last_nonpad uses the attention mask", run)


def check_classifier(m, report):
    def run():
        backbone = tiny_gpt2()
        clf = m.SentimentClassifier(backbone, backbone.config.hidden_size).eval()
        ids = torch.randint(1, 200, (2, 9))
        mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4])
        with torch.no_grad():
            a = clf(input_ids=ids, attention_mask=mask)
            ids2 = torch.cat([ids, torch.zeros(2, 6, dtype=torch.long)], 1)
            mask2 = torch.cat([mask, torch.zeros(2, 6, dtype=torch.long)], 1)
            b = clf(input_ids=ids2, attention_mask=mask2)
            single = clf(input_ids=ids[1:, :5], attention_mask=mask[1:, :5])
        assert a.shape == (2, 2), f"logits shape {tuple(a.shape)} (expected [B, 2] class logits)"
        d1, d2 = float((a - b).abs().max()), float((a[1:] - single).abs().max())
        assert d1 < 1e-5 and d2 < 1e-5, f"padding changed logits: extra-pad {d1:.1e}, batched-vs-single {d2:.1e}"
        return f"[B,2] logits, padding invariant (max |Δ| {max(d1, d2):.1e})"
    run_grading_case(report, "2.2 SentimentClassifier: class logits, padding invariance", run)

    def frozen_mode():
        backbone = tiny_gpt2()
        clf = m.SentimentClassifier(backbone, backbone.config.hidden_size)
        for p in clf.backbone.parameters():
            p.requires_grad_(False)
        clf.backbone_eval = True
        clf.train()
        assert not clf.backbone.training and clf.classifier.training, "frozen backbone must stay in eval mode"
        return "backbone_eval keeps dropout off in train()"
    run_grading_case(report, "2.3 frozen backbone stays in eval mode", frozen_mode)


def check_weights(weights, report):
    def run():
        for key in ("linear_probe", "full", "initial_head"):
            assert key in weights, f"weights.pt missing '{key}'"
        probe, full = weights["linear_probe"]["state"], weights["full"]["state"]
        assert all(k.startswith("classifier.") for k in probe), "linear-probe state must contain only the head"
        assert any(k.startswith("backbone.") for k in full), "full fine-tuning state must contain backbone weights"
        for run_name in ("linear_probe", "full"):
            cfg = weights[run_name]["config"]
            assert cfg.get("backbone") and cfg.get("max_length"), f"{run_name} config incomplete: {cfg}"
        head = weights["initial_head"]
        assert set(head) == {"weight", "bias"} and tuple(head["weight"].shape) == (2, 768)
        return "probe=head only, full=backbone+head, configs and initial head present"
    run_grading_case(report, "2.5 weights.pt structure", run)


def check_metrics(metrics, report):
    check_metrics_block(report, "majority baseline", metrics.get("majority_baseline"))
    for run_name in ("linear_probe", "full"):
        r = metrics.get(run_name, {})
        check_metrics_block(report, f"{run_name} test", r.get("test"))
        check_metrics_block(report, f"{run_name} prompt_test", r.get("prompt_test"))
        costs = r.get("costs", {})
        report.check(f"{run_name} costs recorded",
                     all(is_finite(costs.get(k), 0) for k in ("trainable_params", "total_params", "train_seconds")),
                     f"got {costs}")
    full_acc = metrics.get("full", {}).get("test", {}).get("accuracy")
    report.check("full fine-tuning beats chance on test (accuracy >= 0.6)", is_finite(full_acc) and full_acc >= 0.6,
                 f"accuracy={full_acc}")
    boot = metrics.get("bootstrap_full_minus_probe", {})
    report.check("paired bootstrap interval recorded",
                 all(is_finite(boot.get(k)) for k in ("point_diff", "ci95_low", "ci95_high"))
                 and boot["ci95_low"] <= boot["ci95_high"], f"got {boot}")
    rc = metrics.get("resume_check", {})
    report.check("resume check: resumed run matches uninterrupted run",
                 is_finite(rc.get("max_param_diff"), 0) and rc["max_param_diff"] < 1e-4, f"got {rc}")
    pad = metrics.get("padding_invariance_max_abs_diff")
    report.check("padding invariance recorded (< 1e-3)", is_finite(pad, 0) and pad < 1e-3, f"got {pad}")


def main(student_dir, skip_reload=False):
    report = GradeReport("Question 2")
    sub = SubmissionDir(student_dir, "q2")
    try:
        m = sub.load_modules("q2_classifier")["q2_classifier"]
        report.check("q2_classifier.py imports", True)
    except Exception as exc:
        report.check("q2_classifier.py imports", False, repr(exc))
        return report.summary()
    check_pooling(m, report)
    check_classifier(m, report)
    try:
        metrics, weights = sub.metrics(), sub.weights()
        report.check("submission artifacts load", True)
    except Exception as exc:
        report.check("submission artifacts load", False, repr(exc))
        return report.summary()
    check_weights(weights, report)
    check_metrics(metrics, report)
    if skip_reload:
        report.skip("reload checks", "--skip_reload")
    else:
        for run_name in ("linear_probe", "full"):
            check_reload(report, run_name, lambda r=run_name: m.reload_q2(r, root=sub.root), metrics["reload_check"])
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--skip_reload", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir, args.skip_reload) else 1)
