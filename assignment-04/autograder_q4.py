#!/usr/bin/env python3
"""
Autograder for Question 4: soft prompts, adapters, LoRA and LoRMA implemented from scratch.

Usage:
    python autograder_q4.py --student_dir /path/to/student [--skip_reload]

Expects submissions/q4/{q4_peft.py, weights.pt, metrics.json} plus q2_classifier.py
(submissions/q2/ or the student root). Mechanism checks use a random tiny GPT-2; the reload
check downloads openai-community/gpt2 (skip with --skip_reload).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, SubmissionDir, check_metrics_block, check_reload, is_finite,  # noqa: E402
                            notebook_source, run_grading_case, tiny_gpt2)

METHODS = ("soft_prompt", "adapter", "lora", "lorma")
CFG = {"num_virtual_tokens": 4, "init_token_ids": [5, 6, 7, 8], "bottleneck": 4, "rank": 2, "alpha": 4}


def frozen_tiny(q2, seed=0):
    backbone = tiny_gpt2(seed=seed)
    clf = q2.SentimentClassifier(backbone, backbone.config.hidden_size)
    torch.manual_seed(100 + seed)
    for p in clf.backbone.parameters():
        p.requires_grad_(False)
    clf.backbone_eval = True
    return clf


def batch():
    torch.manual_seed(3)
    ids = torch.randint(10, 200, (3, 11))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    mask[2, 5:] = 0
    return ids, mask


def check_mechanisms(q2, m, report):
    ids, mask = batch()
    for method in METHODS:
        def run(method=method):
            base = frozen_tiny(q2).eval()
            clf = frozen_tiny(q2)
            clf.classifier.load_state_dict(base.classifier.state_dict())
            original = {id(p) for p in clf.parameters()} - {id(p) for p in clf.classifier.parameters()}
            model, names = m.attach_peft(clf, method, dict(CFG))
            model = model.cpu().float().eval()
            assert names and all(dict(model.named_parameters())[n].requires_grad for n in names), "trainable names"
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert not any(id(p) in original for p in trainable), "an original backbone weight is trainable"
            assert any("classifier" in n for n in names), "the classification head must be trainable"
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
                ref = base(input_ids=ids, attention_mask=mask)
            assert out.shape == (3, 2), f"logits shape {tuple(out.shape)}"
            detail = []
            if method != "soft_prompt":
                d = float((out - ref).abs().max())
                assert d < 1e-5, f"initialisation must preserve the frozen model (max |Δ| = {d:.1e})"
                detail.append(f"init preserves base (|Δ| {d:.0e})")
            model.train()
            torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask), torch.tensor([0, 1, 1])).backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad), "frozen params got gradients"
            peft = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
            assert peft and any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in peft), \
                "no gradient reaches the PEFT parameters"
            n_peft = sum(p.numel() for _, p in peft)
            expected = {"adapter": 2 * (2 * 32 * 4 + 4 + 32), "lora": 2 * 2 * (32 + 96), "soft_prompt": 4 * 32}.get(method)
            if expected is not None:
                assert n_peft == expected, f"{n_peft} PEFT parameters, expected {expected} for the tiny model"
            detail.append(f"{n_peft} PEFT params, gradients flow, backbone frozen")
            return "; ".join(detail)
        run_grading_case(report, f"4.x attach_peft('{method}') mechanism", run)


def check_soft_prompt(m, report):
    def run():
        torch.manual_seed(0)
        sp = m.SoftPrompt(torch.randn(4, 8))
        emb, mask = torch.randn(2, 5, 8), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        e2, m2 = sp(emb, mask)
        assert e2.shape == (2, 9, 8) and m2.shape == (2, 9), "prompt must be prepended"
        assert torch.equal(m2[:, :4], torch.ones(2, 4, dtype=m2.dtype)) and torch.equal(m2[:, 4:], mask.to(m2.dtype))
        assert torch.equal(e2[:, 4:], emb), "text embeddings must follow the prompt unchanged"
        e2.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in sp.parameters()), "prompt not trainable"
        return "prepended positions, extended mask, gradients"
    run_grading_case(report, "4.1 SoftPrompt insertion", run)


def check_lora(m, report):
    def run():
        from transformers.pytorch_utils import Conv1D
        torch.manual_seed(1)
        base = Conv1D(12, 6)                      # in=6, out=12 -> weight [6, 12]
        w0 = base.weight.detach().clone()
        layer = m.LoRAConv1D(base, rank=3, alpha=6)
        x = torch.randn(2, 4, 6)
        with torch.no_grad():
            d0 = float((layer(x) - base(x)).abs().max())
            assert d0 < 1e-6, "B = 0 initialisation must leave the output unchanged"
            for name, p in layer.named_parameters():
                if p.requires_grad:
                    p.copy_(torch.randn_like(p))
            merged = layer.merged_copy()
            d1 = float((layer(x) - merged(x)).abs().max())
        assert tuple(merged.weight.shape) == (6, 12), "merged Conv1D must keep the [in, out] orientation"
        assert d1 < 1e-4, f"merged vs unmerged max |Δ| = {d1:.1e}"
        assert torch.equal(layer.base.weight.detach(), w0), "merging modified the shared base weight"
        assert merged is not layer.base and merged.weight.data_ptr() != layer.base.weight.data_ptr()
        return f"zero-init equivalence; merge exact (|Δ| {d1:.1e}); base untouched"
    run_grading_case(report, "4.3 LoRA orientation, zero init and merge", run)


def check_lorma(m, report):
    def run():
        g = torch.Generator().manual_seed(7)
        d_in, d_out, r = 12, 16, 3
        W = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
        x0 = torch.randn(5, d_in, generator=g, dtype=torch.float64)
        factors = [f.detach().to(torch.float64) for f in m.make_lorma_factors(d_in, d_out, r, init=False, generator=g)]
        outs = []
        for fn in (m.lorma_dense, m.lorma_reordered):
            fs = [f.clone().requires_grad_() for f in factors]
            x = x0.clone().requires_grad_()
            y = fn(x, W, tuple(fs), getattr(m, "LORMA_VARIANT", None))
            y.sin().sum().backward()
            outs.append([y.detach(), x.grad] + [f.grad for f in fs])
        diffs = [float((a - b).abs().max()) for a, b in zip(*outs)]
        assert max(diffs) < 1e-9, f"dense vs reordered differ (max {max(diffs):.1e}) in outputs/gradients"
        init = [f.detach().to(torch.float64) for f in m.make_lorma_factors(d_in, d_out, r, init=True, generator=g)]
        d_init = float((m.lorma_reordered(x0, W, tuple(init)) - x0 @ W.T).abs().max())
        assert d_init < 1e-9, f"base-preserving initialisation violated (|Δ| {d_init:.1e})"
        adapted_T = m.lorma_dense(torch.eye(d_in, dtype=torch.float64), W, tuple(factors))      # = W'.T
        rank = int(torch.linalg.matrix_rank(adapted_T))
        assert rank == min(d_in, d_out), f"adapted transformation rank {rank}: rank inflation missing"
        return f"float64 outputs/grads equal (max {max(diffs):.0e}); identity init; full-rank W' ({rank})"
    run_grading_case(report, "4.4 LoRMA dense == reordered, init, rank inflation", run)


def check_weights_and_metrics(weights, metrics, report):
    def run():
        for method in METHODS:
            entry = weights[method]
            keys = list(entry["state"])
            assert any("classifier" in k for k in keys), f"{method}: head missing from saved state"
            assert not any(k.endswith(("wte.weight", "wpe.weight")) for k in keys), \
                f"{method}: frozen embeddings were saved -- save trainable modules + head only"
            assert sum(v.numel() for v in entry["state"].values()) < 5_000_000, f"{method}: state too large for PEFT"
            assert entry["config"].get("backbone"), f"{method}: config must record the frozen backbone"
        return "trainable modules + head only, with configs"
    run_grading_case(report, "4.5 weights.pt contains only PEFT state + head", run)
    table = metrics.get("methods", {})
    for name in ("linear_probe", "full") + METHODS:
        row = table.get(name, {})
        check_metrics_block(report, f"{name} test", row.get("test"))
        costs = row.get("costs", {})
        report.check(f"{name} cost columns recorded",
                     all(is_finite(costs.get(k), 0) for k in ("trainable_params", "total_params", "train_seconds",
                                                              "latency_s_batch16")), f"got {costs}")


def main(student_dir, skip_reload=False):
    report = GradeReport("Question 4")
    sub = SubmissionDir(student_dir, "q4")
    try:
        mods = sub.load_modules("q2_classifier", "q4_peft")
        q2, m = mods["q2_classifier"], mods["q4_peft"]
        report.check("q4_peft.py (and q2_classifier.py) import", True)
    except Exception as exc:
        report.check("q4_peft.py (and q2_classifier.py) import", False, repr(exc))
        return report.summary()
    src = open(sub.dir / "q4_peft.py").read() if (sub.dir / "q4_peft.py").is_file() else ""
    report.check("q4_peft.py does not use the peft library", "import peft" not in src and "from peft" not in src)
    check_soft_prompt(m, report)
    check_lora(m, report)
    check_lorma(m, report)
    check_mechanisms(q2, m, report)
    try:
        metrics, weights = sub.metrics(), sub.weights()
    except Exception as exc:
        report.check("submission artifacts load", False, repr(exc))
        return report.summary()
    check_weights_and_metrics(weights, metrics, report)
    if skip_reload:
        report.skip("reload checks", "--skip_reload")
    else:
        for method in METHODS:
            check_reload(report, method, lambda mm=method: m.reload_q4(mm, root=sub.root), metrics["reload_check"])
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--skip_reload", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir, args.skip_reload) else 1)
