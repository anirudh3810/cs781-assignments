#!/usr/bin/env python3
"""
Autograder for Question 4: SGD, RMSProp, AdamW, and a from-scratch Muon optimizer.

Usage:
    python autograder_q4.py --student_dir /path/to/students/alice

Expects under --student_dir/submissions/q4: optim_muon.py, attention.py,
model_arch.py, model.pt, config.json, and metrics.json.

Like Question 3's RoPE check, the core Muon check is a deterministic unit test on the
student's own newton_schulz_orthogonalize function -- no training or corpus required.
"""
import argparse
import os
import math
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
    check_metric_curve,
    metric_dict,
    run_grading_case,
    assert_tensors_close,
)


def _ns_reference(gradient, steps=5, eps=1e-7):
    x = gradient / (gradient.norm() + eps)
    transposed = gradient.shape[0] > gradient.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x - 4.7750 * (gram @ x) + 2.0315 * (gram @ gram @ x)
    return x.T if transposed else x


def check_muon(optim_mod, report):
    """Requires b <- momentum*b + g and sqrt(max(1, rows/cols)) scaling."""
    import torch
    generator = torch.Generator().manual_seed(784)
    fixtures = {
        "wide": torch.randn(4, 9, generator=generator),
        "tall": torch.randn(9, 4, generator=generator),
        "square": torch.randn(5, 5, generator=generator),
        "zero": torch.zeros(4, 9),
        "rank_one": torch.arange(1, 5).float()[:, None] * torch.ones(1, 9),
    }
    for label, gradient in fixtures.items():
        for steps in (0, 1, 3, 5):
            def run(gradient=gradient, steps=steps):
                with torch.no_grad():
                    actual = optim_mod.newton_schulz_orthogonalize(gradient.clone(), steps=steps, eps=1e-7)
                    return assert_tensors_close(actual, _ns_reference(gradient, steps))
            run_grading_case(report, f"Q4 NS recurrence: {label}, steps={steps}", run)

    for shape in ((4, 9), (9, 4), (5, 5)):
        for momentum, steps, lr in ((0.0, 1, 0.013), (0.8, 5, 0.031)):
            def run(shape=shape, momentum=momentum, steps=steps, lr=lr):
                rng = torch.Generator().manual_seed(785)
                parameter = torch.nn.Parameter(torch.randn(shape, generator=rng))
                expected = parameter.detach().clone()
                buffer = torch.zeros_like(parameter)
                optimizer = optim_mod.Muon([parameter], lr=lr, momentum=momentum, ns_steps=steps)
                gradients = [torch.randn(shape, generator=rng), torch.randn(shape, generator=rng),
                             None, torch.zeros(shape), torch.randn(shape, generator=rng)]
                scale = math.sqrt(max(1.0, shape[0] / shape[1]))
                for gradient in gradients:
                    parameter.grad = None if gradient is None else gradient.clone()
                    if gradient is not None:
                        buffer = momentum * buffer + gradient
                        expected = expected - lr * scale * _ns_reference(buffer, steps)
                    optimizer.step()
                    assert_tensors_close(parameter.detach(), expected)
                # Fresh zero gradient must not change parameters or create NaNs.
                fresh = torch.nn.Parameter(torch.ones(shape))
                zero_optimizer = optim_mod.Muon([fresh], lr=lr, momentum=momentum, ns_steps=steps)
                fresh.grad = torch.zeros_like(fresh)
                zero_optimizer.step()
                assert_tensors_close(fresh.detach(), torch.ones(shape))
                return "matched 5 steps including missing and zero gradients"
            run_grading_case(report, f"Q4 Muon trajectory: shape={shape}, momentum={momentum}, ns_steps={steps}", run)


def check_q4_experiment_evidence(metrics, report, *, smoke_test=False):
    metrics = metric_dict(report, "Q4 metrics", metrics)

    tables = {
        key: metric_dict(report, key, metrics.get(key))
        for key in ( "per_optimizer_best_ppl", "per_optimizer_loss_curves",
            "per_optimizer_time_curves", "lr_sweep_results")
    }

    tried_lrs = {}

    for name in ("sgd", "rmsprop", "adamw", "muon"):
        best = tables["per_optimizer_best_ppl"].get(name)

        report.check( f"{name}: best PPL", metric_is_finite(best, minimum=1), "expected finite PPL >= 1")

        losses = tables["per_optimizer_loss_curves"].get(name)
        times = tables["per_optimizer_time_curves"].get(name)

        loss_ok = check_metric_curve( report, f"{name}: loss curve", losses)
        time_ok = check_metric_curve( report, f"{name}: time curve", times)

        if loss_ok and time_ok:
            report.check( f"{name}: aligned loss/time curves", len(losses) == len(times), "one cumulative time per recorded loss",)

            report.check( f"{name}: cumulative times never decrease", all(b >= a for a, b in zip(times, times[1:])))

        trials = tables["lr_sweep_results"].get(name)
        trials_ok = isinstance(trials, list) and len(trials) > 0
        report.check(f"{name}: nonempty HPO trials", trials_ok)

        tried_lrs[name] = set()
        successful_ppls = []

        if not trials_ok:
            continue

        for i, trial in enumerate(trials):
            label = f"{name}: trial {i + 1}"

            if not isinstance(trial, dict):
                report.check(label, False, "expected a dictionary")
                continue

            lr = trial.get("lr")
            ppl = trial.get("held_out_ppl")

            lr_ok = metric_is_finite(lr) and lr > 0
            report.check( f"{label}: learning rate", lr_ok, "expected finite lr > 0")

            # Preserve evidence of divergent searches without selecting them.
            # None is still an unfilled value, not a recorded numerical result.
            divergent = type(ppl) is float and (math.isnan(ppl) or ppl == float("inf"))
            ppl_ok = metric_is_finite(ppl, minimum=1)

            report.check( f"{label}: recorded held_out_ppl", ppl_ok or divergent, "expected PPL >= 1, or numeric NaN/+inf for a divergent run")

            if lr_ok and (ppl_ok or divergent):
                tried_lrs[name].add(lr)

            if lr_ok and ppl_ok:
                successful_ppls.append(ppl)
            elif divergent:
                print(f"    ({label}: divergent run; excluded from best-PPL selection)")

        report.check( f"{name}: at least one successful HPO trial", bool(successful_ppls))

        if successful_ppls and metric_is_finite(best, minimum=1):
            report.check(
                f"{name}: best PPL matches recorded trial minimum",
                math.isclose(best, min(successful_ppls), rel_tol=1e-4, abs_tol=1e-3),
            )

    # Q4.5 asks for LR sensitivity for SGD and an adaptive optimizer.
    # Two distinct rates are the minimum needed to compare rates.
    if not smoke_test:
        report.check( "Q4.5: SGD has at least two distinct tested learning rates",
            len(tried_lrs["sgd"]) >= 2)

        report.check( "Q4.5: RMSProp or AdamW has at least two distinct tested learning rates",
            any(len(tried_lrs[name]) >= 2 for name in ("rmsprop", "adamw")))
    else:
        print(" (smoke test: skipping LR-variation checks)")


def main(student_dir: str):
    report = GradeReport("Question 4")
    sub_dir = os.path.join(student_dir, "submissions", "q4")
    optim_py = os.path.join(sub_dir, "optim_muon.py")

    if not os.path.exists(optim_py):
        report.check("optim_muon.py present", False)
        return report.summary()
    report.check("optim_muon.py present", True)
    if not check_forbidden_implementations(report, sub_dir, question=4):
        return report.summary()

    try:
        optim_mod = load_student_module(optim_py, "student_optim_muon_q4")
        report.check("optim_muon.py defines newton_schulz_orthogonalize", hasattr(optim_mod, "newton_schulz_orthogonalize"))
        report.check("optim_muon.py defines a Muon optimizer class", hasattr(optim_mod, "Muon"))
        report.check("Muon subclasses torch.optim.Optimizer",
                     hasattr(optim_mod, "Muon") and issubclass(optim_mod.Muon, torch.optim.Optimizer))
    except Exception as e:
        report.check("optim_muon.py imports cleanly", False, repr(e))
        return report.summary()

    check_muon(optim_mod, report)

    # --- submission artifacts ----------------------------------------------------------
    try:
        config, metrics, model_path = load_submission_dir(sub_dir)
        report.check("submission artifacts load", True)
    except FileNotFoundError as e:
        report.check("submission artifacts load", False, str(e))
        return report.summary()

    check_q4_experiment_evidence( metrics, report,smoke_test=(config.get("smoke_test") is True))
   

    # --- load and forward-pass the saved best model -----------------------------------
    try:
        stack = load_full_student_stack(sub_dir, need=("attention", "model_arch"))
        model_mod = stack["model_arch"]
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
        report.check("saved Q4 model forward pass shape correct",
                     logits.shape == (2, min(16, cfg.max_seq_len), cfg.vocab_size))
    except Exception as e:
        report.check("saved Q4 model loads and runs", False, repr(e))

    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    ok = main(args.student_dir)
    sys.exit(0 if ok else 1)
