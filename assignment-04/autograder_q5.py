#!/usr/bin/env python3
"""
Autograder for Question 5: quantization from scratch and QLoRA instruction tuning.

Usage:
    python autograder_q5.py --student_dir /path/to/student [--skip_tokenizer] [--skip_reload]

Expects submissions/q5/{q5_qlora.py, weights.pt, metrics.json}.
  * Quantizers are tested on random CPU tensors (no download).
  * The instruction format is tested with the Qwen2.5 tokenizer (small download; skip with
    --skip_tokenizer).
  * Reloading the 4-bit model needs a CUDA GPU with bitsandbytes; on CPU it is skipped.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common_grading import (GradeReport, SubmissionDir, check_metrics_block, check_reload, is_finite,  # noqa: E402
                            run_grading_case)


def reference_nf4():
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677083
    pos = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    levels = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return levels / levels.abs().max()


def check_quantizers(m, report):
    g = torch.Generator().manual_seed(11)
    w = torch.randn(64, 96, generator=g) * 0.05

    def absmax():
        q = m.absmax_int8(w)
        scale = float(w.abs().max()) / 127
        assert q.shape == w.shape and len(torch.unique(q)) <= 255
        err = float((q - w).abs().max())
        assert err <= scale / 2 + 1e-7, f"max error {err:.2e} > scale/2 {scale / 2:.2e}"
        return f"≤255 levels, max error {err:.1e} ≤ scale/2"
    run_grading_case(report, "5.1 absmax_int8", absmax)

    def per_channel():
        w2 = w.clone(); w2[:, 3] *= 100           # one output channel ([in, out] layout) is huge
        q = m.per_channel_int8(w2, out_dim=-1)
        col_scale = w2.abs().amax(0) / 127
        err = (q - w2).abs().amax(0)
        assert torch.all(err <= col_scale / 2 + 1e-7), "per-output-channel error bound violated"
        assert float((q - w2)[:, :3].abs().max()) < float((m.absmax_int8(w2) - w2)[:, :3].abs().max()), \
            "per-channel scales should beat per-tensor scales when one channel is an outlier"
        return "one scale per output channel"
    run_grading_case(report, "5.1 per_channel_int8", per_channel)

    def group4():
        q = m.group_int4(w, group_size=64)
        blocks_q, blocks_w = q.reshape(-1, 64), w.reshape(-1, 64)
        for bq, bw in zip(blocks_q, blocks_w):
            assert len(torch.unique(bq)) <= 15, "more than 15 levels in a block"
            assert float((bq - bw).abs().max()) <= float(bw.abs().max()) / 7 / 2 + 1e-7
        return f"{len(blocks_q)} blocks, ≤15 levels each, error ≤ scale/2"
    run_grading_case(report, "5.1 group_int4", group4)

    def nf4():
        levels = torch.as_tensor(m.nf4_levels()).float().flatten()
        ref = reference_nf4()
        assert levels.numel() == 16, f"{levels.numel()} levels"
        assert torch.all(levels[1:] > levels[:-1]), "levels must be sorted"
        assert abs(float(levels.min()) + 1) < 1e-6 and abs(float(levels.max()) - 1) < 1e-6, "levels must span [-1, 1]"
        assert float(levels.abs().min()) < 1e-7, "NF4 must represent zero exactly"
        assert torch.allclose(levels, ref, atol=2e-3), f"levels differ from QLoRA NF4: {levels.tolist()}"
        q = m.nf4_quantize(w, block_size=64)
        for bq, bw in zip(q.reshape(-1, 64), w.reshape(-1, 64)):
            amax = float(bw.abs().max())
            normed = bq / amax
            dist = (normed.unsqueeze(-1) - ref).abs().min(-1).values
            assert float(dist.max()) < 1e-4, "dequantised values must be NF4 levels x block absmax"
            nearest = ref[((bw / amax).unsqueeze(-1) - ref).abs().argmin(-1)] * amax
            assert torch.allclose(bq, nearest, atol=1e-5), "values must snap to the NEAREST level"
        return "16 sorted levels incl. 0 and ±1, nearest-level block quantisation"
    run_grading_case(report, "5.1 NF4 levels and quantisation", nf4)

    def bits():
        b = m.storage_bits_per_param("nf4", 10 ** 6)
        bdq = m.storage_bits_per_param("nf4", 10 ** 6, double_quant=True)
        assert abs(b - 4.25) < 0.01, f"NF4 with FP16 block scales: {b}"
        assert 4.12 < bdq < 4.135, f"NF4 + double quantization: {bdq}"
        return f"nf4 {b:.3f} bits, +DQ {bdq:.3f} bits"
    run_grading_case(report, "5.1 storage accounting", bits)


def check_instruction_format(m, report, model_id):
    def run():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        codes = m.verify_answer_codes(tok)
        code_ids = [codes[c] for c in m.CODES]
        assert len(set(code_ids)) == 2 and "end_id" in codes
        long_review = "The film starts slowly but the final act is wonderful. " * 80
        for cls in (0, 1):
            ex = m.make_instruction_example(tok, long_review, cls, code_ids, max_length=256)
            ids, labels = list(ex["input_ids"]), list(ex["labels"])
            assert len(ids) == len(labels) == len(ex["attention_mask"]) and len(ids) <= 256, "length/truncation"
            sup = [i for i, l in enumerate(labels) if l != -100]
            assert len(sup) == 2 and sup == [len(ids) - 2, len(ids) - 1], f"supervised positions {sup}"
            assert labels[-2] == code_ids[cls] and labels[-1] == codes["end_id"], "targets must be code + end-of-turn"
            text = tok.decode(ids[:-2])
            assert "final act" in text or "starts slowly" in text, "review text missing from the prompt"
        return "single-token codes, 2 supervised positions, ≤256 tokens, review truncated not instruction"
    run_grading_case(report, "5.2 instruction format and response-only labels", run)


def check_weights(weights, report):
    def run():
        for run_name, precision in (("fp16_pilot", "fp16"), ("nf4_pilot", "nf4"), ("nf4_final", "nf4")):
            entry = weights[run_name]
            assert entry["config"].get("precision") == precision, f"{run_name}: precision {entry['config'].get('precision')}"
            keys = list(entry["state"])
            for target in ("q_proj", "v_proj"):
                a = [k for k in keys if target in k and "lora_A" in k]
                b = [k for k in keys if target in k and "lora_B" in k]
                assert a and len(a) == len(b), f"{run_name}: LoRA A/B for {target} missing"
                assert all(entry["state"][k].shape[0] == 8 for k in a), f"{run_name}: rank must be 8"
            assert not any("base_layer" in k for k in keys), f"{run_name}: frozen base weights were saved"
        b_final = [v for k, v in weights["nf4_final"]["state"].items() if "lora_B" in k]
        assert any(float(v.abs().sum()) > 0 for v in b_final), "nf4_final adapters are untrained (all B == 0)"
        return "three adapter states, q/v targets, rank 8, trained final adapter"
    run_grading_case(report, "5.3 weights.pt adapters", run)


def check_metrics(metrics, report):
    q = metrics.get("quantization", {})
    ppl = q.get("perplexity", {})
    report.check("5.1 perplexity for fp32 and all four schemes",
                 all(is_finite(ppl.get(k), 1) for k in ("fp32", "absmax_int8", "per_channel_int8", "group_int4", "nf4")),
                 f"got {ppl}")
    pilot = metrics.get("pilot", {})
    for arm in ("fp16", "nf4"):
        p = pilot.get(arm, {})
        report.check(f"5.3 {arm} pilot: updates, validation F1, memory, time",
                     p.get("updates") == metrics.get("q5_pilot") and is_finite(p.get("val_macro_f1"), 0, 1)
                     and is_finite(p.get("peak_MB"), 0) and is_finite(p.get("train_s"), 0), f"got {p}")
    for name in ("fp16_untuned", "nf4_untuned", "fp16_pilot", "nf4_pilot", "nf4_final"):
        check_metrics_block(report, f"5.3 test: {name}", metrics.get("test", {}).get(name),
                            keys=("accuracy", "macro_f1", "code_mass"))
    fg = metrics.get("free_generation", {})
    report.check("5.3 free-generation compliance recorded", is_finite(fg.get("compliance"), 0, 1), f"got {fg}")
    wd = metrics.get("wording", {})
    report.check("5.4 wording robustness recorded", is_finite(wd.get("agreement"), 0, 1), f"got {wd}")


def main(student_dir, skip_tokenizer=False, skip_reload=False, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    report = GradeReport("Question 5")
    sub = SubmissionDir(student_dir, "q5")
    try:
        m = sub.load_modules("q5_qlora")["q5_qlora"]
        report.check("q5_qlora.py imports", True)
    except Exception as exc:
        report.check("q5_qlora.py imports", False, repr(exc))
        return report.summary()
    check_quantizers(m, report)
    if skip_tokenizer:
        report.skip("5.2 instruction format", "--skip_tokenizer")
    else:
        check_instruction_format(m, report, model_id)
    try:
        metrics, weights = sub.metrics(), sub.weights()
    except Exception as exc:
        report.check("submission artifacts load", False, repr(exc))
        return report.summary()
    check_weights(weights, report)
    check_metrics(metrics, report)
    if skip_reload:
        report.skip("reload check", "--skip_reload")
    elif not torch.cuda.is_available():
        report.skip("reload check", "needs a CUDA GPU for the 4-bit base model")
    else:
        common.configure(profile=metrics.get("profile", "core"))
        check_reload(report, "nf4_final", lambda: m.reload_q5("nf4_final", root=sub.root), metrics["reload_check"],
                     atol=5e-3)
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--skip_tokenizer", action="store_true")
    parser.add_argument("--skip_reload", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir, args.skip_tokenizer, args.skip_reload) else 1)
