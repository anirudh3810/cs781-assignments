#!/usr/bin/env python3
"""
Autograder for Question 1: decoding strategies, KV cache and speculative sampling.

Usage:
    python autograder_q1.py --student_dir /path/to/student

Expects submissions/q1/{q1_decoding.py, metrics.json}. All numerical checks run on tiny
synthetic models (a bigram toy LM and a random A-3-layout Transformer), so the student's
A-3 checkpoint is not needed and the grader finishes in well under a minute on a CPU.
"""
import argparse
import itertools
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, ReferenceBuiltinLM, RefLMConfig, SubmissionDir, ToyBigramLM,  # noqa: E402
                            ToyDraft, is_finite, is_fraction, run_grading_case, tv_distance)

TABLE = torch.tensor([[0.5, 2.0, 1.0, -1.0],      # greedy from 0: 1, 2, 2, 2
                      [0.1, 1.4, 1.5, 0.0],
                      [1.0, 0.0, 2.5, 0.3],
                      [0.0, 0.0, 0.0, 3.0]])


def check_filtering(m, report):
    def normalisation():
        z = torch.tensor([2.0, 1.0, 0.5, -1.0, -3.0])
        for kw in ({}, {"temperature": 0.5}, {"top_k": 2}, {"top_p": 0.6}, {"temperature": 1.7, "top_k": 3, "top_p": 0.9}):
            p = torch.as_tensor(m.filtered_probs(z, **kw)).float()
            assert p.shape == z.shape and torch.all(p >= 0) and abs(float(p.sum()) - 1) < 1e-5, kw
        ref = torch.softmax(z, -1)
        p = torch.as_tensor(m.filtered_probs(z)).float()
        assert torch.allclose(p, ref, atol=1e-6), "top_k=None, top_p=1 must equal plain softmax"
        return "probabilities normalised; unrestricted == softmax"
    run_grading_case(report, "1.2 filtered_probs normalisation", normalisation)

    def support():
        logp = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        p = torch.as_tensor(m.filtered_probs(logp, top_k=2)).float()
        assert (p > 0).sum() == 2 and p[0] > 0 and p[1] > 0, f"top_k=2 support {p}"
        p = torch.as_tensor(m.filtered_probs(logp, top_p=0.85)).float()   # 0.5+0.3 < 0.85 <= 0.95
        assert (p > 0).sum() == 3 and p[2] > 0, f"top_p must include the boundary-crossing token: {p}"
        p = torch.as_tensor(m.filtered_probs(logp, top_p=0.4)).float()
        assert (p > 0).sum() == 1 and abs(float(p[0]) - 1) < 1e-6, "top_p below the top probability keeps 1 token"
        assert torch.allclose(torch.as_tensor(m.filtered_probs(logp, top_p=0.85)).float()[:3],
                              torch.tensor([0.5, 0.3, 0.15]) / 0.95, atol=1e-5), "renormalisation"
        cold = torch.as_tensor(m.filtered_probs(logp, temperature=1e-3)).float()
        assert float(cold[0]) > 0.999, "T -> 0 should approach one-hot"
        hot = torch.as_tensor(m.filtered_probs(logp, temperature=1e3)).float()
        assert float(hot.max() - hot.min()) < 0.01, "large T should approach uniform"
        return "top-k / top-p boundary / temperature limits"
    run_grading_case(report, "1.2 filtered_probs support and boundary token", support)

    def invalid():
        bad = [{"temperature": 0.0}, {"temperature": -1.0}, {"top_k": 0}, {"top_p": 0.0}, {"top_p": 1.5}]
        for kw in bad:
            try:
                m.filtered_probs(torch.zeros(4), **kw)
            except Exception:
                continue
            raise AssertionError(f"accepted invalid parameters {kw}")
        return "invalid T / k / p rejected"
    run_grading_case(report, "1.2 invalid sampling parameters rejected", invalid)


def check_generate(m, report):
    model = ToyBigramLM(TABLE)

    def greedy_and_eos():
        ids, _ = m.generate(model, [0], 4, strategy="greedy")
        expected, cur = [], 0
        for _ in range(4):
            cur = int(TABLE[cur].argmax()); expected.append(cur)
        assert list(ids) == expected, f"greedy {list(ids)} != {expected}"
        ids, trace = m.generate(model, [3], 6, strategy="greedy", eos_id=3)
        assert list(ids) == [3] and trace.get("stop_reason") == "eos", f"EOS stop: {ids}, {trace}"
        ids, trace = m.generate(model, [1], 5, strategy="greedy", eos_id=3)
        assert len(ids) == 5 and trace.get("stop_reason") == "max_new_tokens", f"length cap: {ids}, {trace}"
        return "greedy chain, EOS stop, length cap"
    run_grading_case(report, "1.2 greedy decoding and stopping", greedy_and_eos)

    def sampling_distribution():
        g = torch.Generator().manual_seed(781)
        counts = Counter()
        n = 4000
        for _ in range(n):
            ids, _ = m.generate(model, [0], 1, strategy="sample", temperature=0.8, top_k=3, generator=g)
            counts[int(ids[0])] += 1
        expected = torch.as_tensor(m.filtered_probs(TABLE[0], temperature=0.8, top_k=3)).float()
        ref = torch.softmax(TABLE[0] / 0.8, -1)
        ref[ref.argsort()[:1]] = 0; ref = ref / ref.sum()
        assert torch.allclose(expected, ref, atol=1e-5), "filtered_probs mismatch for T=0.8,k=3"
        tv = tv_distance({k: v / n for k, v in counts.items()}, {i: float(ref[i]) for i in range(4)})
        assert tv < 0.04, f"TV distance {tv:.3f} between empirical and expected one-step distribution"
        return f"TV = {tv:.3f} over {n} draws"
    run_grading_case(report, "1.2 sampling follows filtered distribution", sampling_distribution)


def gnmt(logp, L, alpha):
    return logp / (((5 + L) / 6) ** alpha)


def check_beam(m, report):
    model = ToyBigramLM(TABLE)

    def width_one():
        best, _ = m.beam_search(model, [0], 4, beam_width=1, length_penalty=0.0)
        greedy, _ = m.generate(model, [0], 4, strategy="greedy")
        assert list(best) == list(greedy), f"width-1 beam {list(best)} != greedy {list(greedy)}"
        return "width 1, alpha 0 == greedy"
    run_grading_case(report, "1.3 beam width 1 equals greedy", width_one)

    def exhaustive():
        logp = torch.log_softmax(TABLE, -1)
        for alpha in (0.0, 0.6):
            best_score, best_path = -1e9, None
            for path in itertools.product(range(4), repeat=3):
                s, cur = 0.0, 0
                for t in path:
                    s += float(logp[cur, t]); cur = t
                sc = gnmt(s, 3, alpha)
                if sc > best_score:
                    best_score, best_path = sc, list(path)
            got, _ = m.beam_search(model, [0], 3, beam_width=64, length_penalty=alpha)
            assert list(got) == best_path, f"alpha={alpha}: beam {list(got)} != exhaustive {best_path}"
        return "wide beam == exhaustive optimum (alpha 0 and 0.6)"
    run_grading_case(report, "1.3 wide beam matches exhaustive search", exhaustive)

    def eos_and_penalty():
        # Token 3 is EOS; from 0 an early EOS has fewer factors. alpha changes the ranking scale.
        _, cands = m.beam_search(model, [0], 3, beam_width=4, length_penalty=0.6, eos_id=3)
        assert isinstance(cands, list) and cands, "candidates list required"
        for c in cands:
            for key in ("ids", "raw_logprob", "score", "finished"):
                assert key in c, f"candidate missing {key}"
            L = len(c["ids"])
            assert abs(c["score"] - gnmt(c["raw_logprob"], L, 0.6)) < 1e-4, "score must be the GNMT-normalised raw log-prob"
            if c["finished"]:
                assert list(c["ids"])[-1] == 3, "finished candidates end with EOS"
        return f"{len(cands)} candidates with consistent scores"
    run_grading_case(report, "1.3 candidate bookkeeping (EOS, normalised scores)", eos_and_penalty)


def check_speculative(m, report):
    target_table = torch.tensor([[1.0, 0.2, -0.5], [0.0, 1.2, 0.3], [-1.0, 0.5, 1.5]])
    draft_table = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])   # deliberately bad draft
    target = ToyBigramLM(target_table)
    draft = ToyDraft(draft_table)

    def exact_distribution():
        logp = torch.log_softmax(target_table, -1)
        exact = {(a, b): math.exp(float(logp[0, a] + logp[a, b])) for a in range(3) for b in range(3)}
        g = torch.Generator().manual_seed(782)
        n, counts, stats_seen = 6000, Counter(), None
        for i in range(n):
            ids, stats = m.speculative_sample(target, draft, [0], 2, gamma=2 if i % 2 else 3, generator=g)
            ids = list(ids)
            assert len(ids) == 2, f"must return exactly max_new_tokens without EOS, got {ids}"
            counts[tuple(ids)] += 1
            stats_seen = stats
        tv = tv_distance({k: v / n for k, v in counts.items()}, exact)
        assert tv < 0.05, f"TV distance {tv:.3f} to the exact target distribution"
        for key in ("proposed", "accepted", "target_calls", "blocks"):
            assert key in stats_seen, f"stats missing {key}"
        return f"TV = {tv:.3f} over {n} length-2 samples (rejection-heavy draft)"
    run_grading_case(report, "1.6 speculative sampling is exact", exact_distribution)

    def identical_draft():
        same = ToyDraft(target_table)
        g = torch.Generator().manual_seed(783)
        acc = prop = 0
        for _ in range(200):
            _, s = m.speculative_sample(target, same, [0], 8, gamma=4, generator=g)
            acc += s["accepted"]; prop += s["proposed"]
        assert prop > 0 and acc / prop > 0.99, f"identical draft acceptance {acc}/{prop}"
        return f"acceptance {acc}/{prop} with draft == target"
    run_grading_case(report, "1.6 identical draft is always accepted", identical_draft)

    def one_call_per_block():
        same = ToyDraft(target_table)
        target.calls = 0
        _, s = m.speculative_sample(target, same, [0], 8, gamma=4, generator=torch.Generator().manual_seed(1))
        assert s["target_calls"] <= s["blocks"] + 1 and target.calls <= s["blocks"] + 2, \
            f"target_calls={s['target_calls']}, blocks={s['blocks']}, actual forward calls={target.calls}"
        return f"{target.calls} target forward calls for {s['blocks']} blocks"
    run_grading_case(report, "1.6 one target forward pass per proposed block", one_call_per_block)


def check_bigram_draft(m, report):
    def run():
        d = m.BigramDraft(5)
        d.fit([0, 1, 0, 1, 0, 2])
        p = torch.as_tensor(d.next_probs([3, 0])).float()
        expected = torch.tensor([1, 3, 2, 1, 1], dtype=torch.float) / 8      # counts after 0: {1:2, 2:1} + add-one
        assert p.shape == (5,) and torch.allclose(p, expected, atol=1e-6), f"{p} != {expected}"
        p = torch.as_tensor(d.next_probs([4])).float()
        assert torch.allclose(p, torch.full((5,), 0.2)), "unseen context -> uniform"
        return "add-one smoothing over the full vocabulary"
    run_grading_case(report, "1.6 BigramDraft add-one smoothing", run)


def check_cache(m, report):
    for norm_first in (False, True):
        def run(norm_first=norm_first):
            torch.manual_seed(5 + int(norm_first))
            ref = ReferenceBuiltinLM(RefLMConfig(norm_first=norm_first)).eval()
            cached = m.CachedA3LM(ref).eval()
            ids = torch.randint(0, ref.cfg.vocab_size, (2, 14))
            with torch.no_grad():
                full = ref(ids)
                logits, past = cached(ids[:, :6])
                outs = [logits]
                for t in range(6, 14):
                    step, past = cached(ids[:, t:t + 1], past_key_values=past)
                    outs.append(step)
                inc = torch.cat(outs, 1)
            diff = float((inc - full).abs().max())
            assert inc.shape == full.shape and diff < 1e-4, f"max |Δlogit| = {diff:.2e}"
            k0 = past[0][0]
            assert tuple(k0.shape) == (2, ref.cfg.n_heads, 14, ref.cfg.d_model // ref.cfg.n_heads), \
                f"K cache shape {tuple(k0.shape)}"
            try:
                with torch.no_grad():
                    cached(torch.zeros(2, ref.cfg.max_seq_len - 13, dtype=torch.long), past_key_values=past)
            except Exception:
                pass
            else:
                raise AssertionError("context overflow was not rejected")
            return f"prefill + 8 incremental steps match full recompute (max |Δ| = {diff:.1e})"
        run_grading_case(report, f"1.5 KV cache equivalence (norm_first={norm_first})", run)


def check_metrics(metrics, report):
    def run():
        grid = metrics["decoding_grid"]
        assert isinstance(grid, list) and len(grid) == 12, "decoding_grid must have the 12 prescribed settings"
        for row in grid:
            assert isinstance(row.get("setting"), str)
            assert is_fraction(row.get("token_f1_mean")) and is_fraction(row.get("distinct2_mean")), row
        assert is_fraction(metrics["forced_choice_accuracy"])
        assert is_finite(metrics["a3_held_out_perplexity"], 1.0)
        ce = metrics["cache_equivalence"]
        assert is_finite(ce["max_abs_diff"], 0.0) and ce["max_abs_diff"] < 1e-3 and ce["greedy_agreement"] is True
        bench = metrics["cache_benchmark"]
        assert len(bench) >= 3 and all(is_finite(b["cached_ms_per_token"], 0) and is_finite(b["uncached_ms_per_token"], 0)
                                       for b in bench)
        spec = {r["gamma"]: r for r in metrics["speculative"]}
        assert {2, 4} <= set(spec), "speculative rows for gamma 2 and 4"
        assert all(is_fraction(r["acceptance_rate"]) and is_finite(r["target_calls_per_token"], 0) for r in spec.values())
        assert is_finite(metrics["speculative_tv_distance"], 0.0) and metrics["speculative_tv_distance"] < 0.1
        return "decoding grid, forced choice, cache and speculative results recorded"
    run_grading_case(report, "Q1 metrics.json complete and valid", run)


def main(student_dir):
    report = GradeReport("Question 1")
    sub = SubmissionDir(student_dir, "q1")
    try:
        m = sub.load_modules("q1_decoding")["q1_decoding"]
        report.check("q1_decoding.py imports", True)
    except Exception as exc:
        report.check("q1_decoding.py imports", False, repr(exc))
        return report.summary()
    check_filtering(m, report)
    check_generate(m, report)
    check_beam(m, report)
    check_bigram_draft(m, report)
    check_speculative(m, report)
    check_cache(m, report)
    try:
        check_metrics(sub.metrics(), report)
    except FileNotFoundError as exc:
        report.check("submissions/q1/metrics.json present", False, str(exc))
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir) else 1)
