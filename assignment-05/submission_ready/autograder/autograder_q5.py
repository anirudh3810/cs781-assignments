"""autograder_q5.py — Attention-head analysis of your grokked HP model (CPU only).

    python autograder/autograder_q5.py --student_dir .

Score functions are checked on synthetic attention patterns with closed-form answers; ablation
is checked against an independent temporary-hook route. Model-dependent scores and interventions
are reported without requiring a particular head, ranking, or circuit strength.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch, load_decoder,
                            load_split, validate_q1_handoff, require, defines_function, is_implemented, synthetic_cache,
                            zero_head_copy, repeat_acc)

FUNCS = ["previous_token_scores", "induction_scores", "ablate_head", "head_ablation_effect", "shift_test"]


def _original_main(sd):
    rep = GradeReport("Q5 — Attention heads")
    ipy = os.path.join(sd, "interp.py"); q1 = os.path.join(sd, "submissions", "q4_q6_model")
    if not require(rep, [ipy, os.path.join(q1, "model.pt"), os.path.join(q1, "memorised.pt"),
                         os.path.join(q1, "split.pt"), os.path.join(q1, "tokenizer.json"),
                         os.path.join(q1, "metrics.json"), os.path.join(q1, "model_arch.py")]):
        return rep.summary()
    for f in FUNCS:
        rep.check(f"implements {f}", defines_function(ipy, f) and is_implemented(ipy, f))
    try:
        MA = load_model_arch(sd)
    except Exception as e:
        rep.check("loads the exact Q1 architecture snapshot", False, f"{type(e).__name__}: {e}")
        return rep.summary()
    IT = load_student_module(ipy, "student_interp")
    split = load_split(q1); P = int(split["P"]); X = split["eval"][:256]
    lm, _ = load_decoder(os.path.join(q1, "model.pt"), MA)
    memo, _ = load_decoder(os.path.join(q1, "memorised.pt"), MA)
    rep.guard("Q1 tokenizer, split, metrics, and model vocabulary are consistent",
              lambda: (True, validate_q1_handoff(q1, lm, split)))
    try:
        hm, hm_mem = IT.to_hooked(lm), IT.to_hooked(memo)
    except Exception as e:  # noqa: BLE001
        rep.check("to_hooked works (needed for Q5)", False, str(e))
        return rep.summary()
    from transformer_lens import HookedTransformer
    if not isinstance(hm, HookedTransformer) or not isinstance(hm_mem, HookedTransformer):
        rep.check("Q5 requires real TransformerLens ports", False,
                  "to_hooked must return HookedTransformer for both Q1 checkpoints")
        return rep.summary()
    parity_g = (hm(X[:, :-1]) - lm(X[:, :-1])).abs().max().item()
    parity_m = (hm_mem(X[:, :-1]) - memo(X[:, :-1])).abs().max().item()
    if max(parity_g, parity_m) >= 1e-3:
        rep.check("Q4 port parity prerequisite", False,
                  f"grokked max error={parity_g:.3e}; memorised max error={parity_m:.3e}; require <1e-3")
        return rep.summary()
    rep.check("Q4 port parity prerequisite", True,
              f"grokked={parity_g:.2e}, memorised={parity_m:.2e}")
    L, H = hm.cfg.n_layers, hm.cfg.n_heads

    def t_port_contract():
        from port_contract import validate_transformerlens_port
        errors = validate_transformerlens_port(hm, X[:2, :min(8, X.shape[1]-1)])
        return not errors, "; ".join(errors[:4]) if errors else "semantic cache and intervention interface valid"
    rep.guard("student port satisfies the Q4-Q6 cache/hook/readout contract", t_port_contract)

    cache_syn, exp = synthetic_cache(hm, P)
    rep.guard("previous_token_scores matches closed form on synthetic patterns",
              lambda: (torch.allclose(IT.previous_token_scores(cache_syn, L).float(), exp["prev"], atol=1e-4), ""))
    rep.guard("induction_scores matches closed form on synthetic patterns",
              lambda: (torch.allclose(IT.induction_scores(cache_syn, L, P).float(), exp["induction"], atol=1e-4), ""))

    def t_ablate():
        worst = 0.0
        for (l, h) in [(0, 0), (L - 1, H - 1)]:
            hook_name = f"blocks.{l}.attn.hook_z"
            def zero_selected(z, hook, h=h):
                z = z.clone(); z[:, :, h, :] = 0
                return z
            ref = hm.run_with_hooks(X[:64, :-1], fwd_hooks=[(hook_name, zero_selected)])
            got = IT.ablate_head(hm, X[:64], l, h)
            worst = max(worst, (ref - got).abs().max().item())
        intact = (hm(X[:64, :-1]) - lm(X[:64, :-1])).abs().max().item()
        return worst < 1e-3 and intact < 1e-3, f"max diff vs independent z-hook route {worst:.2e}; hooks removed: {intact < 1e-3}"
    rep.guard("ablate_head matches an independent temporary z-hook intervention", t_ablate)

    rows = []
    def t_effect():
        nonlocal rows
        rows = IT.head_ablation_effect(hm, X, P)
        base = repeat_acc(hm(X[:, :-1]), X, P)
        ok = len(rows) == L * H
        for r in rows[:: max(1, len(rows) // 3)]:
            acc = repeat_acc(IT.ablate_head(hm, X, r["layer"], r["head"]), X, P)
            ok &= abs(acc - r["acc_after"]) < 1e-3 and abs((base - acc) - r["acc_drop"]) < 1e-3
        return ok, f"{len(rows)} rows"
    rep.guard("head_ablation_effect: one row per head, consistent with independent ablation", t_effect)

    # behaviour on the student's grokked model
    _, cache = hm.run_with_cache(X[:, :-1]); _, cache_m = hm_mem.run_with_cache(X[:, :-1])
    try:
        ind = IT.induction_scores(cache, L, P); ind_m = IT.induction_scores(cache_m, L, P)
        prev = IT.previous_token_scores(cache, L)
    except Exception as e:  # noqa: BLE001
        rep.check("score functions run on the real cache", False, str(e)); return rep.summary()
    l_star, h_star = divmod(int(ind.argmax()), H)
    rep.info("largest grokked induction score", f"L{l_star}H{h_star} = {ind.max().item():.3f}")
    base = repeat_acc(hm(X[:, :-1]), X, P)
    drop = base - repeat_acc(IT.ablate_head(hm, X, l_star, h_star), X, P)
    rep.info("repeat-accuracy change after ablating the largest-induction head",
             f"L{l_star}H{h_star}: {drop:.3f}")

    # shift_test: layout and consistency
    Xs = X[:128]
    sep = int(Xs[0, P])
    filler = torch.randint(1, int(Xs.max()) + 1, (len(Xs), 3), generator=torch.Generator().manual_seed(0))
    filler[filler == sep] = 1
    def t_shift0():
        r = IT.shift_test(hm, Xs, P, 0, l_star, h_star, filler[:, :0])
        ind_lh = IT.induction_scores(hm.run_with_cache(Xs[:, :-1])[1], L, P)[l_star, h_star].item()
        ok = (torch.equal(r["Xk"], Xs) and abs(r["attn_content"] - r["attn_offset"]) < 1e-6
              and abs(r["attn_content"] - ind_lh) < 1e-4 and abs(r["copy_acc"] - repeat_acc(hm(Xs[:, :-1]), Xs, P)) < 1e-4)
        return ok, f"k=0: copy acc {r['copy_acc']:.3f}, attn {r['attn_content']:.3f}"
    rep.guard("shift_test with k=0 reproduces the unshifted copy accuracy and induction score", t_shift0)
    res3 = {}
    def t_shift3():
        nonlocal res3
        k = 3
        res3 = IT.shift_test(hm, Xs, P, k, l_star, h_star, filler)
        Xk = res3["Xk"]
        layout = (tuple(Xk.shape) == tuple(Xs.shape) and torch.equal(Xk[:, :P + 1], Xs[:, :P + 1])
                  and torch.equal(Xk[:, P + 1:P + 1 + k], filler) and torch.equal(Xk[:, P + 1 + k:], Xs[:, P + 1:2 * P + 1 - k]))
        logits, cache = hm.run_with_cache(Xk[:, :-1])
        q = torch.arange(P + 1 + k, 2 * P)
        acc = (logits[:, q].argmax(-1) == Xk[:, 1:][:, q]).float().mean().item()
        pat = cache[f"blocks.{l_star}.attn.hook_pattern"][:, h_star]
        ac, ao = pat[:, q, q - P - k].mean().item(), pat[:, q, q - P].mean().item()
        ok = layout and abs(acc - res3["copy_acc"]) < 1e-4 and abs(ac - res3["attn_content"]) < 1e-4 \
            and abs(ao - res3["attn_offset"]) < 1e-4
        return ok, f"layout ok: {layout}"
    rep.guard("shift_test with k=3: correct layout, accuracy and attention targets", t_shift3)
    if res3:
        verdict = "content (induction)" if res3["attn_content"] > res3["attn_offset"] else "position (fixed offset)"
        rep.info("shift test, k=3", f"copy acc {res3['copy_acc']:.3f} | attn to content target {res3['attn_content']:.2f} "
                 f"vs offset target {res3['attn_offset']:.2f} -> copies by {verdict}")
    rep.info("max induction score: memorised vs grokked", f"{ind_m.max().item():.2f} vs {ind.max().item():.2f}")
    rep.info("previous-token scores (layer 0)", str([round(v, 2) for v in prev[0].tolist()]))
    if rows:
        top = sorted(rows, key=lambda r: -r["acc_drop"])[:3]
        rep.info("largest ablation drops", ", ".join(f"L{r['layer']}H{r['head']}:{r['acc_drop']:.2f}" for r in top))
    return rep.summary()

from expanded_grader_common import check, load_student, main_for, make_model, torch, tl_utils

def _expanded_checks(student):
    model = make_model(seed=29)
    clean = torch.randint(0, model.cfg.d_vocab, (3, 9))
    corrupt = clean.roll(1, dims=1)
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt)

    def own_corrupt_patch():
        z_full = corrupt_cache["blocks.0.attn.hook_z"][:, :, 1, :]
        got_full = student.patch_head_z(model, corrupt, 0, 1, z_full)
        _, prefix_cache = model.run_with_cache(corrupt[:, :-1])
        z_prefix = prefix_cache["blocks.0.attn.hook_z"][:, :, 1, :]
        prefix_logits = model(corrupt[:, :-1])
        got_prefix = student.patch_head_z(model, corrupt, 0, 1, z_prefix)
        ok = (torch.allclose(got_full, corrupt_logits, atol=1e-6, rtol=1e-6)
              and torch.allclose(got_prefix, prefix_logits, atol=1e-6, rtol=1e-6))
        err = max((got_full-corrupt_logits).abs().max().item(),
                  (got_prefix-prefix_logits).abs().max().item())
        return ok, f"max diff across full/prefix activation inputs={err:.2e}"

    def mean_replace():
        layer, head = 1, 2
        z = clean_cache[tl_utils.get_act_name("z", layer)][:, :, head, :]
        mean_z = z.mean(dim=(0, 1))
        got = student.mean_ablate_head(model, clean, layer, head, mean_z)
        def independent(z_all, hook):
            changed = z_all.clone()
            changed[:, :, head, :] = mean_z
            return changed
        expected = model.run_with_hooks(clean[:, :-1], fwd_hooks=[(tl_utils.get_act_name("z", layer), independent)])
        return torch.allclose(got, expected, atol=1e-6, rtol=1e-6), f"max diff={(got-expected).abs().max().item():.2e}"

    def full_component_restore():
        hooks = [("hook_embed", lambda x, hook: clean_cache["hook_embed"])]
        for layer in range(model.cfg.n_layers):
            hooks.append((tl_utils.get_act_name("z", layer),
                          lambda x, hook, layer=layer: clean_cache[tl_utils.get_act_name("z", layer)]))
            hooks.append((tl_utils.get_act_name("mlp_out", layer),
                          lambda x, hook, layer=layer: clean_cache[tl_utils.get_act_name("mlp_out", layer)]))
        got = model.run_with_hooks(corrupt, fwd_hooks=hooks)
        return torch.allclose(got, clean_logits, atol=1e-6, rtol=1e-6), f"max diff={(got-clean_logits).abs().max().item():.2e}"

    def resid_pre_restore():
        name = "blocks.0.hook_resid_pre"
        ref = clean_cache[name]
        got = model.run_with_hooks(corrupt, fwd_hooks=[(name, lambda x, hook: ref)])
        return torch.allclose(got, clean_logits, atol=1e-6, rtol=1e-6), f"max diff={(got-clean_logits).abs().max().item():.2e}"

    def undefined_normalization():
        result = student.head_patching_effect(model, clean, clean, P=3)
        ok = result.shape == (model.cfg.n_layers, model.cfg.n_heads) and torch.isnan(result).all()
        return ok, f"shape={tuple(result.shape)}"

    def gqa_hooks():
        gqa = make_model(seed=37, gqa=True)
        x = torch.randint(0, gqa.cfg.d_vocab, (2, 9))
        _, cache = gqa.run_with_cache(x)
        z = cache["blocks.0.attn.hook_z"][:, :, 3, :]
        base = gqa(x)
        got = student.patch_head_z(gqa, x, 0, 3, z)
        _, prefix_cache = gqa.run_with_cache(x[:, :-1])
        z_prefix = prefix_cache["blocks.0.attn.hook_z"][:, :, 3, :]
        base_prefix = gqa(x[:, :-1])
        got_prefix = student.patch_head_z(gqa, x, 0, 3, z_prefix)
        ok = (torch.allclose(got, base, atol=1e-6, rtol=1e-6)
              and torch.allclose(got_prefix, base_prefix, atol=1e-6, rtol=1e-6))
        return ok, f"z full/prefix shapes={tuple(z.shape)}/{tuple(z_prefix.shape)}"

    tests = [
        check("patching a head with its corrupted activation changes nothing", own_corrupt_patch),
        check("mean_ablate_head replaces the selected head with a broadcast mean", mean_replace),
        check("patching embedding, all z, and MLP outputs restores clean logits", full_component_restore),
        check("patching layer-0 resid_pre restores clean logits", resid_pre_restore),
        check("degenerate clean/corrupt margin is reported as NaN", undefined_normalization),
        check("hook_z patching supports grouped-query attention", gqa_hooks),
    ]
    return tests


def main(sd):
    print("\n=== Additional assignment checks ===")
    original = _original_main(sd)
    try:
        student = load_student(sd)
        expanded_results = _expanded_checks(student)
    except Exception as exc:
        print(f"FAIL additional-check setup: {type(exc).__name__}: {exc}")
        expanded_results = [False]
    original_passed = original.get("passed", 0) if isinstance(original, dict) else 0
    original_total = original.get("total", 0) if isinstance(original, dict) else 0
    expanded_passed = sum(bool(result) for result in expanded_results)
    expanded_total = len(expanded_results)
    print(f"\nCombined Q5: {original_passed + expanded_passed}/{original_total + expanded_total} checks passed")
    return 0 if original_passed == original_total and all(expanded_results) else 1


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
