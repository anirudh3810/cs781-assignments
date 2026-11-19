"""autograder_q6.py — Residual stream of your grokked HP model (CPU only).

    python autograder/autograder_q6.py --student_dir .

Checks the logit lens end points and that your direct-logit-attribution components sum EXACTLY
to the model's logit for the true next token (and that a head with W_O = 0 contributes 0).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch, load_decoder,
                            load_split, validate_q1_handoff, require, defines_function, is_implemented, repeat_acc)

FUNCS = ["logit_lens", "decompose_correct_logit"]


def _original_main(sd):
    rep = GradeReport("Q6 — Residual stream")
    ipy = os.path.join(sd, "interp.py"); q1 = os.path.join(sd, "submissions", "q4_q6_model")
    if not require(rep, [ipy, os.path.join(q1, "model.pt"), os.path.join(q1, "memorised.pt"),
                         os.path.join(q1, "split.pt"),
                         os.path.join(q1, "tokenizer.json"), os.path.join(q1, "metrics.json"),
                         os.path.join(q1, "model_arch.py")]):
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
    rep.check("memorised checkpoint loads from the same Q1 architecture", memo is not None,
              "strict state-dict load")
    rep.guard("Q1 tokenizer, split, metrics, and model vocabulary are consistent",
              lambda: (True, validate_q1_handoff(q1, lm, split)))
    try:
        hm, hm_mem = IT.to_hooked(lm), IT.to_hooked(memo)
    except Exception as e:  # noqa: BLE001
        rep.check("to_hooked works (needed for Q6)", False, str(e)); return rep.summary()
    from transformer_lens import HookedTransformer
    if not isinstance(hm, HookedTransformer) or not isinstance(hm_mem, HookedTransformer):
        rep.check("Q6 requires real TransformerLens ports", False,
                  "to_hooked must return HookedTransformer for both Q1 checkpoints")
        return rep.summary()
    parity = (hm(X[:, :-1]) - lm(X[:, :-1])).abs().max().item()
    parity_mem = (hm_mem(X[:, :-1]) - memo(X[:, :-1])).abs().max().item()
    if max(parity, parity_mem) >= 1e-3:
        rep.check("Q4 port parity prerequisite", False,
                  f"grokked max error {parity:.3e}; memorised max error {parity_mem:.3e}; require <1e-3")
        return rep.summary()
    rep.check("Q4 port parity prerequisite", True,
              f"grokked={parity:.2e}, memorised={parity_mem:.2e}")
    L, H = hm.cfg.n_layers, hm.cfg.n_heads
    logits, cache = hm.run_with_cache(X[:, :-1])

    def t_port_contract():
        from port_contract import validate_transformerlens_port
        errors = validate_transformerlens_port(hm, X[:2, :min(8, X.shape[1]-1)])
        return not errors, "; ".join(errors[:4]) if errors else "semantic cache and readout interface valid"
    rep.guard("student port satisfies the Q4-Q6 cache/hook/readout contract", t_port_contract)

    ll = []
    def t_lens():
        nonlocal ll
        ll = IT.logit_lens(hm, cache, X, P)
        final = repeat_acc(logits, X, P)
        residual = cache["blocks.0.hook_resid_pre"]
        readout = getattr(hm, "residual_to_logits", None)
        emb_logits = readout(residual, cache) if callable(readout) else hm.unembed(hm.ln_final(residual))
        emb_only = repeat_acc(emb_logits, X, P)
        ok = (len(ll) == L + 1 and [d["depth"] for d in ll] == list(range(L + 1))
              and abs(ll[-1]["acc_repeat"] - final) < 1e-3 and abs(ll[0]["acc_repeat"] - emb_only) < 1e-3)
        return ok, " ".join(f"d{d['depth']}:{d['acc_repeat']:.3f}" for d in ll)
    rep.guard("logit_lens: one entry per depth; ends match embeddings-only and full model", t_lens)

    dec = {}
    def t_recon():
        nonlocal dec
        dec = IT.decompose_correct_logit(hm, cache, X)
        y = X[:, 1:]
        true_logit = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        recon = dec["contrib"].sum(0) + dec["bias"]
        err = (recon - true_logit).abs().max().item()
        return err < 1e-3 and torch.allclose(dec["logit"], true_logit, atol=1e-3, rtol=0.0), \
            f"max reconstruction error {err:.2e}"
    rep.guard("decompose_correct_logit: components + bias sum exactly to the true-token logit", t_recon)

    def t_names():
        names = dec["names"]
        n_pos = 1 if "hook_pos_embed" in cache else 0
        need = ["embed"] + [f"L{l}H{h}" for l in range(L) for h in range(H)] + [f"mlp_{l}" for l in range(L)]
        return (all(n in names for n in need) and len(names) == 1 + n_pos + L * (H + 2)
                and tuple(dec["contrib"].shape) == (len(names),) + tuple(X[:, 1:].shape)), f"{len(names)} components"
    rep.guard("components: embed, (pos_embed), every head, attn bias and MLP per layer", t_names)

    def t_zero_head():
        # Re-port the student's loaded A3 model and zero one TransformerLens W_O
        # slice to check the shared DLA property on the actual submitted model.
        import copy
        layer, head = hm.cfg.n_layers - 1, 0
        test_port = IT.to_hooked(copy.deepcopy(lm))
        sample = X[:2, :9]
        with torch.no_grad():
            test_port.blocks[layer].attn.W_O[head].zero_()
        _, cz = test_port.run_with_cache(sample[:, :-1])
        dz = IT.decompose_correct_logit(test_port, cz, sample)
        i = dz["names"].index(f"L{layer}H{head}")
        return dz["contrib"][i].abs().max().item() < 1e-5, ""
    rep.guard("a head whose W_O is zero contributes exactly 0", t_zero_head)

    if ll:
        rep.info("logit lens repeated-copy accuracy, embeddings to final",
                 f"{ll[0]['acc_repeat']:.2f} -> {ll[-1]['acc_repeat']:.2f}")
    if dec:
        rep_sl = slice(P + 1, 2 * P)
        means = {n: dec["contrib"][i][:, rep_sl].mean().item() for i, n in enumerate(dec["names"])}
        top = sorted(means.items(), key=lambda kv: -kv[1])[:4]
        rep.info("top components on repeated-half positions", ", ".join(f"{k}:{v:.2f}" for k, v in top))
    return rep.summary()

from expanded_grader_common import check, load_student, main_for, make_model, torch, tl_utils

def _expanded_checks(student):
    def profile_closed_form():
        score = torch.full((2, 1, 5, 5), float("-inf"))
        for q in range(5):
            for k in range(q + 1):
                score[:, 0, q, k] = float(q - k)
        cache = {"blocks.0.attn.hook_attn_scores": score}
        offsets, means = student.offset_score_profile(cache, 0, 0)
        expected_offsets = torch.arange(5)
        expected_means = torch.tensor([0., 1., 2., 3., 4.])
        ok = torch.equal(offsets, expected_offsets) and torch.allclose(means, expected_means)
        return ok, f"offsets={offsets.tolist()}, means={means.tolist()}"

    return [check("offset profile masks non-finite entries and averages by q-k", profile_closed_form)]


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
    print(f"\nCombined Q6: {original_passed + expanded_passed}/{original_total + expanded_total} checks passed")
    return 0 if original_passed == original_total and all(expanded_results) else 1


if __name__ == "__main__":
    raise SystemExit(cli_main(main))

