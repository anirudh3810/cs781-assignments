"""autograder_q4.py — student TransformerLens port of the Q1 DecoderLM (CPU only).

    python autograder/autograder_q4.py --student_dir .

Your to_hooked() must return a real TransformerLens HookedTransformer, reproduce the native
DecoderLM logits on both Q1 checkpoints, and expose the common analysis contract for Q4–Q6.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch, load_decoder,
                            load_split, validate_q1_handoff, require, defines_function, is_implemented, synthetic_cache)

FUNCS = ["to_hooked", "logits_match", "logits_and_cache", "attention_entropy"]


def _original_main(sd):
    rep = GradeReport("Q4 — TransformerLens port")
    ipy = os.path.join(sd, "interp.py"); q1 = os.path.join(sd, "submissions", "q4_q6_model")
    if not require(rep, [ipy, os.path.join(q1, "model.pt"), os.path.join(q1, "memorised.pt"),
                         os.path.join(q1, "split.pt"), os.path.join(q1, "tokenizer.json"),
                         os.path.join(q1, "metrics.json"), os.path.join(q1, "model_arch.py")]):
        return rep.summary()
    for f in FUNCS:
        rep.check(f"implements {f}", defines_function(ipy, f) and is_implemented(ipy, f))
    try:
        MA = load_model_arch(sd)
    except Exception as exc:
        rep.check("loads the exact Q1 architecture snapshot", False, f"{type(exc).__name__}: {exc}")
        return rep.summary()
    IT = load_student_module(ipy, "student_interp")
    split = load_split(q1); X = split["eval"][:64]

    def t_handoff():
        lm, _ = load_decoder(os.path.join(q1, "model.pt"), MA)
        return True, validate_q1_handoff(q1, lm, split)
    rep.guard("Q1 tokenizer, split, metrics, and model vocabulary are consistent", t_handoff)

    for name in ["model.pt", "memorised.pt"]:
        def t(name=name):
            lm, _ = load_decoder(os.path.join(q1, name), MA)
            hm = IT.to_hooked(lm)
            from transformer_lens import HookedTransformer
            if not isinstance(hm, HookedTransformer):
                return False, f"to_hooked returned {type(hm).__name__}, not HookedTransformer"
            d = (lm(X[:, :-1]) - hm(X[:, :-1])).abs().max().item()
            return d < 1e-3, f"max |logit diff| {d:.2e}"
        rep.guard(f"real HookedTransformer port reproduces {name} within 1e-3", t)

    def t_port_contract():
        from port_contract import validate_transformerlens_port
        lm, _ = load_decoder(os.path.join(q1, "model.pt"), MA)
        hm = IT.to_hooked(lm)
        probe = X[:2, :min(8, X.shape[1] - 1)]
        errors = validate_transformerlens_port(hm, probe)
        return not errors, "; ".join(errors[:4]) if errors else "semantic cache and hook interface valid"
    rep.guard("student port satisfies the Q4-Q6 cache/hook/readout contract", t_port_contract)

    def t_match():
        a = torch.randn(2, 3, 5)
        return (IT.logits_match(a, a.clone()) is True or IT.logits_match(a, a.clone()) == True) and \
               not IT.logits_match(a, a + 0.01) and not IT.logits_match(a, a + 1.0), ""
    rep.guard("logits_match accepts equal and rejects different logits", t_match)

    lm, _ = load_decoder(os.path.join(q1, "model.pt"), MA)
    def t_cache():
        hm = IT.to_hooked(lm)
        logits, cache = IT.logits_and_cache(hm, X)
        return (tuple(logits.shape[:2]) == tuple(X[:, :-1].shape) and "blocks.0.attn.hook_pattern" in cache
                and torch.allclose(logits, lm(X[:, :-1]), atol=1e-3)), f"logits {tuple(logits.shape)}"
    rep.guard("logits_and_cache runs on X[:, :-1] and returns a full cache", t_cache)

    def t_entropy():
        hm = IT.to_hooked(lm)
        P = (X.shape[1] - 1) // 2
        cache, exp = synthetic_cache(hm, P)
        got = IT.attention_entropy(cache, hm.cfg.n_layers)
        return torch.allclose(got.float(), exp["entropy"], atol=1e-4), f"got {got[0].tolist()} exp {exp['entropy'][0].tolist()}"
    rep.guard("attention_entropy matches closed-form values on synthetic patterns", t_entropy)
    return rep.summary()

from expanded_grader_common import check, load_student, main_for, make_model, torch, tl_utils

def _expanded_checks(student):
    model = make_model(seed=17)
    tokens = torch.randint(0, model.cfg.d_vocab, (2, 9))

    def anatomy():
        fn = getattr(student, "block0_by_hand")
        got = fn(model, tokens)
        _, cache = model.run_with_cache(tokens)
        refs = {
            "resid_pre": cache["blocks.0.hook_resid_pre"],
            "ln1.hook_normalized": cache["blocks.0.ln1.hook_normalized"],
            "attn_out": cache["blocks.0.hook_attn_out"],
            "resid_mid": cache["blocks.0.hook_resid_mid"],
            "ln2.hook_normalized": cache["blocks.0.ln2.hook_normalized"],
            "mlp_out": cache["blocks.0.hook_mlp_out"],
            "resid_post": cache["blocks.0.hook_resid_post"],
        }
        err = max((got[key] - value).abs().max().item() for key, value in refs.items())
        ok = set(got) == set(refs) and all(torch.allclose(got[k], v, atol=1e-5, rtol=1e-5) for k, v in refs.items())
        return ok, f"max activation error={err:.2e}; keys={len(got)}"

    return [check("block0_by_hand matches all seven cached activations", anatomy)]


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
    print(f"\nCombined Q4: {original_passed + expanded_passed}/{original_total + expanded_total} checks passed")
    return 0 if original_passed == original_total and all(expanded_results) else 1


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
