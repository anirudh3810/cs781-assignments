"""autograder_q1.py — Grokking on Harry Potter (CPU only, no retraining).

    python autograder/autograder_q1.py --student_dir .

Expects in --student_dir:
    grokking.py, model_arch.py (optional — falls back to the repo copy)
    submissions/q1/{model.pt, memorised.pt, tokenizer.json, split.pt, metrics.json}
"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch,
                            load_decoder, load_split, load_metrics, require, defines_function,
                            is_implemented, OracleLM, ce_of_oracle, approx, try_corpus_split)

FUNCS = ["make_repeated_passages", "scale_init_", "evaluate_lm", "evaluate_repeat"]


def main(sd):
    rep = GradeReport("Q1 — Grokking on Harry Potter")
    gpy = os.path.join(sd, "grokking.py"); q1 = os.path.join(sd, "submissions", "q1")
    if not require(rep, [gpy] + [os.path.join(q1, f) for f in
                                 ["model.pt", "memorised.pt", "tokenizer.json", "split.pt", "metrics.json"]]):
        return rep.summary()
    for f in FUNCS:
        rep.check(f"implements {f}", defines_function(gpy, f) and is_implemented(gpy, f))

    import common_a5 as C
    MA = load_model_arch(sd)
    G = load_student_module(gpy, "student_grokking")

    # ---------------------------------------------------------- unit: make_repeated_passages
    def t_passages():
        ids = torch.arange(1000) % 97
        P, n, sep = 7, 16, 99
        X = G.make_repeated_passages(ids, n, P, sep, torch.Generator().manual_seed(0))
        X2 = G.make_repeated_passages(ids, n, P, sep, torch.Generator().manual_seed(0))
        X3 = G.make_repeated_passages(ids, n, P, sep, torch.Generator().manual_seed(1))
        ok = (tuple(X.shape) == (n, 2 * P + 1) and bool((X[:, P] == sep).all())
              and torch.equal(X[:, :P], X[:, P + 1:]) and torch.equal(X, X2) and not torch.equal(X, X3))
        # each passage must be a contiguous window of ids (consecutive values mod 97)
        contiguous = bool(((X[:, 1:P] - X[:, :P - 1]) % 97 == 1).all())
        return ok and contiguous, f"shape {tuple(X.shape)}"
    rep.guard("make_repeated_passages: shape, separator, identical halves, contiguous, seeded", t_passages)

    # ---------------------------------------------------------- unit: scale_init_
    def t_scale():
        torch.manual_seed(0)
        m = MA.DecoderLM(MA.LMConfig(vocab_size=30, d_model=16, n_layers=2, n_heads=2, n_kv_heads=2,
                                     d_ff=32, max_seq_len=8, dropout=0.0))
        with torch.no_grad():
            for n_, p_ in m.named_parameters():
                if "ln" in n_: p_.add_(0.3)            # make LN params distinguishable from 1/0
        before = {k: v.clone() for k, v in m.named_parameters()}
        out = G.scale_init_(m, 2.5)
        good = True
        for k, v in m.named_parameters():
            target = before[k] if ("ln" in k) else 2.5 * before[k]
            good &= torch.allclose(v, target, atol=1e-6)
        return good and out is m, "non-LN params x alpha, LN params untouched, returns model"
    rep.guard("scale_init_: scales every non-LayerNorm parameter by alpha", t_scale)

    # ---------------------------------------------------------- unit: metrics on an oracle model
    V, P, B = 11, 5, 6
    Xo = torch.randint(0, V, (B, 2 * P + 1))
    T = 2 * P
    def t_eval_repeat():
        mask = torch.zeros(T, dtype=torch.bool); mask[P + 1:2 * P] = True     # right only on repeats
        r = G.evaluate_repeat(OracleLM(Xo, mask, V), Xo, P)
        mask2 = torch.zeros(T, dtype=torch.bool); mask2[: P - 1] = True       # right only on copy 1
        r2 = G.evaluate_repeat(OracleLM(Xo, mask2, V), Xo, P)
        ok = (approx(r["acc_repeat"], 1, 1e-6) and approx(r["acc_first"], 0, 1e-6)
              and approx(r2["acc_repeat"], 0, 1e-6) and approx(r2["acc_first"], 1, 1e-6)
              and approx(r["loss_repeat"], ce_of_oracle(V, 6.0, True), 1e-4)
              and approx(r2["loss_repeat"], ce_of_oracle(V, 6.0, False), 1e-4))
        return ok, f"oracle1 {r} | oracle2 {r2}"
    rep.guard("evaluate_repeat: correct positions & loss on oracle models", t_eval_repeat)

    def t_eval_lm():
        mask = torch.zeros(T, dtype=torch.bool); mask[::2] = True
        r = G.evaluate_lm(OracleLM(Xo, mask, V), Xo)
        exp_loss = 0.5 * ce_of_oracle(V, 6.0, True) + 0.5 * ce_of_oracle(V, 6.0, False)
        return approx(r["acc"], 0.5, 1e-6) and approx(r["loss"], exp_loss, 1e-4), str(r)
    rep.guard("evaluate_lm: accuracy & loss over all positions on an oracle model", t_eval_lm)

    # ---------------------------------------------------------- behaviour: forward passes on CPU
    split = load_split(q1); P = int(split["P"]); Xtr, Xev = split["train"], split["eval"]
    tok = C.CharTokenizer.load(os.path.join(q1, "tokenizer.json"))
    rep.check("split: doubled passages with the tokenizer's <SEP> in the middle",
              tuple(Xtr.shape[1:]) == (2 * P + 1,) and bool((Xtr[:, P] == tok.sep_id).all())
              and bool((Xev[:, P] == tok.sep_id).all()))
    tr_set = {tuple(r.tolist()) for r in Xtr[:, :P]}
    overlap = sum(tuple(r.tolist()) in tr_set for r in Xev[:, :P])
    rep.check("no held-out passage is also a training passage", overlap == 0, f"{overlap} overlaps")
    rep.check("training set is small (<= 2048 passages) — grokking needs a reduced subset",
              len(Xtr) <= 2048, f"{len(Xtr)} passages")

    model, _ = load_decoder(os.path.join(q1, "model.pt"), MA)
    memo, mextra = load_decoder(os.path.join(q1, "memorised.pt"), MA)
    rep.check("model fits the 50M parameter budget", C.count_parameters(model) <= C.MAX_PARAMS)
    try:
        g_tr, g_ev = G.evaluate_repeat(model, Xtr, P), G.evaluate_repeat(model, Xev, P)
        m_tr, m_ev = G.evaluate_repeat(memo, Xtr, P), G.evaluate_repeat(memo, Xev, P)
    except Exception as e:  # noqa: BLE001
        rep.check("behaviour checks (need a working evaluate_repeat)", False, f"{type(e).__name__}: {e}")
        return rep.summary()
    rep.check("grokked model: training repeats memorised (acc >= 0.95)", g_tr["acc_repeat"] >= 0.95,
              f"{g_tr['acc_repeat']:.3f}")
    rep.check("grokked model: generalises to UNSEEN held-out passages (repeat acc >= 0.80)",
              g_ev["acc_repeat"] >= 0.80, f"{g_ev['acc_repeat']:.3f}")
    rep.check("grokked model: held-out FIRST copy stays hard (acc < 0.6) — it copies, not recalls",
              g_ev["acc_first"] < 0.6, f"{g_ev['acc_first']:.3f}")
    rep.check("memorised checkpoint: training repeats fitted (acc >= 0.85)", m_tr["acc_repeat"] >= 0.85,
              f"{m_tr['acc_repeat']:.3f}")
    rep.check("memorised checkpoint generalises far worse than the grokked one (gap >= 0.3)",
              g_ev["acc_repeat"] - m_ev["acc_repeat"] >= 0.3,
              f"memorised {m_ev['acc_repeat']:.3f} vs grokked {g_ev['acc_repeat']:.3f}")

    met = load_metrics(q1)
    hist, gap = met.get("history", {}), met.get("grokking_gap", {})
    rep.check("logged final held-out repeat acc matches the saved model",
              bool(hist) and approx(hist["test_acc_repeat"][-1], g_ev["acc_repeat"], 0.02),
              f"logged {hist.get('test_acc_repeat', [None])[-1]} vs recomputed {g_ev['acc_repeat']:.3f}")
    rep.check("grokking gap is positive (generalisation came after memorisation)",
              isinstance(gap.get("gap"), (int, float)) and gap["gap"] > 0, f"{gap}")
    rep.info("memorised checkpoint step", str(mextra.get("step")))
    if "partA_history" in met:
        h = met["partA_history"]
        rep.info("Part A (plain LM) final train / held-out acc",
                 f"{h['train_acc'][-1]:.3f} / {h['test_acc'][-1]:.3f}")

    # ---------------------------------------------------------- integrity vs. the TA's copy of the books
    sp = try_corpus_split(sd)
    if sp is None:
        rep.info("HP corpus unavailable (offline, no data/raw) — skipped corpus-membership check")
    else:
        tr_text, te_text = sp
        dec = lambda r: "".join(tok.itos[i] for i in r.tolist())
        n_ev = min(100, len(Xev)); n_tr = min(100, len(Xtr))
        f_ev = sum(dec(r) in te_text for r in Xev[:n_ev, :P]) / n_ev
        f_tr = sum(dec(r) in tr_text for r in Xtr[:n_tr, :P]) / n_tr
        rep.info("held-out passages found in the held-out segments", f"{f_ev:.0%}")
        rep.info("training passages found in the training segments", f"{f_tr:.0%}")
    return rep.summary()


if __name__ == "__main__":
    cli_main(main)
