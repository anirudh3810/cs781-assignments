"""autograder_q2.py — Double descent on Harry Potter (CPU only, no retraining).

    python autograder/autograder_q2.py --student_dir .

Part A is deterministic: the grader re-runs your random-feature sweep on the saved HP id slices
(a few seconds on CPU) and checks the double-descent shape and that it reproduces your log.
Part B: every saved width checkpoint is forward-passed on the saved held-out windows to verify
the curve you report. Whether your curve shows a visible bump is reported as [info].
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch, load_decoder,
                            load_split, load_metrics, require, defines_function, is_implemented, approx)

FUNCS = ["random_feature_experiment", "interpolation_threshold", "add_label_noise", "train_one_width"]


def main(sd):
    rep = GradeReport("Q2 — Double descent on Harry Potter")
    dpy = os.path.join(sd, "double_descent.py"); q2 = os.path.join(sd, "submissions", "q2")
    if not require(rep, [dpy, os.path.join(q2, "split.pt"), os.path.join(q2, "metrics.json"),
                         os.path.join(q2, "tokenizer.json")]):
        return rep.summary()
    for f in FUNCS:
        rep.check(f"implements {f}", defines_function(dpy, f) and is_implemented(dpy, f))
    import common_a5 as C
    MA = load_model_arch(sd)
    DD = load_student_module(dpy, "student_double_descent")
    split, met = load_split(q2), load_metrics(q2)
    tok = C.CharTokenizer.load(os.path.join(q2, "tokenizer.json"))

    # ---------------------------------------------------------- Part A: random features
    logged = met.get("random_feature", {})
    widths = logged.get("widths") or [10, 25, 50, 100, 150, 175, 190, 200, 210, 250, 300, 500, 1000, 2000, 4000]
    n_train = logged.get("n_train", 200)
    res = {}
    def t_rerun():
        nonlocal res
        res = DD.random_feature_experiment(split["rf_train"], split["rf_test"], tok.vocab_size, widths,
                                           n_train=n_train)
        keys = {"widths", "train_mse", "test_mse", "test_acc", "n_train", "n_unique"}
        return keys <= set(res) and len(res["test_mse"]) == len(widths), f"keys {sorted(res)}"
    if not rep.guard("random_feature_experiment runs on the saved HP slices", t_rerun):
        res = None
    if res:
        test = np.array(res["test_mse"]); tr = np.array(res["train_mse"])
        k = int(np.argmax(test)); peak = widths[k]; nu = res["n_unique"]
        rep.check("n_unique counts distinct training contexts (<= n_train)", 0 < nu <= n_train, f"{nu}")
        rep.check("test error has an interior peak (rises, then falls)", 0 < k < len(widths) - 1,
                  f"peak at P={peak}")
        rep.check("peak sits at the interpolation threshold (0.7–1.15 x n_unique)",
                  0.7 * nu <= peak <= 1.15 * nu, f"peak P={peak}, n_unique={nu}")
        rep.check("past the threshold the model interpolates the training set",
                  tr[-1] <= tr[0] * 0.5 and tr[-1] <= tr[k] + 1e-3, f"train mse {tr[0]:.3f} -> {tr[-1]:.3f}")
        rep.check("second descent: widest model's test error far below the peak",
                  test[-1] < test[k] / 3, f"{test[-1]:.2f} vs peak {test[k]:.2f}")
        rep.guard("interpolation_threshold returns the argmax width",
                  lambda: (DD.interpolation_threshold(res) == peak, ""))
        rep.check("re-run reproduces the logged test curve (seeded RNG)",
                  bool(logged) and np.allclose(logged["test_mse"], res["test_mse"], rtol=1e-3, atol=1e-4))

    # ---------------------------------------------------------- unit: add_label_noise
    def t_noise():
        y = torch.zeros(200, 50, dtype=torch.long)
        y0 = y.clone()
        z = DD.add_label_noise(y, 1000, 0.3, torch.Generator().manual_seed(0))
        z2 = DD.add_label_noise(y, 1000, 0.0, torch.Generator().manual_seed(0))
        frac = (z != 0).float().mean().item()
        return (torch.equal(y, y0) and z.shape == y.shape and 0.27 <= frac <= 0.33
                and torch.equal(z2, y0)), f"changed fraction {frac:.3f} (expect ~0.30)"
    rep.guard("add_label_noise: corrupts ~frac of targets, copies input", t_noise)

    # ---------------------------------------------------------- Part B: verify every checkpoint
    sweep = met.get("width_sweep", [])
    rep.check("width sweep logged for >= 6 widths", len(sweep) >= 6, f"{[r.get('d_model') for r in sweep]}")
    Xtr, Ynoisy, Xev = split["sweep_train"], split["sweep_train_noisy"], split["sweep_eval"]
    rep.check("training targets really are noisy (5–40% differ from the text)",
              0.05 <= (Ynoisy != Xtr[:, 1:]).float().mean().item() <= 0.40,
              f"{(Ynoisy != Xtr[:, 1:]).float().mean().item():.3f}")
    verified, curve = 0, []
    for row in sweep:
        p = os.path.join(q2, "widths", f"width_{row['d_model']}.pt")
        if not os.path.exists(p):
            continue
        m, _ = load_decoder(p, MA)
        err = 1 - (m(Xev[:, :-1]).argmax(-1) == Xev[:, 1:]).float().mean().item()
        tra = (m(Xtr[:, :-1]).argmax(-1) == Ynoisy).float().mean().item()
        if approx(err, row["test_err"], 0.01) and approx(tra, row["train_acc_noisy"], 0.01):
            verified += 1
        curve.append((row["d_model"], C.count_parameters(m), tra, err))
    rep.check("every logged width point is reproduced by its saved checkpoint",
              len(sweep) > 0 and verified == len(sweep), f"{verified}/{len(sweep)} verified")
    if curve:
        rep.check("the largest model interpolates the noisy training set (train acc >= 0.9)",
                  curve[-1][2] >= 0.9, f"{curve[-1][2]:.3f}")
        errs = [c[3] for c in curve]
        j = int(np.argmax(errs))
        bump = 0 < j < len(errs) - 1 and errs[j] > min(errs[:j]) and errs[j] > errs[-1]
        rep.info("width sweep test error", " ".join(f"{c[0]}:{c[3]:.3f}" for c in curve))
        rep.info("non-monotonic (double-descent bump) visible", f"{bump} (peak at d_model={curve[j][0]})")
    return rep.summary()


if __name__ == "__main__":
    cli_main(main)
