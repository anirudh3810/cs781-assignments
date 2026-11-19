"""common_grading.py — shared utilities for the A-5 autograders (CPU only).

Design rules (from A-3):
  * Student code is imported from the .py files written by the notebook's %%writefile cells.
  * Graders never retrain. They load the saved checkpoints with map_location="cpu" and run
    forward passes. Everything runs on a laptop CPU in seconds.
  * This folder is shared with students, so it contains NO solution code. Student functions are
    checked with property tests whose answers are known in closed form (oracle models,
    synthetic attention patterns, independent routes such as zeroing W_O instead of hooking z),
    and with exact consistency checks on the student's own model.
  * Stochastic quantities (exact grok step, the size of a double-descent bump) are [info].
"""
import os
import ast
import sys
import json
import importlib.util

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------- report
class GradeReport:
    def __init__(self, title):
        self.title, self.rows, self.soft = title, [], []
        print(f"\n### {title}")

    def check(self, name, passed, detail=""):
        passed = bool(passed)
        self.rows.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        return passed

    def guard(self, name, fn):
        """fn() -> (passed, detail). An exception in student code is a FAIL, not a crash."""
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
        return self.check(name, ok, detail)

    def info(self, name, detail=""):
        self.soft.append((name, detail))
        print(f"  [info] {name}" + (f"  — {detail}" if detail else ""))

    def summary(self):
        n, k = len(self.rows), sum(p for _, p, _ in self.rows)
        print(f"=== {self.title}: {k}/{n} checks passed ===")
        return {"title": self.title, "passed": k, "total": n, "rows": self.rows, "soft": self.soft}


# --------------------------------------------------------------------- loading
def setup_paths(student_dir):
    """Student dir first (their modules), repo root second (common_a5, fallback model_arch)."""
    student_dir = os.path.abspath(student_dir)
    for p in (REPO_ROOT, student_dir):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return student_dir


def load_student_module(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model_arch(student_dir):
    p = os.path.join(student_dir, "model_arch.py")
    if not os.path.exists(p):
        p = os.path.join(REPO_ROOT, "model_arch.py")
    return load_student_module(p, "model_arch")


def load_decoder(path, model_arch):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    m = model_arch.DecoderLM(model_arch.LMConfig(**payload["config"]))
    m.load_state_dict(payload["state_dict"], strict=True)
    return m.eval(), payload.get("extra", {})


def load_split(qdir):
    return torch.load(os.path.join(qdir, "split.pt"), map_location="cpu", weights_only=False)


def load_metrics(qdir):
    return json.load(open(os.path.join(qdir, "metrics.json")))


def require(rep, paths):
    """Check that every path exists; return True if all do."""
    ok = True
    for p in paths:
        ok &= rep.check(f"{os.path.relpath(p)} present", os.path.exists(p))
    return ok


# --------------------------------------------------------------------- AST
def _tree(path):
    return ast.parse(open(path).read())


def defines_function(path, name):
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(_tree(path)))


def is_implemented(path, name):
    """False if the function body is still just `raise NotImplementedError`."""
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            body = [b for b in n.body
                    if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))]
            return not (len(body) == 1 and isinstance(body[0], ast.Raise))
    return False


def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------- property-test helpers
class OracleLM:
    """Fake language model for testing metric functions. Given the full sequences X it emits,
    at every input position q, a confident prediction that is CORRECT where mask[q] is True and
    wrong (target+1 mod V) elsewhere. The true accuracies/losses are therefore known."""
    def __init__(self, X, mask, vocab_size, margin=6.0):
        self.X, self.mask, self.V, self.margin = X, mask, vocab_size, margin

    def __call__(self, inp):
        y = self.X[:, 1:]
        pred = torch.where(self.mask.expand_as(y), y, (y + 1) % self.V)
        logits = torch.zeros(*y.shape, self.V)
        logits.scatter_(-1, pred.unsqueeze(-1), self.margin)
        assert inp.shape == y.shape, "model must be called on X[:, :-1]"
        return logits

    def eval(self):
        return self


def ce_of_oracle(V, margin, correct):
    """Cross-entropy of one OracleLM prediction (correct or wrong)."""
    lse = torch.logsumexp(torch.tensor([margin] + [0.0] * (V - 1)), 0).item()
    return lse - (margin if correct else 0.0)


def synthetic_cache(hm, P, n_layers=None):
    """A real ActivationCache whose attention patterns are hand-built so that previous-token,
    induction and entropy scores are known in closed form. Uses B=1, 3 heads repeated over the
    model's head count:
        head 0 — all mass on q-1 (q>=1)             prev=1, ind=0, entropy=0
        head 1 — uniform over keys 0..q              prev=mean 1/(q+1), entropy=mean log(q+1)
        head 2 — mass on q-P for second-copy q, on q itself elsewhere   ind=1
    Returns (cache, expected) with expected tensors of shape (L, H)."""
    from transformer_lens import ActivationCache
    from transformer_lens import utils as U
    T = 2 * P
    L = n_layers or hm.cfg.n_layers
    H = hm.cfg.n_heads
    base = torch.zeros(3, T, T)
    for q in range(T):
        base[0, q, max(q - 1, 0)] = 1.0
        base[1, q, : q + 1] = 1.0 / (q + 1)
        base[2, q, q - P if P + 1 <= q <= 2 * P - 1 else q] = 1.0
    pat = base[torch.arange(H) % 3].unsqueeze(0)                      # (1, H, T, T)
    qs = torch.arange(1, T)
    exp_prev_h = torch.tensor([1.0, (1.0 / (qs + 1.0)).mean().item(), 0.0])
    exp_ent_h = torch.tensor([0.0, torch.log(qs + 1.0).mean().item(), 0.0])
    exp_ind_h = torch.tensor([0.0, (1.0 / torch.arange(P + 2, 2 * P + 1).float()).mean().item(), 1.0])
    # head 2's previous-token score: rows q with key q-1 only when P==1 (never for P>=2)
    sel = torch.arange(H) % 3
    d = {U.get_act_name("pattern", l): pat.clone() for l in range(L)}
    cache = ActivationCache(d, hm)
    expected = {"prev": exp_prev_h[sel].repeat(L, 1), "entropy": exp_ent_h[sel].repeat(L, 1),
                "induction": exp_ind_h[sel].repeat(L, 1)}
    return cache, expected


def zero_head_copy(hm, layer, head):
    """Independent ablation route: a copy of the hooked model with W_O[head] = 0 in `layer`.
    With b_O = 0 this is identical to zero-ablating that head's z."""
    import copy
    m = copy.deepcopy(hm)
    m.blocks[layer].attn.W_O.data[head] = 0.0
    return m


def repeat_acc(logits, X, P):
    rep = slice(P + 1, 2 * P)
    return (logits[:, rep].argmax(-1) == X[:, 1:][:, rep]).float().mean().item()


def try_corpus_split(student_dir):
    """(train_text, heldout_text) for the optional Q1 provenance check: data/raw/*.txt if present,
    else the Hugging Face dataset (needs internet; cached after the first download). None if
    neither is available — grading never depends on it."""
    import common_a5 as C
    for raw in (os.path.join(student_dir, "data", "raw"), os.path.join(REPO_ROOT, "data", "raw")):
        if os.path.isdir(raw) and any(f.endswith(".txt") for f in os.listdir(raw)):
            try:
                tr, te, _ = C.split_corpus(C.load_corpus("raw", raw))
                return tr, te
            except Exception:  # noqa: BLE001
                pass
    try:
        tr, te, _ = C.split_corpus(C.load_corpus("hf"))
        return tr, te
    except Exception:  # noqa: BLE001
        return None


def cli_main(main):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", default=".")
    a = ap.parse_args()
    sd = setup_paths(a.student_dir)
    return main(sd)
