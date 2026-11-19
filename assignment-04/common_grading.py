"""
Shared utilities for the A-4 autograders.

Design goals (same as A-3):
  - NEVER retrain a student's model. Graders run the student's source on tiny synthetic
    models (CPU, seconds) and reload the saved weights to reproduce a handful of stored
    predictions.
  - The official common.py from THIS package is always used (never the student's copy),
    so data splits, metrics and checkpoint formats cannot be changed by a submission.
  - Broken student code fails individual checks, not the whole grader.

Usage pattern (see autograder_qN.py):
    report = GradeReport("Question 2")
    sub = SubmissionDir(student_dir, "q2")
    mods = sub.load_modules("q2_classifier")
"""
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
import common  # noqa: E402  (official copy)

MODULE_ORDER = ["q2_classifier", "q5_qlora", "q1_decoding", "q3_prompting", "q4_peft", "q6_distill"]


class GradeReport:
    """Accumulates pass/fail checks with messages; prints a summary at the end."""

    def __init__(self, question):
        self.question = question
        self.checks = []

    def check(self, name, passed, detail=""):
        self.checks.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def skip(self, name, reason):
        print(f"  [SKIP] {name} — {reason}")

    def summary(self):
        n_pass = sum(1 for _, p, _ in self.checks if p)
        ok = n_pass == len(self.checks)
        print(f"\n{self.question}: {n_pass}/{len(self.checks)} checks passed {'✅' if ok else '❌'}")
        return ok


def run_grading_case(report, name, fn):
    """A broken student implementation fails this check, not the whole grader."""
    try:
        detail = fn()
        report.check(name, True, detail or "")
        return True
    except Exception as exc:  # noqa: BLE001
        report.check(name, False, f"{type(exc).__name__}: {exc}")
        return False


class SubmissionDir:
    """submissions/<q>/ with metrics.json, optional weights.pt and the module source snapshots."""

    def __init__(self, student_dir, question):
        self.student_dir = Path(student_dir).resolve()
        self.root = self.student_dir / "submissions"
        self.dir = self.root / question
        self.question = question

    def metrics(self):
        with open(self.dir / "metrics.json") as f:
            return json.load(f)

    def weights(self):
        return torch.load(self.dir / "weights.pt", map_location="cpu", weights_only=False)

    def load_modules(self, *names):
        """Import student modules (snapshot in this question's folder first, then the student
        root, then other question folders) in dependency order. The official common.py is
        pinned in sys.modules so `import common` inside student code resolves to it."""
        sys.modules["common"] = common
        wanted = [n for n in MODULE_ORDER if n in names]
        loaded = {}
        for name in wanted:
            candidates = [self.dir / f"{name}.py", self.student_dir / f"{name}.py"]
            candidates += sorted(self.root.glob(f"q*/{name}.py"))
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"{name}.py not found in {self.dir} or {self.student_dir}")
            sys.modules.pop(name, None)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        return loaded


# ----------------------------------------------------------------------------------
# Metric validation helpers
# ----------------------------------------------------------------------------------
def is_finite(value, minimum=None, maximum=None):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (minimum is None or value >= minimum) and (maximum is None or value <= maximum))
    except OverflowError:
        return False


def is_fraction(value):
    return is_finite(value, 0.0, 1.0)


def check_metrics_block(report, name, block, keys=("accuracy", "macro_f1")):
    ok = isinstance(block, dict) and all(is_fraction(block.get(k)) for k in keys)
    report.check(f"{name}: {', '.join(keys)} present and in [0, 1]", ok,
                 "" if ok else f"got {({k: (block or {}).get(k) for k in keys} if isinstance(block, dict) else block)!r}")
    return ok


def check_reload(report, name, predictor_fn, record, atol=2e-2):
    """Reload via the student's function and compare with probabilities stored in metrics.json.
    Tolerance covers GPU fp16 autocast (training machine) vs CPU fp32 (grading machine)."""
    def run():
        texts = record["texts"]
        stored = np.asarray(record["probs"][name])
        predictor = predictor_fn()
        labels, probs = predictor.predict(texts)
        probs = np.asarray(probs, dtype=float)
        if probs.shape != stored.shape:
            raise AssertionError(f"probs shape {probs.shape} != stored {stored.shape}")
        diff = float(np.abs(probs - stored).max())
        if diff > atol or not np.array_equal(np.asarray(labels), stored.argmax(-1)):
            raise AssertionError(f"max |Δp| = {diff:.3e} (atol {atol}) or labels differ")
        return f"max |Δp| = {diff:.2e} on {len(texts)} stored validation reviews"
    return run_grading_case(report, f"reload '{name}' reproduces stored predictions", run)


# ----------------------------------------------------------------------------------
# Tiny reference models used by the numerical checks
# ----------------------------------------------------------------------------------
class ToyBigramLM(nn.Module):
    """model(idx[B,T]) -> logits[B,T,V] where logits depend only on the current token."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))
        self.calls = 0

    def forward(self, idx):
        self.calls += 1
        return self.table[idx]


class ToyDraft:
    def __init__(self, table):
        self.table = torch.as_tensor(table, dtype=torch.float32)
        self.vocab_size = self.table.shape[0]

    def next_probs(self, prefix_ids):
        return self.table[int(list(prefix_ids)[-1])].softmax(-1)


def sinusoidal(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


@dataclass
class RefLMConfig:
    vocab_size: int = 61
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 64
    max_seq_len: int = 32
    dropout: float = 0.0
    norm_first: bool = False


class ReferenceBuiltinLM(nn.Module):
    """Same layout as the A-3 Q5 skeleton (BuiltinDecoderLM): tok_emb + sinusoidal pos_emb ->
    nn.TransformerEncoder(batch_first, gelu) with a causal mask -> ln_f -> tied head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer("pos_emb", sinusoidal(cfg.max_seq_len, cfg.d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=cfg.norm_first)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb[:T].unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(self.ln_f(x))


def tiny_gpt2(seed=0, n_layer=2, n_embd=32, n_head=2, vocab=211, n_positions=96):
    """Random tiny GPT-2 backbone (no download)."""
    from transformers import GPT2Config, GPT2Model
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2Model(cfg).eval()


def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def notebook_source(student_dir):
    """Concatenated code-cell source of the submitted notebook ('' if missing)."""
    path = Path(student_dir) / "notebook.ipynb"
    if not path.is_file():
        return ""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
