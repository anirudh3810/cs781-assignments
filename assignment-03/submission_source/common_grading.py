"""
Shared utilities for the A-3 autograders.

Design goals (per the assignment spec — must run on a laptop with minimal resources):
  - NEVER retrain a student's model. Every grader loads a saved checkpoint
    (submissions/qN/model.pt + config.json) and, at most, runs a handful of forward
    passes on a short held-out sentence list. On a <=50M-parameter model on CPU this
    takes a few seconds per question.
  - Static (AST-based) source inspection is used to check "did you actually implement
    X" style requirements (e.g. "no nn.Transformer in Q1-4", "nn.TransformerEncoderLayer
    used in Q5") without executing untrusted code.
  - Numeric checks use generous tolerances / thresholds, because this file mainly catches
    broken or missing implementations. Hyperparameter-search quality can be graded
    separately from the recorded perplexity/downstream metrics.
"""
import ast
import inspect
import importlib.util
import json
import os
import sys
import time
import math

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (  # noqa: E402
    load_corpus, held_out_sentences, remove_sentences_from_text,
    BPETokenizerWrapper, load_a1_tokenizer_or_fallback, perplexity_of_texts,
    count_parameters, MAX_PARAMS,
)


class GradeReport:
    """Accumulates pass/fail checks with messages; prints a summary at the end."""

    def __init__(self, question: str):
        self.question = question
        self.checks = []  # list of (name, passed: bool, detail: str)

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> bool:
        n_pass = sum(1 for _, p, _ in self.checks if p)
        n_total = len(self.checks)
        all_ok = n_pass == n_total
        print(f"\n{self.question}: {n_pass}/{n_total} checks passed "
              f"{'✅' if all_ok else '❌'}")
        return all_ok


def load_submission_dir(submission_dir: str):
    """Load config.json + metrics.json (but NOT the model — callers load that themselves,
    since the class to instantiate differs per question)."""
    cfg_path = os.path.join(submission_dir, "config.json")
    metrics_path = os.path.join(submission_dir, "metrics.json")
    model_path = os.path.join(submission_dir, "model.pt")
    missing = [p for p in (cfg_path, metrics_path, model_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required submission files: {missing}")
    with open(cfg_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    return config, metrics, model_path


def load_student_module(py_path: str, module_name: str):
    """Dynamically import a student-submitted .py file (e.g. model_arch.py) without
    executing anything beyond module-level code (no arbitrary shell commands etc. are
    run — this is standard Python import, same trust level as importing any local
    module the TA already has to run to grade code at all)."""
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Expected source file not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def source_contains(py_path: str, *substrings: str) -> bool:
    with open(py_path) as f:
        src = f.read()
    return any(s in src for s in substrings)


def ast_defines_class(py_path: str, class_name: str) -> bool:
    with open(py_path) as f:
        tree = ast.parse(f.read())
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))

FORBIDDEN_SCRATCH_APIS = (
    "torch.nn.Transformer", "torch.nn.TransformerEncoder",
    "torch.nn.TransformerDecoder", "torch.nn.TransformerEncoderLayer",
    "torch.nn.TransformerDecoderLayer", "torch.nn.MultiheadAttention",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.functional.multi_head_attention_forward",
)


def ast_forbids(py_path: str, *forbidden_attrs: str):
    """Find listed APIs through ordinary imports, aliases and references."""
    with open(py_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(py_path))

    forbidden = set(forbidden_attrs)
    aliases = {}
    found = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".")[0]
                aliases[item.asname or root] = ( item.name if item.asname else root)
                if item.name in forbidden:
                    found.add(item.name)

        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for item in node.names:
                if item.name == "*":
                    for api in forbidden:
                        module, name = api.rsplit(".", 1)
                        if module == node.module:
                            aliases.setdefault(name, api)
                else:
                    api = f"{node.module}.{item.name}"
                    aliases[item.asname or item.name] = api
                    if api in forbidden:
                        found.add(api)

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            api = resolve(node)
            if api in forbidden:
                found.add(api)

    return sorted(found)


def check_forbidden_implementations(report, sub_dir, question):
    required = {
        1: ("attention.py", "model_arch.py"),
        2: ("attention.py", "model_arch.py"),
        3: ("attention.py", "model_arch.py", "rope.py"),
        4: ("attention.py", "model_arch.py", "optim_muon.py"),
    }
    filenames = list(required[question])

    # Later attention snapshots may depend on rope.py, even outside Q3.
    if "rope.py" not in filenames and os.path.isfile(os.path.join(sub_dir, "rope.py")):
        filenames.append("rope.py")

    forbidden = FORBIDDEN_SCRATCH_APIS
    if question == 4:
        # Q4 requires the specified simplified Muon implementation.
        forbidden += ("torch.optim.Muon",)

    all_ok = True

    for filename in filenames:
        path = os.path.join(sub_dir, filename)
        try:
            hits = ast_forbids(path, *forbidden)
            passed = not hits
            detail = "Found: " + ", ".join(hits) if hits else ""
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            passed = False
            detail = f"Cannot inspect source: {exc}"

        report.check(f"{filename}: no forbidden implementation", passed, detail)
        all_ok = all_ok and passed

    return all_ok


_CORPUS_CACHE = {}


def get_shared_eval_set(raw_dir=None, n_chapters=5, sentences_per_chapter=6):
    """
    Returns the same held-out sentence list every student's Section-0 cell would have
    produced (same seed, same sampling logic in common.py) PLUS an independent
    TA-only held-out set sampled from *different* intermediate chapters, so the
    autograder can optionally re-check perplexity on sentences the student could not
    have seen or tuned against even indirectly. Both require data/raw/ to be populated
    on the grading machine (see data/README.md) — if it is not, perplexity checks are
    skipped gracefully (see `try_compute_perplexity` below) rather than crashing, since
    a TA may sometimes grade code-structure-only without the full corpus on hand.
    """
    key = (raw_dir, n_chapters, sentences_per_chapter)
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]
    raw_dir = raw_dir or os.path.join(os.path.dirname(__file__), "data", "raw")
    corpus = load_corpus(raw_dir)
    student_visible = held_out_sentences(corpus, n_chapters=n_chapters,
                                          sentences_per_chapter=sentences_per_chapter)
    ta_only = held_out_sentences(corpus, n_chapters=n_chapters,
                                  sentences_per_chapter=sentences_per_chapter, seed=99991)
    result = (corpus, student_visible, ta_only)
    _CORPUS_CACHE[key] = result
    return result


def try_compute_perplexity(model, tokenizer, sentences, device="cpu", max_len=128):
    """Never raises — returns None (and prints a warning) if the corpus isn't available
    on the grading machine, or if the forward pass errors out for any reason. A grader
    should treat None as 'skipped', not as a failed check."""
    try:
        t0 = time.time()
        ppl = perplexity_of_texts(model, tokenizer, sentences, device=device, max_len=max_len)
        print(f"    (perplexity computed on {len(sentences)} sentences in {time.time()-t0:.1f}s)")
        return ppl
    except FileNotFoundError:
        print("    (skipping perplexity recheck: data/raw/ not populated on this machine)")
        return None
    except Exception as e:  # pragma: no cover - defensive
        print(f"    (skipping perplexity recheck due to error: {e})")
        return None


def build_tokenizer_for_grading(student_dir: str, vocab_size_hint: int):
    """
    Recreate the tokenizer used for grading.

    The grader first looks for the required Assignment-1 hook:
        a1_tokenizer.py::train_tokenizer(train_text, vocab_size)

    It checks both the submission root and the Q1 source snapshot. If the hook/file or
    returned encode/decode/vocabulary-size interface is unavailable, diagnostics are
    printed and grading continues with the course fallback BPE. The return value is
    `(tokenizer_or_none, used_student_tokenizer, diagnostics)`.
    """
    try:
        corpus, student_visible, _ = get_shared_eval_set()
    except FileNotFoundError:
        print("    (tokenizer check skipped: data/raw/ not populated on this machine)")
        return None, False, ["grading corpus unavailable"]

    train_text = remove_sentences_from_text(corpus.raw_text, student_visible)
    candidate_paths = [
        os.path.join(student_dir, "a1_tokenizer.py"),
        os.path.join(student_dir, "submissions", "q1", "a1_tokenizer.py"),
    ]
    module_path = next((path for path in candidate_paths if os.path.isfile(path)), candidate_paths[0])

    tok, used_student, diagnostics = load_a1_tokenizer_or_fallback(
        train_text,
        vocab_size=vocab_size_hint,
        module_path=module_path,
        verbose=True,
    )
    return tok, used_student, diagnostics


def load_full_student_stack(student_dir: str, need: tuple = ("attention", "model_arch")):
    """
    Loads the student's .py files in dependency order and registers each in sys.modules
    BEFORE loading the next, so `from rope import ...` / `from attention import ...` /
    `from model_arch import ...` statements inside later files resolve correctly. This
    mirrors the real dependency chain used throughout the assignment:

        rope.py  ->  attention.py  ->  model_arch.py  ->  model_builtin.py

    (rope.py is optional -- only Questions 3+ submissions require it; attention.py only
    needs `from rope import ...` once a student has done Question 3.)

    `need` lists which modules the caller actually requires to exist (raises
    FileNotFoundError with a clear message if a required one is missing); modules not
    in `need` are loaded opportunistically if present, silently skipped otherwise.

    Returns a dict: {"rope": module_or_None, "attention": module_or_None,
                      "model_arch": module_or_None, "model_builtin": module_or_None}
    """
    order = [
        ("rope", "rope.py"),
        ("attention", "attention.py"),
        ("model_arch", "model_arch.py"),
        ("model_builtin", "model_builtin.py"),
    ]
    loaded = {}
    for mod_name, fname in order:
        path = os.path.join(student_dir, fname)
        if os.path.exists(path):
            module = load_student_module(path, f"student_{mod_name}")
            sys.modules[mod_name] = module
            loaded[mod_name] = module
        else:
            loaded[mod_name] = None
            if mod_name in need:
                raise FileNotFoundError(f"Required file missing for this question: {fname}")
    return loaded


def check_param_budget(model, report: GradeReport):
    n = count_parameters(model)
    report.check("parameter budget (<= 50,000,000)", n <= MAX_PARAMS, f"{n:,} parameters")
    return n

@torch.no_grad()


def check_causality(model, cfg, report: GradeReport, *, max_test_len=16, trials=2, atol=1e-4, rtol=1e-5):
    """Check that changing future tokens leaves earlier logits unchanged."""
    name = "causality: future tokens do not affect earlier logits"
    modes = [(module, module.training) for module in model.modules()]

    try:
        vocab_size = int(cfg.vocab_size)
        T = min(int(cfg.max_seq_len), int(max_test_len))
        if vocab_size < 2 or T < 2 or trials < 1:
            raise ValueError( "Need vocab_size >= 2, sequence length >= 2, trials >= 1")

        parameter = next(model.parameters(), None)
        device = (parameter.device if parameter is not None else torch.device("cpu"))

        # Local generator: does not change the grader's global random seed.
        rng = torch.Generator(device="cpu").manual_seed(781)
        batch_size = 2
        cutoffs = sorted({0, (T - 1) // 2, T - 2})
        expected_shape = (batch_size, T, vocab_size)
        max_diff = 0.0
        comparisons = 0
        model.eval()

        def checked_forward(ids):
            logits = model(ids.to(device))
            
            if not isinstance(logits, torch.Tensor):
                raise TypeError("model must return a logits tensor")
            
            if tuple(logits.shape) != expected_shape:
                raise ValueError( f"Expected logits {expected_shape}, got {tuple(logits.shape)}")

            if (not logits.is_floating_point() or not torch.isfinite(logits).all().item()):
                raise ValueError("Logits must be floating-point and finite")

            return logits

        for trial in range(trials):
            original = torch.randint(vocab_size, (batch_size, T), generator=rng)
            reference = checked_forward(original).clone()

            for cutoff in cutoffs:
                changed = original.clone()
                suffix = changed[:, cutoff + 1:]

                # Guarantee that every token AFTER the cutoff changes.
                offsets = torch.randint(1, vocab_size, suffix.shape, generator=rng)
                changed[:, cutoff + 1:] = (suffix + offsets) % vocab_size

                altered = checked_forward(changed)

                # Positions 0 through cutoff must remain unaffected.
                before = reference[:, :cutoff + 1, :]
                after = altered[:, :cutoff + 1, :]

                diff = (before - after).abs().max().item()
                max_diff = max(max_diff, diff)
                comparisons += 1

                if not torch.allclose( before, after, atol=atol, rtol=rtol):
                    report.check(name, False,
                        f"trial={trial + 1}, cutoff={cutoff}, max abs diff={diff:.3e}; atol={atol}, rtol={rtol}",)
                    return False

        report.check(name, True, f"{comparisons} comparisons at T={T}; max abs diff={max_diff:.3e}")
        return True

    except Exception as exc:
        report.check(name, False, f"{type(exc).__name__}: {exc}")
        return False

    finally:
        # Restore each module's original train/eval state.
        for module, training in modes:
            module.training = training


def metric_is_finite(value, minimum=None):
    """Accept JSON numbers, rejecting bool, None, strings, NaN and infinity."""
    try:
        return (type(value) in (int, float) and math.isfinite(value) and (minimum is None or value >= minimum))
    except OverflowError:
        return False


def check_metric_curve(report, name, values):
    ok = metric_is_curve(values)
    report.check( name, ok, "expected a nonempty list of finite, nonnegative numbers")
    return ok


def metric_dict(report, name, value):
    ok = isinstance(value, dict)
    report.check(name, ok, "expected a dictionary")
    return value if ok else {}


# Shared validation and numerical-test utilities.


def metric_is_integer(x, minimum=1):
    return type(x) is int and x >= minimum


def metric_is_curve(values):
    """Nonempty list of finite, nonnegative JSON numbers."""
    return isinstance(values, list) and bool(values) and all(
        metric_is_finite(v, minimum=0) for v in values
    )


def run_grading_case(report, name, fn):
    """A broken student implementation fails this check, not the whole grader."""
    try:
        detail = fn()
        report.check(name, True, detail or "")
    except Exception as exc:
        report.check(name, False, f"{type(exc).__name__}: {exc}")


def assert_tensors_close(actual, expected, atol=2e-5, rtol=2e-4):
    import torch
    if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
        raise AssertionError("wrong output type or shape")
    if not torch.isfinite(actual).all().item():
        raise AssertionError("nonfinite output")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return f"max abs diff={(actual - expected).abs().max().item():.3e}"


def make_test_attention(cls, kv=8, rope=False, require_rope=False):
    """Supports the Q1 signature and the extended Q2/Q3 signatures."""
    signature = inspect.signature(cls.__init__).parameters
    kwargs = dict(d_model=64, n_heads=8, max_seq_len=16, dropout=0.0)
    if "n_kv_heads" in signature:
        kwargs["n_kv_heads"] = kv
    elif kv != 8:
        raise AssertionError("missing n_kv_heads argument")
    if "use_rope" in signature:
        kwargs["use_rope"] = rope
    elif require_rope:
        raise AssertionError("missing use_rope argument")
    if "attention_scale" in signature:
        kwargs["attention_scale"] = None
    return cls(**kwargs).cpu().float().eval()


def randomize_attention_projections(module, generator):
    import torch
    # Nonzero independent weights defeat default-zero and symmetric fixtures.
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            projection = getattr(module, name)
            projection.weight.copy_(torch.randn(projection.weight.shape, generator=generator) * 0.12)
            if projection.bias is not None:
                projection.bias.zero_()


def reference_causal_attention(module, x, kv, *, n_heads=8, transform_qk=None):
    import torch
    import torch.nn.functional as F
    batch, length, width = x.shape
    heads, dim = n_heads, width // n_heads

    def project(name, count):
        layer = getattr(module, name)
        # Use the copied weights directly, not the student's projection forward.
        return F.linear(x, layer.weight, layer.bias).reshape(batch, length, count, dim).transpose(1, 2)

    q, k, v = project("q_proj", heads), project("k_proj", kv), project("v_proj", kv)
    if transform_qk is not None:
        q, k = transform_qk(q, k)
    # Explicit indexing is independent of the student's repetition implementation.
    index = torch.arange(heads) // (heads // kv)
    k, v = k[:, index], v[:, index]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
    blocked = torch.ones(length, length, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(blocked, float("-inf"))
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
    return F.linear(out, module.out_proj.weight, module.out_proj.bias)
