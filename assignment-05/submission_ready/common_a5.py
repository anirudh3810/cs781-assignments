"""
common_a5.py — shared infrastructure for Assignment 5
(Post-training, Alignment, Grokking, Double Descent & Mechanistic Interpretability).

Everything in A-5 runs on the SAME Harry Potter corpus used in A-3. This module is
imported by the questions notebook, the solutions notebook, and every autograder, so
splits, tokenisation and I/O are identical everywhere. Do NOT modify it.

Students train on a GPU (Kaggle / Colab). Autograders only load the saved checkpoints
and run forward passes on CPU, so every save/load here is device-agnostic.
"""
import os
import importlib.util
import shutil
import sys
import re
import glob
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

SEED = 1337


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MAX_PARAMS = 50_000_000   # same ceiling as A-3


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_within_budget(model, max_params: int = MAX_PARAMS) -> int:
    n = count_parameters(model)
    assert n <= max_params, f"{n:,} parameters exceeds the {max_params:,} budget."
    return n


# ======================================================================================
# 1. Harry Potter corpus — the same Hugging Face dataset as A-3
# ======================================================================================
# WutYee/HarryPotter_books_1to7: one `text` column, one wrapped line of a book per row.
# Its splits are contiguous books in reading order: train = books 1-5, validation = book 6,
# test = book 7. We concatenate them in that order to recover the whole series.
# The text is downloaded at runtime and never shipped with the assignment.
# Offline alternative: put plaintext books in data/raw/*.txt and pass source="raw".

HF_DATASET = "WutYee/HarryPotter_books_1to7"
HF_SPLITS = ("train", "validation", "test")
N_SEGMENTS = 100          # the series is cut into 100 equal-length segments ...
N_HELDOUT = 5             # ... and 5 of them, spread through the middle 70%, are held out

# Scanning artefacts that appear as their own lines in some books.
_PAGE_RE = re.compile(r"^(P\s*a\s*g\s*e\s*\|?\s*\d+|page\s+\d+(\s+of\s+\d+)?|\d{1,4})$", re.IGNORECASE)
_RUNNING_HEADER_RE = re.compile(r"(^HP\s*\d+\s*[-–]|Harry Potter and the .*Rowling\s*$)", re.IGNORECASE)


@dataclass
class Corpus:
    raw_text: str            # cleaned, unwrapped text of all seven books
    segments: List[str]      # raw_text cut into N_SEGMENTS equal-length pieces
    source: str = "hf"


def _lines_from_hf():
    from datasets import load_dataset          # installed with transformer_lens
    ds = load_dataset(HF_DATASET)
    for split in HF_SPLITS:
        if split in ds:
            yield from ds[split]["text"]


def _lines_from_raw(raw_dir):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files in {raw_dir}/ (see data/README.md).")
    for p in paths:
        yield from open(p, encoding="utf-8", errors="ignore").read().splitlines()
        yield ""


def clean_lines(lines) -> str:
    """Drop page numbers / running headers, then unwrap: consecutive non-empty lines are
    joined with spaces; blank lines become paragraph breaks ('\\n')."""
    lines = [("" if l is None else str(l)).strip() for l in lines]
    counts = {}
    for s in lines:
        if s:
            counts[s] = counts.get(s, 0) + 1
    paras, cur = [], []
    for s in lines:
        frequent_header = counts.get(s, 0) >= 50 and len(s) >= 8 and s[-1] not in '.!?"\'”’'
        if not s:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        if _PAGE_RE.match(s) or _RUNNING_HEADER_RE.search(s) or frequent_header:
            continue
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return re.sub(r"[ \t]+", " ", "\n".join(paras)).strip()


def load_corpus(source: str = "hf", raw_dir: str = "data/raw") -> Corpus:
    """source="hf": download WutYee/HarryPotter_books_1to7 (cached by `datasets` after the
    first call). source="raw": read data/raw/*.txt instead (offline)."""
    lines = _lines_from_hf() if source == "hf" else _lines_from_raw(raw_dir)
    text = clean_lines(lines)
    if len(text) < N_SEGMENTS * 100:
        raise ValueError(f"corpus is suspiciously short ({len(text)} characters)")
    bounds = [round(i * len(text) / N_SEGMENTS) for i in range(N_SEGMENTS + 1)]
    segments = [text[bounds[i]:bounds[i + 1]] for i in range(N_SEGMENTS)]
    return Corpus(raw_text=text, segments=segments, source=source)


def heldout_segment_ids(n_segments: int = N_SEGMENTS, k: int = N_HELDOUT) -> List[int]:
    """k segments evenly spaced through the MIDDLE 70% of the series (skipping the first and
    last 15%) — A-3's 'held-out text from intermediate chapters', made independent of how
    each book in the dataset marks its chapters."""
    lo, hi = int(0.15 * n_segments), int(0.85 * n_segments)
    return [lo + round(i * (hi - 1 - lo) / (k - 1)) for i in range(k)]


def split_corpus(corpus: Corpus) -> Tuple[str, str, List[int]]:
    """Return (train_text, heldout_text, heldout_ids). Held-out segments never appear in any
    training set in this assignment."""
    held = heldout_segment_ids(len(corpus.segments))
    train = "\n".join(s for i, s in enumerate(corpus.segments) if i not in held)
    test = "\n".join(corpus.segments[i] for i in held)
    return train, test, held


# ======================================================================================
# 2. Character tokenizer (saved alongside every submission)
# ======================================================================================
# A-5 works at the CHARACTER level. With the tiny training subsets that grokking and
# double descent need, a BPE vocabulary would leave most token embeddings untrained;
# character accuracy is also a clean memorisation-vs-generalisation metric. The vocab is
# saved with each submission so autograders never need to rebuild it.

SEP_TOKEN = "<SEP>"


class CharTokenizer:
    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.sep_id = self.stoi.get(SEP_TOKEN)
        self.pad_id = self.stoi.get("\x00", 0)

    @classmethod
    def from_text(cls, text: str, add_sep: bool = True):
        itos = ["\x00"] + sorted(set(text) - {"\x00"})
        if add_sep:
            itos.append(SEP_TOKEN)
        return cls(itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] if self.itos[int(i)] != SEP_TOKEN else " | "
                       for i in ids if int(i) != self.pad_id)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"itos": self.itos}, open(path, "w"))

    @classmethod
    def load(cls, path: str):
        return cls(json.load(open(path))["itos"])


# ======================================================================================
# 3. Dataset builders
# ======================================================================================
def random_windows(ids: torch.Tensor, n: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """n random contiguous windows of `length` tokens from a 1-D id tensor -> (n, length)."""
    starts = torch.randperm(len(ids) - length - 1, generator=generator)[:n]
    return torch.stack([ids[s:s + length] for s in starts])


# ======================================================================================
# 4. Splits / checkpoints / metrics I/O
# ======================================================================================
def save_split(qdir: str, **tensors):
    os.makedirs(qdir, exist_ok=True)
    torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tensors.items()},
               os.path.join(qdir, "split.pt"))


def load_split(qdir: str) -> dict:
    return torch.load(os.path.join(qdir, "split.pt"), map_location="cpu", weights_only=False)


def save_decoder_model(qdir: str, model, name: str = "model.pt", extra: Optional[dict] = None,
                       architecture_source=None):
    """Save a student's decoder checkpoint and its reproducibility metadata.

    ``architecture_source`` may be the model file, a sequence containing that file and
    its local Python dependencies, or a directory containing those files. The file named
    ``model_arch.py`` is saved under that name; dependencies retain their basenames.
    """
    os.makedirs(qdir, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"kind": "decoder", "state_dict": sd, "config": asdict(model.cfg),
                "extra": extra or {}}, os.path.join(qdir, name))
    if architecture_source is not None:
        if isinstance(architecture_source, (str, os.PathLike)) and os.path.isdir(architecture_source):
            sources = [os.path.join(os.fspath(architecture_source), n)
                       for n in os.listdir(architecture_source) if n.endswith(".py")]
        elif isinstance(architecture_source, (str, os.PathLike)):
            sources = [os.fspath(architecture_source)]
        else:
            sources = [os.fspath(p) for p in architecture_source]
        sources = [os.path.abspath(p) for p in sources]
        missing = [p for p in sources if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("Architecture source file(s) do not exist: " + ", ".join(missing))
        arch = next((p for p in sources if os.path.basename(p) == "model_arch.py"), sources[0] if sources else None)
        if arch is None:
            raise ValueError("architecture_source must include model_arch.py or at least one Python file.")
        for source in sources:
            target = "model_arch.py" if source == arch else os.path.basename(source)
            destination = os.path.abspath(os.path.join(qdir, target))
            if source != destination:
                shutil.copy2(source, destination)


def load_decoder_model(path: str, model_arch_module):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg_payload = payload["config"]
        state_dict = payload["state_dict"]
    except Exception as exc:
        raise ValueError(
            f"Could not read decoder checkpoint {path!r}. Save Q1 checkpoints with "
            "common_a5.save_decoder_model so config and state_dict are stored together."
        ) from exc
    if not isinstance(cfg_payload, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {path!r} must contain dictionary fields 'config' and 'state_dict'.")
    for attr in ("LMConfig", "DecoderLM"):
        if not hasattr(model_arch_module, attr):
            raise TypeError(f"Student model_arch.py must define {attr} to load Q1 checkpoints.")
    try:
        cfg = model_arch_module.LMConfig(**cfg_payload)
        m = model_arch_module.DecoderLM(cfg)
        m.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint {path!r} does not match the submitted model_arch.py. Use the same "
            "architecture source and config for Q1 training and Q4–Q6."
        ) from exc
    return m.eval(), payload.get("extra", {})


def load_student_model_arch(q1_dir: str):
    """Load the exact model definition saved with a student's Q1 artifacts."""
    path = os.path.join(q1_dir, "model_arch.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Save/copy the architecture used for Q1 alongside the Q1 checkpoints."
        )
    # Resolve local imports (for example, A3's attention.py and rope.py) from this
    # checkpoint bundle, even when another module with the same name is installed.
    q1_dir = os.path.abspath(q1_dir)
    local_modules = {os.path.splitext(n)[0] for n in os.listdir(q1_dir)
                     if n.endswith(".py") and n != "model_arch.py"}
    old_modules = {n: sys.modules.pop(n) for n in local_modules if n in sys.modules}
    old_path = list(sys.path)
    sys.path.insert(0, q1_dir)
    try:
        spec = importlib.util.spec_from_file_location("student_q1_model_arch", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load student architecture module at {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Could not import {path}. Include every local Python dependency imported by "
            "model_arch.py (for example attention.py and rope.py) beside the Q1 checkpoints."
        ) from exc
    finally:
        sys.path[:] = old_path
        for n in local_modules:
            sys.modules.pop(n, None)
        sys.modules.update(old_modules)
    for attr in ("LMConfig", "DecoderLM"):
        if not hasattr(module, attr):
            raise TypeError(f"{path} must define {attr}.")
    return module


def save_metrics(qdir: str, metrics: dict):
    os.makedirs(qdir, exist_ok=True)
    json.dump(metrics, open(os.path.join(qdir, "metrics.json"), "w"), indent=2, default=_json_default)


def load_metrics(qdir: str) -> dict:
    return json.load(open(os.path.join(qdir, "metrics.json")))


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if torch.is_tensor(o): return o.detach().cpu().tolist()
    return str(o)


ANALYSIS_LOG_PATH = "analysis_log.json"


def log_result(key, value, path: str = ANALYSIS_LOG_PATH):
    log = json.load(open(path)) if os.path.exists(path) else {}
    log[key] = value
    json.dump(log, open(path, "w"), indent=2, default=_json_default)
    print(f"[logged] {key} = {value}")


# ======================================================================================
# 5. Harry-Potter safety preference data (Q3) — ships with the repo
# ======================================================================================
def load_preferences(path: str = "data/preferences.jsonl") -> List[Dict[str, str]]:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in recs:
        assert {"prompt", "chosen", "rejected"} <= set(r), f"bad record: {r}"
    return recs


def build_char_tokenizer_from_preferences(recs) -> CharTokenizer:
    corpus = "".join(r["prompt"] + " " + r["chosen"] + " " + r["rejected"] + " " for r in recs)
    return CharTokenizer.from_text(corpus, add_sep=False)


def make_pref_batch(recs, tokenizer: CharTokenizer, max_len: int = 192):
    """Padded (prompt + response) ids and a mask that is 1 on response tokens only."""
    def build(side):
        seqs, masks = [], []
        for r in recs:
            p = tokenizer.encode(r["prompt"] + " ")
            a = tokenizer.encode(r[side])
            ids = (p + a)[:max_len]
            m = ([0] * len(p) + [1] * len(a))[:max_len]
            pad = max_len - len(ids)
            seqs.append(ids + [tokenizer.pad_id] * pad)
            masks.append(m + [0] * pad)
        return torch.tensor(seqs), torch.tensor(masks, dtype=torch.float)
    ci, cm = build("chosen")
    ri, rm = build("rejected")
    return {"chosen_ids": ci, "chosen_mask": cm, "rejected_ids": ri, "rejected_mask": rm}
