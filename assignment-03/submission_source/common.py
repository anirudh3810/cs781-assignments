"""
common.py — shared utilities for Assignment A-3 (Decoder-only Transformer from scratch)

This module is imported by both `questions.ipynb` and `solutions.ipynb`, and by every
autograder script in this package. Keeping it in one place guarantees that students,
the reference solution, and the autograder all compute perplexity, count parameters,
and split train/held-out data in EXACTLY the same way — which is what makes autograding
on a laptop possible without retraining anything.

You should not need to modify this file for the assignment itself, but you are welcome
to read it end-to-end; understanding it is part of the assignment.
"""

import os
import re
import json
import glob
import math
import time
import random
import shutil
import importlib.util
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------------
# 0. Parameter / compute budget
# ----------------------------------------------------------------------------------
MAX_PARAMS = 50_000_000  # 50M parameter ceiling for this assignment


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_within_budget(model: nn.Module, max_params: int = MAX_PARAMS) -> int:
    n = count_parameters(model)
    assert n <= max_params, (
        f"Model has {n:,} parameters, which exceeds the {max_params:,} parameter budget "
        f"for this assignment. Shrink d_model, n_layers, or vocab_size."
    )
    return n


# ----------------------------------------------------------------------------------
# 1. Data loading: Harry Potter corpus
# ----------------------------------------------------------------------------------
# We deliberately do NOT ship the book text in this repository (copyright). The
# notebook setup uses download_data.py to prepare the course-selected external dataset;
# a user may instead supply compatible plaintext files under `data/raw/`.
#
# Expected layout:
#   data/raw/book1.txt
#   data/raw/book2.txt
#   ...
# (any filenames ending in .txt under data/raw/ will be picked up, sorted alphabetically
#  so that book order is preserved if you name them book1.txt, book2.txt, ...)

CHAPTER_HEADER_RE = re.compile(r"^\s*CHAPTER\s+[A-Z0-9\-]+\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Corpus:
    raw_text: str
    chapters: List[str]  # raw_text split by chapter boundaries (chapter 0 = front matter)


def load_corpus(raw_dir: str = "data/raw") -> Corpus:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {raw_dir}/. Run `python download_data.py` from the "
            f"assignment directory, or see data/README.md."
        )
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    full_text = "\n\n".join(texts)
    # Split on chapter headers, keeping the header text attached to the chapter that follows.
    splits = CHAPTER_HEADER_RE.split(full_text)
    headers = CHAPTER_HEADER_RE.findall(full_text)
    chapters = [splits[0]] + [h + "\n" + c for h, c in zip(headers, splits[1:])]
    chapters = [c.strip() for c in chapters if c.strip()]
    return Corpus(raw_text=full_text, chapters=chapters)


def held_out_sentences(
    corpus: Corpus,
    n_chapters: int = 5,
    sentences_per_chapter: int = 6,
    seed: int = SEED,
) -> List[str]:
    """
    Sample sentences from a handful of INTERMEDIATE chapters (skipping the first/last
    ~15% of chapters, which tend to be prologue/epilogue-ish and unusually short) to use
    as a held-out perplexity evaluation set. These sentences are excluded from training.
    """
    rng = random.Random(seed)
    n = len(corpus.chapters)
    lo, hi = int(n * 0.15), int(n * 0.85)
    candidate_idxs = list(range(lo, hi))
    rng.shuffle(candidate_idxs)
    chosen = candidate_idxs[:n_chapters]

    sent_split_re = re.compile(r"(?<=[.!?])\s+")
    held_out = []
    for idx in chosen:
        sents = [s.strip() for s in sent_split_re.split(corpus.chapters[idx]) if len(s.strip()) > 30]
        rng.shuffle(sents)
        held_out.extend(sents[:sentences_per_chapter])
    return held_out


def remove_sentences_from_text(text: str, sentences: List[str]) -> str:
    """Strip held-out sentences out of the training text so there's no leakage."""
    for s in sentences:
        text = text.replace(s, " ")
    return text


# ----------------------------------------------------------------------------------
# 2. Tokenizer: wraps the BPE tokenizer you built in A-1
# ----------------------------------------------------------------------------------
class BPETokenizerWrapper:
    """
    Thin wrapper so the rest of the assignment can use either:
      (a) a tokenizer retrained from the student's Assignment-1 implementation on the
          Assignment-3 training text, or
      (b) the course fallback ByteLevel BPE tokenizer.

    Any student tokenizer passed here must expose:
      - encode(str) -> List[int]
      - decode(List[int]) -> str
      - a visible vocabulary size via `.vocab_size` or `.get_vocab_size()`
    """

    def __init__(self, a1_tokenizer=None, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self._impl = a1_tokenizer
        self._hf_tok = None

        if a1_tokenizer is not None:
            inferred = _visible_vocab_size(a1_tokenizer)
            if inferred is not None:
                self.vocab_size = inferred

    @property
    def uses_a1_tokenizer(self) -> bool:
        return self._impl is not None

    def train_fallback(self, text: str):
        from tokenizers import ByteLevelBPETokenizer

        tok = ByteLevelBPETokenizer()
        tmp_path = "/tmp/_a3_corpus_for_bpe.txt"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        tok.train(
            files=[tmp_path],
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        self._hf_tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        if self._impl is not None:
            ids = self._impl.encode(text)
            if hasattr(ids, "ids"):
                ids = ids.ids
            return list(ids)
        assert self._hf_tok is not None, "Call train_fallback() first, or pass your retrained A-1 tokenizer."
        return self._hf_tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self._impl is not None:
            return self._impl.decode(list(ids))
        assert self._hf_tok is not None
        return self._hf_tok.decode(list(ids))


def _visible_vocab_size(tokenizer) -> Optional[int]:
    """Return a tokenizer vocabulary size when it is exposed through the agreed API."""
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value

    getter = getattr(tokenizer, "get_vocab_size", None)
    if callable(getter):
        try:
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
        except Exception:
            pass
    return None


def load_a1_tokenizer_or_fallback(
    train_text: str,
    vocab_size: int = 8000,
    module_path: str = "a1_tokenizer.py",
    verbose: bool = True,
):
    """
    Try the Assignment-1 tokenizer hook used by Assignment 3.

    Required student-facing interface:
        a1_tokenizer.py
            train_tokenizer(train_text: str, vocab_size: int) -> tokenizer

    The returned tokenizer must expose encode(), decode(), and a visible vocabulary size.
    If any part of that interface is missing or fails, the function prints the missing
    pieces and falls back to the course ByteLevel BPE so the rest of the notebook/grader
    can continue.

    Returns:
        (wrapped_tokenizer, used_student_tokenizer: bool, diagnostics: List[str])
    """
    diagnostics = []
    student_tok = None

    if not os.path.isfile(module_path):
        diagnostics.append(f"missing file: {module_path}")
    else:
        try:
            spec = importlib.util.spec_from_file_location("_student_a1_tokenizer", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            diagnostics.append(f"{module_path} could not be imported: {e!r}")
            module = None

        if module is not None:
            factory = getattr(module, "train_tokenizer", None)
            if not callable(factory):
                diagnostics.append("missing exposed function: train_tokenizer(train_text, vocab_size)")
            else:
                try:
                    student_tok = factory(train_text, vocab_size)
                except NotImplementedError:
                    diagnostics.append("train_tokenizer(...) is still NotImplemented")
                    student_tok = None
                except Exception as e:
                    diagnostics.append(f"train_tokenizer(...) failed: {e!r}")
                    student_tok = None

    if student_tok is not None:
        if not callable(getattr(student_tok, "encode", None)):
            diagnostics.append("returned tokenizer is missing callable encode(text)")
        if not callable(getattr(student_tok, "decode", None)):
            diagnostics.append("returned tokenizer is missing callable decode(ids)")
        if _visible_vocab_size(student_tok) is None:
            diagnostics.append("returned tokenizer does not expose vocab_size or get_vocab_size()")

        if not diagnostics:
            try:
                probe_ids = student_tok.encode("A3 tokenizer interface check.")
                if hasattr(probe_ids, "ids"):
                    probe_ids = probe_ids.ids
                probe_ids = list(probe_ids)
                if not all(isinstance(i, int) for i in probe_ids):
                    raise TypeError("encode() did not return integer token ids")
                student_tok.decode(probe_ids)
            except Exception as e:
                diagnostics.append(f"encode/decode interface check failed: {e!r}")

    if not diagnostics and student_tok is not None:
        wrapped = BPETokenizerWrapper(a1_tokenizer=student_tok, vocab_size=vocab_size)
        if verbose:
            print(f"[tokenizer] using retrained A-1 tokenizer from {module_path}")
        return wrapped, True, []

    if verbose:
        print("[tokenizer] A-1 tokenizer interface unavailable; falling back to course BPE.")
        print("  Required exposed interface:")
        print("  - a1_tokenizer.py::train_tokenizer(train_text, vocab_size)")
        print("  - returned tokenizer.encode(text)")
        print("  - returned tokenizer.decode(ids)")
        print("  - returned tokenizer.vocab_size or tokenizer.get_vocab_size()")
        print("  Problems detected:")
        for item in diagnostics:
            print(f"  - {item}")

    fallback = BPETokenizerWrapper(vocab_size=vocab_size)
    fallback.train_fallback(train_text)
    return fallback, False, diagnostics


# ----------------------------------------------------------------------------------
# 3. Dataset
# ----------------------------------------------------------------------------------
class TokenDataset(torch.utils.data.Dataset):
    """Fixed-length chunks of a tokenized corpus for next-token-prediction training."""

    def __init__(self, token_ids: List[int], seq_len: int):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, i):
        start = i * self.seq_len
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + self.seq_len + 1]
        return x, y


# ----------------------------------------------------------------------------------
# 4. Perplexity
# ----------------------------------------------------------------------------------
@torch.no_grad()
def perplexity_of_texts(model, tokenizer: BPETokenizerWrapper, texts: List[str], device="cpu", max_len=256) -> float:
    """
    Compute the course-defined held-out perplexity over a list of text samples.

    Assignment 3 uses a sentence-level evaluation convention. Use this shared helper
    everywhere rather than substituting a different dataset-level aggregation, so every
    experiment is comparable.
    """
    model.eval()
    cfg = getattr(model, "cfg", getattr(model, "config", None))
    model_max_seq_len = (
        cfg.get("max_seq_len") if isinstance(cfg, dict)
        else getattr(cfg, "max_seq_len", None)
    )
    token_limit = min(max_len, model_max_seq_len + 1) if model_max_seq_len else max_len

    sentence_ppls = []
    for text in texts:
        ids = tokenizer.encode(text)[:token_limit]
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="mean",
        )
        sentence_ppls.append(math.exp(loss.item()))

    if not sentence_ppls:
        return float("inf")
    return sum(sentence_ppls) / len(sentence_ppls)


# ----------------------------------------------------------------------------------
# 5. The interface every student model MUST implement (this is what the autograder relies on)
# ----------------------------------------------------------------------------------
class LMInterface:
    """
    Not meant to be subclassed directly — this documents the contract the autograder
    expects from `model` objects saved by students:

      model(idx: LongTensor[B, T]) -> logits: FloatTensor[B, T, vocab_size]
      model.config: an object/dict with at least {d_model, n_layers, n_heads, n_kv_heads,
                     vocab_size, max_seq_len, pos_emb_type in {"sinusoidal","rope"}}

    Students should save both a checkpoint (state_dict) AND a small `config.json`
    describing the architecture, plus the exact class definition file, so the autograder
    can reconstruct the model and run a forward pass without retraining.
    """
    pass


def save_submission(out_dir: str, model: nn.Module, config: dict, metrics: dict, tokenizer_vocab_size: int):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    config = dict(config)
    config["vocab_size"] = tokenizer_vocab_size
    config["param_count"] = count_parameters(model)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Preserve the exact source version used to build this checkpoint. Later questions
    # overwrite several root-level modules, so grading only those final files could load
    # an earlier checkpoint against the wrong architecture.
    copied_sources = []
    for filename in ("a1_tokenizer.py", "attention.py", "model_arch.py", "rope.py", "optim_muon.py", "model_builtin.py"):
        if os.path.isfile(filename):
            shutil.copy2(filename, os.path.join(out_dir, filename))
            copied_sources.append(filename)

    print(
        f"Saved submission artifacts to {out_dir}/ "
        f"(model.pt, config.json, metrics.json; sources: {', '.join(copied_sources)})"
    )
