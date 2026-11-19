from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import time
import zipfile
from collections import Counter
from getpass import getpass
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset


nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


LANGS = ("en", "hi")
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
SPECIAL_SET = set(SPECIALS)

SHARED_SEED = 42
SHARED_N_PAIRS = None
MIN_TOKENS = 3
SHARED_MIN_COUNT = 2

EN_CACHE = "shared_english_corpus.csv"
HI_CACHE = "shared_hindi_corpus.csv"

EMBED_DIM = 100
DEFAULT_Q4_N = 4
DEFAULT_Q4_HIDDEN = 100
DEFAULT_Q4_EPOCHS = 30
DEFAULT_Q4_BATCH_SIZE = 512
DEFAULT_Q4_LR = 1e-3
DEFAULT_Q4_WEIGHT_DECAY = 1e-4
DEFAULT_Q4_PATIENCE = 3
DEFAULT_DIRECT = False

STAGE1_BUNDLE = "q4_stage1_bundle.zip"
STAGE2_BUNDLE = "q4_stage2_bundle.zip"
FINAL_BUNDLE = "question_4_bundle.zip"
CHECKPOINT_EVERY = 5


def build_default_config():
    return {
        "n": DEFAULT_Q4_N,
        "m": EMBED_DIM,
        "h": DEFAULT_Q4_HIDDEN,
        "epochs": DEFAULT_Q4_EPOCHS,
        "batch_size": DEFAULT_Q4_BATCH_SIZE,
        "lr": DEFAULT_Q4_LR,
        "weight_decay": DEFAULT_Q4_WEIGHT_DECAY,
        "patience": DEFAULT_Q4_PATIENCE,
        "min_freq": SHARED_MIN_COUNT,
        "direct": DEFAULT_DIRECT,
        "max_pairs": None,
        "preferred_q3_method": {"en": None, "hi": None},
        "hf_token": os.environ.get("HF_TOKEN", "").strip(),
    }


def language_label(lang: str) -> str:
    return "English" if lang == "en" else "Hindi"


def set_shared_seed(seed: int = SHARED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def shared_tokenize(text):
    return nltk.word_tokenize(str(text).lower())


def get_hf_token(config: dict | None = None) -> str:
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token

    if config and config.get("hf_token"):
        token = str(config["hf_token"]).strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    token = getpass("Enter Hugging Face token: ").strip()
    if not token:
        raise ValueError("A Hugging Face token is required to download the assignment dataset.")
    os.environ["HF_TOKEN"] = token
    return token


def prime_hf_token(config: dict | None = None):
    if Path(EN_CACHE).exists() and Path(HI_CACHE).exists():
        return
    token = get_hf_token(config)
    os.environ["HF_TOKEN"] = token


def load_shared_corpus(
    n_pairs=SHARED_N_PAIRS,
    seed: int = SHARED_SEED,
    en_cache: str = EN_CACHE,
    hi_cache: str = HI_CACHE,
    config: dict | None = None,
):
    if os.path.exists(en_cache) and os.path.exists(hi_cache):
        en_df = pd.read_csv(en_cache, keep_default_na=False)
        hi_df = pd.read_csv(hi_cache, keep_default_na=False)
        print(f"Loaded English corpus from {en_cache}")
        print(f"Loaded Hindi corpus from {hi_cache}")
        return en_df, hi_df

    ds = load_dataset(
        "Exploration-Lab/CS781",
        data_files={"train": "Assignment-1/train.parquet"},
        split="train",
        token=get_hf_token(config),
    )

    translations = ds["translation"]
    en_df = pd.DataFrame({"text": [row["en"] for row in translations]})
    hi_df = pd.DataFrame({"text": [row["hi"] for row in translations]})

    en_df = en_df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    hi_df = hi_df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

    en_df = en_df[en_df["text"].apply(lambda x: len(shared_tokenize(x)) >= MIN_TOKENS)]
    hi_df = hi_df[hi_df["text"].apply(lambda x: len(shared_tokenize(x)) >= MIN_TOKENS)]

    en_order = np.random.RandomState(seed).permutation(len(en_df))
    hi_order = np.random.RandomState(seed).permutation(len(hi_df))
    en_df = en_df.iloc[en_order].reset_index(drop=True)
    hi_df = hi_df.iloc[hi_order].reset_index(drop=True)

    if n_pairs is not None:
        en_df = en_df.iloc[:n_pairs].reset_index(drop=True)
        hi_df = hi_df.iloc[:n_pairs].reset_index(drop=True)

    en_df.to_csv(en_cache, index=False)
    hi_df.to_csv(hi_cache, index=False)
    print(f"Wrote {len(en_df):,} English sentences to {en_cache}")
    print(f"Wrote {len(hi_df):,} Hindi sentences to {hi_cache}")
    return en_df, hi_df


def load_language_texts(lang: str, config: dict) -> list[str]:
    set_shared_seed()
    en_df, hi_df = load_shared_corpus(config=config)
    texts = en_df["text"].tolist() if lang == "en" else hi_df["text"].tolist()
    max_pairs = config.get("max_pairs")
    return texts[:max_pairs] if max_pairs is not None else texts


def tokenize(text):
    return shared_tokenize(text)


def split_sentences(sentences, ratios=(0.8, 0.1, 0.1)):
    total = len(sentences)
    train_end = int(total * ratios[0])
    valid_end = train_end + int(total * ratios[1])
    train = sentences[:train_end]
    valid = sentences[train_end:valid_end]
    test = sentences[valid_end:]
    return train, valid, test


def build_vocabulary(sentences, min_freq=SHARED_MIN_COUNT):
    counter = Counter(token for sent in sentences for token in sent)
    kept = [tok for tok, freq in counter.items() if freq >= min_freq and tok not in SPECIAL_SET]
    kept.sort(key=lambda tok: (-counter[tok], tok))
    itos = SPECIALS + kept
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return stoi, itos, counter


def unk_rate(sentences, stoi):
    total = sum(len(sent) for sent in sentences)
    if total == 0:
        return 0.0
    unknown = sum(1 for sent in sentences for token in sent if token not in stoi)
    return unknown / total


def encode_sentence(sentence, stoi):
    unk_id = stoi["<unk>"]
    return [stoi.get(token, unk_id) for token in sentence]


def map_sentence_to_vocab(sentence, stoi):
    return [token if token in stoi else "<unk>" for token in sentence]


def make_windows(sentences, stoi, n):
    bos = [stoi["<s>"]] * (n - 1)
    eos = stoi["</s>"]
    xs = []
    ys = []
    for sentence in sentences:
        ids = bos + encode_sentence(sentence, stoi) + [eos]
        for end in range(n - 1, len(ids)):
            gram = ids[end - (n - 1): end + 1]
            xs.append(gram[:-1])
            ys.append(gram[-1])

    if not xs:
        return (
            np.empty((0, n - 1), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def padded_token_sentences(sentences, stoi, n):
    bos = ["<s>"] * (n - 1)
    eos = ["</s>"]
    return [bos + map_sentence_to_vocab(sent, stoi) + eos for sent in sentences]


def prepare_language_resources(lang: str, config: dict):
    texts = load_language_texts(lang, config)
    tokenized = [tokenize(text) for text in texts if str(text).strip()]
    train_sentences, valid_sentences, test_sentences = split_sentences(tokenized)
    stoi, itos, counter = build_vocabulary(train_sentences, min_freq=config["min_freq"])

    n = config["n"]
    Xtr, Ytr = make_windows(train_sentences, stoi, n=n)
    Xva, Yva = make_windows(valid_sentences, stoi, n=n)
    Xte, Yte = make_windows(test_sentences, stoi, n=n)

    return {
        "texts": texts,
        "tokenized": tokenized,
        "train_sentences": train_sentences,
        "valid_sentences": valid_sentences,
        "test_sentences": test_sentences,
        "stoi": stoi,
        "itos": itos,
        "counter": counter,
        "Xtr": Xtr,
        "Ytr": Ytr,
        "Xva": Xva,
        "Yva": Yva,
        "Xte": Xte,
        "Yte": Yte,
        "unk_rates": {
            "train": unk_rate(train_sentences, stoi),
            "valid": unk_rate(valid_sentences, stoi),
            "test": unk_rate(test_sentences, stoi),
        },
        "split_sizes": {
            "sentences_train": len(train_sentences),
            "sentences_valid": len(valid_sentences),
            "sentences_test": len(test_sentences),
        },
        "window_counts": {
            "train": int(len(Ytr)),
            "valid": int(len(Yva)),
            "test": int(len(Yte)),
        },
    }


def resolve_device(device_id: int | None):
    if device_id is not None and device_id >= 0 and torch.cuda.is_available():
        usable = min(device_id, torch.cuda.device_count() - 1)
        torch.cuda.set_device(usable)
        device = torch.device(f"cuda:{usable}")
        print(f"Using {device} -> {torch.cuda.get_device_name(usable)}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"Using {device} -> {torch.cuda.get_device_name(0)}")
        return device

    print("CUDA not available, using CPU.")
    return torch.device("cpu")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class NNLM(nn.Module):
    def __init__(self, V, m=100, h=100, n=4, direct=False):
        super().__init__()
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.direct = direct
        self.context_dim = (n - 1) * m

        self.C = nn.Embedding(V, m)
        self.H = nn.Linear(self.context_dim, h, bias=False)
        self.d = nn.Parameter(torch.zeros(h))
        self.U = nn.Linear(h, V, bias=False)
        self.W = nn.Linear(self.context_dim, V, bias=False) if direct else None
        self.b = nn.Parameter(torch.zeros(V))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.C.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.H.weight)
        nn.init.xavier_uniform_(self.U.weight)
        if self.W is not None:
            nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.d)
        nn.init.zeros_(self.b)

    def forward(self, context):
        x = self.C(context).reshape(context.size(0), -1)
        hidden_pre = self.H(x) + self.d
        hidden = torch.tanh(hidden_pre)
        hidden_term = self.U(hidden)
        direct_term = self.W(x) if self.W is not None else 0.0
        return self.b + direct_term + hidden_term

    def embeddings(self):
        return self.C.weight.detach().cpu().numpy().astype(np.float32).copy()


def batches(X, Y, batch_size, device, shuffle=True):
    size = len(Y)
    if size == 0:
        return
    indices = np.random.permutation(size) if shuffle else np.arange(size)
    for start in range(0, size, batch_size):
        sl = indices[start:start + batch_size]
        xb = torch.from_numpy(X[sl]).long().to(device, non_blocking=True)
        yb = torch.from_numpy(Y[sl]).long().to(device, non_blocking=True)
        yield xb, yb


@torch.no_grad()
def mean_nll(model, X, Y, device, batch_size=2048):
    model.eval()
    total_nll = 0.0
    total_items = 0
    for xb, yb in batches(X, Y, batch_size=batch_size, device=device, shuffle=False):
        logits = model(xb)
        loss_sum = F.cross_entropy(logits, yb, reduction="sum")
        total_nll += float(loss_sum.detach().cpu())
        total_items += int(yb.size(0))
    if total_items == 0:
        return float("inf")
    return total_nll / total_items


@torch.no_grad()
def perplexity(model, X, Y, device, batch_size=2048):
    avg_nll = mean_nll(model, X, Y, device=device, batch_size=batch_size)
    return float(np.exp(min(avg_nll, 50.0)))


def train_model(model, Xtr, Ytr, Xva, Yva, device, config, verbose=True, checkpoint_lang=None, stoi=None, itos=None):
    decay_params = [model.C.weight, model.H.weight, model.U.weight]
    if model.W is not None:
        decay_params.append(model.W.weight)

    no_decay_params = [model.d, model.b]
    optimizer = optim.Adam(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["lr"],
    )

    history = {
        "train_loss": [],
        "train_ppl": [],
        "valid_loss": [],
        "valid_ppl": [],
        "epoch_seconds": [],
        "elapsed_seconds": 0.0,
        "best_epoch": 0,
        "best_valid_ppl": float("inf"),
        "converged": False,
        "stopped_epoch": 0,
    }

    start_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    if checkpoint_lang is not None and stoi is not None and itos is not None:
        start_epoch, resume_state = load_resume_checkpoint(
            checkpoint_lang, model, optimizer, device, stoi, itos, config
        )
        if resume_state is not None:
            history = resume_state["history"]
            history.setdefault("epoch_seconds", [])
            history.setdefault("elapsed_seconds", 0.0)
            best_state = resume_state["best_state_dict"]
            history["best_valid_ppl"] = resume_state.get("best_valid_ppl", history["best_valid_ppl"])
            stale_epochs = int(resume_state.get("stale_epochs", 0))
            print(
                f"[checkpoint] resumed {checkpoint_lang} from epoch {start_epoch} "
                f"(best valid ppl {history['best_valid_ppl']:.2f}, "
                f"logged time {format_duration(history['elapsed_seconds'])})",
                flush=True,
            )

    total_batches = max(1, math.ceil(len(Ytr) / max(config["batch_size"], 1)))
    if verbose:
        print(
            f"Training setup | epochs={config['epochs']} | batch_size={config['batch_size']} | "
            f"train_windows={len(Ytr):,} | valid_windows={len(Yva):,} | batches/epoch={total_batches}",
            flush=True,
        )

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        epoch_start = time.time()
        progress_step = max(1, total_batches // 20)
        model.train()
        total_nll = 0.0
        total_items = 0
        for batch_idx, (xb, yb) in enumerate(
            batches(Xtr, Ytr, batch_size=config["batch_size"], device=device, shuffle=True),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="mean")
            loss.backward()
            optimizer.step()

            batch_items = int(yb.size(0))
            total_items += batch_items
            total_nll += float(loss.detach().cpu()) * batch_items

            if verbose and (batch_idx == 1 or batch_idx % progress_step == 0 or batch_idx == total_batches):
                batch_elapsed = time.time() - epoch_start
                epoch_eta = (total_batches - batch_idx) * (batch_elapsed / max(batch_idx, 1))
                overall_fraction = ((epoch - 1) + (batch_idx / total_batches)) / max(config["epochs"], 1)
                overall_elapsed = history["elapsed_seconds"] + batch_elapsed
                project_eta = (
                    (overall_elapsed / overall_fraction) - overall_elapsed
                    if overall_fraction > 0
                    else float("inf")
                )
                project_eta_text = format_duration(project_eta) if math.isfinite(project_eta) else "--:--:--"
                print(
                    f"[progress] epoch {epoch:02d}/{config['epochs']} | "
                    f"batch {batch_idx:04d}/{total_batches} ({100.0 * batch_idx / total_batches:5.1f}%) | "
                    f"epoch elapsed {format_duration(batch_elapsed)} | "
                    f"epoch ETA {format_duration(epoch_eta)} | "
                    f"total elapsed {format_duration(overall_elapsed)} | "
                    f"project ETA {project_eta_text}",
                    flush=True,
                )

        train_loss = total_nll / max(total_items, 1)
        valid_loss = mean_nll(model, Xva, Yva, device=device, batch_size=config["batch_size"])
        train_ppl = float(np.exp(min(train_loss, 50.0)))
        valid_ppl = float(np.exp(min(valid_loss, 50.0)))
        epoch_seconds = time.time() - epoch_start
        history["epoch_seconds"].append(epoch_seconds)
        history["elapsed_seconds"] += epoch_seconds

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["valid_loss"].append(valid_loss)
        history["valid_ppl"].append(valid_ppl)

        if verbose:
            avg_epoch_seconds = history["elapsed_seconds"] / max(len(history["epoch_seconds"]), 1)
            remaining_epochs = max(config["epochs"] - epoch, 0)
            print(
                f"Epoch {epoch:02d} | train loss={train_loss:.4f} | train ppl={train_ppl:.2f} | "
                f"valid loss={valid_loss:.4f} | valid ppl={valid_ppl:.2f} | "
                f"best epoch={history['best_epoch'] or epoch} | "
                f"epoch time={format_duration(epoch_seconds)} | "
                f"elapsed={format_duration(history['elapsed_seconds'])} | "
                f"remaining~={format_duration(remaining_epochs * avg_epoch_seconds)}"
            )

        if valid_ppl + 1e-8 < history["best_valid_ppl"]:
            history["best_valid_ppl"] = valid_ppl
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                history["converged"] = True
                history["stopped_epoch"] = epoch
                if checkpoint_lang is not None and stoi is not None and itos is not None:
                    save_resume_checkpoint(
                        checkpoint_lang,
                        epoch,
                        model,
                        optimizer,
                        history,
                        best_state,
                        history["best_valid_ppl"],
                        stale_epochs,
                        config,
                        stoi,
                        itos,
                    )
                break

        if checkpoint_lang is not None and stoi is not None and itos is not None:
            if epoch % CHECKPOINT_EVERY == 0 or epoch == config["epochs"]:
                save_resume_checkpoint(
                    checkpoint_lang,
                    epoch,
                    model,
                    optimizer,
                    history,
                    best_state,
                    history["best_valid_ppl"],
                    stale_epochs,
                    config,
                    stoi,
                    itos,
                )

    if not history["stopped_epoch"]:
        history["stopped_epoch"] = len(history["train_ppl"])

    model.load_state_dict(best_state)
    return history


def save_figure(fig, path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_convergence(history, lang, save_path):
    epochs = np.arange(1, len(history["train_ppl"]) + 1)
    best_epoch = history["best_epoch"]
    status = "converged" if history["converged"] else "max-epochs reached"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="valid loss")
    axes[0].axvline(best_epoch, color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_title(f"{language_label(lang)} | Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_ppl"], marker="o", label="train ppl")
    axes[1].plot(epochs, history["valid_ppl"], marker="o", label="valid ppl")
    axes[1].axvline(best_epoch, color="tab:red", linestyle="--", alpha=0.7)
    axes[1].set_title(f"{language_label(lang)} | Perplexity ({status})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    return save_figure(fig, save_path)


def count_parameters(V, m, h, n, direct=False):
    context_dim = (n - 1) * m
    counts = {
        "C": V * m,
        "H": h * context_dim,
        "d": h,
        "U": V * h,
        "W": V * context_dim if direct else 0,
        "b": V,
    }
    counts["TOTAL"] = sum(counts.values())
    counts["QUESTION_FORMULA_IF_DIRECT"] = V * (1 + n * m + h) + h * (1 + context_dim)
    counts["FORMULA_WITH_CURRENT_SETTING"] = (
        V * (1 + m + h + (context_dim if direct else 0)) + h * (1 + context_dim)
    )
    return counts


def generate_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class Q2_NGramLM:
    def __init__(self, n, V):
        self.n = n
        self.V = V
        self.ngram_counts = {order: Counter() for order in range(1, n + 1)}
        self.context_counts = {order: Counter() for order in range(1, n)}
        self.total_unigrams = 0

    def fit(self, padded_sentences):
        for sent in padded_sentences:
            for full_ngram in generate_ngrams(sent, self.n):
                for order in range(1, self.n + 1):
                    gram = tuple(full_ngram[-order:])
                    self.ngram_counts[order][gram] += 1

        self.total_unigrams = sum(self.ngram_counts[1].values())
        for order in range(2, self.n + 1):
            for gram, count in self.ngram_counts[order].items():
                self.context_counts[order - 1][gram[:-1]] += count
        return self

    def prob(self, ctx, w, mode="interp", alpha=1.0):
        ctx = list(ctx)[-(self.n - 1):]
        w = str(w)
        if mode == "laplace":
            top_ctx = tuple(ctx[-(self.n - 1):])
            gram = top_ctx + (w,)
            numerator = self.ngram_counts[self.n].get(gram, 0) + alpha
            denominator = self.context_counts[self.n - 1].get(top_ctx, 0) + alpha * self.V
            return float(numerator / denominator)

        pieces = [1.0 / self.V]
        unigram_prob = (
            self.ngram_counts[1].get((w,), 0) / self.total_unigrams
            if self.total_unigrams
            else 1.0 / self.V
        )
        pieces.append(unigram_prob)
        for order in range(2, self.n + 1):
            local_ctx = tuple(ctx[-(order - 1):])
            gram = local_ctx + (w,)
            denom = self.context_counts[order - 1].get(local_ctx, 0)
            prob = self.ngram_counts[order].get(gram, 0) / denom if denom else 0.0
            pieces.append(prob)
        return float(sum(pieces) / len(pieces))

    def token_probs(self, padded_sentences, mode="interp"):
        probs = []
        for sent in padded_sentences:
            for full_ngram in generate_ngrams(sent, self.n):
                ctx = full_ngram[:-1]
                w = full_ngram[-1]
                probs.append(self.prob(ctx, w, mode=mode))
        return np.asarray(probs, dtype=np.float64)

    def perplexity(self, padded_sentences, mode="interp"):
        return ppl_from_probs(self.token_probs(padded_sentences, mode=mode))

    def distinct_ngrams_total(self):
        return sum(len(counter) for counter in self.ngram_counts.values())


def ppl_from_probs(p):
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return float("inf")
    p = np.clip(p, 1e-12, 1.0)
    return float(np.exp(-np.mean(np.log(p))))


@torch.no_grad()
def nnlm_token_probs(model, X, Y, device, batch_size=2048):
    model.eval()
    parts = []
    for xb, yb in batches(X, Y, batch_size=batch_size, device=device, shuffle=False):
        logits = model(xb)
        log_probs = F.log_softmax(logits, dim=1)
        chosen = log_probs.gather(1, yb.unsqueeze(1)).squeeze(1).exp()
        parts.append(chosen.detach().cpu().numpy())
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts).astype(np.float64)


def mixture_report_from_probs(nnlm_probs, ngram_probs, lam=0.5):
    nnlm_probs = np.asarray(nnlm_probs, dtype=np.float64)
    ngram_probs = np.asarray(ngram_probs, dtype=np.float64)
    assert len(nnlm_probs) == len(ngram_probs), "Probability arrays must align token-by-token."

    grid_rows = []
    best_lambda = None
    best_ppl = float("inf")
    for candidate in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        mix = candidate * nnlm_probs + (1.0 - candidate) * ngram_probs
        ppl = ppl_from_probs(mix)
        grid_rows.append({"lambda": float(candidate), "mixture_ppl": float(ppl)})
        if ppl < best_ppl:
            best_lambda = float(candidate)
            best_ppl = float(ppl)

    lam_mix = lam * nnlm_probs + (1.0 - lam) * ngram_probs
    return {
        "nnlm_ppl": ppl_from_probs(nnlm_probs),
        "interp_ngram_ppl": ppl_from_probs(ngram_probs),
        "mixture_ppl_at_lambda": ppl_from_probs(lam_mix),
        "best_lambda": best_lambda,
        "best_mixture_ppl": best_ppl,
        "grid": pd.DataFrame(grid_rows),
    }


def sparsity_table(Xtr, Ytr, Xte, Yte):
    rows = []
    max_order = Xtr.shape[1] + 1
    test_tokens = int(len(Yte))
    for order in range(2, max_order + 1):
        ctx_len = order - 1
        train_contexts = {tuple(row[:ctx_len].tolist()) for row in Xtr}
        train_ngrams = {
            tuple(list(row[:ctx_len]) + [int(y)])
            for row, y in zip(Xtr, Ytr)
        }
        unseen_context = 0
        unseen_ngram = 0
        for row, y in zip(Xte, Yte):
            ctx = tuple(row[:ctx_len].tolist())
            gram = ctx + (int(y),)
            if ctx not in train_contexts:
                unseen_context += 1
            if gram not in train_ngrams:
                unseen_ngram += 1
        rows.append(
            {
                "n": order,
                "train_contexts_seen": len(train_contexts),
                "train_ngrams_seen": len(train_ngrams),
                "test_tokens": test_tokens,
                "pct_unseen_context": 100.0 * unseen_context / max(test_tokens, 1),
                "pct_unseen_ngram": 100.0 * unseen_ngram / max(test_tokens, 1),
            }
        )
    return pd.DataFrame(rows)


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def cosine(u, v):
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(u, v) / denom)


def is_content_token(token):
    if token in SPECIAL_SET:
        return False
    if token == "\u0964":
        return False
    return re.search(r"\w", token, flags=re.UNICODE) is not None


def pick_words(itos, start=50, stop=1500, k=18):
    candidates = [tok for tok in itos if is_content_token(tok)]
    if not candidates:
        return []
    upper = min(stop, len(candidates))
    lower = min(start, upper - 1) if upper > 0 else 0
    pool = candidates[lower:upper] if upper > lower else candidates
    if len(pool) <= k:
        return pool
    indices = np.linspace(0, len(pool) - 1, num=k, dtype=int)
    return [pool[idx] for idx in indices]


def plot_pca(E, stoi, words, title, save_path):
    labels = [word for word in words if word in stoi]
    if len(labels) < 2:
        print(f"Skipping PCA for {title}: need at least 2 in-vocabulary words.")
        return None
    vectors = np.stack([E[stoi[word]] for word in labels], axis=0)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.8, color="tab:blue")
    for idx, word in enumerate(labels):
        ax.text(coords[idx, 0], coords[idx, 1], word, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return save_figure(fig, save_path)


def load_word2vec(path):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    raw_itos = data.get("itos", data.get("vocab"))
    if isinstance(raw_itos, dict):
        itos = [raw_itos[idx] for idx in range(len(raw_itos))]
    else:
        itos = list(raw_itos)
    return {
        "path": str(path),
        "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        "stoi": data["stoi"],
        "itos": itos,
    }


def nearest_neighbours(word, E, stoi, itos, k=10):
    word = str(word).lower()
    if word not in stoi:
        return []
    norm = normalize_rows(E)
    sims = norm @ norm[stoi[word]]
    sims[stoi[word]] = -np.inf
    for special in SPECIALS:
        if special in stoi:
            sims[stoi[special]] = -np.inf
    ordered = np.argsort(-sims)
    out = []
    for idx in ordered:
        token = itos[int(idx)]
        if not is_content_token(token):
            continue
        out.append((token, float(sims[int(idx)])))
        if len(out) == k:
            break
    return out


def same_word_similarity(E_a, stoi_a, E_b, stoi_b, words):
    scores = []
    for word in words:
        if word in stoi_a and word in stoi_b:
            scores.append(cosine(E_a[stoi_a[word]], E_b[stoi_b[word]]))
    return float(np.mean(scores)) if scores else None


def load_analogy_set(lang, config):
    split = load_dataset(
        "Exploration-Lab/CS781",
        data_files={lang: f"Assignment-1/analogy_test_{lang}.parquet"},
        split=lang,
        token=get_hf_token(config),
    )
    df = split.to_pandas()
    if {"word1", "word2", "word3"}.issubset(df.columns):
        return df[["word1", "word2", "word3"]].copy()
    cols = list(df.columns[:3])
    renamed = df[cols].copy()
    renamed.columns = ["word1", "word2", "word3"]
    return renamed


def embedding_analogy_predict(E, E_norm, stoi, itos, w1, w2, w3):
    words = [str(w1).lower(), str(w2).lower(), str(w3).lower()]
    if any(word not in stoi for word in words):
        return "OOV"
    idx1, idx2, idx3 = (stoi[word] for word in words)
    query = E[idx1] - E[idx2] + E[idx3]
    qnorm = np.linalg.norm(query)
    if qnorm <= 1e-12:
        return "OOV"
    sims = E_norm @ (query / qnorm)
    for idx in (idx1, idx2, idx3):
        sims[idx] = -np.inf
    for special in SPECIALS:
        if special in stoi:
            sims[stoi[special]] = -np.inf
    best = int(np.argmax(sims))
    return itos[best] if np.isfinite(sims[best]) else "OOV"


def extract_bundle_if_present(bundle_name):
    bundle = Path(bundle_name)
    if bundle.exists():
        with zipfile.ZipFile(bundle, "r") as zf:
            zf.extractall(".")
        print(f"Extracted {bundle_name}")


def stage_kaggle_input_files(required_names):
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return

    for name in required_names:
        local_path = Path(name)
        if local_path.exists():
            continue

        matches = list(input_root.rglob(name))
        if not matches:
            continue

        source = matches[0]
        shutil.copy2(source, local_path)
        print(f"Copied input file {source} -> {local_path}")


def train_files(lang):
    return {
        "model": Path(f"nnlm_{lang}.pt"),
        "checkpoint": Path(f"checkpoint_nnlm_{lang}.pt"),
        "state": Path(f"q4_train_state_{lang}.pkl"),
        "summary": Path(f"q4_train_summary_{lang}.json"),
        "plot": Path(f"q4_convergence_{lang}.png"),
        "bundle": Path(f"q4_{lang}_artifacts.zip"),
    }


def metrics_files(lang):
    return {
        "perplexity": Path(f"q4_perplexity_table_{lang}.csv"),
        "parameter": Path(f"q4_parameter_table_{lang}.csv"),
        "sparsity": Path(f"q4_sparsity_table_{lang}.csv"),
        "mixture": Path(f"q4_mixture_grid_{lang}.csv"),
        "summary": Path(f"q4_metrics_summary_{lang}.json"),
    }


def analysis_files(lang):
    return {
        "analogy": Path(f"analogy_results_nnlm_{lang}.csv"),
        "pca": Path(f"q4_embedding_pca_{lang}.png"),
        "word2vec_compare": Path(f"q4_word2vec_compare_{lang}.csv"),
        "word2vec_proxy": Path(f"q4_word2vec_proxy_{lang}.csv"),
        "embed_compare_plot": Path(f"q4_embedding_compare_{lang}.png"),
        "written_answers": Path(f"Q4_WRITTEN_ANSWERS_{lang}.md"),
        "summary": Path(f"q4_analysis_summary_{lang}.json"),
    }


def json_ready(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def save_json(path, data):
    Path(path).write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=False), encoding="utf-8")


def save_model_checkpoint(lang, model, config, stoi, itos):
    files = train_files(lang)
    cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(
        {
            "state_dict": cpu_state,
            "config": {
                "V": len(itos),
                "m": config["m"],
                "h": config["h"],
                "n": config["n"],
                "direct": config["direct"],
            },
            "stoi": stoi,
            "itos": itos,
        },
        files["model"],
    )


def save_resume_checkpoint(
    lang,
    epoch,
    model,
    optimizer,
    history,
    best_state,
    best_valid_ppl,
    stale_epochs,
    config,
    stoi,
    itos,
):
    checkpoint_path = train_files(lang)["checkpoint"]
    model_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    best_state_cpu = {name: tensor.detach().cpu() for name, tensor in best_state.items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer_state.get("state", {}).values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "history": history,
            "best_state_dict": best_state_cpu,
            "best_valid_ppl": best_valid_ppl,
            "stale_epochs": stale_epochs,
            "config": {
                "n": config["n"],
                "m": config["m"],
                "h": config["h"],
                "direct": config["direct"],
                "min_freq": config["min_freq"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "patience": config["patience"],
            },
            "stoi": stoi,
            "itos": itos,
        },
        checkpoint_path,
    )
    print(f"[checkpoint] saved -> {checkpoint_path.name} (epoch {epoch})")


def load_resume_checkpoint(lang, model, optimizer, device, stoi, itos, config):
    checkpoint_path = train_files(lang)["checkpoint"]
    if not checkpoint_path.exists():
        return 0, None

    ckpt = torch.load(checkpoint_path, map_location=device)
    if ckpt.get("stoi") != stoi or ckpt.get("itos") != itos:
        raise ValueError(f"[checkpoint] vocabulary mismatch for {lang}.")

    saved_cfg = ckpt.get("config", {})
    for key in ("n", "m", "h", "direct", "min_freq"):
        if saved_cfg.get(key) != config.get(key):
            raise ValueError(f"[checkpoint] config mismatch for {lang} on '{key}'.")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

    return int(ckpt.get("epoch", 0)), ckpt


def load_model_checkpoint(lang, device):
    files = train_files(lang)
    ckpt = torch.load(files["model"], map_location=device)
    cfg = ckpt["config"]
    model = NNLM(V=cfg["V"], m=cfg["m"], h=cfg["h"], n=cfg["n"], direct=cfg["direct"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["stoi"], ckpt["itos"], cfg


def train_language_worker(lang: str, device_id: int, config: dict):
    set_shared_seed()
    device = resolve_device(device_id)
    resources = prepare_language_resources(lang, config)

    model = NNLM(
        V=len(resources["itos"]),
        m=config["m"],
        h=config["h"],
        n=config["n"],
        direct=config["direct"],
    ).to(device)

    history = train_model(
        model,
        resources["Xtr"],
        resources["Ytr"],
        resources["Xva"],
        resources["Yva"],
        device=device,
        config=config,
        verbose=True,
        checkpoint_lang=lang,
        stoi=resources["stoi"],
        itos=resources["itos"],
    )

    plot_path = plot_convergence(history, lang=lang, save_path=train_files(lang)["plot"])
    save_model_checkpoint(lang, model, config, resources["stoi"], resources["itos"])

    summary = {
        "lang": lang,
        "language": language_label(lang),
        "vocab_size": len(resources["itos"]),
        "unk_rates": resources["unk_rates"],
        "split_sizes": resources["split_sizes"],
        "window_counts": resources["window_counts"],
        "history": history,
        "probe_words": pick_words(resources["itos"]),
        "config": {
            "n": config["n"],
            "m": config["m"],
            "h": config["h"],
            "direct": config["direct"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "min_freq": config["min_freq"],
            "device_id": device_id,
        },
        "plot_path": plot_path,
        "top_tokens": resources["counter"].most_common(20),
    }

    with train_files(lang)["state"].open("wb") as f:
        pickle.dump(summary, f)
    save_json(train_files(lang)["summary"], summary)
    print(f"Training stage complete for {language_label(lang)}")


def detect_gpu_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def run_parallel_by_language(worker_fn, config: dict, stage_name: str):
    set_shared_seed()
    num_gpus = detect_gpu_count()
    print(f"{stage_name}: visible GPUs = {num_gpus}")
    if num_gpus > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        processes = []
        for idx, lang in enumerate(LANGS):
            process = mp.Process(target=worker_fn, args=(lang, idx, copy.deepcopy(config)))
            process.start()
            processes.append(process)
            print(f"Launched {stage_name} worker for {language_label(lang)} on cuda:{idx}")

        exitcodes = []
        for process in processes:
            process.join()
            exitcodes.append(process.exitcode)
        if any(code != 0 for code in exitcodes):
            raise RuntimeError(f"{stage_name} multiprocessing failed with exit codes {exitcodes}")
    else:
        device_id = 0 if num_gpus == 1 else -1
        for lang in LANGS:
            print(f"Running {stage_name} for {language_label(lang)} on device_id={device_id}")
            worker_fn(lang, device_id, copy.deepcopy(config))


def bundle_files(bundle_name: str, members: list[str | Path]):
    with zipfile.ZipFile(bundle_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            path = Path(member)
            if path.exists():
                zf.write(path, arcname=path.name)
    print(f"Created {bundle_name}")


def stage1_train_all(config: dict):
    prime_hf_token(config)
    run_parallel_by_language(train_language_worker, config, "stage1_train")

    members = [EN_CACHE, HI_CACHE]
    for lang in LANGS:
        members.extend(train_files(lang).values())
    bundle_files(STAGE1_BUNDLE, members)
    print("Stage 1 outputs:")
    for member in members + [STAGE1_BUNDLE]:
        if Path(member).exists():
            print(" -", Path(member).name)


def ensure_stage1_ready(config: dict):
    if not all(train_files(lang)["model"].exists() for lang in LANGS):
        extract_bundle_if_present(STAGE1_BUNDLE)
    missing = [lang for lang in LANGS if not train_files(lang)["model"].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing stage 1 model files for: " + ", ".join(language_label(lang) for lang in missing)
        )


def metrics_language_worker(lang: str, device_id: int, config: dict):
    set_shared_seed()
    device = resolve_device(device_id)
    resources = prepare_language_resources(lang, config)
    model, stoi, itos, model_cfg = load_model_checkpoint(lang, device)
    assert stoi == resources["stoi"], f"Vocabulary mismatch for {lang}; stage1 artifacts do not match current split."
    assert itos == resources["itos"], f"Vocabulary mismatch for {lang}; stage1 artifacts do not match current split."

    test_ppl = perplexity(model, resources["Xte"], resources["Yte"], device=device, batch_size=config["batch_size"])
    train_padded = padded_token_sentences(resources["train_sentences"], stoi, n=config["n"])
    test_padded = padded_token_sentences(resources["test_sentences"], stoi, n=config["n"])
    ngram_model = Q2_NGramLM(n=config["n"], V=len(itos)).fit(train_padded)
    laplace_ppl = ngram_model.perplexity(test_padded, mode="laplace")
    interp_ppl = ngram_model.perplexity(test_padded, mode="interp")

    nnlm_probs = nnlm_token_probs(
        model,
        resources["Xte"],
        resources["Yte"],
        device=device,
        batch_size=config["batch_size"],
    )
    interp_probs = ngram_model.token_probs(test_padded, mode="interp")
    mixture = mixture_report_from_probs(nnlm_probs, interp_probs, lam=0.5)

    perplexity_table = pd.DataFrame(
        [
            {"model": "NNLM", "perplexity": test_ppl, "lambda": np.nan},
            {"model": "Laplace n-gram", "perplexity": laplace_ppl, "lambda": np.nan},
            {"model": "Interpolated n-gram", "perplexity": interp_ppl, "lambda": np.nan},
            {"model": "Mixture @ 0.50", "perplexity": mixture["mixture_ppl_at_lambda"], "lambda": 0.50},
            {"model": "Best mixture", "perplexity": mixture["best_mixture_ppl"], "lambda": mixture["best_lambda"]},
        ]
    )

    param_counts = count_parameters(
        V=len(itos),
        m=model_cfg["m"],
        h=model_cfg["h"],
        n=model_cfg["n"],
        direct=model_cfg["direct"],
    )

    parameter_table = pd.DataFrame(
        [
            {
                "lang": lang,
                "language": language_label(lang),
                "vocab_size": len(itos),
                "n": model_cfg["n"],
                "m": model_cfg["m"],
                "h": model_cfg["h"],
                "direct": model_cfg["direct"],
                **param_counts,
                "ngram_entries_total": ngram_model.distinct_ngrams_total(),
                **{f"distinct_{order}gram": len(ngram_model.ngram_counts[order]) for order in range(1, config["n"] + 1)},
            }
        ]
    )

    sparsity = sparsity_table(resources["Xtr"], resources["Ytr"], resources["Xte"], resources["Yte"])
    sparsity["lang"] = lang
    sparsity["language"] = language_label(lang)
    sparsity = sparsity[
        [
            "lang",
            "language",
            "n",
            "train_contexts_seen",
            "train_ngrams_seen",
            "test_tokens",
            "pct_unseen_context",
            "pct_unseen_ngram",
        ]
    ]

    files = metrics_files(lang)
    perplexity_table.to_csv(files["perplexity"], index=False)
    parameter_table.to_csv(files["parameter"], index=False)
    sparsity.to_csv(files["sparsity"], index=False)
    mixture["grid"].to_csv(files["mixture"], index=False)

    summary = {
        "lang": lang,
        "language": language_label(lang),
        "perplexities": {
            "nnlm": test_ppl,
            "laplace_ngram": laplace_ppl,
            "interp_ngram": interp_ppl,
            "mixture_0.5": mixture["mixture_ppl_at_lambda"],
            "best_mixture": mixture["best_mixture_ppl"],
            "best_lambda": mixture["best_lambda"],
        },
        "parameter_total": int(parameter_table.iloc[0]["TOTAL"]),
        "ngram_entries_total": int(parameter_table.iloc[0]["ngram_entries_total"]),
        "vocab_size": int(len(itos)),
        "sparsity_rows": sparsity.to_dict(orient="records"),
    }
    save_json(files["summary"], summary)
    print(f"Metrics stage complete for {language_label(lang)}")


def stage2_metrics_all(config: dict):
    ensure_stage1_ready(config)
    run_parallel_by_language(metrics_language_worker, config, "stage2_metrics")

    members = []
    for lang in LANGS:
        members.extend(metrics_files(lang).values())
    bundle_files(STAGE2_BUNDLE, members)
    print("Stage 2 outputs:")
    for member in members + [STAGE2_BUNDLE]:
        if Path(member).exists():
            print(" -", Path(member).name)


def ensure_stage2_ready(config: dict):
    needed = [metrics_files(lang)["summary"] for lang in LANGS]
    if not all(path.exists() for path in needed):
        extract_bundle_if_present(STAGE2_BUNDLE)
    if not all(path.exists() for path in needed):
        print("Stage 2 outputs missing, computing them now so stage 3 can finish end-to-end.")
        stage2_metrics_all(config)


def discover_word2vec_artifacts(lang):
    found = {}
    for method in ("softmax", "ns", "hs"):
        path = Path(f"{lang}_embeddings_{method}.pkl")
        if path.exists():
            found[method] = load_word2vec(path)
    return found


def compare_spaces(lang, embeddings, stoi, itos, artifact, method, probe_words):
    probe_words = [word for word in probe_words if word in stoi and word in artifact["stoi"]]
    rows = []
    overlaps = []
    for word in probe_words:
        nnlm_neighbors = nearest_neighbours(word, embeddings, stoi, itos, k=10)
        w2v_neighbors = nearest_neighbours(word, artifact["embeddings"], artifact["stoi"], artifact["itos"], k=10)
        nnlm_tokens = [tok for tok, _ in nnlm_neighbors]
        w2v_tokens = [tok for tok, _ in w2v_neighbors]
        overlap = len(set(nnlm_tokens) & set(w2v_tokens))
        overlaps.append(overlap / 10.0 if (nnlm_tokens or w2v_tokens) else 0.0)
        rows.append(
            {
                "probe_word": word,
                "nnlm_neighbors": ", ".join(nnlm_tokens),
                "word2vec_method": method,
                "word2vec_neighbors": ", ".join(w2v_tokens),
                "overlap_at_10": overlap,
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(analysis_files(lang)["word2vec_compare"], index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, title, E, local_stoi, words in [
        (axes[0], f"{lang.upper()} NNLM PCA", embeddings, stoi, probe_words),
        (axes[1], f"{lang.upper()} word2vec ({method}) PCA", artifact["embeddings"], artifact["stoi"], probe_words),
    ]:
        labels = [word for word in words if word in local_stoi]
        if len(labels) >= 2:
            vectors = np.stack([E[local_stoi[word]] for word in labels], axis=0)
            centered = vectors - vectors.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ vt[:2].T
            ax.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.8)
            for idx, word in enumerate(labels):
                ax.text(coords[idx, 0], coords[idx, 1], word, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    save_figure(fig, analysis_files(lang)["embed_compare_plot"])

    return {
        "method": method,
        "avg_same_word_similarity": same_word_similarity(
            embeddings, stoi, artifact["embeddings"], artifact["stoi"], probe_words
        ),
        "avg_neighbor_overlap": float(np.mean(overlaps)) if overlaps else None,
        "probe_words": probe_words,
        "compare_csv": str(analysis_files(lang)["word2vec_compare"]),
        "compare_plot": str(analysis_files(lang)["embed_compare_plot"]),
    }


def choose_word2vec_baseline(lang, config, analogies, embeddings, stoi, itos):
    preferred = config.get("preferred_q3_method", {}).get(lang)
    artifacts = discover_word2vec_artifacts(lang)
    if not artifacts:
        return None, None, "No Q3 embedding artifacts found in the working directory."

    if preferred and preferred in artifacts:
        artifact = artifacts[preferred]
        method = preferred
        note = f"Using user-selected Q3 baseline: {preferred}."
    else:
        best = None
        scored = []
        for method, artifact in artifacts.items():
            norm = normalize_rows(artifact["embeddings"])
            preds = [
                embedding_analogy_predict(
                    artifact["embeddings"], norm, artifact["stoi"], artifact["itos"], row.word1, row.word2, row.word3
                )
                for row in analogies.itertuples(index=False)
            ]
            answered = int(sum(pred != "OOV" for pred in preds))
            scored.append((answered, {"softmax": 1, "ns": 2, "hs": 3}[method], method))
        _, _, method = max(scored)
        artifact = artifacts[method]
        note = (
            f"No explicit Q3 baseline was provided, so '{method}' was selected as the proxy best model "
            "because it answered the most analogy rows."
        )

    artifact_norm = normalize_rows(artifact["embeddings"])
    proxy_df = analogies.copy()
    proxy_df["word2vec"] = [
        embedding_analogy_predict(
            artifact["embeddings"], artifact_norm, artifact["stoi"], artifact["itos"], row.word1, row.word2, row.word3
        )
        for row in analogies.itertuples(index=False)
    ]
    proxy_df.to_csv(analysis_files(lang)["word2vec_proxy"], index=False)

    comparison = compare_spaces(lang, embeddings, stoi, itos, artifact, method, pick_words(itos))
    return comparison, proxy_df, note


def analysis_language_worker(lang: str, device_id: int, config: dict):
    set_shared_seed()
    device = resolve_device(device_id)
    resources = prepare_language_resources(lang, config)
    model, stoi, itos, _ = load_model_checkpoint(lang, device)
    assert stoi == resources["stoi"], f"Vocabulary mismatch for {lang}; stage1 artifacts do not match current split."
    embeddings = model.embeddings()

    probe_words = pick_words(itos)
    pca_path = plot_pca(
        embeddings,
        stoi,
        probe_words,
        title=f"{language_label(lang)} NNLM embeddings (PCA)",
        save_path=analysis_files(lang)["pca"],
    )

    analogies = load_analogy_set(lang, config)
    norm = normalize_rows(embeddings)
    analogy_df = analogies.copy()
    analogy_df["cbow"] = [
        embedding_analogy_predict(embeddings, norm, stoi, itos, row.word1, row.word2, row.word3)
        for row in analogies.itertuples(index=False)
    ]
    analogy_df.to_csv(analysis_files(lang)["analogy"], index=False)

    comparison, proxy_df, note = choose_word2vec_baseline(lang, config, analogies, embeddings, stoi, itos)
    both_answered = 0
    agreement_rate = None
    if proxy_df is not None:
        merged = analogy_df.merge(proxy_df, on=["word1", "word2", "word3"], how="inner")
        joint = merged[(merged["cbow"] != "OOV") & (merged["word2vec"] != "OOV")].copy()
        both_answered = int(len(joint))
        if both_answered:
            agreement_rate = float((joint["cbow"] == joint["word2vec"]).mean())

    summary = {
        "lang": lang,
        "language": language_label(lang),
        "analogy_answered": int((analogy_df["cbow"] != "OOV").sum()),
        "analogy_oov": int((analogy_df["cbow"] == "OOV").sum()),
        "analogy_rows": int(len(analogy_df)),
        "probe_words": probe_words,
        "pca_path": pca_path,
        "word2vec_note": note,
        "word2vec_comparison": comparison,
        "word2vec_agreement_rows": both_answered,
        "word2vec_agreement_rate": agreement_rate,
    }
    save_json(analysis_files(lang)["summary"], summary)
    print(f"Analysis stage complete for {language_label(lang)}")


def build_language_bundle(lang: str):
    members = [
        EN_CACHE if lang == "en" else HI_CACHE,
        train_files(lang)["model"],
        train_files(lang)["checkpoint"],
        train_files(lang)["state"],
        train_files(lang)["summary"],
        train_files(lang)["plot"],
        metrics_files(lang)["perplexity"],
        metrics_files(lang)["parameter"],
        metrics_files(lang)["sparsity"],
        metrics_files(lang)["mixture"],
        metrics_files(lang)["summary"],
        analysis_files(lang)["analogy"],
        analysis_files(lang)["pca"],
        analysis_files(lang)["word2vec_compare"],
        analysis_files(lang)["word2vec_proxy"],
        analysis_files(lang)["embed_compare_plot"],
        analysis_files(lang)["written_answers"],
        analysis_files(lang)["summary"],
    ]
    bundle_files(train_files(lang)["bundle"], members)
    return train_files(lang)["bundle"]


def build_language_answer_draft(lang: str):
    train_state = load_train_state(lang)
    metric = load_json(metrics_files(lang)["summary"])
    analysis = load_json(analysis_files(lang)["summary"])

    hist = train_state["history"]
    if hist["converged"]:
        convergence_note = (
            f"The model early-stopped at epoch {hist['stopped_epoch']} and reached its best validation "
            f"perplexity at epoch {hist['best_epoch']}."
        )
    else:
        convergence_note = (
            f"The run hit the epoch cap at {hist['stopped_epoch']}; check the convergence plot to decide "
            "whether more epochs are still needed."
        )

    text = f"""# Question 4 {language_label(lang)} draft

## Step 7

1. Mixture perplexity at lambda = 0.5: {metric['perplexities']['mixture_0.5']:.4f}
2. Best mixture perplexity: {metric['perplexities']['best_mixture']:.4f} at lambda = {metric['perplexities']['best_lambda']:.2f}
3. The mixture {"does" if metric['perplexities']['best_mixture'] < min(metric['perplexities']['nnlm'], metric['perplexities']['interp_ngram']) else "does not"} beat both standalone components for {language_label(lang)}.

## Observations

1. NNLM perplexity = {metric['perplexities']['nnlm']:.4f}, Laplace n-gram = {metric['perplexities']['laplace_ngram']:.4f}, interpolated n-gram = {metric['perplexities']['interp_ngram']:.4f}.
2. {convergence_note}
3. NNLM parameters = {metric['parameter_total']:,}; stored n-gram entries = {metric['ngram_entries_total']:,}.
4. Unseen-context percentages are listed in `{metrics_files(lang)['sparsity'].name}`.
5. Cross-language hardness still needs both English and Hindi notebooks together before writing the final combined answer.
6. {analysis['word2vec_note']}

Additional notes:
- Analogy rows answered: {analysis['analogy_answered']} / {analysis['analogy_rows']}
- Analogy OOV rows: {analysis['analogy_oov']}
- Agreement rows with selected Q3 word2vec model: {analysis['word2vec_agreement_rows']}
- Agreement rate with selected Q3 word2vec model: {fmt_optional(analysis['word2vec_agreement_rate'])}
"""
    analysis_files(lang)["written_answers"].write_text(text, encoding="utf-8")
    return analysis_files(lang)["written_answers"]


def run_full_language_pipeline(lang: str, config: dict, device_id: int | None = None):
    expected_inputs = [
        EN_CACHE,
        HI_CACHE,
        train_files(lang)["checkpoint"].name,
        train_files(lang)["model"].name,
    ]
    for method in ("softmax", "ns", "hs"):
        expected_inputs.append(f"{lang}_embeddings_{method}.pkl")

    stage_kaggle_input_files(expected_inputs)
    prime_hf_token(config)
    if device_id is None:
        device_id = 0 if detect_gpu_count() > 0 else -1

    print(f"Running full Q4 pipeline for {language_label(lang)} on device_id={device_id}")
    train_language_worker(lang, device_id, copy.deepcopy(config))
    metrics_language_worker(lang, device_id, copy.deepcopy(config))
    analysis_language_worker(lang, device_id, copy.deepcopy(config))
    build_language_answer_draft(lang)
    bundle = build_language_bundle(lang)

    print(f"Completed full Q4 pipeline for {language_label(lang)}")
    print("Artifacts:")
    for path in [
        train_files(lang)["model"],
        train_files(lang)["checkpoint"],
        train_files(lang)["plot"],
        metrics_files(lang)["perplexity"],
        metrics_files(lang)["parameter"],
        metrics_files(lang)["sparsity"],
        metrics_files(lang)["mixture"],
        analysis_files(lang)["analogy"],
        analysis_files(lang)["pca"],
        analysis_files(lang)["word2vec_compare"],
        analysis_files(lang)["word2vec_proxy"],
        analysis_files(lang)["embed_compare_plot"],
        analysis_files(lang)["written_answers"],
        bundle,
    ]:
        if Path(path).exists():
            print(" -", Path(path).name)


def fmt_optional(value):
    return "nan" if value is None else f"{value:.4f}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_train_state(lang):
    with train_files(lang)["state"].open("rb") as f:
        return pickle.load(f)


def draft_answers(train_states, metric_summaries, analysis_summaries):
    en_train = train_states["en"]
    hi_train = train_states["hi"]
    en_metric = metric_summaries["en"]
    hi_metric = metric_summaries["hi"]
    en_analysis = analysis_summaries["en"]
    hi_analysis = analysis_summaries["hi"]

    def conv_text(state):
        hist = state["history"]
        if hist["converged"]:
            return (
                f"Yes. Early stopping triggered after epoch {hist['stopped_epoch']}, "
                f"and the best validation perplexity occurred at epoch {hist['best_epoch']}."
            )
        return (
            f"Not fully guaranteed. The run hit the epoch cap at {hist['stopped_epoch']}, "
            "so the reported perplexity should be treated as a lower bound unless the validation curve is flat."
        )

    harder = "Hindi" if hi_metric["perplexities"]["nnlm"] > en_metric["perplexities"]["nnlm"] else "English"
    easier = "English" if harder == "Hindi" else "Hindi"
    harder_train = hi_train if harder == "Hindi" else en_train
    easier_train = en_train if harder == "Hindi" else hi_train
    harder_metric = hi_metric if harder == "Hindi" else en_metric
    easier_metric = en_metric if harder == "Hindi" else hi_metric

    return f"""# Question 4 written-answer draft

## Step 7

### 1. Mixture perplexity at $\\lambda = 0.5$
- English: the NNLM/interpolated n-gram mixture reached perplexity {en_metric['perplexities']['mixture_0.5']:.4f} at $\\lambda = 0.5$.
- Hindi: the NNLM/interpolated n-gram mixture reached perplexity {hi_metric['perplexities']['mixture_0.5']:.4f} at $\\lambda = 0.5$.

### 2. Best mixture after sweeping $\\lambda$
- English: the best mixture perplexity was {en_metric['perplexities']['best_mixture']:.4f} at $\\lambda = {en_metric['perplexities']['best_lambda']:.2f}$.
- Hindi: the best mixture perplexity was {hi_metric['perplexities']['best_mixture']:.4f} at $\\lambda = {hi_metric['perplexities']['best_lambda']:.2f}$.

### 3. Does the mixture beat both components?
- English: {"yes" if en_metric['perplexities']['best_mixture'] < min(en_metric['perplexities']['nnlm'], en_metric['perplexities']['interp_ngram']) else "no"}. This suggests the NNLM and the n-gram model make partly different mistakes, so interpolation benefits from their complementarity.
- Hindi: {"yes" if hi_metric['perplexities']['best_mixture'] < min(hi_metric['perplexities']['nnlm'], hi_metric['perplexities']['interp_ngram']) else "no"}. A weaker standalone model can still improve the mixture if it is strong on tokens where the other model fails.

## Observations

### Question 1
- English: NNLM perplexity = {en_metric['perplexities']['nnlm']:.4f}, Laplace n-gram = {en_metric['perplexities']['laplace_ngram']:.4f}, interpolated n-gram = {en_metric['perplexities']['interp_ngram']:.4f}.
- Hindi: NNLM perplexity = {hi_metric['perplexities']['nnlm']:.4f}, Laplace n-gram = {hi_metric['perplexities']['laplace_ngram']:.4f}, interpolated n-gram = {hi_metric['perplexities']['interp_ngram']:.4f}.

### Question 2
- English: {conv_text(en_train)}
- Hindi: {conv_text(hi_train)}

### Question 3
- English: the NNLM used {en_metric['parameter_total']:,} learned parameters, while the count model stored {en_metric['ngram_entries_total']:,} distinct n-gram entries across orders.
- Hindi: the NNLM used {hi_metric['parameter_total']:,} learned parameters, while the count model stored {hi_metric['ngram_entries_total']:,} distinct n-gram entries across orders.

### Question 4
- English unseen-context / unseen-ngram percentages by order:
{pd.DataFrame(en_metric['sparsity_rows'])[['n', 'pct_unseen_context', 'pct_unseen_ngram']].to_string(index=False)}

- Hindi unseen-context / unseen-ngram percentages by order:
{pd.DataFrame(hi_metric['sparsity_rows'])[['n', 'pct_unseen_context', 'pct_unseen_ngram']].to_string(index=False)}

The NNLM can still assign sensible probability to unseen contexts because it uses shared word embeddings and shared network parameters instead of storing one isolated probability table entry per context.

### Question 5
- Within this shared preprocessing pipeline, {harder} was harder to model because its NNLM perplexity was higher ({harder_metric['perplexities']['nnlm']:.4f} vs {easier_metric['perplexities']['nnlm']:.4f} for {easier}). Part of that difference is explained by vocabulary size ({harder_metric['vocab_size']:,} vs {easier_metric['vocab_size']:,}) and test `<unk>` rate ({harder_train['unk_rates']['test']:.4%} vs {easier_train['unk_rates']['test']:.4%}).

### Question 6
- English word2vec baseline note: {en_analysis['word2vec_note']}
- Hindi word2vec baseline note: {hi_analysis['word2vec_note']}
- English same-word cosine similarity vs selected word2vec space: {fmt_optional(None if en_analysis['word2vec_comparison'] is None else en_analysis['word2vec_comparison']['avg_same_word_similarity'])}.
- Hindi same-word cosine similarity vs selected word2vec space: {fmt_optional(None if hi_analysis['word2vec_comparison'] is None else hi_analysis['word2vec_comparison']['avg_same_word_similarity'])}.
- English average top-10 neighbour overlap with selected word2vec space: {fmt_optional(None if en_analysis['word2vec_comparison'] is None else en_analysis['word2vec_comparison']['avg_neighbor_overlap'])}.
- Hindi average top-10 neighbour overlap with selected word2vec space: {fmt_optional(None if hi_analysis['word2vec_comparison'] is None else hi_analysis['word2vec_comparison']['avg_neighbor_overlap'])}.

The NNLM and word2vec spaces usually agree on broad semantic neighborhoods, but they differ in exact neighbour rankings because the objectives are different: word2vec optimizes local embedding relations directly, while the NNLM learns embeddings only as part of next-word prediction.
"""


def combine_final_outputs(config: dict):
    train_states = {lang: load_train_state(lang) for lang in LANGS}
    metric_summaries = {lang: load_json(metrics_files(lang)["summary"]) for lang in LANGS}
    analysis_summaries = {lang: load_json(analysis_files(lang)["summary"]) for lang in LANGS}

    parameter_table = pd.concat(
        [pd.read_csv(metrics_files(lang)["parameter"]) for lang in LANGS],
        ignore_index=True,
    )
    parameter_table.to_csv("q4_parameter_count_table.csv", index=False)

    sparsity = pd.concat(
        [pd.read_csv(metrics_files(lang)["sparsity"]) for lang in LANGS],
        ignore_index=True,
    )
    sparsity.to_csv("q4_sparsity_table.csv", index=False)

    summary_rows = []
    for lang in LANGS:
        t = train_states[lang]
        m = metric_summaries[lang]
        a = analysis_summaries[lang]
        summary_rows.append(
            {
                "lang": lang,
                "language": language_label(lang),
                "vocab_size": m["vocab_size"],
                "unk_rate_train": t["unk_rates"]["train"],
                "unk_rate_valid": t["unk_rates"]["valid"],
                "unk_rate_test": t["unk_rates"]["test"],
                "nnlm_ppl": m["perplexities"]["nnlm"],
                "laplace_ppl": m["perplexities"]["laplace_ngram"],
                "interp_ppl": m["perplexities"]["interp_ngram"],
                "mixture_0_5_ppl": m["perplexities"]["mixture_0.5"],
                "best_mixture_ppl": m["perplexities"]["best_mixture"],
                "best_lambda": m["perplexities"]["best_lambda"],
                "analogy_answered": a["analogy_answered"],
                "analogy_oov": a["analogy_oov"],
                "word2vec_agreement_rows": a["word2vec_agreement_rows"],
                "word2vec_agreement_rate": a["word2vec_agreement_rate"],
            }
        )

    pd.DataFrame(summary_rows).to_csv("q4_language_summary.csv", index=False)
    draft = draft_answers(train_states, metric_summaries, analysis_summaries)
    Path("Q4_WRITTEN_ANSWERS.md").write_text(draft, encoding="utf-8")
    print(draft)

    final_members = [
        "analogy_results_nnlm_en.csv",
        "analogy_results_nnlm_hi.csv",
        "q4_convergence_en.png",
        "q4_convergence_hi.png",
        "q4_perplexity_table_en.csv",
        "q4_perplexity_table_hi.csv",
        "q4_parameter_table_en.csv",
        "q4_parameter_table_hi.csv",
        "q4_sparsity_table_en.csv",
        "q4_sparsity_table_hi.csv",
        "q4_mixture_grid_en.csv",
        "q4_mixture_grid_hi.csv",
        "q4_parameter_count_table.csv",
        "q4_sparsity_table.csv",
        "q4_embedding_pca_en.png",
        "q4_embedding_pca_hi.png",
        "q4_word2vec_compare_en.csv",
        "q4_word2vec_compare_hi.csv",
        "q4_word2vec_proxy_en.csv",
        "q4_word2vec_proxy_hi.csv",
        "q4_embedding_compare_en.png",
        "q4_embedding_compare_hi.png",
        "q4_language_summary.csv",
        "Q4_WRITTEN_ANSWERS.md",
        "nnlm_en.pt",
        "nnlm_hi.pt",
    ]
    bundle_files(FINAL_BUNDLE, final_members)
    print("Final Question 4 bundle contents:")
    for member in final_members:
        if Path(member).exists():
            print(" -", member)


def stage3_analysis_and_bundle(config: dict):
    ensure_stage1_ready(config)
    ensure_stage2_ready(config)
    run_parallel_by_language(analysis_language_worker, config, "stage3_analysis")
    combine_final_outputs(config)
