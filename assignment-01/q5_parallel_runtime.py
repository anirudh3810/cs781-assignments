from __future__ import annotations

import copy
import heapq
import json
import math
import os
import pickle
import random
import re
import shutil
import time
import zipfile
from array import array
from collections import Counter, defaultdict
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
MIN_TOKENS = 3

EN_CACHE = "shared_english_corpus.csv"
HI_CACHE = "shared_hindi_corpus.csv"

EMBED_DIM = 100
DEFAULT_Q5_N = 4
DEFAULT_Q5_HIDDEN = 100
DEFAULT_Q5_EPOCHS = 30
DEFAULT_Q5_BATCH_SIZE = 512
DEFAULT_Q5_LR = 1e-3
DEFAULT_Q5_WEIGHT_DECAY = 1e-4
DEFAULT_Q5_PATIENCE = 3
DEFAULT_DIRECT = False
DEFAULT_TARGET_VOCAB_SIZE = 5000
CHECKPOINT_EVERY = 5

TOKENIZER_SUFFIX = "</w>"
MERGE_PROGRESS_EVERY = 100
TOKENIZE_PROGRESS_EVERY = 50000


def build_default_config():
    return {
        "n": DEFAULT_Q5_N,
        "m": EMBED_DIM,
        "h": DEFAULT_Q5_HIDDEN,
        "epochs": DEFAULT_Q5_EPOCHS,
        "batch_size": DEFAULT_Q5_BATCH_SIZE,
        "lr": DEFAULT_Q5_LR,
        "weight_decay": DEFAULT_Q5_WEIGHT_DECAY,
        "patience": DEFAULT_Q5_PATIENCE,
        "direct": DEFAULT_DIRECT,
        "target_vocab_size": DEFAULT_TARGET_VOCAB_SIZE,
        "max_pairs": None,
        "hf_token": os.environ.get("HF_TOKEN", "").strip(),
    }


def language_label(lang: str) -> str:
    return "English" if lang == "en" else "Hindi"


def cache_path_for_lang(lang: str) -> Path:
    return Path(EN_CACHE if lang == "en" else HI_CACHE)


def q5_files(lang: str):
    return {
        "tokenizer": Path(f"bpe_{lang}.json"),
        "model": Path(f"bpe_nnlm_{lang}.pt"),
        "checkpoint": Path(f"checkpoint_bpe_nnlm_{lang}.pt"),
        "train_log": Path(f"bpe_train_log_{lang}.json"),
        "train_state": Path(f"q5_train_state_{lang}.pkl"),
        "train_summary": Path(f"q5_train_summary_{lang}.json"),
        "bpe_summary": Path(f"q5_bpe_summary_{lang}.json"),
        "plot": Path(f"q5_convergence_{lang}.png"),
        "perplexity_compare": Path(f"q5_perplexity_compare_{lang}.csv"),
        "parameter_compare": Path(f"q5_parameter_compare_{lang}.csv"),
        "embedding_compare": Path(f"q5_embedding_compare_{lang}.csv"),
        "analysis_summary": Path(f"q5_analysis_summary_{lang}.json"),
        "written_answers": Path(f"Q5_WRITTEN_ANSWERS_{lang}.md"),
        "bundle": Path(f"q5_{lang}_artifacts.zip"),
    }


def set_shared_seed(seed: int = SHARED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def shared_tokenize(text):
    return nltk.word_tokenize(str(text).lower())


def surface_tokenize(text):
    return nltk.word_tokenize(str(text))


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


def build_shared_corpus_files(seed: int = SHARED_SEED, config: dict | None = None):
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

    en_df.to_csv(EN_CACHE, index=False)
    hi_df.to_csv(HI_CACHE, index=False)
    print(f"Wrote {len(en_df):,} English sentences to {EN_CACHE}")
    print(f"Wrote {len(hi_df):,} Hindi sentences to {HI_CACHE}")


def load_language_texts(lang: str, config: dict) -> list[str]:
    path = cache_path_for_lang(lang)
    if not path.exists():
        build_shared_corpus_files(config=config)
    df = pd.read_csv(path, keep_default_na=False)
    texts = df["text"].tolist()
    max_pairs = config.get("max_pairs")
    return texts[:max_pairs] if max_pairs is not None else texts


def split_texts(texts, ratios=(0.8, 0.1, 0.1)):
    total = len(texts)
    train_end = int(total * ratios[0])
    valid_end = train_end + int(total * ratios[1])
    train = texts[:train_end]
    valid = texts[train_end:valid_end]
    test = texts[valid_end:]
    return train, valid, test


def make_symbol(byte_values, terminal):
    return (tuple(int(b) for b in byte_values), bool(terminal))


def symbol_bytes(symbol):
    return symbol[0]


def symbol_terminal(symbol):
    return bool(symbol[1])


def merge_symbols(left, right):
    return make_symbol(symbol_bytes(left) + symbol_bytes(right), symbol_terminal(right))


def initial_symbols_for_word(word: str):
    payload = str(word).encode("utf-8")
    if not payload:
        return []
    if len(payload) == 1:
        return [make_symbol((payload[0],), True)]
    seq = [make_symbol((value,), False) for value in payload[:-1]]
    seq.append(make_symbol((payload[-1],), True))
    return seq


def sequence_pair_counts(symbols):
    counts = Counter()
    for idx in range(len(symbols) - 1):
        counts[(symbols[idx], symbols[idx + 1])] += 1
    return counts


def merge_pair_in_sequence(symbols, pair, merged_symbol):
    out = []
    idx = 0
    left, right = pair
    while idx < len(symbols):
        if idx + 1 < len(symbols) and symbols[idx] == left and symbols[idx + 1] == right:
            out.append(merged_symbol)
            idx += 2
        else:
            out.append(symbols[idx])
            idx += 1
    return out


def bytes_to_display(byte_values):
    payload = bytes(byte_values)
    try:
        text = payload.decode("utf-8")
        if text and TOKENIZER_SUFFIX not in text and all(ch not in "\x00\r\n\t" for ch in text):
            return text
    except UnicodeDecodeError:
        pass
    return "".join(f"<0x{value:02X}>" for value in payload)


def symbol_to_token(symbol):
    body = bytes_to_display(symbol_bytes(symbol))
    return body + TOKENIZER_SUFFIX if symbol_terminal(symbol) else body


def token_to_symbol(token: str):
    token = str(token)
    terminal = token.endswith(TOKENIZER_SUFFIX)
    body = token[: -len(TOKENIZER_SUFFIX)] if terminal else token
    payload = bytearray()
    idx = 0
    while idx < len(body):
        match = re.match(r"<0x([0-9A-Fa-f]{2})>", body[idx:])
        if match:
            payload.append(int(match.group(1), 16))
            idx += len(match.group(0))
            continue
        payload.extend(body[idx].encode("utf-8"))
        idx += 1
    return make_symbol(tuple(payload), terminal)


def token_to_surface_word(token: str):
    symbol = token_to_symbol(token)
    if not symbol_terminal(symbol):
        return None
    try:
        return bytes(symbol_bytes(symbol)).decode("utf-8")
    except UnicodeDecodeError:
        return None


def full_base_byte_symbols():
    symbols = []
    for terminal in (False, True):
        for value in range(256):
            symbols.append(make_symbol((value,), terminal))
    return symbols


def tokenizer_has_full_byte_coverage(tokenizer) -> bool:
    required = {symbol_to_token(symbol) for symbol in full_base_byte_symbols()}
    return required.issubset(set(tokenizer.stoi))


def piece_ids_with_fallback(pieces, stoi):
    unk_id = stoi["<unk>"]
    piece_ids = []
    for piece in pieces:
        token_id = stoi.get(piece)
        if token_id is not None:
            piece_ids.append(token_id)
            continue

        symbol = token_to_symbol(piece)
        raw_bytes = symbol_bytes(symbol)
        if not raw_bytes:
            piece_ids.append(unk_id)
            continue

        for idx, value in enumerate(raw_bytes):
            terminal = idx == len(raw_bytes) - 1 and symbol_terminal(symbol)
            byte_token = symbol_to_token(make_symbol((value,), terminal))
            piece_ids.append(stoi.get(byte_token, unk_id))
    return piece_ids


class ByteLevelBPE:
    def __init__(self, target_vocab_size=DEFAULT_TARGET_VOCAB_SIZE, lang: str | None = None):
        self.target_vocab_size = int(target_vocab_size)
        self.lang = lang
        self.base_symbols = []
        self.merged_symbols = []
        self.merge_rules = []
        self.merge_lookup = {}
        self.merge_rank = {}
        self.itos = list(SPECIALS)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.word_cache = {}
        self.lowercase_representatives = {}
        self.summary = {}

    @property
    def vocab_size(self):
        return len(self.itos)

    def fit(self, texts, verbose=True):
        word_counter = Counter()
        word_first_seen = {}
        lower_variant_counts = defaultdict(Counter)
        next_word_order = 0

        base_symbols = full_base_byte_symbols()
        base_seen = set(base_symbols)
        total_surface_tokens = 0

        started = time.time()
        for text_idx, text in enumerate(texts, start=1):
            tokens = surface_tokenize(text)
            total_surface_tokens += len(tokens)
            for token in tokens:
                word_counter[token] += 1
                lower_variant_counts[token.lower()][token] += 1
                if token not in word_first_seen:
                    word_first_seen[token] = next_word_order
                    next_word_order += 1
                    for symbol in initial_symbols_for_word(token):
                        if symbol not in base_seen:
                            base_seen.add(symbol)
                            base_symbols.append(symbol)
            if verbose and text_idx % TOKENIZE_PROGRESS_EVERY == 0:
                print(
                    f"[BPE setup] {language_label(self.lang or 'en')} | "
                    f"sentences {text_idx:,}/{len(texts):,} | "
                    f"surface tokens {total_surface_tokens:,} | "
                    f"unique surface tokens {len(word_counter):,} | "
                    f"elapsed {format_duration(time.time() - started)}",
                    flush=True,
                )

        words = sorted(word_counter, key=word_first_seen.get)
        word_freqs = [int(word_counter[word]) for word in words]
        word_sequences = [initial_symbols_for_word(word) for word in words]

        pair_freqs = Counter()
        pair_to_words = defaultdict(set)
        for word_idx, seq in enumerate(word_sequences):
            local_counts = sequence_pair_counts(seq)
            if not local_counts:
                continue
            weight = word_freqs[word_idx]
            for pair, count in local_counts.items():
                pair_freqs[pair] += weight * count
                pair_to_words[pair].add(word_idx)

        heap = [(-int(freq), pair) for pair, freq in pair_freqs.items()]
        heapq.heapify(heap)

        merged_symbols = []
        merged_seen = set()
        merge_rules = []
        merge_steps = 0
        target = max(self.target_vocab_size, len(SPECIALS) + len(base_symbols))

        while len(SPECIALS) + len(base_symbols) + len(merged_symbols) < target and heap:
            neg_freq, pair = heapq.heappop(heap)
            current_freq = int(pair_freqs.get(pair, 0))
            if current_freq != -neg_freq:
                continue
            if current_freq <= 1:
                break

            merged_symbol = merge_symbols(pair[0], pair[1])
            merge_rules.append((pair[0], pair[1], merged_symbol))
            merge_steps += 1

            if merged_symbol not in base_seen and merged_symbol not in merged_seen:
                merged_seen.add(merged_symbol)
                merged_symbols.append(merged_symbol)

            affected = list(pair_to_words.get(pair, ()))
            for word_idx in affected:
                old_seq = word_sequences[word_idx]
                old_counts = sequence_pair_counts(old_seq)
                if pair not in old_counts:
                    bucket = pair_to_words.get(pair)
                    if bucket is not None:
                        bucket.discard(word_idx)
                        if not bucket:
                            pair_to_words.pop(pair, None)
                    continue

                weight = word_freqs[word_idx]
                for old_pair, count in old_counts.items():
                    updated = int(pair_freqs.get(old_pair, 0) - weight * count)
                    if updated > 0:
                        pair_freqs[old_pair] = updated
                        heapq.heappush(heap, (-updated, old_pair))
                    else:
                        pair_freqs.pop(old_pair, None)

                    bucket = pair_to_words.get(old_pair)
                    if bucket is not None:
                        bucket.discard(word_idx)
                        if not bucket:
                            pair_to_words.pop(old_pair, None)

                new_seq = merge_pair_in_sequence(old_seq, pair, merged_symbol)
                word_sequences[word_idx] = new_seq
                new_counts = sequence_pair_counts(new_seq)
                for new_pair, count in new_counts.items():
                    updated = int(pair_freqs.get(new_pair, 0) + weight * count)
                    pair_freqs[new_pair] = updated
                    pair_to_words[new_pair].add(word_idx)
                    heapq.heappush(heap, (-updated, new_pair))

            if verbose and (merge_steps == 1 or merge_steps % MERGE_PROGRESS_EVERY == 0):
                effective_vocab = len(SPECIALS) + len(base_symbols) + len(merged_symbols)
                print(
                    f"[BPE merge] {language_label(self.lang or 'en')} | "
                    f"merge {merge_steps:,} | vocab {effective_vocab:,}/{target:,} | "
                    f"best pair freq {current_freq:,} | elapsed {format_duration(time.time() - started)}",
                    flush=True,
                )

        merge_lookup = {}
        merge_rank = {}
        for rank, (left, right, merged) in enumerate(merge_rules):
            merge_lookup[(left, right)] = merged
            merge_rank[(left, right)] = rank

        representatives = {}
        for lower, variants in lower_variant_counts.items():
            if lower in variants:
                representatives[lower] = lower
                continue
            representative = sorted(
                variants.items(),
                key=lambda item: (-item[1], word_first_seen[item[0]], item[0]),
            )[0][0]
            representatives[lower] = representative

        self.base_symbols = base_symbols
        self.merged_symbols = merged_symbols
        self.merge_rules = merge_rules
        self.merge_lookup = merge_lookup
        self.merge_rank = merge_rank
        self.lowercase_representatives = representatives
        self.itos = (
            list(SPECIALS)
            + [symbol_to_token(symbol) for symbol in base_symbols]
            + [symbol_to_token(symbol) for symbol in merged_symbols]
        )
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.word_cache = {}
        self.summary = {
            "language": language_label(self.lang or "en"),
            "lang": self.lang,
            "target_vocab_size": self.target_vocab_size,
            "effective_vocab_size": len(self.itos),
            "initial_symbol_count": len(base_symbols),
            "learned_merges": len(merge_rules),
            "unique_surface_tokens_train": len(words),
            "surface_tokens_train": total_surface_tokens,
            "elapsed_seconds": time.time() - started,
        }
        return self

    def encode_word(self, word: str):
        word = str(word)
        cached = self.word_cache.get(word)
        if cached is not None:
            return list(cached)

        symbols = initial_symbols_for_word(word)
        if not symbols:
            return []

        while len(symbols) > 1:
            best_rank = None
            best_pos = None
            best_symbol = None
            for idx in range(len(symbols) - 1):
                pair = (symbols[idx], symbols[idx + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pos = idx
                    best_symbol = self.merge_lookup[pair]
            if best_pos is None:
                break
            symbols = symbols[:best_pos] + [best_symbol] + symbols[best_pos + 2 :]

        tokens = tuple(symbol_to_token(symbol) for symbol in symbols)
        self.word_cache[word] = tokens
        return list(tokens)

    def tokenize_text(self, text):
        pieces = []
        for token in surface_tokenize(text):
            pieces.extend(self.encode_word(token))
        return pieces

    def to_export_dict(self):
        return {
            "language": self.lang,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": len(self.itos),
            "itos": self.itos,
            "base_tokens": [symbol_to_token(symbol) for symbol in self.base_symbols],
            "merge_tokens": [symbol_to_token(symbol) for symbol in self.merged_symbols],
            "merges": [
                {
                    "left": symbol_to_token(left),
                    "right": symbol_to_token(right),
                    "merged": symbol_to_token(merged),
                }
                for left, right, merged in self.merge_rules
            ],
            "byte_level": True,
            "case_sensitive": True,
            "terminal_suffix": TOKENIZER_SUFFIX,
            "summary": self.summary,
        }

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_export_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(target_vocab_size=int(data.get("target_vocab_size", data["vocab_size"])), lang=data.get("language"))
        obj.itos = list(data["itos"])
        obj.stoi = {token: idx for idx, token in enumerate(obj.itos)}
        obj.base_symbols = [token_to_symbol(token) for token in data.get("base_tokens", [])]
        merged_tokens = data.get("merge_tokens", [])
        obj.merged_symbols = [token_to_symbol(token) for token in merged_tokens]

        merge_rules = []
        for item in data.get("merges", []):
            if isinstance(item, dict):
                left = token_to_symbol(item["left"])
                right = token_to_symbol(item["right"])
                merged = token_to_symbol(item["merged"])
            elif len(item) == 3:
                left = token_to_symbol(item[0])
                right = token_to_symbol(item[1])
                merged = token_to_symbol(item[2])
            else:
                raise ValueError("Unsupported BPE merge format.")
            merge_rules.append((left, right, merged))

        obj.merge_rules = merge_rules
        obj.merge_lookup = {}
        obj.merge_rank = {}
        for rank, (left, right, merged) in enumerate(merge_rules):
            obj.merge_lookup[(left, right)] = merged
            obj.merge_rank[(left, right)] = rank

        obj.summary = data.get("summary", {})
        obj.word_cache = {}
        return obj


def load_or_train_bpe(lang: str, train_texts, config: dict):
    files = q5_files(lang)
    tokenizer_path = files["tokenizer"]
    if tokenizer_path.exists():
        tokenizer = ByteLevelBPE.load(tokenizer_path)
        saved_target = int(tokenizer.summary.get("target_vocab_size", tokenizer.target_vocab_size))
        if saved_target != int(config["target_vocab_size"]):
            raise ValueError(
                f"Existing {tokenizer_path.name} was trained with target_vocab_size={saved_target}, "
                f"but current config wants {config['target_vocab_size']}."
            )
        if not tokenizer_has_full_byte_coverage(tokenizer):
            print(
                f"Existing {tokenizer_path.name} is missing full byte coverage; retraining tokenizer.",
                flush=True,
            )
            tokenizer_path.unlink()
        else:
            print(f"Loaded existing tokenizer from {tokenizer_path.name}")
            save_json(files["bpe_summary"], tokenizer.summary or {"target_vocab_size": tokenizer.target_vocab_size})
            return tokenizer

    tokenizer = ByteLevelBPE(target_vocab_size=config["target_vocab_size"], lang=lang)
    tokenizer.fit(train_texts, verbose=True)
    tokenizer.save(tokenizer_path)
    save_json(files["bpe_summary"], tokenizer.summary)
    print(f"Saved tokenizer to {tokenizer_path.name}")
    return tokenizer


def build_surface_stats(texts):
    lower_counts = Counter()
    lower_variant_counts = defaultdict(Counter)
    first_seen = {}
    order = 0
    total_surface_tokens = 0

    for text in texts:
        for token in surface_tokenize(text):
            total_surface_tokens += 1
            lower = token.lower()
            lower_counts[lower] += 1
            lower_variant_counts[lower][token] += 1
            if token not in first_seen:
                first_seen[token] = order
                order += 1

    representatives = {}
    for lower, variants in lower_variant_counts.items():
        if lower in variants:
            representatives[lower] = lower
            continue
        representative = sorted(
            variants.items(),
            key=lambda item: (-item[1], first_seen[item[0]], item[0]),
        )[0][0]
        representatives[lower] = representative

    return {
        "lower_counts": lower_counts,
        "representatives": representatives,
        "total_surface_tokens": total_surface_tokens,
        "surface_vocab_size": len(lower_counts),
    }


def vectorize_bpe_texts(texts, tokenizer: ByteLevelBPE, stoi, n: int, label: str):
    bos = [stoi["<s>"]] * (n - 1)
    eos = stoi["</s>"]

    xbuf = array("i")
    ybuf = array("i")
    surface_token_total = 0
    sentence_total = 0
    subword_token_total = 0
    single_piece_token_total = 0

    started = time.time()
    for text_idx, text in enumerate(texts, start=1):
        surface_tokens = surface_tokenize(text)
        if not surface_tokens:
            continue

        sentence_total += 1
        surface_token_total += len(surface_tokens)
        sent_ids = list(bos)
        for token in surface_tokens:
            pieces = tokenizer.encode_word(token)
            subword_token_total += len(pieces)
            if len(pieces) == 1:
                single_piece_token_total += 1
            sent_ids.extend(piece_ids_with_fallback(pieces, stoi))
        sent_ids.append(eos)

        for end in range(n - 1, len(sent_ids)):
            xbuf.extend(sent_ids[end - (n - 1) : end])
            ybuf.append(sent_ids[end])

        if text_idx % TOKENIZE_PROGRESS_EVERY == 0:
            print(
                f"[vectorize] {label} | sentences {text_idx:,}/{len(texts):,} | "
                f"surface tokens {surface_token_total:,} | subword tokens {subword_token_total:,} | "
                f"elapsed {format_duration(time.time() - started)}",
                flush=True,
            )

    if len(ybuf) == 0:
        X = np.empty((0, n - 1), dtype=np.int32)
        Y = np.empty((0,), dtype=np.int32)
    else:
        X = np.frombuffer(xbuf, dtype=np.int32).copy().reshape(-1, n - 1)
        Y = np.frombuffer(ybuf, dtype=np.int32).copy()

    word_units = surface_token_total + sentence_total
    stats = {
        "sentences": sentence_total,
        "surface_tokens": surface_token_total,
        "surface_word_units": word_units,
        "subword_tokens": subword_token_total,
        "subword_targets": int(len(Y)),
        "avg_subwords_per_surface_token": (
            subword_token_total / surface_token_total if surface_token_total else 0.0
        ),
        "single_piece_rate": (
            single_piece_token_total / surface_token_total if surface_token_total else 0.0
        ),
    }
    return X, Y, stats


def prepare_language_resources(lang: str, config: dict):
    texts = load_language_texts(lang, config)
    train_texts, valid_texts, test_texts = split_texts(texts)

    tokenizer = load_or_train_bpe(lang, train_texts, config)
    stoi = tokenizer.stoi
    itos = tokenizer.itos
    surface_stats = build_surface_stats(train_texts)

    Xtr, Ytr, train_stats = vectorize_bpe_texts(
        train_texts,
        tokenizer=tokenizer,
        stoi=stoi,
        n=config["n"],
        label=f"{language_label(lang)} train",
    )
    Xva, Yva, valid_stats = vectorize_bpe_texts(
        valid_texts,
        tokenizer=tokenizer,
        stoi=stoi,
        n=config["n"],
        label=f"{language_label(lang)} valid",
    )
    Xte, Yte, test_stats = vectorize_bpe_texts(
        test_texts,
        tokenizer=tokenizer,
        stoi=stoi,
        n=config["n"],
        label=f"{language_label(lang)} test",
    )

    return {
        "texts": texts,
        "train_texts": train_texts,
        "valid_texts": valid_texts,
        "test_texts": test_texts,
        "tokenizer": tokenizer,
        "stoi": stoi,
        "itos": itos,
        "Xtr": Xtr,
        "Ytr": Ytr,
        "Xva": Xva,
        "Yva": Yva,
        "Xte": Xte,
        "Yte": Yte,
        "train_stats": train_stats,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "surface_stats": surface_stats,
    }


def resolve_device(device_id: int | None):
    if device_id is not None and device_id >= 0 and torch.cuda.is_available():
        usable = min(device_id, torch.cuda.device_count() - 1)
        torch.cuda.set_device(usable)
        device = torch.device(f"cuda:{usable}")
        print(f"Using {device} -> {torch.cuda.get_device_name(usable)}")
        return device

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Using {device} -> {torch.cuda.get_device_name(0)}")
        return device

    print("CUDA not available, using CPU.")
    return torch.device("cpu")


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
    order = np.random.permutation(size) if shuffle else np.arange(size)
    for start in range(0, size, batch_size):
        indices = order[start : start + batch_size]
        xb = torch.from_numpy(X[indices]).long().to(device, non_blocking=True)
        yb = torch.from_numpy(Y[indices]).long().to(device, non_blocking=True)
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
    return float(np.exp(min(mean_nll(model, X, Y, device=device, batch_size=batch_size), 50.0)))


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_model_checkpoint(lang, model, config, stoi, itos):
    files = q5_files(lang)
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
    best_valid_nll,
    stale_epochs,
    config,
    itos,
):
    files = q5_files(lang)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer_state.get("state", {}).values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "optimizer_state_dict": optimizer_state,
            "history": history,
            "best_state_dict": {name: tensor.detach().cpu() for name, tensor in best_state.items()},
            "best_valid_nll": float(best_valid_nll),
            "stale_epochs": int(stale_epochs),
            "config": {
                "n": config["n"],
                "m": config["m"],
                "h": config["h"],
                "direct": config["direct"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "patience": config["patience"],
                "target_vocab_size": config["target_vocab_size"],
            },
            "itos": itos,
        },
        files["checkpoint"],
    )
    print(f"[checkpoint] saved -> {files['checkpoint'].name} (epoch {epoch})")


def load_resume_checkpoint(lang, model, optimizer, device, itos, config):
    files = q5_files(lang)
    if not files["checkpoint"].exists():
        return 0, None

    ckpt = torch.load(files["checkpoint"], map_location="cpu")
    if ckpt.get("itos") != itos:
        raise ValueError(f"[checkpoint] vocabulary mismatch for {lang}.")

    saved_cfg = ckpt.get("config", {})
    for key in ("n", "m", "h", "direct", "target_vocab_size"):
        if saved_cfg.get(key) != config.get(key):
            raise ValueError(f"[checkpoint] config mismatch for {lang} on '{key}'.")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    return int(ckpt.get("epoch", 0)), ckpt


def train_model(model, resources, device, config, lang):
    decay_params = [model.C.weight, model.H.weight, model.U.weight]
    if model.W is not None:
        decay_params.append(model.W.weight)

    optimizer = optim.Adam(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": [model.d, model.b], "weight_decay": 0.0},
        ],
        lr=config["lr"],
    )

    history = {
        "train_loss": [],
        "train_ppl_token": [],
        "valid_loss": [],
        "valid_ppl_token": [],
        "epoch_seconds": [],
        "elapsed_seconds": 0.0,
        "best_epoch": 0,
        "best_valid_nll": float("inf"),
        "converged": False,
        "stopped_epoch": 0,
    }

    start_epoch = 0
    stale_epochs = 0
    best_state = copy.deepcopy(model.state_dict())

    start_epoch, resume_state = load_resume_checkpoint(
        lang=lang,
        model=model,
        optimizer=optimizer,
        device=device,
        itos=resources["itos"],
        config=config,
    )
    if resume_state is not None:
        history = resume_state["history"]
        history.setdefault("epoch_seconds", [])
        history.setdefault("elapsed_seconds", 0.0)
        history["best_valid_nll"] = resume_state.get("best_valid_nll", history["best_valid_nll"])
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        best_state = resume_state["best_state_dict"]
        print(
            f"[checkpoint] resumed {lang} from epoch {start_epoch} "
            f"(best valid nll {history['best_valid_nll']:.4f}, "
            f"elapsed {format_duration(history['elapsed_seconds'])})"
        )

    total_batches = max(1, math.ceil(len(resources["Ytr"]) / max(config["batch_size"], 1)))
    print(
        f"Training setup | epochs={config['epochs']} | batch_size={config['batch_size']} | "
        f"train_targets={len(resources['Ytr']):,} | valid_targets={len(resources['Yva']):,} | "
        f"batches/epoch={total_batches}",
        flush=True,
    )

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        epoch_started = time.time()
        progress_step = max(1, total_batches // 20)
        total_nll = 0.0
        total_items = 0

        model.train()
        for batch_idx, (xb, yb) in enumerate(
            batches(resources["Xtr"], resources["Ytr"], batch_size=config["batch_size"], device=device, shuffle=True),
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

            if batch_idx == 1 or batch_idx % progress_step == 0 or batch_idx == total_batches:
                batch_elapsed = time.time() - epoch_started
                epoch_eta = (total_batches - batch_idx) * (batch_elapsed / max(batch_idx, 1))
                print(
                    f"[progress] epoch {epoch:02d}/{config['epochs']} | "
                    f"batch {batch_idx:04d}/{total_batches} ({100.0 * batch_idx / total_batches:5.1f}%) | "
                    f"epoch elapsed {format_duration(batch_elapsed)} | epoch ETA {format_duration(epoch_eta)}",
                    flush=True,
                )

        train_loss = total_nll / max(total_items, 1)
        valid_loss = mean_nll(
            model,
            resources["Xva"],
            resources["Yva"],
            device=device,
            batch_size=config["batch_size"],
        )
        train_ppl_token = float(np.exp(min(train_loss, 50.0)))
        valid_ppl_token = float(np.exp(min(valid_loss, 50.0)))

        epoch_seconds = time.time() - epoch_started
        history["train_loss"].append(train_loss)
        history["train_ppl_token"].append(train_ppl_token)
        history["valid_loss"].append(valid_loss)
        history["valid_ppl_token"].append(valid_ppl_token)
        history["epoch_seconds"].append(epoch_seconds)
        history["elapsed_seconds"] += epoch_seconds

        if valid_loss + 1e-8 < history["best_valid_nll"]:
            history["best_valid_nll"] = valid_loss
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        remaining_epochs = max(config["epochs"] - epoch, 0)
        avg_epoch = history["elapsed_seconds"] / max(len(history["epoch_seconds"]), 1)
        print(
            f"Epoch {epoch:02d} | train loss={train_loss:.4f} | train ppl(token)={train_ppl_token:.2f} | "
            f"valid loss={valid_loss:.4f} | valid ppl(token)={valid_ppl_token:.2f} | "
            f"best epoch={history['best_epoch'] or epoch} | elapsed={format_duration(history['elapsed_seconds'])} | "
            f"remaining~={format_duration(remaining_epochs * avg_epoch)}",
            flush=True,
        )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == config["epochs"]:
            save_resume_checkpoint(
                lang=lang,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                best_state=best_state,
                best_valid_nll=history["best_valid_nll"],
                stale_epochs=stale_epochs,
                config=config,
                itos=resources["itos"],
            )

        if stale_epochs >= config["patience"]:
            history["converged"] = True
            history["stopped_epoch"] = epoch
            break

    if not history["stopped_epoch"]:
        history["stopped_epoch"] = len(history["train_loss"])

    model.load_state_dict(best_state)
    return history


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


def normalized_ppl_from_mean_nll(mean_loss, target_count, word_units):
    if not math.isfinite(mean_loss) or target_count <= 0 or word_units <= 0:
        return float("inf")
    total_nll = float(mean_loss) * float(target_count)
    return float(np.exp(min(total_nll / float(word_units), 50.0)))


def save_figure(fig, path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_convergence(history, resources, lang, save_path):
    train_stats = resources["train_stats"]
    valid_stats = resources["valid_stats"]
    train_word_ppl = [
        normalized_ppl_from_mean_nll(loss, train_stats["subword_targets"], train_stats["surface_word_units"])
        for loss in history["train_loss"]
    ]
    valid_word_ppl = [
        normalized_ppl_from_mean_nll(loss, valid_stats["subword_targets"], valid_stats["surface_word_units"])
        for loss in history["valid_loss"]
    ]
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="valid loss")
    axes[0].axvline(history["best_epoch"], color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_title(f"{language_label(lang)} | Cross-Entropy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean NLL per subword target")
    axes[0].legend()

    axes[1].plot(epochs, train_word_ppl, marker="o", label="train ppl / word-unit")
    axes[1].plot(epochs, valid_word_ppl, marker="o", label="valid ppl / word-unit")
    axes[1].axvline(history["best_epoch"], color="tab:red", linestyle="--", alpha=0.7)
    axes[1].set_title(f"{language_label(lang)} | Per-Word Normalized Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    return save_figure(fig, save_path)


def json_ready(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(val) for val in value]
    return value


def save_json(path, data):
    Path(path).write_text(
        json.dumps(json_ready(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def public_config(config: dict):
    return {key: value for key, value in config.items() if key != "hf_token"}


def find_local_variant(*names):
    for name in names:
        if not name:
            continue
        path = Path(name)
        if path.exists():
            return path
    return None


def load_q4_metric_summary(lang):
    json_path = find_local_variant(
        f"q4_metrics_summary_{lang}.json",
        f"q4_metrics_summary_{lang}",
    )
    if json_path is not None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        best_ngram = min(
            float(data["perplexities"]["laplace_ngram"]),
            float(data["perplexities"]["interp_ngram"]),
        )
        return {
            "source": json_path.name,
            "word_nnlm_ppl": float(data["perplexities"]["nnlm"]),
            "best_ngram_ppl": best_ngram,
            "laplace_ppl": float(data["perplexities"]["laplace_ngram"]),
            "interp_ppl": float(data["perplexities"]["interp_ngram"]),
        }

    csv_path = find_local_variant(
        f"q4_perplexity_table_{lang}.csv",
        f"q4_perplexity_table_{lang}",
    )
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        row_map = {str(row["model"]).strip(): float(row["perplexity"]) for _, row in df.iterrows()}
        nnlm = row_map.get("NNLM")
        laplace = row_map.get("Laplace n-gram")
        interp = row_map.get("Interpolated n-gram")
        if nnlm is not None and laplace is not None and interp is not None:
            return {
                "source": csv_path.name,
                "word_nnlm_ppl": nnlm,
                "best_ngram_ppl": min(laplace, interp),
                "laplace_ppl": laplace,
                "interp_ppl": interp,
            }
    return None


def load_q4_model_artifact(lang):
    path = find_local_variant(f"nnlm_{lang}.pt", f"nnlm_{lang}")
    if path is None:
        return None

    ckpt = torch.load(path, map_location="cpu")
    embeddings = np.asarray(ckpt["state_dict"]["C.weight"], dtype=np.float32)
    itos = list(ckpt["itos"])
    return {
        "source": path.name,
        "embeddings": embeddings,
        "stoi": ckpt["stoi"],
        "itos": itos,
        "config": ckpt["config"],
    }


def cosine(u, v):
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(u, v) / denom)


def is_content_word(token):
    if token in SPECIAL_SET:
        return False
    return re.search(r"\w", str(token), flags=re.UNICODE) is not None


def build_embedding_comparison(lang, resources, model_embeddings, q4_model):
    files = q5_files(lang)
    if q4_model is None:
        empty = pd.DataFrame(
            [
                {
                    "status": "missing_q4_model",
                    "note": "Upload nnlm_<lang>.pt from Question 4 to enable embedding comparisons.",
                }
            ]
        )
        empty.to_csv(files["embedding_compare"], index=False)
        return {
            "status": "missing_q4_model",
            "rows": 0,
        }

    lower_counts = resources["surface_stats"]["lower_counts"]
    representatives = resources["surface_stats"]["representatives"]
    tokenizer = resources["tokenizer"]
    q4_stoi = q4_model["stoi"]
    q4_embeddings = q4_model["embeddings"]

    candidates = [
        word
        for word, _ in sorted(lower_counts.items(), key=lambda item: (-item[1], item[0]))
        if word in q4_stoi and is_content_word(word)
    ]

    rows = []
    for word in candidates[:300]:
        surface_form = representatives.get(word, word)
        pieces = tokenizer.encode_word(surface_form)
        if not pieces:
            continue

        piece_ids = [resources["stoi"][piece] for piece in pieces]
        whole_word = len(pieces) == 1 and token_to_surface_word(pieces[0]) == surface_form
        direct_cosine = None
        if whole_word:
            direct_cosine = cosine(q4_embeddings[q4_stoi[word]], model_embeddings[piece_ids[0]])

        proxy_vector = model_embeddings[piece_ids].mean(axis=0)
        proxy_cosine = cosine(q4_embeddings[q4_stoi[word]], proxy_vector)
        rows.append(
            {
                "word_q4": word,
                "surface_form_q5": surface_form,
                "frequency": int(lower_counts[word]),
                "subword_count": len(pieces),
                "subword_tokens": " | ".join(pieces),
                "is_whole_token": bool(whole_word),
                "direct_cosine_if_whole_token": direct_cosine,
                "mean_subword_cosine_proxy": proxy_cosine,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame(
            [
                {
                    "status": "no_overlap",
                    "note": "No overlapping content words were found between Q4 and the subword model.",
                }
            ]
        ).to_csv(files["embedding_compare"], index=False)
        return {
            "status": "no_overlap",
            "rows": 0,
        }

    df.to_csv(files["embedding_compare"], index=False)
    whole = df[df["is_whole_token"] & df["direct_cosine_if_whole_token"].notna()].copy()
    split = df[df["subword_count"] > 1].copy()

    def mean_or_none(series):
        series = pd.to_numeric(series, errors="coerce").dropna()
        return float(series.mean()) if not series.empty else None

    whole_frequent = whole.sort_values(["frequency", "word_q4"], ascending=[False, True]).head(50)
    split_frequent = split.sort_values(["frequency", "word_q4"], ascending=[False, True]).head(50)

    return {
        "status": "ok",
        "rows": int(len(df)),
        "whole_token_rows": int(len(whole)),
        "split_rows": int(len(split)),
        "mean_whole_token_cosine": mean_or_none(whole["direct_cosine_if_whole_token"]),
        "mean_split_proxy_cosine": mean_or_none(split["mean_subword_cosine_proxy"]),
        "mean_top50_whole_token_cosine": mean_or_none(whole_frequent["direct_cosine_if_whole_token"]),
        "mean_top50_split_proxy_cosine": mean_or_none(split_frequent["mean_subword_cosine_proxy"]),
        "csv": str(files["embedding_compare"]),
    }


def build_written_answers(lang, analysis_summary):
    files = q5_files(lang)
    q4_metrics = analysis_summary.get("q4_metrics")
    q4_model = analysis_summary.get("q4_model")
    embed = analysis_summary.get("embedding_comparison", {})
    ppl = analysis_summary["perplexities"]
    params = analysis_summary["parameter_comparison"]
    split_stats = analysis_summary["split_stats"]

    lines = [
        f"# Question 5 draft for {language_label(lang)}",
        "",
        "## 1. Perplexity",
        (
            f"- Subword NNLM test perplexity (per-word normalized) = "
            f"{ppl['subword_per_word_normalized']:.4f}."
        ),
        (
            f"- Raw subword-token perplexity = {ppl['subword_token_level']:.4f}; "
            f"average subwords per surface token on the test split = "
            f"{split_stats['test']['avg_subwords_per_surface_token']:.4f}."
        ),
    ]

    if q4_metrics is not None:
        lines.append(
            f"- Word-level Q4 NNLM perplexity = {q4_metrics['word_nnlm_ppl']:.4f}; "
            f"best Q4 n-gram perplexity = {q4_metrics['best_ngram_ppl']:.4f}."
        )
        lines.append(
            "- The subword model should usually trade a smaller vocabulary and near-zero OOV risk "
            "for longer prediction sequences."
        )
    else:
        lines.append(
            "- Upload `q4_metrics_summary_<lang>.json` or `q4_perplexity_table_<lang>.csv` if you want "
            "the notebook to fill in the exact comparison against your earlier word-level results."
        )

    lines.extend(
        [
            "",
            "## 2. Embedding Cosine Similarity",
        ]
    )
    if embed.get("status") == "ok":
        lines.append(
            f"- Whole-token overlap rows = {embed['whole_token_rows']}; "
            f"mean cosine on those rows = {fmt_optional(embed['mean_whole_token_cosine'])}."
        )
        lines.append(
            f"- Split-word rows = {embed['split_rows']}; "
            f"proxy cosine from the mean of subword pieces = {fmt_optional(embed['mean_split_proxy_cosine'])}."
        )
        lines.append(
            "- Frequent whole-word tokens usually stay closer across the two spaces because the "
            "subword model can dedicate one vector to the full form, while split words distribute "
            "their signal across multiple pieces."
        )
    else:
        lines.append(
            "- Upload `nnlm_<lang>.pt` from Question 4 to enable the embedding-space comparison."
        )

    lines.extend(
        [
            "",
            "## 3. Sparsity and Parameter Count",
            (
                f"- Subword vocabulary size = {params['subword_vocab_size']:,}; "
                f"subword NNLM total parameters = {params['subword_total_params']:,}."
            ),
        ]
    )
    if q4_model is not None:
        lines.append(
            f"- Word-level vocabulary size = {params['word_vocab_size']:,}; "
            f"word-level NNLM total parameters = {params['word_total_params']:,}."
        )
        lines.append(
            "- The subword model reduces vocabulary-driven parameter growth, but it usually needs "
            "more prediction steps per sentence because many words split into multiple pieces."
        )
    else:
        lines.append(
            "- Upload `nnlm_<lang>.pt` from Question 4 if you want the exact parameter comparison filled in automatically."
        )

    files["written_answers"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return files["written_answers"]


def fmt_optional(value):
    return "nan" if value is None else f"{value:.4f}"


def build_analysis_outputs(lang, resources, model, history, config):
    files = q5_files(lang)
    device = next(model.parameters()).device

    test_nll = mean_nll(model, resources["Xte"], resources["Yte"], device=device, batch_size=config["batch_size"])
    test_ppl_token = float(np.exp(min(test_nll, 50.0)))
    test_ppl_word = normalized_ppl_from_mean_nll(
        test_nll,
        resources["test_stats"]["subword_targets"],
        resources["test_stats"]["surface_word_units"],
    )

    train_log = {
        "test_ppl": test_ppl_word,
        "test_ppl_token_level": test_ppl_token,
        "train_ppl_history": [
            normalized_ppl_from_mean_nll(
                loss,
                resources["train_stats"]["subword_targets"],
                resources["train_stats"]["surface_word_units"],
            )
            for loss in history["train_loss"]
        ],
        "valid_ppl_history": [
            normalized_ppl_from_mean_nll(
                loss,
                resources["valid_stats"]["subword_targets"],
                resources["valid_stats"]["surface_word_units"],
            )
            for loss in history["valid_loss"]
        ],
        "raw_train_token_ppl_history": history["train_ppl_token"],
        "raw_valid_token_ppl_history": history["valid_ppl_token"],
        "history": history,
        "split_stats": {
            "train": resources["train_stats"],
            "valid": resources["valid_stats"],
            "test": resources["test_stats"],
        },
    }
    save_json(files["train_log"], train_log)

    q4_metrics = load_q4_metric_summary(lang)
    q4_model = load_q4_model_artifact(lang)

    perplexity_rows = [
        {
            "model": "Subword NNLM",
            "perplexity": test_ppl_word,
            "normalization": "per surface token + sentence end",
            "source": files["train_log"].name,
        },
        {
            "model": "Subword NNLM (token-level)",
            "perplexity": test_ppl_token,
            "normalization": "per subword target",
            "source": files["train_log"].name,
        },
    ]
    if q4_metrics is not None:
        perplexity_rows.append(
            {
                "model": "Word-level NNLM (Q4)",
                "perplexity": q4_metrics["word_nnlm_ppl"],
                "normalization": "word-level",
                "source": q4_metrics["source"],
            }
        )
        perplexity_rows.append(
            {
                "model": "Best n-gram baseline (Q4)",
                "perplexity": q4_metrics["best_ngram_ppl"],
                "normalization": "word-level",
                "source": q4_metrics["source"],
            }
        )
    pd.DataFrame(perplexity_rows).to_csv(files["perplexity_compare"], index=False)

    subword_counts = count_parameters(
        V=len(resources["itos"]),
        m=config["m"],
        h=config["h"],
        n=config["n"],
        direct=config["direct"],
    )
    parameter_rows = [
        {
            "model": "Subword NNLM",
            "lang": lang,
            "language": language_label(lang),
            "vocab_size": len(resources["itos"]),
            "embedding_params": subword_counts["C"],
            "total_params": subword_counts["TOTAL"],
            "n": config["n"],
            "m": config["m"],
            "h": config["h"],
            "direct": config["direct"],
            "source": files["model"].name,
        }
    ]
    parameter_summary = {
        "subword_vocab_size": int(len(resources["itos"])),
        "subword_total_params": int(subword_counts["TOTAL"]),
        "subword_embedding_params": int(subword_counts["C"]),
        "word_vocab_size": None,
        "word_total_params": None,
        "word_embedding_params": None,
    }

    if q4_model is not None:
        word_cfg = q4_model["config"]
        word_counts = count_parameters(
            V=int(word_cfg["V"]),
            m=int(word_cfg["m"]),
            h=int(word_cfg["h"]),
            n=int(word_cfg["n"]),
            direct=bool(word_cfg["direct"]),
        )
        parameter_rows.append(
            {
                "model": "Word-level NNLM (Q4)",
                "lang": lang,
                "language": language_label(lang),
                "vocab_size": int(word_cfg["V"]),
                "embedding_params": word_counts["C"],
                "total_params": word_counts["TOTAL"],
                "n": int(word_cfg["n"]),
                "m": int(word_cfg["m"]),
                "h": int(word_cfg["h"]),
                "direct": bool(word_cfg["direct"]),
                "source": q4_model["source"],
            }
        )
        parameter_summary["word_vocab_size"] = int(word_cfg["V"])
        parameter_summary["word_total_params"] = int(word_counts["TOTAL"])
        parameter_summary["word_embedding_params"] = int(word_counts["C"])

    pd.DataFrame(parameter_rows).to_csv(files["parameter_compare"], index=False)

    embeddings = model.embeddings()
    embedding_summary = build_embedding_comparison(
        lang=lang,
        resources=resources,
        model_embeddings=embeddings,
        q4_model=q4_model,
    )

    analysis_summary = {
        "lang": lang,
        "language": language_label(lang),
        "config": {
            "n": config["n"],
            "m": config["m"],
            "h": config["h"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "patience": config["patience"],
            "direct": config["direct"],
            "target_vocab_size": config["target_vocab_size"],
        },
        "tokenizer_summary": resources["tokenizer"].summary,
        "split_stats": {
            "train": resources["train_stats"],
            "valid": resources["valid_stats"],
            "test": resources["test_stats"],
        },
        "perplexities": {
            "subword_token_level": test_ppl_token,
            "subword_per_word_normalized": test_ppl_word,
        },
        "q4_metrics": q4_metrics,
        "q4_model": {
            "source": q4_model["source"],
            "config": q4_model["config"],
        }
        if q4_model is not None
        else None,
        "parameter_comparison": parameter_summary,
        "embedding_comparison": embedding_summary,
        "files": {
            "train_log": str(files["train_log"]),
            "perplexity_compare": str(files["perplexity_compare"]),
            "parameter_compare": str(files["parameter_compare"]),
            "embedding_compare": str(files["embedding_compare"]),
        },
    }
    save_json(files["analysis_summary"], analysis_summary)
    build_written_answers(lang, analysis_summary)
    return train_log, analysis_summary


def bundle_files(bundle_name: str | Path, members):
    bundle_path = Path(bundle_name)
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            path = Path(member)
            if path.exists():
                zf.write(path, arcname=path.name)
    print(f"Created {bundle_path.name}")
    return bundle_path


def build_language_bundle(lang):
    files = q5_files(lang)
    members = [
        files["tokenizer"],
        files["model"],
        files["checkpoint"],
        files["train_log"],
        files["train_state"],
        files["train_summary"],
        files["bpe_summary"],
        files["plot"],
        files["perplexity_compare"],
        files["parameter_compare"],
        files["embedding_compare"],
        files["analysis_summary"],
        files["written_answers"],
        cache_path_for_lang(lang),
    ]
    return bundle_files(files["bundle"], members)


def build_problem_5_bundle_if_ready():
    required = [
        Path("bpe_en.json"),
        Path("bpe_hi.json"),
        Path("bpe_nnlm_en.pt"),
        Path("bpe_nnlm_hi.pt"),
        Path("bpe_train_log_en.json"),
        Path("bpe_train_log_hi.json"),
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        print("Combined Q5 zip not created yet. Missing:", ", ".join(missing))
        return None
    bundle_files("problem_5.zip", required)
    bundle_files("question_5.zip", required)
    return Path("problem_5.zip")


def maybe_build_combined_written_answers():
    needed = [q5_files(lang)["analysis_summary"] for lang in LANGS]
    if not all(path.exists() for path in needed):
        return None

    summaries = {
        lang: json.loads(q5_files(lang)["analysis_summary"].read_text(encoding="utf-8"))
        for lang in LANGS
    }
    lines = ["# Question 5 written-answer draft", ""]
    for lang in LANGS:
        summary = summaries[lang]
        ppl = summary["perplexities"]
        params = summary["parameter_comparison"]
        embed = summary["embedding_comparison"]
        lines.extend(
            [
                f"## {language_label(lang)}",
                (
                    f"- Subword NNLM test perplexity (per-word normalized): "
                    f"{ppl['subword_per_word_normalized']:.4f}"
                ),
                (
                    f"- Subword NNLM token-level perplexity: "
                    f"{ppl['subword_token_level']:.4f}"
                ),
            ]
        )
        if summary["q4_metrics"] is not None:
            lines.append(
                f"- Word-level Q4 NNLM: {summary['q4_metrics']['word_nnlm_ppl']:.4f}; "
                f"best Q4 n-gram: {summary['q4_metrics']['best_ngram_ppl']:.4f}"
            )
        if params["word_total_params"] is not None:
            lines.append(
                f"- Subword params: {params['subword_total_params']:,}; "
                f"word-level params: {params['word_total_params']:,}"
            )
        if embed.get("status") == "ok":
            lines.append(
                f"- Whole-token cosine mean: {fmt_optional(embed['mean_whole_token_cosine'])}; "
                f"split-word proxy cosine mean: {fmt_optional(embed['mean_split_proxy_cosine'])}"
            )
        lines.append("")

    output = Path("Q5_WRITTEN_ANSWERS.md")
    output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output


def extract_bundle_if_present(bundle_name):
    path = Path(bundle_name)
    if path.exists():
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(".")
        print(f"Extracted {path.name}")


def copy_or_repack_input(src: Path, dst: Path):
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_file():
        shutil.copy2(src, dst)
        print(f"Copied input file {src} -> {dst}")
        return

    if src.is_dir():
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
            for member in sorted(src.rglob("*")):
                if member.is_file():
                    zf.write(member, arcname=member.relative_to(src).as_posix())
        print(f"Repacked input directory {src} -> {dst}")
        return

    raise FileNotFoundError(f"Input artifact not found: {src}")


def stage_kaggle_input_files(required_names):
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return

    for name in required_names:
        local_path = Path(name)
        if local_path.exists() and local_path.is_file():
            continue
        matches = list(input_root.rglob(name))
        if not matches:
            continue
        file_matches = [path for path in matches if path.is_file()]
        dir_matches = [path for path in matches if path.is_dir()]
        if file_matches:
            copy_or_repack_input(file_matches[0], local_path)
        elif dir_matches:
            copy_or_repack_input(dir_matches[0], local_path)


def run_full_language_pipeline(lang: str, config: dict, device_id: int | None = None):
    set_shared_seed()
    files = q5_files(lang)
    expected_inputs = [
        EN_CACHE,
        HI_CACHE,
        files["tokenizer"].name,
        files["checkpoint"].name,
        f"q4_{lang}_artifacts.zip",
        "question_4_bundle.zip",
        f"nnlm_{lang}.pt",
        f"q4_metrics_summary_{lang}.json",
        f"q4_metrics_summary_{lang}",
        f"q4_perplexity_table_{lang}.csv",
        f"q4_perplexity_table_{lang}",
    ]
    stage_kaggle_input_files(expected_inputs)
    extract_bundle_if_present(f"q4_{lang}_artifacts.zip")
    extract_bundle_if_present("question_4_bundle.zip")

    if device_id is None:
        device_id = 0 if torch.cuda.is_available() else -1
    device = resolve_device(device_id)

    print(f"Running full Q5 pipeline for {language_label(lang)} on device_id={device_id}")
    resources = prepare_language_resources(lang, config)

    model = NNLM(
        V=len(resources["itos"]),
        m=config["m"],
        h=config["h"],
        n=config["n"],
        direct=config["direct"],
    ).to(device)

    history = train_model(model, resources=resources, device=device, config=config, lang=lang)
    save_model_checkpoint(lang, model, config, resources["stoi"], resources["itos"])
    plot_path = plot_convergence(history, resources, lang=lang, save_path=files["plot"])

    train_summary = {
        "lang": lang,
        "language": language_label(lang),
        "config": public_config(config),
        "tokenizer_summary": resources["tokenizer"].summary,
        "split_stats": {
            "train": resources["train_stats"],
            "valid": resources["valid_stats"],
            "test": resources["test_stats"],
        },
        "history": history,
        "plot_path": plot_path,
        "vocab_size": len(resources["itos"]),
    }
    with files["train_state"].open("wb") as handle:
        pickle.dump(train_summary, handle)
    save_json(files["train_summary"], train_summary)

    build_analysis_outputs(lang, resources, model, history, config)
    bundle = build_language_bundle(lang)
    combined = build_problem_5_bundle_if_ready()
    written = maybe_build_combined_written_answers()

    print(f"Completed Q5 pipeline for {language_label(lang)}")
    print("Artifacts:")
    for path in [
        files["tokenizer"],
        files["model"],
        files["checkpoint"],
        files["train_log"],
        files["plot"],
        files["perplexity_compare"],
        files["parameter_compare"],
        files["embedding_compare"],
        files["analysis_summary"],
        files["written_answers"],
        bundle,
        combined,
        written,
    ]:
        if path is not None and Path(path).exists():
            print(" -", Path(path).name)
