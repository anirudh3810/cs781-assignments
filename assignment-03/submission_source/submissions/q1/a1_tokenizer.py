"""Adapter around the byte-level BPE implementation developed for Assignment 1.

The tokenizer is retrained on Assignment-3's training text and persists a corpus-
fingerprinted JSON model so a fresh Kaggle session can reconstruct it exactly.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SPECIAL_SET = set(SPECIALS)
TOKENIZER_SUFFIX = "</w>"
ALGORITHM_VERSION = 2


def _surface_tokenize(text):
    # Deterministic, dependency-free equivalent of the surface-token stage used in A1.
    return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)


def _make_symbol(byte_values, terminal):
    return (tuple(int(value) for value in byte_values), bool(terminal))


def _symbol_bytes(symbol):
    return symbol[0]


def _symbol_terminal(symbol):
    return bool(symbol[1])


def _merge_symbols(left, right):
    return _make_symbol(_symbol_bytes(left) + _symbol_bytes(right), _symbol_terminal(right))


def _initial_symbols_for_word(word):
    payload = str(word).encode("utf-8")
    if not payload:
        return []
    if len(payload) == 1:
        return [_make_symbol((payload[0],), True)]
    return ([_make_symbol((value,), False) for value in payload[:-1]]
            + [_make_symbol((payload[-1],), True)])


def _sequence_pair_counts(symbols):
    counts = Counter()
    for idx in range(len(symbols) - 1):
        counts[(symbols[idx], symbols[idx + 1])] += 1
    return counts


def _merge_pair_in_sequence(symbols, pair, merged_symbol):
    output, idx = [], 0
    left, right = pair
    while idx < len(symbols):
        if idx + 1 < len(symbols) and symbols[idx] == left and symbols[idx + 1] == right:
            output.append(merged_symbol)
            idx += 2
        else:
            output.append(symbols[idx])
            idx += 1
    return output


def _bytes_to_display(byte_values):
    payload = bytes(byte_values)
    try:
        text = payload.decode("utf-8")
        if text and TOKENIZER_SUFFIX not in text and all(ch not in "\x00\r\n\t" for ch in text):
            return text
    except UnicodeDecodeError:
        pass
    return "".join(f"<0x{value:02X}>" for value in payload)


def _symbol_to_token(symbol):
    body = _bytes_to_display(_symbol_bytes(symbol))
    return body + TOKENIZER_SUFFIX if _symbol_terminal(symbol) else body


def _token_to_symbol(token):
    token = str(token)
    terminal = token.endswith(TOKENIZER_SUFFIX)
    body = token[:-len(TOKENIZER_SUFFIX)] if terminal else token
    payload, idx = bytearray(), 0
    while idx < len(body):
        match = re.match(r"<0x([0-9A-Fa-f]{2})>", body[idx:])
        if match:
            payload.append(int(match.group(1), 16))
            idx += len(match.group(0))
        else:
            payload.extend(body[idx].encode("utf-8"))
            idx += 1
    return _make_symbol(tuple(payload), terminal)


def _full_base_byte_symbols():
    return [_make_symbol((value,), terminal) for terminal in (False, True) for value in range(256)]


class Assignment1ByteLevelBPE:
    """Deterministic byte-level BPE, adapted from the submitted A1 implementation."""

    def __init__(self, target_vocab_size=8000):
        self.target_vocab_size = int(target_vocab_size)
        self.base_symbols = []
        self.merged_symbols = []
        self.merge_rules = []
        self.merge_lookup = {}
        self.merge_rank = {}
        self.itos = list(SPECIALS)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.word_cache = {}
        self.metadata = {}

    @property
    def vocab_size(self):
        return len(self.itos)

    def get_vocab_size(self):
        return self.vocab_size

    def fit(self, texts):
        word_counter = Counter()
        word_first_seen = {}
        for text in texts:
            for token in _surface_tokenize(text):
                word_counter[token] += 1
                word_first_seen.setdefault(token, len(word_first_seen))

        base_symbols = _full_base_byte_symbols()
        base_seen = set(base_symbols)
        words = sorted(word_counter, key=word_first_seen.get)
        frequencies = [int(word_counter[word]) for word in words]
        sequences = [_initial_symbols_for_word(word) for word in words]
        pair_freqs, pair_to_words = Counter(), defaultdict(set)
        for word_idx, sequence in enumerate(sequences):
            for pair, count in _sequence_pair_counts(sequence).items():
                pair_freqs[pair] += frequencies[word_idx] * count
                pair_to_words[pair].add(word_idx)
        heap = [(-int(freq), pair) for pair, freq in pair_freqs.items()]
        heapq.heapify(heap)
        merged_symbols, merged_seen, merge_rules = [], set(), []
        target = max(self.target_vocab_size, len(SPECIALS) + len(base_symbols))

        while len(SPECIALS) + len(base_symbols) + len(merged_symbols) < target and heap:
            neg_freq, pair = heapq.heappop(heap)
            current_freq = int(pair_freqs.get(pair, 0))
            if current_freq != -neg_freq:
                continue
            if current_freq <= 1:
                break
            merged = _merge_symbols(pair[0], pair[1])
            merge_rules.append((pair[0], pair[1], merged))
            if merged not in base_seen and merged not in merged_seen:
                merged_seen.add(merged)
                merged_symbols.append(merged)
            for word_idx in list(pair_to_words.get(pair, ())):
                old_sequence = sequences[word_idx]
                old_counts = _sequence_pair_counts(old_sequence)
                if pair not in old_counts:
                    continue
                weight = frequencies[word_idx]
                for old_pair, count in old_counts.items():
                    updated = int(pair_freqs.get(old_pair, 0) - weight * count)
                    if updated > 0:
                        pair_freqs[old_pair] = updated
                        heapq.heappush(heap, (-updated, old_pair))
                    else:
                        pair_freqs.pop(old_pair, None)
                    pair_to_words[old_pair].discard(word_idx)
                new_sequence = _merge_pair_in_sequence(old_sequence, pair, merged)
                sequences[word_idx] = new_sequence
                for new_pair, count in _sequence_pair_counts(new_sequence).items():
                    updated = int(pair_freqs.get(new_pair, 0) + weight * count)
                    pair_freqs[new_pair] = updated
                    pair_to_words[new_pair].add(word_idx)
                    heapq.heappush(heap, (-updated, new_pair))

        self.base_symbols = base_symbols
        self.merged_symbols = merged_symbols
        self.merge_rules = merge_rules
        self.merge_lookup = {(left, right): merged for left, right, merged in merge_rules}
        self.merge_rank = {(left, right): rank for rank, (left, right, _) in enumerate(merge_rules)}
        self.itos = (list(SPECIALS)
                     + [_symbol_to_token(symbol) for symbol in base_symbols]
                     + [_symbol_to_token(symbol) for symbol in merged_symbols])
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.word_cache = {}
        return self

    def encode_word(self, word):
        word = str(word)
        if word in self.word_cache:
            return list(self.word_cache[word])
        symbols = _initial_symbols_for_word(word)
        while len(symbols) > 1:
            candidates = [(self.merge_rank[(symbols[i], symbols[i + 1])], i)
                          for i in range(len(symbols) - 1)
                          if (symbols[i], symbols[i + 1]) in self.merge_rank]
            if not candidates:
                break
            _, position = min(candidates)
            pair = (symbols[position], symbols[position + 1])
            symbols = (symbols[:position] + [self.merge_lookup[pair]] + symbols[position + 2:])
        pieces = tuple(_symbol_to_token(symbol) for symbol in symbols)
        self.word_cache[word] = pieces
        return list(pieces)

    def tokenize_text(self, text):
        pieces = []
        for token in _surface_tokenize(text):
            pieces.extend(self.encode_word(token))
        return pieces

    def encode(self, text):
        unknown = self.stoi["<unk>"]
        ids = []
        for piece in self.tokenize_text(text):
            token_id = self.stoi.get(piece)
            if token_id is not None:
                ids.append(int(token_id))
                continue
            symbol = _token_to_symbol(piece)
            raw = _symbol_bytes(symbol)
            for idx, value in enumerate(raw):
                terminal = idx == len(raw) - 1 and _symbol_terminal(symbol)
                fallback = _symbol_to_token(_make_symbol((value,), terminal))
                ids.append(int(self.stoi.get(fallback, unknown)))
        return ids

    @staticmethod
    def _detokenize(surface_tokens):
        text = ""
        no_space_before = set(".,!?;:%)]}’\"")
        no_space_after = set("([{‘\"")
        for token in surface_tokens:
            if not text:
                text = token
            elif token and token[0] in no_space_before:
                text += token
            elif text[-1] in no_space_after:
                text += token
            else:
                text += " " + token
        return text

    def decode(self, ids):
        words, buffer = [], bytearray()
        for value in ids:
            idx = int(value)
            if idx < 0 or idx >= len(self.itos):
                token = "<unk>"
            else:
                token = self.itos[idx]
            if token in SPECIAL_SET:
                if buffer:
                    words.append(bytes(buffer).decode("utf-8", errors="replace"))
                    buffer.clear()
                if token == "<unk>":
                    words.append(token)
                continue
            symbol = _token_to_symbol(token)
            buffer.extend(_symbol_bytes(symbol))
            if _symbol_terminal(symbol):
                words.append(bytes(buffer).decode("utf-8", errors="replace"))
                buffer.clear()
        if buffer:
            words.append(bytes(buffer).decode("utf-8", errors="replace"))
        return self._detokenize(words)

    def to_dict(self):
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "base_tokens": [_symbol_to_token(symbol) for symbol in self.base_symbols],
            "merge_tokens": [_symbol_to_token(symbol) for symbol in self.merged_symbols],
            "merges": [{"left": _symbol_to_token(left), "right": _symbol_to_token(right),
                        "merged": _symbol_to_token(merged)}
                       for left, right, merged in self.merge_rules],
            "metadata": self.metadata,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(data.get("target_vocab_size", data["vocab_size"]))
        obj.itos = list(data["itos"])
        obj.stoi = {token: idx for idx, token in enumerate(obj.itos)}
        obj.base_symbols = [_token_to_symbol(token) for token in data.get("base_tokens", [])]
        obj.merged_symbols = [_token_to_symbol(token) for token in data.get("merge_tokens", [])]
        obj.merge_rules = [(_token_to_symbol(item["left"]), _token_to_symbol(item["right"]),
                            _token_to_symbol(item["merged"])) for item in data.get("merges", [])]
        obj.merge_lookup = {(left, right): merged for left, right, merged in obj.merge_rules}
        obj.merge_rank = {(left, right): rank for rank, (left, right, _) in enumerate(obj.merge_rules)}
        obj.metadata = dict(data.get("metadata", {}))
        return obj


def train_tokenizer(train_text: str, vocab_size: int):
    """Retrain/load the A1 BPE on the Assignment-3 training corpus."""
    corpus_hash = hashlib.sha256(train_text.encode("utf-8")).hexdigest()
    cache_dir = Path(os.environ.get("A3_TOKENIZER_DIR", "artifacts/shared/tokenizer"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "a1_bpe_tokenizer.json"
    expected = {
        "algorithm_version": ALGORITHM_VERSION,
        "training_corpus_sha256": corpus_hash,
        "requested_vocab_size": int(vocab_size),
    }
    if model_path.is_file():
        candidate = Assignment1ByteLevelBPE.load(model_path)
        if all(candidate.metadata.get(key) == value for key, value in expected.items()):
            return candidate
        raise RuntimeError(
            "An incompatible persisted tokenizer exists. Move it aside instead of silently "
            f"overwriting it: {model_path}"
        )
    tokenizer = Assignment1ByteLevelBPE(vocab_size).fit([train_text])
    tokenizer.metadata = {
        **expected,
        "effective_vocab_size": tokenizer.vocab_size,
        "special_tokens": {token: tokenizer.stoi[token] for token in SPECIALS},
    }
    tokenizer.save(model_path)
    serialized = model_path.read_bytes()
    tokenizer.metadata["model_sha256"] = hashlib.sha256(serialized).hexdigest()
    (cache_dir / "tokenizer_metadata.json").write_text(
        json.dumps(tokenizer.metadata, indent=2), encoding="utf-8"
    )
    return tokenizer
