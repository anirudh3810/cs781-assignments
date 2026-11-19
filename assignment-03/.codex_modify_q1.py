import json
import textwrap
from pathlib import Path


NOTEBOOK = Path("copy_assignment3.ipynb")
SOURCE_NOTEBOOK = Path("cs781_assignment3_questions.ipynb")


def src(text: str):
    text = textwrap.dedent(text).lstrip("\n")
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def code_cell(cell_id: str, text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def markdown_cell(cell_id: str, text: str):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src(text),
    }


# Always regenerate the working copy from the untouched starter. This also makes the
# implementation script idempotent while it is being locally validated.
nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))


def index(cell_id: str) -> int:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == cell_id:
            return i
    raise KeyError(cell_id)


def replace(cell_id: str, text: str):
    nb["cells"][index(cell_id)]["source"] = src(text)


def insert_after(cell_id: str, cells):
    at = index(cell_id) + 1
    nb["cells"][at:at] = cells


# A stdlib-only project-root bootstrap must precede the original setup guard.  It makes
# a read-only Kaggle Dataset copy writable before the %%writefile cells execute.
insert_after("7facd988", [code_cell("q1-project-bootstrap", r'''
# Student-added bootstrap: locate the complete Assignment-3 package in Colab/Kaggle.
# This cell uses only the Python standard library so it can run before common.py exists.
import os as _bootstrap_os
import shutil as _bootstrap_shutil
from pathlib import Path as _BootstrapPath

_A3_PROJECT_ROOT = _BootstrapPath.cwd().resolve()
if not (_A3_PROJECT_ROOT / "common.py").is_file():
    _direct_candidates = [
        _BootstrapPath("/content/assignment_3"),
        _BootstrapPath("/kaggle/working/assignment_3"),
        _BootstrapPath("/kaggle/working"),
    ]
    _source_candidates = [
        p.resolve() for p in _direct_candidates
        if (p / "common.py").is_file() and (p / "download_data.py").is_file()
    ]
    _kaggle_input = _BootstrapPath("/kaggle/input")
    if not _source_candidates and _kaggle_input.is_dir():
        _source_candidates = sorted({
            p.parent.resolve() for p in _kaggle_input.rglob("common.py")
            if (p.parent / "download_data.py").is_file()
        })
    if len(_source_candidates) > 1:
        raise RuntimeError(
            "Multiple Assignment-3 support packages were found. Keep one attached package "
            f"or start the notebook inside the intended folder: {_source_candidates}"
        )
    if len(_source_candidates) == 1:
        _source_root = _source_candidates[0]
        if str(_source_root).startswith("/kaggle/input/"):
            _writable_root = _BootstrapPath("/kaggle/working/assignment_3")
            _bootstrap_shutil.copytree(_source_root, _writable_root, dirs_exist_ok=True)
            _A3_PROJECT_ROOT = _writable_root.resolve()
        else:
            _A3_PROJECT_ROOT = _source_root
        _bootstrap_os.chdir(_A3_PROJECT_ROOT)

print("Assignment-3 project root:", _BootstrapPath.cwd().resolve())
''')])


# Restore a previously exported archive before tokenizer construction.  Source snapshots
# remain under artifacts/ and never overwrite the active root-level Q1/Q2 implementation.
insert_after("2e746d22", [code_cell("q1-recovery-bootstrap", r'''
# Optional crash/cross-account recovery bootstrap.
# Set this to an exact attached ZIP when more than one non-identical candidate is present.
Q1_RECOVERY_ARCHIVE = None  # e.g. "/kaggle/input/my-a3-q1/q1_all_artifacts.zip"

import hashlib as _recovery_hashlib
import tempfile as _recovery_tempfile
import zipfile as _recovery_zipfile
import shutil as _recovery_shutil

def _recovery_sha256(path, chunk_size=1024 * 1024):
    digest = _recovery_hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _archive_mode(path):
    try:
        with _recovery_zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("recovery/integrity_manifest.json"))
            return manifest.get("mode")
    except Exception:
        return None

def _discover_q1_archives(explicit=None, expected_mode=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured Q1 recovery archive does not exist: {path}")
        mode = _archive_mode(path)
        if mode != expected_mode:
            raise ValueError(
                f"Configured recovery archive mode is {mode!r}, expected {expected_mode!r}: {path}"
            )
        return [path]
    primary = "q1_smoke_all_artifacts.zip" if expected_mode == "smoke" else "q1_all_artifacts.zip"
    names = {primary, primary.replace(".zip", ".previous.zip")}
    roots = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in names:
            candidates.extend(root.rglob(name) if root.name == "input" else root.glob(name))
    unique = {}
    for path in candidates:
        if path.is_file() and _archive_mode(path) == expected_mode:
            unique.setdefault(_recovery_sha256(path), path.resolve())
    return sorted(unique.values(), key=lambda p: str(p))

def _restore_q1_archive(archive_path, destination=None):
    archive_path = Path(archive_path).resolve()
    destination = Path(destination or Path.cwd()).resolve()
    with _recovery_zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt Q1 recovery ZIP member: {bad}")
        names = set(archive.namelist())
        manifest_name = "recovery/integrity_manifest.json"
        if manifest_name not in names:
            raise ValueError("Recovery ZIP has no integrity manifest.")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        with _recovery_tempfile.TemporaryDirectory(prefix="a3_q1_restore_") as tmp:
            tmp_root = Path(tmp).resolve()
            for member in archive.infolist():
                if member.is_dir() or member.filename.startswith("recovery/"):
                    continue
                target = (tmp_root / member.filename).resolve()
                if tmp_root not in target.parents:
                    raise ValueError(f"Unsafe path in recovery ZIP: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(target, "wb") as sink:
                    _recovery_shutil.copyfileobj(source, sink)
            for rel, info in manifest.get("files", {}).items():
                restored = (tmp_root / rel).resolve()
                if tmp_root not in restored.parents:
                    raise ValueError(f"Unsafe manifest path in recovery ZIP: {rel}")
                if not restored.is_file():
                    raise ValueError(f"Manifest member is missing: {rel}")
                if restored.stat().st_size != int(info["size"]):
                    raise ValueError(f"Size mismatch while restoring {rel}")
                if _recovery_sha256(restored) != info["sha256"]:
                    raise ValueError(f"Checksum mismatch while restoring {rel}")
            conflicts = []
            for rel in manifest.get("files", {}):
                incoming = (tmp_root / rel).resolve()
                current = (destination / rel).resolve()
                if destination not in current.parents:
                    raise ValueError(f"Unsafe restore destination in manifest: {rel}")
                if current.is_file() and _recovery_sha256(current) != _recovery_sha256(incoming):
                    conflicts.append(rel)
            if conflicts:
                raise RuntimeError(
                    "Recovery would overwrite non-identical current work. Resolve or explicitly "
                    f"move these files first: {conflicts[:20]}"
                )
            for rel in manifest.get("files", {}):
                incoming = (tmp_root / rel).resolve()
                current = (destination / rel).resolve()
                if not current.exists():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    staged = current.with_name(current.name + ".restore.tmp")
                    _recovery_shutil.copy2(incoming, staged)
                    if _recovery_sha256(staged) != _recovery_sha256(incoming):
                        staged.unlink(missing_ok=True)
                        raise ValueError(f"Restore staging checksum failed for {rel}")
                    os.replace(staged, current)
    print("Restored and checksum-validated Q1 archive:", archive_path)

_q1_expected_mode = "smoke" if SMOKE_TEST else "full"
_q1_local_state = Path("artifacts") / _q1_expected_mode / "q1"
if Q1_RECOVERY_ARCHIVE is None and _q1_local_state.is_dir() and any(_q1_local_state.rglob("*")):
    _q1_archive_candidates = []
    print("Existing local Q1 state found; automatic ZIP restore skipped:", _q1_local_state)
else:
    _q1_archive_candidates = _discover_q1_archives(Q1_RECOVERY_ARCHIVE, _q1_expected_mode)
if len(_q1_archive_candidates) == 1:
    _restore_q1_archive(_q1_archive_candidates[0])
    for _label in ("standard", "unscaled"):
        _status_path = Path("artifacts") / _q1_expected_mode / "q1" / _label / "status.json"
        _checkpoint_path = Path("artifacts") / _q1_expected_mode / "q1" / _label / "checkpoint.latest.pt"
        if _status_path.is_file():
            print(f"Restored {_label} status:", json.loads(_status_path.read_text()).get("status"))
        elif _checkpoint_path.is_file():
            print(f"Restored {_label}: resumable checkpoint present")
        else:
            print(f"Restored {_label}: no experiment state in this archive")
elif len(_q1_archive_candidates) > 1:
    print("Multiple non-identical Q1 recovery archives found; set Q1_RECOVERY_ARCHIVE explicitly:")
    for _candidate in _q1_archive_candidates:
        print(" -", _candidate)
else:
    print("No Q1 recovery archive selected; starting from current local artifacts.")
''')])


replace("27e8915c", r'''
%%writefile a1_tokenizer.py
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
''')


replace("7ddd06bc", r'''
# Try the required A-1 hook first. The implemented hook is persistent and corpus-fingerprinted.
import hashlib
sys.modules.pop("a1_tokenizer", None)
os.environ["A3_TOKENIZER_DIR"] = str(Path("artifacts/shared/tokenizer").resolve())
tokenizer, USING_A1_TOKENIZER, TOKENIZER_DIAGNOSTICS = load_a1_tokenizer_or_fallback(
    train_text,
    vocab_size=TOKENIZER_VOCAB_SIZE,
    module_path="a1_tokenizer.py",
)
if not USING_A1_TOKENIZER:
    raise RuntimeError("The required retrained Assignment-1 tokenizer was not usable: "
                       + "; ".join(TOKENIZER_DIAGNOSTICS))

print("using retrained A-1 tokenizer:", USING_A1_TOKENIZER)
print("vocab size:", tokenizer.vocab_size)
ids = tokenizer.encode(held_out[0])
print("round trip check:", tokenizer.decode(ids)[:100])

VOCAB_SIZE = tokenizer.vocab_size
_tokenizer_impl = getattr(tokenizer, "_impl", None)
TOKENIZER_FINGERPRINT = getattr(_tokenizer_impl, "metadata", {}).get("model_sha256")
if not TOKENIZER_FINGERPRINT:
    TOKENIZER_FINGERPRINT = hashlib.sha256(
        Path("artifacts/shared/tokenizer/a1_bpe_tokenizer.json").read_bytes()
    ).hexdigest()

_ids_cache = Path("artifacts/shared/tokenizer/train_ids.pt")
_ids_meta = Path("artifacts/shared/tokenizer/train_ids_metadata.json")
_train_text_hash = hashlib.sha256(train_text.encode("utf-8")).hexdigest()
_can_load_ids = False
if _ids_cache.is_file() and _ids_meta.is_file():
    _cached_meta = json.loads(_ids_meta.read_text(encoding="utf-8"))
    _can_load_ids = (_cached_meta.get("tokenizer_fingerprint") == TOKENIZER_FINGERPRINT
                     and _cached_meta.get("training_corpus_sha256") == _train_text_hash)
if _can_load_ids:
    try:
        _cached_ids = torch.load(_ids_cache, map_location="cpu", weights_only=True)
    except TypeError:
        _cached_ids = torch.load(_ids_cache, map_location="cpu")
    train_ids = [int(value) for value in _cached_ids.tolist()]
else:
    train_ids = tokenizer.encode(train_text)
    _ids_cache.parent.mkdir(parents=True, exist_ok=True)
    _ids_tmp = Path(str(_ids_cache) + ".tmp")
    torch.save(torch.tensor(train_ids, dtype=torch.long), _ids_tmp)
    os.replace(_ids_tmp, _ids_cache)
    _ids_meta.write_text(json.dumps({
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "training_corpus_sha256": _train_text_hash,
        "token_count": len(train_ids),
    }, indent=2), encoding="utf-8")
print(f"{len(train_ids):,} training tokens | tokenizer fingerprint: {TOKENIZER_FINGERPRINT[:12]}")
''')


replace("3175e597", r'''
%%writefile attention.py
"""Multi-head causal self-attention, implemented from raw tensor operations."""
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.attention_scale = (1.0 / math.sqrt(self.d_head)
                                if attention_scale is None else float(attention_scale))
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.causal_mask.size(0)}"
            )
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length]
        scores = scores.masked_fill(~allowed.view(1, 1, length, length), float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))
''')


replace("8ef5fbfe", r'''
%%writefile model_arch.py
"""Decoder-only Transformer LM, built from scratch (no built-in Transformer blocks)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from attention import CausalSelfAttention


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return classic Transformer sinusoidal positions with shape (max_len, d_model)."""
    if max_len <= 0 or d_model <= 0:
        raise ValueError("max_len and d_model must be positive")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * even_indices / d_model)
    angles = position * frequencies.unsqueeze(0)
    encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Configurable pre-norm or post-norm causal decoder block."""
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm_first = bool(cfg.norm_first)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout,
            attention_scale=cfg.attention_scale,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class DecoderLM(nn.Module):
    """The model class imported by the autograder."""
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            self.register_buffer(
                "pos_emb",
                sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
                persistent=False,
            )
        elif cfg.pos_emb_type != "rope":
            raise ValueError("pos_emb_type must be 'sinusoidal' or 'rope'")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        _, length = idx.shape
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        x = self.tok_emb(idx)
        if self.cfg.pos_emb_type == "sinusoidal":
            x = x + self.pos_emb[:length].to(dtype=x.dtype).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
''')


insert_after("382d781c", [
    markdown_cell("q1-runtime-heading", r'''
### Question 1 execution, dual-GPU, and recovery helpers

The following student-added runtime keeps the two scaling experiments independent and
resumable. Full training remains an explicit action in Q1.3; importing these helpers does
not start a training run.
'''),
    code_cell("q1-runtime-writefile", r'''
%%writefile q1_runtime.py
"""Restart-safe, deterministic two-GPU runtime for Assignment 3 Question 1."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from common import TokenDataset, assert_within_budget, perplexity_of_texts
from model_arch import DecoderLM, LMConfig


SCHEMA_VERSION = 1
SAMPLER_SCHEME = "epoch_randperm_seed_plus_epoch_v1"
_ARCHIVE_LOCK = threading.Lock()
_PLOT_LOCK = threading.Lock()
_TQDM_LOCK = threading.RLock()
tqdm.set_lock(_TQDM_LOCK)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_hash(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int_sequence(values, chunk_items=100_000):
    digest = hashlib.sha256()
    for start in range(0, len(values), chunk_items):
        digest.update(array("q", (int(x) for x in values[start:start + chunk_items])).tobytes())
    return digest.hexdigest()


def _hash_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        payload = str(text).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _state_dict_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_plain_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    _torch_load(temporary, "cpu")
    os.replace(temporary, path)


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class DeterministicStepBatchSampler(Sampler):
    """Map every global optimization step to a deterministic shuffled batch."""

    def __init__(self, dataset_size, batch_size, seed, start_step, total_steps):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError(
                f"Dataset has {self.dataset_size} chunks, fewer than batch_size={self.batch_size}."
            )

    def __len__(self):
        return max(0, self.total_steps - self.start_step)

    def __iter__(self):
        cached_epoch, order = None, None
        for step in range(self.start_step, self.total_steps):
            epoch = step // self.batches_per_epoch
            batch_in_epoch = step % self.batches_per_epoch
            if epoch != cached_epoch:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed + epoch)
                order = torch.randperm(self.dataset_size, generator=generator)
                cached_epoch = epoch
            start = batch_in_epoch * self.batch_size
            yield order[start:start + self.batch_size].tolist()

    def position_after(self, completed_steps):
        return {
            "scheme": SAMPLER_SCHEME,
            "seed": self.seed,
            "epoch": int(completed_steps) // self.batches_per_epoch,
            "batch_in_epoch": int(completed_steps) % self.batches_per_epoch,
            "batches_per_epoch": self.batches_per_epoch,
        }


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def config(self): return self.root / "config.json"
    @property
    def latest(self): return self.root / "checkpoint.latest.pt"
    @property
    def previous(self): return self.root / "checkpoint.previous.pt"
    @property
    def latest_checksum(self): return self.root / "checkpoint.latest.sha256"
    @property
    def previous_checksum(self): return self.root / "checkpoint.previous.sha256"
    @property
    def best(self): return self.root / "best.pt"
    @property
    def final(self): return self.root / "final.pt"
    @property
    def history(self): return self.root / "history.json"
    @property
    def metrics(self): return self.root / "metrics.json"
    @property
    def status(self): return self.root / "status.json"


class Q1Runtime:
    def __init__(self, *, tokenizer, train_ids, held_out, base_config, train_steps,
                 batch_size, seed, tokenizer_fingerprint, smoke_test,
                 project_root=".", checkpoint_steps=50, checkpoint_seconds=300,
                 archive_seconds=600, learning_rate=3e-4, use_amp=True):
        self.tokenizer = tokenizer
        self.train_ids = [int(value) for value in train_ids]
        self.held_out = list(held_out)
        self.base_config = dict(base_config)
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.smoke_test = bool(smoke_test)
        self.checkpoint_steps = int(checkpoint_steps)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.archive_seconds = float(archive_seconds)
        self.learning_rate = float(learning_rate)
        self.use_amp = bool(use_amp)
        self.project_root = Path(project_root).resolve()
        self.mode = "smoke" if self.smoke_test else "full"
        self.root = self.project_root / "artifacts" / self.mode / "q1"
        self.analysis_dir = self.root / "analysis"
        self.plot_dir = self.root / "plots"
        self.source_dir = self.root / "source_snapshots"
        self.manifest_dir = self.root / "manifests"
        for directory in (self.root, self.analysis_dir, self.plot_dir,
                          self.source_dir, self.manifest_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.train_ids_hash = _hash_int_sequence(self.train_ids)
        self.held_out_hash = _hash_texts(self.held_out)
        self.dataset = TokenDataset(self.train_ids, int(self.base_config["max_seq_len"]))
        if len(self.dataset) < self.batch_size:
            raise ValueError("TokenDataset is too small for the configured training batch size.")
        self.eval_steps = sorted({
            max(1, int(round(self.train_steps * fraction)))
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)
        })
        self._last_archive_time = 0.0
        self._stop_event = threading.Event()
        self.snapshot_sources()

    def paths(self, label):
        return ExperimentPaths(self.root / label)

    def _cfg(self, scale):
        values = dict(self.base_config)
        values["attention_scale"] = scale
        return LMConfig(**values)

    def _source_hashes(self):
        hashes = {}
        for name in ("a1_tokenizer.py", "attention.py", "model_arch.py", "q1_runtime.py"):
            path = self.project_root / name
            if path.is_file():
                hashes[name] = _sha256_file(path)
        return hashes

    def snapshot_sources(self):
        for name in ("a1_tokenizer.py", "attention.py", "model_arch.py", "q1_runtime.py"):
            source = self.project_root / name
            destination = self.source_dir / name
            if source.is_file():
                if destination.is_file() and _sha256_file(destination) != _sha256_file(source):
                    versioned = self.source_dir / f"{source.stem}.{_sha256_file(source)[:12]}{source.suffix}"
                    shutil.copy2(source, versioned)
                else:
                    shutil.copy2(source, destination)

    def _canonical_initial_state(self):
        path = self.manifest_dir / "canonical_initial_state.pt"
        signature = {
            "base_config_without_scale": {
                key: value for key, value in self.base_config.items() if key != "attention_scale"
            },
            "seed": self.seed,
            "source_hashes": self._source_hashes(),
        }
        signature_hash = _json_hash(signature)
        if path.is_file():
            payload = _torch_load(path, "cpu")
            if payload.get("signature_hash") != signature_hash:
                raise RuntimeError(
                    "Existing canonical initialization is incompatible with current code/config. "
                    f"Move aside {path}; it was not overwritten."
                )
            return payload["state_dict"], payload["state_dict_hash"]
        cpu_rng_before = torch.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            model = DecoderLM(self._cfg(None)).cpu()
            state = _cpu_tree(model.state_dict())
        finally:
            torch.set_rng_state(cpu_rng_before)
        state_hash = _state_dict_hash(state)
        _atomic_plain_torch(path, {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "signature_hash": signature_hash,
            "state_dict": state,
            "state_dict_hash": state_hash,
            "created_utc": _utc_now(),
        })
        return state, state_hash

    def _spec(self, label, scale, init_hash):
        cfg = asdict(self._cfg(scale))
        strict = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": f"q1-{self.mode}-{label}",
            "label": label,
            "architecture": cfg,
            "optimizer": {"name": "Adam", "lr": self.learning_rate},
            "target_steps": self.train_steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "sampler_scheme": SAMPLER_SCHEME,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "train_ids_hash": self.train_ids_hash,
            "held_out_hash": self.held_out_hash,
            "initial_state_hash": init_hash,
            "smoke_test": self.smoke_test,
            "source_hashes": self._source_hashes(),
        }
        strict["trajectory_hash"] = _json_hash(strict)
        return strict

    def _write_config(self, spec):
        path = self.paths(spec["label"]).config
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("trajectory_hash") != spec["trajectory_hash"]:
                raise RuntimeError(f"Incompatible experiment config already exists: {path}")
        else:
            _atomic_json(path, spec)

    @staticmethod
    def _validate_checkpoint(payload, expected_hash):
        required = {
            "schema_version", "trajectory_hash", "model_state", "optimizer_state",
            "completed_step", "tokens_processed", "history", "eval_history",
            "cumulative_active_seconds", "sampler_state",
        }
        missing = required - set(payload) if isinstance(payload, dict) else required
        if missing:
            raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version")
        if payload["trajectory_hash"] != expected_hash:
            raise ValueError("Checkpoint trajectory/configuration mismatch")
        if int(payload["completed_step"]) < 0:
            raise ValueError("Invalid completed step")
        return payload

    def _validated_checkpoint_file(self, path, checksum_path, expected_hash):
        path, checksum_path = Path(path), Path(checksum_path)
        if not path.is_file():
            return None
        if checksum_path.is_file():
            expected = checksum_path.read_text(encoding="ascii").strip()
            if _sha256_file(path) != expected:
                return None
        try:
            return self._validate_checkpoint(_torch_load(path, "cpu"), expected_hash)
        except Exception:
            return None

    def _load_checkpoint(self, spec):
        paths = self.paths(spec["label"])
        latest = self._validated_checkpoint_file(
            paths.latest, paths.latest_checksum, spec["trajectory_hash"]
        )
        if latest is not None:
            return latest, "latest"
        previous = self._validated_checkpoint_file(
            paths.previous, paths.previous_checksum, spec["trajectory_hash"]
        )
        return (previous, "previous") if previous is not None else (None, None)

    def _save_checkpoint(self, spec, payload):
        paths = self.paths(spec["label"])
        paths.root.mkdir(parents=True, exist_ok=True)
        candidate = paths.root / f"checkpoint.step_{payload['completed_step']:06d}.{uuid.uuid4().hex}.candidate"
        torch.save(_cpu_tree(payload), candidate)
        loaded = self._validate_checkpoint(_torch_load(candidate, "cpu"), spec["trajectory_hash"])
        if int(loaded["completed_step"]) != int(payload["completed_step"]):
            candidate.unlink(missing_ok=True)
            raise ValueError("Checkpoint validation changed the completed step")
        candidate_hash = _sha256_file(candidate)
        if paths.latest.is_file():
            os.replace(paths.latest, paths.previous)
            if paths.latest_checksum.is_file():
                os.replace(paths.latest_checksum, paths.previous_checksum)
        os.replace(candidate, paths.latest)
        checksum_tmp = paths.latest_checksum.with_suffix(".sha256.tmp")
        checksum_tmp.write_text(candidate_hash, encoding="ascii")
        os.replace(checksum_tmp, paths.latest_checksum)

    def _status(self, spec, status, **extra):
        _atomic_json(self.paths(spec["label"]).status, {
            "experiment_id": spec["experiment_id"],
            "trajectory_hash": spec["trajectory_hash"],
            "status": status,
            "updated_utc": _utc_now(),
            **extra,
        })

    def _is_complete(self, spec):
        paths = self.paths(spec["label"])
        if not (paths.status.is_file() and paths.metrics.is_file() and paths.final.is_file()):
            return False
        status = json.loads(paths.status.read_text(encoding="utf-8"))
        metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
        return (
            status.get("status") == "complete"
            and status.get("trajectory_hash") == spec["trajectory_hash"]
            and int(metrics.get("completed_steps", -1)) == self.train_steps
            and math.isfinite(float(metrics.get("held_out_ppl", float("inf"))))
            and len(metrics.get("train_loss_curve", [])) > 0
        )

    def _reset_if_requested(self, label, force):
        paths = self.paths(label)
        if not force or not paths.root.exists():
            return
        backup = paths.root.with_name(paths.root.name + ".reset_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(paths.root, backup)
        print(f"Explicit reset preserved previous {label} artifacts at {backup}")

    def _checkpoint_payload(self, *, spec, model, optimizer, scaler, step, tokens,
                            history, eval_history, cumulative_seconds, sampler,
                            device, best):
        cuda_rng = None
        if device.type == "cuda":
            cuda_rng = torch.cuda.get_rng_state(device=device).cpu()
        sampler_state = sampler.position_after(step)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": spec["experiment_id"],
            "trajectory_hash": spec["trajectory_hash"],
            "config": spec,
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "scheduler_state": None,
            "scaler_state": _cpu_tree(scaler.state_dict()) if scaler is not None else None,
            "completed_step": int(step),
            "epoch": sampler_state["epoch"],
            "batch_in_epoch": sampler_state["batch_in_epoch"],
            "tokens_processed": int(tokens),
            "history": list(history),
            "eval_history": list(eval_history),
            "best": dict(best),
            "cumulative_active_seconds": float(cumulative_seconds),
            "sampler_state": sampler_state,
            "assigned_device": str(device),
            "cuda_rng_state_for_assigned_device": cuda_rng,
            "created_utc": _utc_now(),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda": torch.version.cuda,
            },
        }

    def _run_worker(self, spec, prepared_model, device_name, position, initial_rng_state):
        label, paths = spec["label"], self.paths(spec["label"])
        device = torch.device(device_name)
        self._write_config(spec)
        if self._is_complete(spec):
            print(f"{label}: compatible completed run found; training skipped.")
            return {"label": label, "status": "complete", "skipped": True}
        checkpoint, checkpoint_source = self._load_checkpoint(spec)
        model = prepared_model.to(device)
        assert_within_budget(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        amp_enabled = self.use_amp and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        step, tokens, history, eval_history, cumulative_before = 0, 0, [], [], 0.0
        best = {"held_out_ppl": float("inf"), "step": None}
        run_kind = "new"
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _move_optimizer_state(optimizer, device)
            if checkpoint.get("scaler_state"):
                scaler.load_state_dict(checkpoint["scaler_state"])
            step = int(checkpoint["completed_step"])
            tokens = int(checkpoint["tokens_processed"])
            history = [row for row in checkpoint["history"] if int(row["step"]) <= step]
            eval_history = [row for row in checkpoint["eval_history"] if int(row["step"]) <= step]
            cumulative_before = float(checkpoint["cumulative_active_seconds"])
            best = dict(checkpoint.get("best", best))
            if device.type == "cuda" and checkpoint.get("cuda_rng_state_for_assigned_device") is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state_for_assigned_device"], device=device)
            run_kind = f"resumed-{checkpoint_source}"
        elif device.type == "cuda":
            torch.cuda.set_rng_state(initial_rng_state, device=device)
        else:
            torch.set_rng_state(initial_rng_state)

        sampler = DeterministicStepBatchSampler(
            len(self.dataset), self.batch_size, self.seed, step, self.train_steps
        )
        loader = DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        self._status(spec, "in_progress", completed_step=step, assigned_device=str(device), run_kind=run_kind)
        session_started = time.perf_counter()
        last_checkpoint_time = session_started
        last_checkpoint_step = step
        completed_normally = False

        try:
            progress = tqdm(
                total=self.train_steps, initial=step, position=position, leave=True,
                desc=f"{label} [{device}] {run_kind}", dynamic_ncols=True,
            )
            model.train()
            for x, y in loader:
                if self._stop_event.is_set():
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    enabled=amp_enabled,
                ):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                tokens += int(x.numel())
                session_elapsed = time.perf_counter() - session_started
                cumulative = cumulative_before + session_elapsed
                history.append({
                    "step": step,
                    "epoch": sampler.position_after(step)["epoch"],
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "tokens_processed": tokens,
                    "elapsed_seconds": cumulative,
                })
                evaluation_happened = False
                if step in self.eval_steps:
                    model.eval()
                    ppl = float(perplexity_of_texts(
                        model, self.tokenizer, self.held_out, device=device,
                        max_len=int(spec["architecture"]["max_seq_len"]),
                    ))
                    model.train()
                    eval_history.append({
                        "step": step,
                        "tokens_processed": tokens,
                        "held_out_ppl": ppl,
                        "elapsed_seconds": cumulative,
                    })
                    evaluation_happened = True
                    if math.isfinite(ppl) and ppl < float(best.get("held_out_ppl", float("inf"))):
                        best = {"held_out_ppl": ppl, "step": step}
                        _atomic_plain_torch(paths.best, _cpu_tree(model.state_dict()))
                now = time.perf_counter()
                checkpoint_due = (
                    step == self.train_steps
                    or evaluation_happened
                    or step - last_checkpoint_step >= self.checkpoint_steps
                    or now - last_checkpoint_time >= self.checkpoint_seconds
                )
                if checkpoint_due:
                    payload = self._checkpoint_payload(
                        spec=spec, model=model, optimizer=optimizer, scaler=scaler,
                        step=step, tokens=tokens, history=history, eval_history=eval_history,
                        cumulative_seconds=cumulative, sampler=sampler, device=device, best=best,
                    )
                    self._save_checkpoint(spec, payload)
                    _atomic_json(paths.history, {
                        "trajectory_hash": spec["trajectory_hash"],
                        "history": history,
                        "eval_history": eval_history,
                    })
                    self._status(spec, "in_progress", completed_step=step,
                                 assigned_device=str(device), last_checkpoint_step=step)
                    last_checkpoint_time, last_checkpoint_step = now, step
                    self.export_recovery_archive(force=False, reason=f"checkpoint-{label}-{step}")
                elapsed = max(1e-9, session_elapsed)
                progress.update(1)
                progress.set_postfix({
                    "loss": f"{float(loss.detach()):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "tok": tokens,
                    "tok/s": f"{(tokens - (checkpoint['tokens_processed'] if checkpoint else 0))/elapsed:.0f}",
                    "total_s": f"{cumulative:.0f}",
                    "ckpt": last_checkpoint_step,
                }, refresh=(step % 5 == 0 or step == self.train_steps))
                if step >= self.train_steps:
                    completed_normally = True
                    break
            progress.close()
            cumulative = cumulative_before + (time.perf_counter() - session_started)
            if not completed_normally:
                payload = self._checkpoint_payload(
                    spec=spec, model=model, optimizer=optimizer, scaler=scaler,
                    step=step, tokens=tokens, history=history, eval_history=eval_history,
                    cumulative_seconds=cumulative, sampler=sampler, device=device, best=best,
                )
                self._save_checkpoint(spec, payload)
                self._status(spec, "interrupted", completed_step=step, assigned_device=str(device))
                return {"label": label, "status": "interrupted", "completed_step": step}

            if not eval_history or int(eval_history[-1]["step"]) != self.train_steps:
                model.eval()
                final_ppl = float(perplexity_of_texts(
                    model, self.tokenizer, self.held_out, device=device,
                    max_len=int(spec["architecture"]["max_seq_len"]),
                ))
                eval_history.append({
                    "step": step, "tokens_processed": tokens,
                    "held_out_ppl": final_ppl, "elapsed_seconds": cumulative,
                })
            final_ppl = float(eval_history[-1]["held_out_ppl"])
            if not math.isfinite(final_ppl):
                raise FloatingPointError("Final held-out perplexity is not finite")
            _atomic_plain_torch(paths.final, _cpu_tree(model.state_dict()))
            metrics = {
                "label": label,
                "attention_scale": spec["architecture"]["attention_scale"],
                "held_out_ppl": final_ppl,
                "train_loss_curve": [float(row["loss"]) for row in history],
                "history": history,
                "eval_history": eval_history,
                "completed_steps": step,
                "tokens_processed": tokens,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "cumulative_active_seconds": cumulative,
                "initial_state_hash": spec["initial_state_hash"],
                "trajectory_hash": spec["trajectory_hash"],
            }
            _atomic_json(paths.metrics, metrics)
            _atomic_json(paths.history, {
                "trajectory_hash": spec["trajectory_hash"],
                "history": history,
                "eval_history": eval_history,
            })
            self._status(spec, "complete", completed_step=step, assigned_device=str(device),
                         final_held_out_ppl=final_ppl)
            print(
                f"{label} completed on {device}: session={time.perf_counter()-session_started:.1f}s, "
                f"cumulative={cumulative:.1f}s, ppl={final_ppl:.3f}"
            )
            return {"label": label, "status": "complete", "metrics": metrics}
        except KeyboardInterrupt:
            self._stop_event.set()
            raise
        except Exception as error:
            # A failed worker never removes its last validated checkpoint or its partner's files.
            self._status(spec, "failed", completed_step=step, assigned_device=str(device),
                         error=repr(error), traceback=traceback.format_exc())
            return {"label": label, "status": "failed", "error": repr(error)}
        finally:
            model.to("cpu")
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def run_scaling_pair(self, *, force_retrain=False, require_two_gpus=True,
                         allow_sequential_fallback=False):
        self._stop_event.clear()
        self._reset_if_requested("standard", force_retrain)
        self._reset_if_requested("unscaled", force_retrain)
        initial_state, init_hash = self._canonical_initial_state()
        specs = {
            "standard": self._spec("standard", None, init_hash),
            "unscaled": self._spec("unscaled", 1.0, init_hash),
        }
        for spec in specs.values():
            self._write_config(spec)
        self.snapshot_sources()
        self.export_recovery_archive(force=True, reason="pre-training")
        completed = {label: self._is_complete(spec) for label, spec in specs.items()}
        pending = [label for label in ("standard", "unscaled") if not completed[label]]
        if not pending:
            print("Both Q1 scaling experiments are already complete; no retraining occurred.")
            return self.collect_results()

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if require_two_gpus and gpu_count < 2 and not allow_sequential_fallback:
            raise RuntimeError(
                f"Q1 full paired training requires two visible CUDA GPUs; found {gpu_count}. "
                "Use a Kaggle dual-GPU runtime. Sequential fallback is intentionally disabled."
            )
        assignments = {"standard": "cuda:0", "unscaled": "cuda:1"}
        if gpu_count < 2:
            fallback = "cuda:0" if gpu_count == 1 else "cpu"
            print(f"CLEAR FALLBACK NOTICE: running pending Q1 jobs sequentially on {fallback}; "
                  "this is not dual-GPU training.")
            assignments = {label: fallback for label in pending}
        models, rng_states = {}, {}
        for label in pending:
            model = DecoderLM(LMConfig(**specs[label]["architecture"]))
            model.load_state_dict(initial_state)
            models[label] = model
            target_device = assignments[label]
            if target_device.startswith("cuda"):
                generator = torch.Generator(device=target_device)
                generator.manual_seed(self.seed)
                rng_states[label] = generator.get_state().cpu()
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed)
                rng_states[label] = generator.get_state()

        outcomes = {}
        if gpu_count >= 2 and len(pending) > 1:
            print("DUAL-GPU Q1: standard -> cuda:0; unscaled -> cuda:1")
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="a3-q1-gpu") as pool:
                futures = {
                    pool.submit(
                        self._run_worker, specs[label], models[label], assignments[label],
                        0 if label == "standard" else 1, rng_states[label]
                    ): label for label in pending
                }
                try:
                    for future in as_completed(futures):
                        label = futures[future]
                        try:
                            outcomes[label] = future.result()
                        except BaseException as error:
                            outcomes[label] = {"label": label, "status": "failed", "error": repr(error)}
                except KeyboardInterrupt:
                    self._stop_event.set()
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                    raise
        else:
            for label in pending:
                outcomes[label] = self._run_worker(
                    specs[label], models[label], assignments[label],
                    0 if label == "standard" else 1, rng_states[label]
                )
        self.export_recovery_archive(force=True, reason="post-pair")
        failures = {label: result for label, result in outcomes.items()
                    if result.get("status") != "complete"}
        if failures:
            raise RuntimeError(
                "One or more Q1 jobs did not complete; successful partner artifacts were preserved: "
                + _canonical_json(failures)
            )
        return self.collect_results()

    def collect_results(self):
        results = {}
        for label in ("standard", "unscaled"):
            path = self.paths(label).metrics
            if not path.is_file():
                raise FileNotFoundError(f"Q1 experiment is not complete: {label} ({path})")
            results[label] = json.loads(path.read_text(encoding="utf-8"))
        return results

    def load_model(self, label="standard", device="cpu"):
        paths = self.paths(label)
        if not paths.config.is_file() or not paths.final.is_file():
            raise FileNotFoundError(f"Completed {label} model artifacts are unavailable.")
        spec = json.loads(paths.config.read_text(encoding="utf-8"))
        model = DecoderLM(LMConfig(**spec["architecture"]))
        model.load_state_dict(_torch_load(paths.final, "cpu"))
        return model.to(device).eval(), LMConfig(**spec["architecture"])

    def export_recovery_archive(self, *, force=False, reason="manual"):
        now = time.monotonic()
        archive_name = "q1_smoke_all_artifacts.zip" if self.smoke_test else "q1_all_artifacts.zip"
        runtime_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else self.project_root
        destination = runtime_root / archive_name
        with _ARCHIVE_LOCK:
            if not force and destination.is_file() and now - self._last_archive_time < self.archive_seconds:
                return destination
            self.snapshot_sources()
            roots = [self.root, self.project_root / "artifacts" / "shared" / "tokenizer"]
            submission = self.project_root / "submissions" / "q1"
            if submission.is_dir():
                roots.append(submission)
            files = []
            for root in roots:
                if not root.is_dir():
                    continue
                for path in root.rglob("*"):
                    if path.is_file() and not any(part.endswith((".tmp", ".candidate")) for part in path.parts):
                        files.append(path.resolve())
            analysis_log = self.project_root / "analysis_log.json"
            if analysis_log.is_file():
                files.append(analysis_log.resolve())
            files = sorted(set(files), key=lambda p: str(p))
            manifest = {"schema_version": 1, "created_utc": _utc_now(), "reason": reason,
                        "mode": self.mode, "files": {}}
            temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
            with zipfile.ZipFile(temporary, "w") as archive:
                for path in files:
                    rel = path.relative_to(self.project_root).as_posix()
                    info = {"sha256": _sha256_file(path), "size": path.stat().st_size}
                    manifest["files"][rel] = info
                    compression = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED
                    archive.write(path, arcname=rel, compress_type=compression)
                archive.writestr(
                    "recovery/integrity_manifest.json",
                    json.dumps(manifest, indent=2),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                archive.writestr(
                    "recovery/recovery_metadata.json",
                    json.dumps({
                        "notebook": "copy_assignment3.ipynb",
                        "reason": reason,
                        "created_utc": _utc_now(),
                        "instructions": "Attach this ZIP plus the Assignment-3 student package, then run setup top-to-bottom.",
                    }, indent=2),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise ValueError("New recovery archive failed CRC validation")
                saved_manifest = json.loads(archive.read("recovery/integrity_manifest.json"))
                for rel, info in saved_manifest["files"].items():
                    payload = archive.read(rel)
                    if len(payload) != int(info["size"]):
                        raise ValueError(f"Recovery ZIP size validation failed for {rel}")
                    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
                        raise ValueError(f"Recovery ZIP checksum validation failed for {rel}")
            previous = destination.with_name(destination.stem + ".previous.zip")
            if destination.is_file():
                os.replace(destination, previous)
            os.replace(temporary, destination)
            self._last_archive_time = now
            print(f"Validated Q1 recovery archive ({reason}): {destination}")
            return destination

    def completion_report(self):
        report = {"mode": self.mode, "experiments": {}, "submission": {}, "sources": {}}
        for label in ("standard", "unscaled"):
            paths = self.paths(label)
            report["experiments"][label] = {
                "config": paths.config.is_file(), "latest": paths.latest.is_file(),
                "final": paths.final.is_file(), "metrics": paths.metrics.is_file(),
                "status": json.loads(paths.status.read_text()).get("status") if paths.status.is_file() else "missing",
            }
        sub = self.project_root / "submissions" / "q1"
        for name in ("model.pt", "config.json", "metrics.json", "attention.py", "model_arch.py"):
            report["submission"][name] = (sub / name).is_file()
        for name in ("a1_tokenizer.py", "attention.py", "model_arch.py", "q1_runtime.py"):
            report["sources"][name] = (self.source_dir / name).is_file()
        return report

    def run_smoke_tests(self):
        smoke_root = self.project_root / "artifacts" / "_smoke" / "q1_runtime"
        if smoke_root.exists():
            shutil.rmtree(smoke_root)
        smoke_root.mkdir(parents=True)
        checks = {}
        cfg_values = dict(self.base_config)
        cfg_values.update({"d_model": 32, "n_layers": 1, "n_heads": 4,
                           "n_kv_heads": 4, "d_ff": 64, "max_seq_len": 16,
                           "dropout": 0.0, "attention_scale": None})
        cfg = LMConfig(**cfg_values)
        model = DecoderLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 12))
        logits = model(x)
        checks["forward_shape"] = list(logits.shape) == [2, 12, cfg.vocab_size]
        x_changed = x.clone(); x_changed[:, 7:] = torch.randint(0, cfg.vocab_size, x_changed[:, 7:].shape)
        with torch.no_grad():
            a, b = model.eval()(x), model.eval()(x_changed)
        checks["causal_prefix"] = bool(torch.allclose(a[:, :7], b[:, :7], atol=1e-5, rtol=1e-5))
        model.train(); loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size), x.reshape(-1))
        loss.backward()
        checks["finite_gradients"] = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        checks["weight_tying"] = model.head.weight is model.tok_emb.weight
        checks["parameter_budget"] = sum(p.numel() for p in model.parameters()) <= 50_000_000
        sampler_a = list(DeterministicStepBatchSampler(100, 4, self.seed, 0, 8))
        sampler_b = list(DeterministicStepBatchSampler(100, 4, self.seed, 3, 8))
        checks["exact_sampler_resume"] = sampler_a[3:] == sampler_b
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_path = smoke_root / "optimizer_roundtrip.pt"
        _atomic_plain_torch(checkpoint_path, {
            "model": _cpu_tree(model.state_dict()), "optimizer": _cpu_tree(optimizer.state_dict())
        })
        restored = _torch_load(checkpoint_path, "cpu")
        model_2 = DecoderLM(cfg); model_2.load_state_dict(restored["model"])
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-3)
        optimizer_2.load_state_dict(restored["optimizer"])
        checks["checkpoint_optimizer_roundtrip"] = _state_dict_hash(model.state_dict()) == _state_dict_hash(model_2.state_dict())
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            good = pool.submit(lambda: "preserved")
            bad = pool.submit(lambda: (_ for _ in ()).throw(RuntimeError("injected")))
            checks["partner_success_survives"] = good.result() == "preserved"
            try: bad.result()
            except RuntimeError: failures.append("injected")
        checks["injected_failure_observed"] = failures == ["injected"]
        if torch.cuda.device_count() >= 2:
            def gpu_probe(device_name):
                local = DecoderLM(cfg).to(device_name).eval()
                value = local(torch.randint(0, cfg.vocab_size, (1, 8), device=device_name))
                return str(value.device)
            with ThreadPoolExecutor(max_workers=2) as pool:
                devices = {pool.submit(gpu_probe, "cuda:0"), pool.submit(gpu_probe, "cuda:1")}
                seen = {future.result() for future in devices}
            checks["dual_gpu_independence"] = seen == {"cuda:0", "cuda:1"}
        else:
            checks["dual_gpu_independence"] = None
        zip_path = smoke_root / "integrity.zip"
        payload_path = smoke_root / "payload.json"
        _atomic_json(payload_path, {"ok": True})
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(payload_path, arcname="payload.json")
        with zipfile.ZipFile(zip_path) as archive:
            checks["zip_crc"] = archive.testzip() is None
        report = {"created_utc": _utc_now(), "checks": checks,
                  "all_available_passed": all(value for value in checks.values() if value is not None)}
        _atomic_json(smoke_root / "smoke_report.json", report)
        print(json.dumps(report, indent=2))
        return report
'''),
    code_cell("q1-runtime-configure", r'''
# Configure the reusable Q1 runtime. This cell only prepares state; it does not train.
import importlib
sys.modules.pop("q1_runtime", None)
q1_runtime_module = importlib.import_module("q1_runtime")

q1_base_config = {
    "vocab_size": VOCAB_SIZE,
    "d_model": MODEL_DIM,
    "n_layers": MODEL_LAYERS,
    "n_heads": 4,
    "n_kv_heads": 4,
    "d_ff": MODEL_FF,
    "max_seq_len": MODEL_SEQ_LEN,
    "dropout": 0.1,
    "pos_emb_type": "sinusoidal",
    "attention_scale": None,
    "norm_first": False,
}
q1_runtime = q1_runtime_module.Q1Runtime(
    tokenizer=tokenizer,
    train_ids=train_ids,
    held_out=held_out,
    base_config=q1_base_config,
    train_steps=TRAIN_STEPS,
    batch_size=TRAIN_BATCH_SIZE,
    seed=SEED,
    tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
    smoke_test=SMOKE_TEST,
    checkpoint_steps=50,
    checkpoint_seconds=5 * 60,
    archive_seconds=10 * 60,
    learning_rate=3e-4,
    use_amp=True,
)
print("Q1 artifact root:", q1_runtime.root)
print("Q1 evaluation steps:", q1_runtime.eval_steps)
'''),
    code_cell("q1-infra-smoke-tests", r'''
# Lightweight infrastructure tests are isolated from real experiment artifacts.
RUN_Q1_INFRA_SMOKE_TESTS = SMOKE_TEST  # set True manually in a full-mode validation session
if RUN_Q1_INFRA_SMOKE_TESTS:
    q1_smoke_report = q1_runtime.run_smoke_tests()
    if not q1_smoke_report["all_available_passed"]:
        raise AssertionError("One or more available Q1 smoke checks failed.")
else:
    print("Q1 infrastructure smoke tests skipped; set RUN_Q1_INFRA_SMOKE_TESTS=True to run them.")
'''),
])


replace("c50760ee", r'''
# Q1.3 + Q1.4 principal training action.
# Standard scaling is the Q1 baseline; it is trained once, concurrently with unscaled.
Q1_FORCE_RETRAIN = False       # explicit reset only; old work is preserved in a timestamped folder
Q1_ALLOW_SEQUENTIAL_FALLBACK = SMOKE_TEST

q1_pair_results = q1_runtime.run_scaling_pair(
    force_retrain=Q1_FORCE_RETRAIN,
    require_two_gpus=not SMOKE_TEST,
    allow_sequential_fallback=Q1_ALLOW_SEQUENTIAL_FALLBACK,
)

q1_analysis_device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
model, cfg = q1_runtime.load_model("standard", device=q1_analysis_device)
q1_standard_result = q1_pair_results["standard"]
n_params = assert_within_budget(model)
train_loss_curve = q1_standard_result["train_loss_curve"]
held_out_perplexity_q1 = float(q1_standard_result["held_out_ppl"])

print(f"Q1 baseline: {n_params:,} parameters")
print(f"Actual optimization steps: {q1_standard_result['completed_steps']}")
print(f"Held-out sentence-level perplexity: {held_out_perplexity_q1:.4f}")
print("Baseline is the standard-scaling member of the paired experiment; it was not retrained twice.")
''')


replace("q1-4-code-final", r'''
# Read and compare the two persisted runs. This cell never launches training.
q1_pair_results = q1_runtime.collect_results()
attention_scaling_results = [
    {
        "label": label,
        "attention_scale": q1_pair_results[label]["attention_scale"],
        "held_out_ppl": float(q1_pair_results[label]["held_out_ppl"]),
        "train_loss_curve": [float(value) for value in q1_pair_results[label]["train_loss_curve"]],
    }
    for label in ("standard", "unscaled")
]

for row in attention_scaling_results:
    effective_scale = row["attention_scale"]
    if effective_scale is None:
        effective_scale = 1.0 / math.sqrt(cfg.d_model // cfg.n_heads)
    print(
        f"{row['label']:>10s} | alpha={effective_scale:.6f} | "
        f"held-out ppl={row['held_out_ppl']:.4f} | logged losses={len(row['train_loss_curve'])}"
    )

with q1_runtime_module._PLOT_LOCK:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row in attention_scaling_results:
        axes[0].plot(range(1, len(row["train_loss_curve"]) + 1), row["train_loss_curve"],
                     label=row["label"])
    axes[0].set(title="Q1 attention scaling: training loss", xlabel="Optimization step",
                ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].bar([row["label"] for row in attention_scaling_results],
                [row["held_out_ppl"] for row in attention_scaling_results])
    axes[1].set(title="Held-out perplexity", ylabel="Sentence-level perplexity")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _scaling_plot = q1_runtime.plot_dir / "attention_scaling_comparison.png"
    fig.savefig(_scaling_plot, dpi=170, bbox_inches="tight")
    plt.show()

_better_scaling = min(attention_scaling_results, key=lambda row: row["held_out_ppl"])
print(
    f"Measured conclusion: {_better_scaling['label']} scaling produced the lower held-out "
    f"perplexity in this controlled run ({_better_scaling['held_out_ppl']:.4f})."
)
''')


replace("8ac85f97", r'''
# Q1.5: corpus-specific closed, forced-choice, and free-generation probes.
def q1_continuation_logprob(prefix, continuation):
    prefix_ids = [int(value) for value in tokenizer.encode(prefix)]
    continuation_ids = [int(value) for value in tokenizer.encode(continuation)]
    if not prefix_ids or not continuation_ids:
        return float("-inf")
    full_ids = prefix_ids + continuation_ids
    full_ids = full_ids[-(cfg.max_seq_len + 1):]
    continuation_start = max(1, len(full_ids) - len(continuation_ids))
    x = torch.tensor(full_ids[:-1], dtype=torch.long, device=q1_analysis_device).unsqueeze(0)
    with torch.no_grad():
        log_probs = F.log_softmax(model.eval()(x), dim=-1)
    score = 0.0
    for token_position in range(continuation_start, len(full_ids)):
        score += float(log_probs[0, token_position - 1, full_ids[token_position]])
    return score


def q1_generate(prompt, max_new_tokens=30, temperature=0.8, top_k=20, seed=SEED):
    ids = [int(value) for value in tokenizer.encode(prompt)]
    generator_device = q1_analysis_device.type if q1_analysis_device.type != "cuda" else q1_analysis_device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    model.eval()
    for _ in range(max_new_tokens):
        context = ids[-cfg.max_seq_len:]
        x = torch.tensor(context, dtype=torch.long, device=q1_analysis_device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0, -1] / max(float(temperature), 1e-6)
        k = min(int(top_k), logits.numel())
        values, indices = torch.topk(logits, k)
        probabilities = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        ids.append(int(indices[sampled].item()))
    return tokenizer.decode(ids)


q1_scored_probes = [
    {"format": "closed_cloze", "prefix": "Harry's owl is named",
     "candidates": [" Hedwig", " Crookshanks", " Scabbers"], "expected": " Hedwig"},
    {"format": "closed_cloze", "prefix": "The headmaster of Hogwarts is",
     "candidates": [" Albus Dumbledore", " Lord Voldemort", " Hagrid"],
     "expected": " Albus Dumbledore"},
    {"format": "closed_cloze", "prefix": "Harry belongs to the house of",
     "candidates": [" Gryffindor", " Slytherin", " Ravenclaw"], "expected": " Gryffindor"},
    {"format": "forced_choice", "prefix": "Harry raised his wand and shouted, Expecto",
     "candidates": [" Patronum", " Toaster"], "expected": " Patronum"},
    {"format": "forced_choice", "prefix": "Platform nine and three-quarters leads to",
     "candidates": [" Hogwarts", " the moon"], "expected": " Hogwarts"},
    {"format": "forced_choice", "prefix": "Ron and Hermione were Harry's",
     "candidates": [" friends", " professors"], "expected": " friends"},
]

q1_probe_records = []
for probe in q1_scored_probes:
    scores = {candidate: q1_continuation_logprob(probe["prefix"], candidate)
              for candidate in probe["candidates"]}
    prediction = max(scores, key=scores.get)
    record = {**probe, "candidate_log_probabilities": scores, "prediction": prediction,
              "correct": prediction == probe["expected"],
              "scoring_rule": "largest sum of log-probabilities over every candidate BPE token"}
    q1_probe_records.append(record)
    print(json.dumps(record, indent=2))

q1_generation_records = []
for probe_index, prompt in enumerate((
    "Harry stepped into the Great Hall and",
    "Hermione opened the old book and whispered",
)):
    continuation = q1_generate(prompt, seed=SEED + probe_index)
    q1_generation_records.append({
        "format": "free_form", "prompt": prompt, "generated_text": continuation,
        "evaluation": "Qualitatively inspect syntax, repetition, entity consistency, and local corpus style.",
    })
    print("\nPROMPT:", prompt, "\nGENERATION:", continuation)

extrinsic_probe_accuracy = (
    sum(int(record["correct"]) for record in q1_probe_records) / len(q1_probe_records)
)
q1_extrinsic_results = {
    "accuracy": extrinsic_probe_accuracy,
    "correct": sum(int(record["correct"]) for record in q1_probe_records),
    "total_scored": len(q1_probe_records),
    "scored_probes": q1_probe_records,
    "qualitative_generations": q1_generation_records,
}
q1_runtime_module._atomic_json(q1_runtime.analysis_dir / "extrinsic_probes.json", q1_extrinsic_results)
print(f"\nExtrinsic scored-probe accuracy: {extrinsic_probe_accuracy:.3f} "
      f"({q1_extrinsic_results['correct']}/{q1_extrinsic_results['total_scored']})")
''')


replace("c01f4da2", r'''
# Q1.6: the training worker persisted held-out evaluations at predetermined milestones.
q1_standard_result = q1_runtime.collect_results()["standard"]
q1_ppl_curve = q1_standard_result["eval_history"]
if len(q1_ppl_curve) < 2 and not SMOKE_TEST:
    raise RuntimeError("Full training did not preserve enough intermediate PPL measurements.")

with q1_runtime_module._PLOT_LOCK:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["tokens_processed"] for row in q1_ppl_curve],
            [row["held_out_ppl"] for row in q1_ppl_curve], marker="o")
    ax.set(title="Q1 held-out perplexity during training", xlabel="Training tokens processed",
           ylabel="Sentence-level held-out perplexity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    q1_ppl_plot = q1_runtime.plot_dir / "perplexity_vs_training_tokens.png"
    fig.savefig(q1_ppl_plot, dpi=170, bbox_inches="tight")
    plt.show()

for row in q1_ppl_curve:
    print(f"step={row['step']:4d} | tokens={row['tokens_processed']:10d} | ppl={row['held_out_ppl']:.4f}")
if len(q1_ppl_curve) >= 2:
    previous, final = q1_ppl_curve[-2]["held_out_ppl"], q1_ppl_curve[-1]["held_out_ppl"]
    relative_change = (previous - final) / max(previous, 1e-12)
    interpretation = "still improving materially" if relative_change > 0.02 else "beginning to plateau"
    print(f"Measured end-of-run change: {relative_change:+.2%}; performance appears {interpretation}.")
''')


replace("343de0b3", r'''
# Q1.7: token-level NLL on several held-out sentences.
q1_token_losses = []
model.eval()
for sentence_index, sentence in enumerate(held_out[:8]):
    sentence_ids = [int(value) for value in tokenizer.encode(sentence)][:cfg.max_seq_len + 1]
    if len(sentence_ids) < 2:
        continue
    x = torch.tensor(sentence_ids[:-1], dtype=torch.long, device=q1_analysis_device).unsqueeze(0)
    y = torch.tensor(sentence_ids[1:], dtype=torch.long, device=q1_analysis_device)
    with torch.no_grad():
        logits = model(x)[0]
        losses = F.cross_entropy(logits, y, reduction="none")
    for position, (token_id, token_loss) in enumerate(zip(sentence_ids[1:], losses.tolist()), start=1):
        rendered = tokenizer.decode([int(token_id)])
        q1_token_losses.append({
            "sentence_index": sentence_index, "target_position": position,
            "token_id": int(token_id), "token": rendered,
            "negative_log_likelihood": float(token_loss),
        })

q1_highest_loss_tokens = sorted(
    q1_token_losses, key=lambda row: row["negative_log_likelihood"], reverse=True
)[:25]
for row in q1_highest_loss_tokens:
    print(f"NLL={row['negative_log_likelihood']:.4f} | sentence={row['sentence_index']} "
          f"| position={row['target_position']} | token={row['token']!r}")
q1_difficult_token_summary = {
    "punctuation_like": sum(not any(ch.isalnum() for ch in row["token"])
                            for row in q1_highest_loss_tokens),
    "contains_uppercase": sum(any(ch.isupper() for ch in row["token"])
                              for row in q1_highest_loss_tokens),
    "short_subword": sum(len(row["token"].strip()) <= 2 for row in q1_highest_loss_tokens),
}
print("Observed high-loss-token categories:", q1_difficult_token_summary)
q1_runtime_module._atomic_json(q1_runtime.analysis_dir / "highest_loss_tokens.json", {
    "evaluated_sentence_count": min(8, len(held_out)),
    "top_tokens": q1_highest_loss_tokens,
    "category_counts": q1_difficult_token_summary,
})
''')


replace("0ec95cb7", r'''
# Q1.8: represent a word by the mean of all of its BPE token embeddings.
q1_embedding_words = [
    "Harry", "Hermione", "Ron", "Dumbledore", "Voldemort",
    "wand", "school", "magic", "friend", "teacher", "book", "house",
]
embedding_table = model.tok_emb.weight.detach()
q1_word_vectors, q1_word_token_ids = {}, {}
for word in q1_embedding_words:
    token_ids = [int(value) for value in tokenizer.encode(word)]
    if not token_ids:
        continue
    ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=embedding_table.device)
    vector = embedding_table.index_select(0, ids_tensor).mean(dim=0)
    q1_word_vectors[word] = F.normalize(vector, dim=0)
    q1_word_token_ids[word] = token_ids

q1_embedding_geometry = {}
for word, vector in q1_word_vectors.items():
    neighbours = sorted(
        ((other, float(torch.dot(vector, other_vector)))
         for other, other_vector in q1_word_vectors.items() if other != word),
        key=lambda item: item[1], reverse=True,
    )[:5]
    q1_embedding_geometry[word] = {
        "bpe_token_ids": q1_word_token_ids[word],
        "nearest_selected_words": [{"word": other, "cosine_similarity": similarity}
                                   for other, similarity in neighbours],
    }
    print(word, "->", ", ".join(f"{other} ({similarity:.3f})" for other, similarity in neighbours))

q1_runtime_module._atomic_json(q1_runtime.analysis_dir / "embedding_geometry.json", {
    "aggregation_rule": "arithmetic mean of the learned embeddings of every BPE token in the word",
    "results": q1_embedding_geometry,
    "interpretation_caution": (
        "At this scale, similarity may reflect local co-occurrence and token frequency rather than stable semantics."
    ),
})
print("Interpretation: inspect whether character names cluster; weak/mixed clustering is evidence "
      "that local co-occurrence and frequency dominate at this training scale.")
''')


replace("b0f5504a", r'''
# Q1.9: a sentence is dialogue-heavy when it contains a straight or curly quotation mark.
q1_dialogue_rule = 'contains at least one of: \", “, ”'
dialogue_sentences = [sentence for sentence in held_out if any(mark in sentence for mark in ('"', '“', '”'))]
narration_sentences = [sentence for sentence in held_out if sentence not in dialogue_sentences]
if not dialogue_sentences or not narration_sentences:
    raise RuntimeError("The stated dialogue rule produced an empty subset; report and revise the rule explicitly.")

dialogue_sentence_ppls = [
    float(perplexity_of_texts(model, tokenizer, [sentence], q1_analysis_device,
                              max_len=cfg.max_seq_len))
    for sentence in dialogue_sentences
]
narration_sentence_ppls = [
    float(perplexity_of_texts(model, tokenizer, [sentence], q1_analysis_device,
                              max_len=cfg.max_seq_len))
    for sentence in narration_sentences
]
dialogue_ppl = sum(dialogue_sentence_ppls) / len(dialogue_sentence_ppls)
narration_ppl = sum(narration_sentence_ppls) / len(narration_sentence_ppls)
q1_dialogue_results = {
    "rule": q1_dialogue_rule,
    "dialogue_count": len(dialogue_sentences),
    "narration_count": len(narration_sentences),
    "dialogue_ppl": dialogue_ppl,
    "narration_ppl": narration_ppl,
    "dialogue_sentence_ppls": dialogue_sentence_ppls,
    "narration_sentence_ppls": narration_sentence_ppls,
}
q1_runtime_module._atomic_json(q1_runtime.analysis_dir / "dialogue_vs_narration.json", q1_dialogue_results)
print(json.dumps(q1_dialogue_results, indent=2))
easier = "dialogue" if dialogue_ppl < narration_ppl else "narration"
print(
    f"Measured result: {easier} has lower average sentence-level perplexity. "
    "A plausible corpus-grounded explanation is that repeated dialogue templates and speaker turns "
    "can be easier, while narration may contain rarer descriptions—or the reverse when quotation "
    "punctuation and short turns fragment the tokenizer context."
)
''')


replace("b64f0293", r'''
# Q1.10: create the official submission only from completed, non-smoke measurements.
import shutil

q1_pair_results = q1_runtime.collect_results()
_extrinsic_path = q1_runtime.analysis_dir / "extrinsic_probes.json"
if not _extrinsic_path.is_file():
    raise FileNotFoundError("Run Q1.5 before finalizing; extrinsic probe evidence is missing.")
q1_extrinsic_results = json.loads(_extrinsic_path.read_text(encoding="utf-8"))
extrinsic_probe_accuracy = float(q1_extrinsic_results["accuracy"])
attention_scaling_results = [
    {
        "label": label,
        "attention_scale": q1_pair_results[label]["attention_scale"],
        "held_out_ppl": float(q1_pair_results[label]["held_out_ppl"]),
        "train_loss_curve": [float(value) for value in q1_pair_results[label]["train_loss_curve"]],
    }
    for label in ("standard", "unscaled")
]
metrics_q1 = {
    "held_out_perplexity": float(q1_pair_results["standard"]["held_out_ppl"]),
    "train_loss_curve": [float(value) for value in q1_pair_results["standard"]["train_loss_curve"]],
    "extrinsic_probe_accuracy": extrinsic_probe_accuracy,
    "attention_scaling_results": attention_scaling_results,
}

model, cfg = q1_runtime.load_model("standard", device="cpu")
if SMOKE_TEST:
    print("SMOKE_TEST=True: official submissions/q1 was intentionally not overwritten with smoke metrics.")
    print("Set SMOKE_TEST=False, rerun setup/Q1 training and Q1.5-Q1.10 for the final submission.")
else:
    save_submission(
        "submissions/q1", model,
        {"question": "q1", "smoke_test": False, "actual_train_steps": TRAIN_STEPS, **cfg.__dict__},
        metrics_q1, VOCAB_SIZE,
    )
    q1_runtime.snapshot_sources()
    for _source_name in ("a1_tokenizer.py", "attention.py", "model_arch.py"):
        _source = Path(_source_name)
        _destination = Path("submissions/q1") / _source_name
        if _source.is_file() and not _destination.is_file():
            shutil.copy2(_source, _destination)
    log_result("q1_held_out_perplexity", metrics_q1["held_out_perplexity"])
    log_result("q1_attention_scaling", attention_scaling_results)
    log_result("q1_extrinsic_probe_accuracy", extrinsic_probe_accuracy)
    q1_official_archive = q1_runtime.export_recovery_archive(force=True, reason="official-q1-submission")
    print("Official Q1 recovery archive:", q1_official_archive)
''')


insert_after("b64f0293", [
    markdown_cell("q1-final-validation-heading", r'''
### Q1 final validation and recovery export

Run this after Q1.10. It audits the completed artifacts, executes the supplied Q1
autograder when present, and refreshes the independently restorable protection ZIP.
'''),
    code_cell("q1-final-validation", r'''
q1_completion_report = q1_runtime.completion_report()
print(json.dumps(q1_completion_report, indent=2))

_missing_experiments = [
    label for label, evidence in q1_completion_report["experiments"].items()
    if evidence["status"] != "complete" or not evidence["final"] or not evidence["metrics"]
]
if _missing_experiments:
    raise RuntimeError(f"Q1 remains incomplete for experiments: {_missing_experiments}")

if not SMOKE_TEST:
    _missing_submission = [name for name, present in q1_completion_report["submission"].items()
                           if not present]
    _missing_sources = [name for name, present in q1_completion_report["sources"].items()
                        if not present]
    if _missing_submission or _missing_sources:
        raise RuntimeError(
            f"Q1 finalization missing submission={_missing_submission}, source snapshots={_missing_sources}"
        )

Q1_RUN_AUTOGRADER = (not SMOKE_TEST) and Path("autograder_q1.py").is_file()
if Q1_RUN_AUTOGRADER:
    _grader = subprocess.run(
        [sys.executable, "-u", "autograder_q1.py", "--student_dir", "."],
        text=True, capture_output=True,
    )
    print(_grader.stdout)
    if _grader.stderr:
        print(_grader.stderr)
    if _grader.returncode != 0:
        raise RuntimeError(f"Q1 autograder failed with status {_grader.returncode}")
else:
    print("Q1 autograder not run (smoke mode or autograder_q1.py is unavailable).")

q1_recovery_archive = q1_runtime.export_recovery_archive(force=True, reason="final-validation")
print("Q1 VALIDATION COMPLETE")
print("Recovery ZIP:", q1_recovery_archive)
print("Official submission folder:", Path("submissions/q1").resolve())
print("To resume elsewhere: attach this ZIP and the complete student package, then run setup top-to-bottom.")
'''),
])


# Write exactly one notebook after every requested replacement/insertion has succeeded.
# The separately created rollback copy remains byte-identical to the pre-edit notebook.
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK} with {len(nb['cells'])} cells")


