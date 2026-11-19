import hashlib
import json
import textwrap
from pathlib import Path


NOTEBOOK = Path("copy_assignment3.ipynb")
ROLLBACK_NOTEBOOK = Path(".codex_backups/copy_assignment3.before_q2_implementation.ipynb")
EXPECTED_ROLLBACK_SHA256 = "d3ab2d1114349b6bed8227f6af8034ed82f941ef7c914d0c857f33fa36a420da"


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


rollback_bytes = ROLLBACK_NOTEBOOK.read_bytes()
if hashlib.sha256(rollback_bytes).hexdigest() != EXPECTED_ROLLBACK_SHA256:
    raise RuntimeError("The immutable pre-Q2 rollback notebook changed; refusing to rebuild")
nb = json.loads(rollback_bytes.decode("utf-8"))


def index(cell_id: str) -> int:
    matches = [i for i, cell in enumerate(nb["cells"]) if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one cell {cell_id!r}, found {matches}")
    return matches[0]


def replace(cell_id: str, text: str):
    cell = nb["cells"][index(cell_id)]
    cell["source"] = src(text)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def insert_after(cell_id: str, cells):
    at = index(cell_id) + 1
    nb["cells"][at:at] = cells


# Q1/Section-0 is immutable. Keep a deep in-memory copy and verify it again before write.
q1_boundary = index("q2-intro-final")
q1_cells_before = json.loads(json.dumps(nb["cells"][:q1_boundary], ensure_ascii=False))
q3_start_id = "q3-intro-final"
q3_cells_before = json.loads(json.dumps(nb["cells"][index(q3_start_id):], ensure_ascii=False))


insert_after("q2-intro-final", [
    markdown_cell("q2-bootstrap-heading", r'''
### Q2 fresh-session bootstrap and immutable Q1 dependency checks

This Q2-only bootstrap can be run in a fresh Kaggle kernel. It locates the course
support package, validates a **full** Q1 recovery archive, restores only the persisted
tokenizer/source metadata needed by Q2, reconstructs the official split, and restores
an optional Q2 recovery archive. It never launches or rewrites a Q1 experiment.

**Source-version rule:** Q1's canonical modules remain immutable under
`artifacts/full/q1/source_snapshots/` and `submissions/q1/`. The Q2 `%%writefile`
cells replace only the active root modules expected by the assignment, and the Q2
runtime snapshots those versions independently under `artifacts/<mode>/q2/source_snapshots/`.
'''),
    code_cell("q2-bootstrap", r'''
# --------------------------- user controls ---------------------------------
Q2_SMOKE_TEST = True  # @param {type:"boolean"}; set False for the real four-model runs
Q2_PROJECT_ROOT = None  # optional exact writable assignment directory
Q2_SUPPORT_ROOT = None  # optional exact read-only directory containing the course utilities
Q1_ARCHIVE_FOR_Q2 = None  # optional exact q1_all_artifacts.zip
Q2_RECOVERY_ARCHIVE = None  # optional exact q2_all_artifacts.zip

# This cell intentionally reconstructs every Q2 dependency without executing Q1 cells.
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from array import array
from pathlib import Path


def _q2_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _q2_json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _q2_hash_ints(values, chunk_items=100_000):
    digest = hashlib.sha256()
    for start in range(0, len(values), chunk_items):
        digest.update(array("q", (int(x) for x in values[start:start + chunk_items])).tobytes())
    return digest.hexdigest()


def _q2_hash_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        payload = str(text).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _q2_safe_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Unsafe archive path: {relative}")
    return target


def _q2_support_signature(path):
    names = ("common.py", "common_grading.py", "download_data.py", "autograder_q2.py")
    return tuple(_q2_sha256(path / name) for name in names)


def _q2_find_support_root():
    explicit = Path(Q2_SUPPORT_ROOT).expanduser().resolve() if Q2_SUPPORT_ROOT else None
    if explicit is not None:
        if not all((explicit / name).is_file() for name in
                   ("common.py", "common_grading.py", "download_data.py", "autograder_q2.py")):
            raise FileNotFoundError(f"Q2_SUPPORT_ROOT is incomplete: {explicit}")
        return explicit
    cwd = Path.cwd().resolve()
    candidates = []
    for candidate in (cwd, cwd / "cs781-a3-support-files",
                      Path("/kaggle/working/assignment_3"), Path("/content/assignment_3")):
        if all((candidate / name).is_file() for name in
               ("common.py", "common_grading.py", "download_data.py", "autograder_q2.py")):
            candidates.append(candidate.resolve())
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(
            path.parent.resolve() for path in kaggle_input.rglob("autograder_q2.py")
            if all((path.parent / name).is_file() for name in
                   ("common.py", "common_grading.py", "download_data.py"))
        )
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if not unique:
        raise FileNotFoundError(
            "Could not locate the complete Assignment-3 support package. Attach the supplied "
            "support dataset (common.py, common_grading.py, download_data.py, autograder_q2.py)."
        )
    signatures = {_q2_support_signature(candidate) for candidate in unique}
    if len(signatures) > 1:
        raise RuntimeError(f"Conflicting Assignment-3 support packages found: {unique}")
    if cwd in unique:
        return cwd
    return unique[0]


_q2_support_root = _q2_find_support_root()
_q2_cwd = Path.cwd().resolve()
if Q2_PROJECT_ROOT:
    _q2_project_root = Path(Q2_PROJECT_ROOT).expanduser().resolve()
elif str(_q2_cwd).startswith("/kaggle/input/"):
    _q2_project_root = Path("/kaggle/working/assignment_3").resolve()
elif Path("/kaggle/working").is_dir() and str(_q2_support_root).startswith("/kaggle/input/"):
    _q2_project_root = Path("/kaggle/working/assignment_3").resolve()
else:
    _q2_project_root = _q2_cwd
_q2_project_root.mkdir(parents=True, exist_ok=True)

# Copy only course support utilities. Q1/Q2 student sources are managed independently.
for _source in _q2_support_root.iterdir():
    if not _source.is_file() or _source.name == "a1_tokenizer.py":
        continue
    _destination = _q2_project_root / _source.name
    if _destination.is_file() and _q2_sha256(_destination) != _q2_sha256(_source):
        raise RuntimeError(f"Refusing to overwrite conflicting support file: {_destination}")
    if not _destination.exists():
        shutil.copy2(_source, _destination)
os.chdir(_q2_project_root)
sys.path.insert(0, str(_q2_project_root))
print("Q2 writable project root:", _q2_project_root)


def _q2_read_archive_manifest(path, expected_mode=None):
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        if "recovery/integrity_manifest.json" not in archive.namelist():
            raise ValueError(f"Archive has no integrity manifest: {path}")
        manifest = json.loads(archive.read("recovery/integrity_manifest.json"))
    if expected_mode is not None and manifest.get("mode") != expected_mode:
        raise ValueError(
            f"Archive mode mismatch for {path}: {manifest.get('mode')!r} != {expected_mode!r}"
        )
    return manifest


def _q2_discover_archive(primary_name, explicit=None, expected_mode=None):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        _q2_read_archive_manifest(candidate, expected_mode)
        return candidate
    roots = [Path.cwd(), _q2_cwd, _q2_support_root, Path("/kaggle/working"), Path("/kaggle/input")]

    def valid_named(name):
        found = []
        for root in roots:
            if root.is_dir():
                found.extend(root.rglob(name))
        by_hash = {}
        for path in (path.resolve() for path in found if path.is_file()):
            try:
                _q2_read_archive_manifest(path, expected_mode)
            except Exception as exc:
                print(f"Ignoring invalid {name} candidate {path}: {type(exc).__name__}: {exc}")
                continue
            by_hash.setdefault(_q2_sha256(path), path)
        return by_hash

    by_hash = valid_named(primary_name)
    if not by_hash:
        previous_name = primary_name.replace(".zip", ".previous.zip")
        by_hash = valid_named(previous_name)
        if by_hash:
            print(f"No valid {primary_name} found; using a validated previous-known-good archive.")
    if len(by_hash) > 1:
        raise RuntimeError(
            f"Multiple non-identical {primary_name} archives found; set the explicit path: "
            + ", ".join(str(path) for path in by_hash.values())
        )
    return next(iter(by_hash.values()), None)


def _q2_restore_archive(path, prefixes, label, strict_conflicts=True):
    """Validate the whole ZIP, then atomically restore only missing/identical selected files."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt member in {path}: {bad_member}")
        manifest = json.loads(archive.read("recovery/integrity_manifest.json"))
        file_manifest = manifest.get("files", {})
        selected = [
            rel for rel in file_manifest
            if any(rel == prefix or rel.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)
            and "__pycache__" not in rel
        ]
        with tempfile.TemporaryDirectory(prefix=f"a3_{label}_restore_") as temp_dir:
            temp_root = Path(temp_dir).resolve()
            for rel in selected:
                staged = _q2_safe_path(temp_root, rel)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(rel) as source, open(staged, "wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                info = file_manifest[rel]
                if staged.stat().st_size != int(info["size"]) or _q2_sha256(staged) != info["sha256"]:
                    raise ValueError(f"Manifest validation failed for {rel}")
            conflicts = []
            for rel in selected:
                staged = _q2_safe_path(temp_root, rel)
                destination = _q2_safe_path(Path.cwd(), rel)
                if destination.is_file() and _q2_sha256(destination) != _q2_sha256(staged):
                    conflicts.append(rel)
            if conflicts:
                if strict_conflicts:
                    raise RuntimeError(
                        f"{label} restore would overwrite non-identical local work: {conflicts[:20]}"
                    )
                print(f"{label}: retained {len(conflicts)} non-identical local files; "
                      f"restoring only missing archive files. Conflicts: {conflicts[:20]}")
            restored, reused = 0, 0
            for rel in selected:
                staged = _q2_safe_path(temp_root, rel)
                destination = _q2_safe_path(Path.cwd(), rel)
                if destination.exists():
                    reused += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                candidate = destination.with_name(destination.name + f".{uuid.uuid4().hex}.restore")
                shutil.copy2(staged, candidate)
                if _q2_sha256(candidate) != _q2_sha256(staged):
                    candidate.unlink(missing_ok=True)
                    raise ValueError(f"Staged restore checksum failed for {rel}")
                os.replace(candidate, destination)
                restored += 1
    print(f"{label} archive validated; restored={restored}, already-identical={reused}: {path}")
    return manifest


# Q1 is always the completed full run. Never accept its smoke archive for Q2.
_q1_required = [
    Path("artifacts/shared/tokenizer/a1_bpe_tokenizer.json"),
    Path("artifacts/shared/tokenizer/tokenizer_metadata.json"),
    Path("artifacts/shared/tokenizer/train_ids.pt"),
    Path("artifacts/shared/tokenizer/train_ids_metadata.json"),
    Path("artifacts/full/q1/standard/config.json"),
    Path("artifacts/full/q1/source_snapshots/a1_tokenizer.py"),
    Path("artifacts/full/q1/source_snapshots/attention.py"),
    Path("artifacts/full/q1/source_snapshots/model_arch.py"),
]
_q1_archive = _q2_discover_archive(
    "q1_all_artifacts.zip", Q1_ARCHIVE_FOR_Q2, expected_mode="full"
)
if not all(path.is_file() for path in _q1_required):
    if _q1_archive is None:
        raise FileNotFoundError(
            "Q2 needs the completed full Q1 tokenizer/source metadata. Attach q1_all_artifacts.zip "
            "or restore those exact artifacts before running this cell."
        )
    _q2_restore_archive(_q1_archive, (
        "artifacts/shared/tokenizer",
        "artifacts/full/q1/source_snapshots",
        "artifacts/full/q1/standard/config.json",
        "artifacts/full/q1/standard/metrics.json",
        "artifacts/full/q1/standard/status.json",
    ), "Q1 dependency")
if not all(path.is_file() for path in _q1_required):
    raise FileNotFoundError("Q1 dependency restoration is incomplete.")

_q2_mode = "smoke" if Q2_SMOKE_TEST else "full"
_q2_archive_name = "q2_smoke_all_artifacts.zip" if Q2_SMOKE_TEST else "q2_all_artifacts.zip"
_q2_existing_root = Path("artifacts") / _q2_mode / "q2"
_q2_archive = _q2_discover_archive(
    _q2_archive_name, Q2_RECOVERY_ARCHIVE, expected_mode=_q2_mode
)
if _q2_archive:
    _q2_restore_prefixes = [f"artifacts/{_q2_mode}/q2"]
    if not Q2_SMOKE_TEST:
        _q2_restore_prefixes.append("submissions/q2")
    _q2_restore_archive(
        _q2_archive, tuple(_q2_restore_prefixes), "Q2", strict_conflicts=False
    )
elif _q2_existing_root.is_dir() and any(_q2_existing_root.rglob("*")):
    print("No Q2 ZIP selected; existing local Q2 state will be validated by the runtime:",
          _q2_existing_root)


# Download only when the seven normalized books are absent. Tokens stay in memory/env only.
for _package in ("huggingface_hub",):
    if importlib.util.find_spec(_package) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _package], check=True)
_raw_dir = Path("data/raw")
if len(list(_raw_dir.glob("book*.txt"))) != 7:
    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "The corpus is missing. Add HF_TOKEN to Kaggle Secrets (with notebook access) "
            "or attach data/raw/book1.txt ... book7.txt."
        )
    subprocess.run([sys.executable, "download_data.py", "--raw-dir", str(_raw_dir)], check=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (
    load_corpus, held_out_sentences, remove_sentences_from_text,
    BPETokenizerWrapper, TokenDataset, perplexity_of_texts,
    count_parameters, assert_within_budget, save_submission, MAX_PARAMS, SEED,
)


def log_result(key, value):
    path = Path("analysis_log.json")
    log = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    log[key] = value
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    print(f"[logged] {key}")


corpus = load_corpus("data/raw")
held_out = held_out_sentences(corpus, n_chapters=5, sentences_per_chapter=6)
train_text = remove_sentences_from_text(corpus.raw_text, held_out)

_q1_spec = json.loads(Path("artifacts/full/q1/standard/config.json").read_text(encoding="utf-8"))
_q1_sources = _q1_spec["source_hashes"]
_q1_snapshot_dir = Path("artifacts/full/q1/source_snapshots")
for _name in ("a1_tokenizer.py", "attention.py", "model_arch.py"):
    _snapshot = _q1_snapshot_dir / _name
    if _q2_sha256(_snapshot) != _q1_sources[_name]:
        raise RuntimeError(f"Immutable Q1 source snapshot hash mismatch: {_snapshot}")

# Make the validated Q1 tokenizer adapter active; Q2 attention/model files are written later.
_a1_snapshot = _q1_snapshot_dir / "a1_tokenizer.py"
_a1_active = Path("a1_tokenizer.py")
if not _a1_active.is_file() or _q2_sha256(_a1_active) != _q2_sha256(_a1_snapshot):
    _a1_candidate = Path("a1_tokenizer.py.q2restore")
    shutil.copy2(_a1_snapshot, _a1_candidate)
    os.replace(_a1_candidate, _a1_active)

_a1_spec = importlib.util.spec_from_file_location("_q2_a1_tokenizer", _a1_active)
_a1_module = importlib.util.module_from_spec(_a1_spec)
_a1_spec.loader.exec_module(_a1_module)
_tokenizer_dir = Path("artifacts/shared/tokenizer")
_tokenizer_model = _tokenizer_dir / "a1_bpe_tokenizer.json"
_tokenizer_metadata = json.loads((_tokenizer_dir / "tokenizer_metadata.json").read_text(encoding="utf-8"))
_train_ids_metadata = json.loads((_tokenizer_dir / "train_ids_metadata.json").read_text(encoding="utf-8"))
_train_text_hash = hashlib.sha256(train_text.encode("utf-8")).hexdigest()
if _tokenizer_metadata.get("training_corpus_sha256") != _train_text_hash:
    raise RuntimeError("Q1 tokenizer corpus fingerprint does not match the reconstructed official split.")
if int(_tokenizer_metadata.get("requested_vocab_size", -1)) != 8000:
    raise RuntimeError("Q2 refuses a smoke/incompatible tokenizer; expected Q1 vocab target 8000.")
if _q2_sha256(_tokenizer_model) != _tokenizer_metadata.get("model_sha256"):
    raise RuntimeError("Persisted Q1 tokenizer model checksum mismatch.")
_tokenizer_impl = _a1_module.Assignment1ByteLevelBPE.load(_tokenizer_model)
tokenizer = BPETokenizerWrapper(a1_tokenizer=_tokenizer_impl, vocab_size=8000)
VOCAB_SIZE = int(tokenizer.vocab_size)
TOKENIZER_FINGERPRINT = _tokenizer_metadata["model_sha256"]

try:
    _train_ids_tensor = torch.load(_tokenizer_dir / "train_ids.pt", map_location="cpu", weights_only=True)
except TypeError:
    _train_ids_tensor = torch.load(_tokenizer_dir / "train_ids.pt", map_location="cpu")
train_ids = [int(value) for value in _train_ids_tensor.tolist()]
if len(train_ids) != int(_train_ids_metadata["token_count"]):
    raise RuntimeError("Persisted Q1 training-token count mismatch.")
if _train_ids_metadata.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT:
    raise RuntimeError("Training-token cache was produced by a different tokenizer.")
if _q2_hash_ints(train_ids) != _q1_spec["train_ids_hash"]:
    raise RuntimeError("Training-token sequence differs from the completed Q1 experiment.")
if _q2_hash_texts(held_out) != _q1_spec["held_out_hash"]:
    raise RuntimeError("Held-out split differs from the completed Q1 experiment.")
if VOCAB_SIZE != int(_q1_spec["architecture"]["vocab_size"]):
    raise RuntimeError("Tokenizer vocabulary size differs from Q1 model configuration.")

Q2_DIALOGUE_RULE = 'contains at least one of: ", “, ”'
q2_dialogue_sentences = [s for s in held_out if any(mark in s for mark in ('"', '“', '”'))]
q2_narration_sentences = [s for s in held_out if not any(mark in s for mark in ('"', '“', '”'))]
if not q2_dialogue_sentences or not q2_narration_sentences:
    raise RuntimeError("The Q1 dialogue/narration rule produced an empty Q2 evaluation subset.")

_q1_arch = _q1_spec["architecture"]
Q2_MODEL_DIM = 64 if Q2_SMOKE_TEST else int(_q1_arch["d_model"])
Q2_MODEL_LAYERS = 1 if Q2_SMOKE_TEST else int(_q1_arch["n_layers"])
Q2_MODEL_FF = 128 if Q2_SMOKE_TEST else int(_q1_arch["d_ff"])
Q2_MODEL_SEQ_LEN = 64 if Q2_SMOKE_TEST else int(_q1_arch["max_seq_len"])
Q2_BATCH_SIZE = 2 if Q2_SMOKE_TEST else int(_q1_spec["batch_size"])
Q2_TRAIN_STEPS = 3 if Q2_SMOKE_TEST else int(_q1_spec["target_steps"])
Q2_LEARNING_RATE = float(_q1_spec["optimizer"]["lr"])
Q2_MLA_LATENT_DIM = Q2_MODEL_DIM // 4  # stated before training: 16 smoke / 64 full
Q2_CHECKPOINT_STEPS = 1 if Q2_SMOKE_TEST else 50
Q2_CHECKPOINT_SECONDS = 30 if Q2_SMOKE_TEST else 5 * 60
Q2_ARCHIVE_SECONDS = 60 if Q2_SMOKE_TEST else 10 * 60

Q1_DEPENDENCY = {
    "q1_archive_sha256": _q2_sha256(_q1_archive) if _q1_archive else None,
    "q1_trajectory_hash": _q1_spec["trajectory_hash"],
    "q1_source_hashes": _q1_sources,
    "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
    "train_ids_hash": _q1_spec["train_ids_hash"],
    "held_out_hash": _q1_spec["held_out_hash"],
    "training_corpus_sha256": _train_text_hash,
}

print(f"Q2 mode: {'SMOKE' if Q2_SMOKE_TEST else 'FULL'}")
print(f"Reused Q1 tokenizer: vocab={VOCAB_SIZE:,}, tokens={len(train_ids):,}, fp={TOKENIZER_FINGERPRINT[:12]}")
print(f"Held-out: {len(held_out)} total, {len(q2_dialogue_sentences)} dialogue, "
      f"{len(q2_narration_sentences)} narration")
print(f"Q2 controls: D={Q2_MODEL_DIM}, L={Q2_MODEL_LAYERS}, H=8, "
      f"FF={Q2_MODEL_FF}, seq={Q2_MODEL_SEQ_LEN}, batch={Q2_BATCH_SIZE}, "
      f"steps={Q2_TRAIN_STEPS}, MLA r={Q2_MLA_LATENT_DIM}")
'''),
])


replace("4202f56a", r'''
%%writefile attention.py
"""Manual MHA/GQA/MQA and Q2 joint-latent attention implementations."""
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Causal attention with independent query-head and key/value-head counts."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("d_model, n_heads, and n_kv_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_head = int(d_model) // self.n_heads
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        )
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.causal_mask.size(0)}"
            )
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.d_head).transpose(1, 2)
        # Contiguous query-head groups share each KV head: [k0...k0, k1...k1, ...].
        if self.n_rep != 1:
            k = torch.repeat_interleave(k, self.n_rep, dim=1)
            v = torch.repeat_interleave(v, self.n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length].view(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = self.attn_dropout(weights)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))


class MultiHeadLatentAttention(nn.Module):
    """Q2 joint low-rank KV compression with direct, uncompressed queries."""
    def __init__(self, d_model: int, n_heads: int, kv_latent_dim: int,
                 max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not isinstance(kv_latent_dim, int) or not 0 < kv_latent_dim < d_model:
            raise ValueError("kv_latent_dim must be an integer strictly between 0 and d_model")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_heads = int(n_heads)
        self.d_head = int(d_model) // self.n_heads
        self.kv_latent_dim = int(kv_latent_dim)
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        )
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, self.kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(self.kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(self.kv_latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.causal_mask.size(0)}"
            )
        latent = self.kv_down_proj(x)  # computed once and shared by both reconstruction branches
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_up_proj(latent).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up_proj(latent).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        allowed = self.causal_mask[:length, :length].view(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        weights = self.attn_dropout(weights)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.out_proj(output))
''')


replace("b3ce8243", r'''
%%writefile model_arch.py
"""Q2 decoder-only LM dispatching between standard MHA/GQA/MQA and Q2 MLA."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from attention import CausalSelfAttention, MultiHeadLatentAttention


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False
    attention_type: str = "standard"
    kv_latent_dim: int = None


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
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
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm_first = bool(cfg.norm_first)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        if cfg.attention_type == "mla":
            if cfg.pos_emb_type != "sinusoidal":
                raise ValueError("Q2 MLA requires sinusoidal positional encoding")
            self.attn = MultiHeadLatentAttention(
                cfg.d_model, cfg.n_heads, cfg.kv_latent_dim, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale,
            )
        elif cfg.attention_type == "standard":
            if cfg.n_kv_heads is None:
                raise ValueError("standard attention requires n_kv_heads")
            self.attn = CausalSelfAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
                cfg.dropout, attention_scale=cfg.attention_scale,
            )
        else:
            raise ValueError("attention_type must be 'standard' or 'mla'")
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class DecoderLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        if cfg.vocab_size <= 0 or cfg.n_layers <= 0:
            raise ValueError("vocab_size and n_layers must be positive")
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


replace("019b7982", r'''
# Reload both Q2 modules so this fresh session cannot retain Q1 class objects.
import importlib, sys
for _name in ("model_arch", "attention"):
    sys.modules.pop(_name, None)
attention = importlib.import_module("attention")
model_arch = importlib.import_module("model_arch")
LMConfig, DecoderLM = model_arch.LMConfig, model_arch.DecoderLM
print("Loaded Q2 attention/model modules:", attention.__file__, model_arch.__file__)
''')


insert_after("q2-3-final", [
    markdown_cell("q2-runtime-heading", r'''
### Q2 experiment runtime, validation, and independently resumable GPU pairs

The runtime below is Q2-specific: each variant owns its model, optimizer, deterministic
sampler, RNG state, history, checksummed latest/previous checkpoint, final weights, and
metrics.  Pair cells retain fixed pair-position device assignment (`cuda:0`, `cuda:1`).
Full mode requires both GPUs; only explicitly labelled smoke mode may fall back to
sequential execution.  Periodic recovery archives never contain the large Q1 ZIP—they
record its validated dependency hashes instead.
'''),
    code_cell("q2-runtime-writefile", r'''
%%writefile q2_runtime.py
"""Independent, crash-resumable Q2 training/evaluation runtime."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from model_arch import DecoderLM, LMConfig


SCHEMA_VERSION = 2
VARIANT_ORDER = ("mha", "gqa", "mqa", "mla")
DEFAULT_PAIRING = (("mha", "mla"), ("gqa", "mqa"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trajectory_hash(history, eval_history):
    return json_hash({"history": history, "eval_history": eval_history})


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cpu_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    candidate.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    json.loads(candidate.read_text(encoding="utf-8"))
    os.replace(candidate, path)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    torch.save(value, candidate)
    torch_load(candidate, "cpu")
    digest = sha256_file(candidate)
    os.replace(candidate, path)
    atomic_json(path.with_name(path.name + ".sha256.json"), {
        "sha256": digest, "size": path.stat().st_size, "updated_utc": utc_now(),
    })
    return digest


def verify_checksummed_file(path):
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        return path.stat().st_size == int(record["size"]) and sha256_file(path) == record["sha256"]
    except Exception:
        return False


class ResumableBatchSampler(Sampler):
    """Step-addressed deterministic shuffled batches; cursor is the completed step."""
    def __init__(self, dataset_size, batch_size, seed, start_step, stop_step):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("Dataset is too small for one complete batch")

    def __iter__(self):
        cached_epoch, permutation = None, None
        for step in range(self.start_step, self.stop_step):
            epoch, within = divmod(step, self.batches_per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
                permutation = torch.randperm(self.dataset_size, generator=generator)
                cached_epoch = epoch
            start = within * self.batch_size
            yield permutation[start:start + self.batch_size].tolist()

    def __len__(self):
        return max(0, self.stop_step - self.start_step)

    def state_dict(self, completed_step):
        epoch, within = divmod(int(completed_step), self.batches_per_epoch)
        return {
            "scheme": "epoch_randperm_seed_plus_epoch_v2",
            "seed": self.seed,
            "completed_step": int(completed_step),
            "epoch": int(epoch),
            "batch_cursor": int(within),
            "batches_per_epoch": int(self.batches_per_epoch),
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
        }


class Q2Runtime:
    def __init__(self, *, smoke_test, train_ids, eval_token_groups, base_config,
                 variant_definitions, target_steps, batch_size, learning_rate, seed,
                 tokenizer_fingerprint, train_ids_hash, held_out_hash, q1_dependency,
                 checkpoint_steps=50, checkpoint_seconds=300, archive_seconds=600,
                 notebook_path="copy_assignment3.ipynb"):
        self.smoke_test = bool(smoke_test)
        self.mode = "smoke" if self.smoke_test else "full"
        self.root = Path("artifacts") / self.mode / "q2"
        self.archive_path = Path(
            "q2_smoke_all_artifacts.zip" if self.smoke_test else "q2_all_artifacts.zip"
        )
        self.notebook_path = Path(notebook_path)
        self.train_ids = [int(value) for value in train_ids]
        # TokenDataset semantics are implemented without making four copies of the corpus.
        self.seq_len = int(base_config["max_seq_len"])
        self.chunk_count = max(0, (len(self.train_ids) - 1) // self.seq_len)
        if self.chunk_count < int(batch_size):
            raise ValueError("Token sequence is too short for the configured Q2 batch")
        self.eval_token_groups = {
            name: [[int(x) for x in ids] for ids in rows]
            for name, rows in eval_token_groups.items()
        }
        self.base_config = dict(base_config)
        self.variant_definitions = json.loads(json.dumps(variant_definitions))
        self.target_steps = int(target_steps)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.train_ids_hash = str(train_ids_hash)
        self.held_out_hash = str(held_out_hash)
        self.q1_dependency = json.loads(json.dumps(q1_dependency))
        self.q1_compatibility = {
            key: value for key, value in self.q1_dependency.items()
            if key != "q1_archive_sha256"
        }
        self.checkpoint_steps = int(checkpoint_steps)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.archive_seconds = float(archive_seconds)
        self.io_lock = threading.RLock()
        self.model_build_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_archive_monotonic = 0.0
        self.source_names = ("a1_tokenizer.py", "attention.py", "model_arch.py", "q2_runtime.py")
        self.source_hashes = {}
        self.initialization = {}

    def architecture(self, label):
        if label not in VARIANT_ORDER:
            raise KeyError(label)
        cfg = dict(self.base_config)
        cfg.update(self.variant_definitions[label])
        return cfg

    def _model_dir(self, label):
        return self.root / label

    def _experiment_id(self, label):
        return f"q2-{self.mode}-{label}"

    def _build_model(self, label, seed_offset=0):
        # Model construction is always serialized and isolated from the caller's CPU RNG.
        with self.model_build_lock, torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + int(seed_offset))
            model = DecoderLM(LMConfig(**self.architecture(label))).cpu()
        return model

    def _expected_initial_state(self, label):
        return torch_load(self._model_dir(label) / "initial.pt", "cpu")

    def _training_contract(self):
        return {
            "target_steps": self.target_steps,
            "batch_size": self.batch_size,
            "sequence_length": self.seq_len,
            "optimizer": {"name": "Adam", "lr": self.learning_rate},
            "seed": self.seed,
            "sampler_scheme": "epoch_randperm_seed_plus_epoch_v2",
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "train_ids_hash": self.train_ids_hash,
            "held_out_hash": self.held_out_hash,
            "smoke_test": self.smoke_test,
        }

    def _spec_payload(self, label):
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self._experiment_id(label),
            "label": label,
            "architecture": self.architecture(label),
            "training": self._training_contract(),
            "source_hashes": self.source_hashes,
            "initialization": self.initialization[label],
            # Archive-container identity is provenance, not semantic compatibility.
            "q1_dependency": self.q1_compatibility,
        }

    def _spec_fingerprint(self, label):
        return json_hash(self._spec_payload(label))

    def configure(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ("analysis", "plots", "source_snapshots", "manifests", "recovery"):
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

        for name in self.source_names:
            source = Path(name)
            if not source.is_file():
                raise FileNotFoundError(f"Required Q2 source is missing: {source}")
            self.source_hashes[name] = sha256_file(source)
            snapshot = self.root / "source_snapshots" / name
            if snapshot.is_file() and sha256_file(snapshot) != self.source_hashes[name]:
                raise RuntimeError(f"Existing immutable Q2 source snapshot conflicts: {snapshot}")
            if not snapshot.exists():
                shutil.copy2(source, snapshot)

        dependency_path = self.root / "manifests" / "q1_dependency.json"
        dependency_record = {
            "compatibility": self.q1_compatibility,
            "selected_archive_sha256": self.q1_dependency.get("q1_archive_sha256"),
            "recorded_utc": utc_now(),
        }
        if dependency_path.is_file():
            previous = json.loads(dependency_path.read_text(encoding="utf-8"))
            previous_compatibility = previous.get("compatibility")
            if previous_compatibility is None:
                previous_compatibility = {
                    key: value for key, value in previous.items() if key != "q1_archive_sha256"
                }
            if previous_compatibility != self.q1_compatibility:
                raise RuntimeError("The restored Q2 state depends on a different Q1 run")
        else:
            atomic_json(dependency_path, dependency_record)

        def load_existing_initial(label):
            model_dir = self._model_dir(label)
            init_path = model_dir / "initial.pt"
            metadata_path = model_dir / "initialization.json"
            if not init_path.exists():
                if metadata_path.exists():
                    raise RuntimeError(f"Initialization metadata exists without weights: {label}")
                return None, None
            if not verify_checksummed_file(init_path):
                raise RuntimeError(f"Initial-state checksum failed: {init_path}")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                old_config = model_dir / "config.json"
                if not old_config.is_file():
                    raise RuntimeError(f"Initial state has no initialization metadata: {label}")
                metadata = json.loads(old_config.read_text(encoding="utf-8"))["initialization"]
                atomic_json(metadata_path, metadata)
            state = torch_load(init_path, "cpu")
            if state_hash(state) != metadata.get("state_sha256"):
                raise RuntimeError(f"Initial-state tensor hash mismatch: {label}")
            shape_model = self._build_model(label)
            expected = shape_model.state_dict()
            if set(state) != set(expected) or any(
                tuple(state[name].shape) != tuple(expected[name].shape) for name in state
            ):
                raise RuntimeError(f"Archived initial state is incompatible with {label}")
            del shape_model
            return state, metadata

        mha_dir = self._model_dir("mha")
        mha_dir.mkdir(parents=True, exist_ok=True)
        canonical_state, canonical_metadata = load_existing_initial("mha")
        if canonical_state is None:
            canonical = self._build_model("mha", seed_offset=0)
            canonical_state = cpu_state_dict(canonical)
            del canonical
        canonical_hash = state_hash(canonical_state)
        if canonical_metadata is None:
            canonical_metadata = {
                "method": "canonical_mha_then_copy_same_name_same_shape_v1",
                "state_sha256": canonical_hash,
                "canonical_mha_state_sha256": canonical_hash,
                "shared_tensor_names": sorted(canonical_state),
                "variant_seed": self.seed,
            }
            atomic_torch(mha_dir / "initial.pt", canonical_state)
            atomic_json(mha_dir / "initialization.json", canonical_metadata)
        if canonical_metadata.get("canonical_mha_state_sha256") != canonical_hash:
            raise RuntimeError("Archived MHA initialization has inconsistent canonical identity")
        self.initialization["mha"] = canonical_metadata

        for label in VARIANT_ORDER[1:]:
            model_dir = self._model_dir(label)
            model_dir.mkdir(parents=True, exist_ok=True)
            state, metadata = load_existing_initial(label)
            if state is None:
                # Each architecture uses the same declared seed; only incompatible tensors differ.
                model = self._build_model(label, seed_offset=0)
                state = cpu_state_dict(model)
                shared = []
                for name, tensor in canonical_state.items():
                    if name in state and tuple(state[name].shape) == tuple(tensor.shape):
                        state[name] = tensor.clone()
                        shared.append(name)
                del model
                metadata = {
                    "method": "canonical_mha_then_copy_same_name_same_shape_v1",
                    "state_sha256": state_hash(state),
                    "canonical_mha_state_sha256": canonical_hash,
                    "shared_tensor_names": shared,
                    "variant_seed": self.seed,
                }
                atomic_torch(model_dir / "initial.pt", state)
                atomic_json(model_dir / "initialization.json", metadata)
            if metadata.get("canonical_mha_state_sha256") != canonical_hash:
                raise RuntimeError(f"{label} initialization belongs to another MHA canonical state")
            for name in metadata.get("shared_tensor_names", []):
                if name not in canonical_state or name not in state or not torch.equal(
                    state[name], canonical_state[name]
                ):
                    raise RuntimeError(f"{label} shared initialization diverges at {name}")
            self.initialization[label] = metadata

        for label in VARIANT_ORDER:
            spec = self._spec_payload(label)
            spec["spec_fingerprint"] = self._spec_fingerprint(label)
            spec_path = self._model_dir(label) / "config.json"
            if spec_path.is_file():
                previous = json.loads(spec_path.read_text(encoding="utf-8"))
                if previous != spec:
                    raise RuntimeError(f"Incompatible existing Q2 experiment config: {label}")
            else:
                atomic_json(spec_path, spec)
            model = self._build_model(label)
            model.load_state_dict(self._expected_initial_state(label), strict=True)
            params = sum(parameter.numel() for parameter in model.parameters())
            if params > 50_000_000:
                raise RuntimeError(f"{label} exceeds the 50M parameter budget: {params}")
            del model

        contract = {
            "mode": self.mode,
            "variant_order": list(VARIANT_ORDER),
            "default_pairing": [list(pair) for pair in DEFAULT_PAIRING],
            "training": self._training_contract(),
            "source_hashes": self.source_hashes,
        }
        atomic_json(self.root / "manifests" / "experiment_contract.json", contract)
        return self.status_report()

    def _chunk(self, index):
        start = int(index) * self.seq_len
        return (
            torch.tensor(self.train_ids[start:start + self.seq_len], dtype=torch.long),
            torch.tensor(self.train_ids[start + 1:start + self.seq_len + 1], dtype=torch.long),
        )

    class _ChunkDataset(torch.utils.data.Dataset):
        def __init__(self, runtime):
            self.runtime = runtime
        def __len__(self):
            return self.runtime.chunk_count
        def __getitem__(self, index):
            return self.runtime._chunk(index)

    def _checkpoint_paths(self, label):
        root = self._model_dir(label)
        return root / "checkpoint.latest.pt", root / "checkpoint.previous.pt"

    def _validate_checkpoint_payload(self, label, payload):
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint schema mismatch")
        if payload.get("experiment_id") != self._experiment_id(label):
            raise ValueError("checkpoint experiment mismatch")
        if payload.get("label") != label:
            raise ValueError("checkpoint label mismatch")
        if payload.get("spec_fingerprint") != self._spec_fingerprint(label):
            raise ValueError("checkpoint configuration fingerprint mismatch")
        if payload.get("architecture") != self.architecture(label):
            raise ValueError("checkpoint architecture mismatch")
        if payload.get("training_contract") != self._training_contract():
            raise ValueError("checkpoint training contract mismatch")
        step = payload.get("completed_step")
        if type(step) is not int or not 0 <= step <= self.target_steps:
            raise ValueError("invalid completed step")
        history = payload.get("history")
        if not isinstance(history, list) or len(history) != step:
            raise ValueError("training-history continuity mismatch")
        if history and [row.get("step") for row in history] != list(range(1, step + 1)):
            raise ValueError("training history has missing/duplicate steps")
        for row_index, row in enumerate(history, start=1):
            if row.get("tokens_processed") != row_index * self.batch_size * self.seq_len:
                raise ValueError("training history token count mismatch")
            if type(row.get("loss")) not in (int, float) or not math.isfinite(row["loss"]):
                raise ValueError("training history contains a non-finite loss")
        expected_tokens = step * self.batch_size * self.seq_len
        if payload.get("tokens_processed") != expected_tokens:
            raise ValueError("token counter does not match completed step")
        sampler = payload.get("sampler_state", {})
        batches_per_epoch = self.chunk_count // self.batch_size
        expected_epoch, expected_cursor = divmod(step, batches_per_epoch)
        expected_sampler = {
            "scheme": "epoch_randperm_seed_plus_epoch_v2",
            "seed": self.seed,
            "completed_step": step,
            "epoch": expected_epoch,
            "batch_cursor": expected_cursor,
            "batches_per_epoch": batches_per_epoch,
            "dataset_size": self.chunk_count,
            "batch_size": self.batch_size,
        }
        if sampler != expected_sampler:
            raise ValueError("sampler state does not match the deterministic step cursor")
        eval_history = payload.get("eval_history")
        if not isinstance(eval_history, list):
            raise ValueError("evaluation history is missing")
        eval_steps = [row.get("step") for row in eval_history]
        if eval_steps != sorted(set(eval_steps)) or any(
            type(value) is not int or not 1 <= value <= step for value in eval_steps
        ):
            raise ValueError("evaluation history has invalid or duplicate steps")
        for row in eval_history:
            if row.get("tokens_processed") != row["step"] * self.batch_size * self.seq_len:
                raise ValueError("evaluation token count mismatch")
            value = row.get("held_out_ppl")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 1:
                raise ValueError("evaluation history contains an invalid perplexity")
        if payload.get("trajectory_hash") != trajectory_hash(history, eval_history):
            raise ValueError("checkpoint trajectory hash mismatch")
        state = payload.get("model_state")
        if not isinstance(state, dict) or state_hash(state) != payload.get("model_state_sha256"):
            raise ValueError("checkpoint model state checksum mismatch")
        initial = self._expected_initial_state(label)
        if set(state) != set(initial):
            raise ValueError("checkpoint state keys mismatch")
        for name in state:
            if tuple(state[name].shape) != tuple(initial[name].shape):
                raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        if not isinstance(payload.get("optimizer_state"), dict):
            raise ValueError("optimizer state is missing")
        if payload.get("scheduler_state") is not None:
            raise ValueError("unexpected scheduler state for the fixed Adam experiment")
        if not isinstance(payload.get("amp_scaler_state"), (dict, type(None))):
            raise ValueError("invalid AMP scaler state")
        for key in ("cpu_generator_state", "device_rng_state"):
            rng = payload.get(key)
            if not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() == 0:
                raise ValueError(f"invalid model-specific RNG state: {key}")
        cumulative = payload.get("cumulative_active_seconds")
        if type(cumulative) not in (int, float) or not math.isfinite(cumulative) or cumulative < 0:
            raise ValueError("invalid cumulative training duration")
        best = payload.get("best_checkpoint_metadata")
        if eval_history:
            expected_best = min(eval_history, key=lambda row: (row["held_out_ppl"], row["step"]))
            if not isinstance(best, dict) or best.get("step") != expected_best["step"]:
                raise ValueError("best-checkpoint step does not match evaluation history")
            if best.get("held_out_ppl") != expected_best["held_out_ppl"]:
                raise ValueError("best-checkpoint metric does not match evaluation history")
            if best.get("path") != "best.pt" or not best.get("state_sha256") or not best.get("file_sha256"):
                raise ValueError("best-checkpoint identity is incomplete")
        elif best is not None:
            raise ValueError("best-checkpoint metadata exists before any evaluation")
        return payload

    def _load_checkpoint_candidate(self, label, path):
        path = Path(path)
        if not verify_checksummed_file(path):
            raise ValueError(f"checkpoint file checksum failed: {path}")
        payload = self._validate_checkpoint_payload(label, torch_load(path, "cpu"))
        model = self._build_model(label)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        optimizer.load_state_dict(payload["optimizer_state"])
        best = payload.get("best_checkpoint_metadata")
        # best.pt advances independently; only the latest payload must identify the live file.
        if best is not None and path.name == "checkpoint.latest.pt":
            best_path = self._model_dir(label) / best["path"]
            if not verify_checksummed_file(best_path) or sha256_file(best_path) != best["file_sha256"]:
                raise ValueError("best-checkpoint file checksum mismatch")
            if state_hash(torch_load(best_path, "cpu")) != best["state_sha256"]:
                raise ValueError("best-checkpoint tensor hash mismatch")
        del optimizer, model
        return payload

    def load_checkpoint(self, label):
        latest, previous = self._checkpoint_paths(label)
        errors = []
        for path in (latest, previous):
            if not path.exists():
                continue
            try:
                payload = self._load_checkpoint_candidate(label, path)
                if path == previous:
                    print(f"[{label}] latest invalid/unavailable; recovered previous checkpoint")
                return payload
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError(f"No valid checkpoint for {label}: " + " | ".join(errors))
        return None

    def _save_checkpoint(self, label, model, optimizer, scaler, *, step, history,
                         eval_history, cumulative_seconds, device, cpu_generator,
                         sampler_state, last_checkpoint_step, reason):
        state = cpu_state_dict(model)
        device_rng_state = (
            torch.cuda.get_rng_state(device).cpu()
            if device.type == "cuda" else torch.get_rng_state().cpu()
        )
        best_metadata = None
        if eval_history:
            best_row = min(eval_history, key=lambda row: (row["held_out_ppl"], row["step"]))
            best_path = self._model_dir(label) / "best.pt"
            if not verify_checksummed_file(best_path):
                raise RuntimeError(f"Best checkpoint is missing or corrupt for {label}")
            best_state = torch_load(best_path, "cpu")
            best_metadata = {
                "step": int(best_row["step"]),
                "held_out_ppl": float(best_row["held_out_ppl"]),
                "path": "best.pt",
                "state_sha256": state_hash(best_state),
                "file_sha256": sha256_file(best_path),
            }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self._experiment_id(label),
            "label": label,
            "spec_fingerprint": self._spec_fingerprint(label),
            "architecture": self.architecture(label),
            "training_contract": self._training_contract(),
            "completed_step": int(step),
            "epoch": int(sampler_state["epoch"]),
            "batch_cursor": int(sampler_state["batch_cursor"]),
            "tokens_processed": int(step * self.batch_size * self.seq_len),
            "sampler_state": sampler_state,
            "model_state": state,
            "model_state_sha256": state_hash(state),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "amp_scaler_state": scaler.state_dict() if scaler is not None else None,
            "cpu_generator_state": cpu_generator.get_state(),
            "device_rng_state": device_rng_state,
            "assigned_device": str(device),
            "history": history,
            "eval_history": eval_history,
            "trajectory_hash": trajectory_hash(history, eval_history),
            "best_checkpoint_metadata": best_metadata,
            "cumulative_active_seconds": float(cumulative_seconds),
            "last_checkpoint_step": int(last_checkpoint_step),
            "checkpoint_reason": str(reason),
            "saved_utc": utc_now(),
        }
        latest, previous = self._checkpoint_paths(label)
        with self.io_lock:
            candidate = latest.with_name(latest.name + f".{uuid.uuid4().hex}.candidate")
            torch.save(payload, candidate)
            loaded = torch_load(candidate, "cpu")
            self._validate_checkpoint_payload(label, loaded)
            candidate_hash = sha256_file(candidate)
            if latest.is_file() and verify_checksummed_file(latest):
                previous_candidate = previous.with_name(previous.name + f".{uuid.uuid4().hex}.tmp")
                shutil.copy2(latest, previous_candidate)
                if sha256_file(previous_candidate) != sha256_file(latest):
                    previous_candidate.unlink(missing_ok=True)
                    raise IOError("previous-checkpoint copy verification failed")
                os.replace(previous_candidate, previous)
                atomic_json(previous.with_name(previous.name + ".sha256.json"), {
                    "sha256": sha256_file(previous), "size": previous.stat().st_size,
                    "updated_utc": utc_now(),
                })
            os.replace(candidate, latest)
            atomic_json(latest.with_name(latest.name + ".sha256.json"), {
                "sha256": candidate_hash, "size": latest.stat().st_size,
                "updated_utc": utc_now(),
            })
            atomic_json(self._model_dir(label) / "status.json", {
                "status": "running", "experiment_id": self._experiment_id(label),
                "completed_step": int(step), "assigned_device": str(device),
                "latest_checkpoint_sha256": candidate_hash,
                "best_checkpoint_metadata": best_metadata,
                "updated_utc": utc_now(),
            })
        return int(step)

    def _save_plain_state(self, path, model):
        state = cpu_state_dict(model)
        atomic_torch(path, state)
        return state_hash(state)

    def _evaluate(self, model, token_rows, device):
        modes = [(module, module.training) for module in model.modules()]
        values = []
        try:
            model.eval()
            with torch.no_grad():
                for ids in token_rows:
                    if self.stop_event.is_set():
                        raise InterruptedError("Q2 pair stop requested during evaluation")
                    # Match common.perplexity_of_texts(..., max_len=256) exactly.
                    clipped = ids[: min(256, self.seq_len + 1)]
                    if len(clipped) < 2:
                        continue
                    x = torch.tensor(clipped[:-1], dtype=torch.long, device=device).unsqueeze(0)
                    y = torch.tensor(clipped[1:], dtype=torch.long, device=device).unsqueeze(0)
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    value = math.exp(float(loss.item()))
                    if not math.isfinite(value):
                        raise FloatingPointError("non-finite sentence perplexity")
                    values.append(value)
        finally:
            for module, training in modes:
                module.training = training
        if not values:
            raise ValueError("evaluation subset has no sentence with at least two tokens")
        return float(sum(values) / len(values))

    def _kv_counts(self, label):
        cfg = self.architecture(label)
        d, h, layers = cfg["d_model"], cfg["n_heads"], cfg["n_layers"]
        if label == "mla":
            latent = cfg["kv_latent_dim"]
            return int(3 * d * latent * layers), int(latent * layers * 4)
        kv = cfg["n_kv_heads"]
        return int(2 * d * kv * (d // h) * layers), int(2 * kv * (d // h) * layers * 4)

    def _optimizer_to(self, optimizer, device):
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _make_scaler(self, enabled):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _train_worker(self, label, device_name, position, model, restored):
        device = torch.device(device_name)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scaler = self._make_scaler(device.type == "cuda")
        start_step = 0
        history, eval_history = [], []
        cumulative_base = 0.0
        # Independent generator objects/devices, initialized from the same matched seed.
        cpu_generator = torch.Generator(device="cpu").manual_seed(self.seed + 10_000)
        resumed = restored is not None
        if restored is not None:
            model.load_state_dict(restored["model_state"], strict=True)
            optimizer.load_state_dict(restored["optimizer_state"])
            self._optimizer_to(optimizer, device)
            if restored.get("amp_scaler_state"):
                scaler.load_state_dict(restored["amp_scaler_state"])
            start_step = int(restored["completed_step"])
            history = list(restored["history"])
            eval_history = list(restored["eval_history"])
            cumulative_base = float(restored.get("cumulative_active_seconds", 0.0))
            cpu_generator.set_state(restored["cpu_generator_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(restored["device_rng_state"], device)
            else:
                torch.set_rng_state(restored["device_rng_state"])
        elif device.type == "cuda":
            generator = torch.Generator(device=device).manual_seed(self.seed + 20_000)
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.manual_seed(self.seed + 20_000)

        sampler = ResumableBatchSampler(
            self.chunk_count, self.batch_size, self.seed,
            start_step=start_step, stop_step=self.target_steps,
        )
        loader = DataLoader(
            self._ChunkDataset(self), batch_sampler=sampler, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        last_checkpoint_step = start_step
        last_checkpoint_time = time.monotonic()
        session_start = time.monotonic()
        current_step = start_step
        safe_to_checkpoint = True
        status_path = self._model_dir(label) / "status.json"
        with self.io_lock:
            atomic_json(status_path, {
                "status": "running", "experiment_id": self._experiment_id(label),
                "completed_step": start_step, "assigned_device": str(device),
                "resumed": resumed, "training_rng_seed": self.seed + 20_000,
                "updated_utc": utc_now(),
            })
        milestones = {max(1, math.ceil(self.target_steps * fraction / 5)) for fraction in range(1, 6)}
        best_ppl = min(
            (float(row["held_out_ppl"]) for row in eval_history if math.isfinite(float(row["held_out_ppl"]))),
            default=float("inf"),
        )
        progress = tqdm(
            total=self.target_steps, initial=start_step, position=position, leave=True,
            desc=f"{label.upper()} | {device}", dynamic_ncols=True,
        )
        try:
            for x, y in loader:
                if self.stop_event.is_set():
                    raise InterruptedError("Q2 pair stop requested; preserving the last safe boundary")
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {current_step + 1}")
                scaler.scale(loss).backward()
                safe_to_checkpoint = False
                scaler.step(optimizer)
                scaler.update()
                current_step += 1
                session_elapsed = time.monotonic() - session_start
                cumulative = cumulative_base + session_elapsed
                tokens = current_step * self.batch_size * self.seq_len
                history.append({
                    "step": current_step,
                    "epoch": sampler.state_dict(current_step)["epoch"],
                    "loss": float(loss.item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "tokens_processed": tokens,
                    "elapsed_seconds": cumulative,
                })
                safe_to_checkpoint = True
                with self.io_lock:
                    atomic_json(self._model_dir(label) / "history.json", {
                        "history": history, "eval_history": eval_history,
                        "completed_step": current_step, "cumulative_active_seconds": cumulative,
                    })

                evaluated = current_step in milestones and not any(
                    row.get("step") == current_step for row in eval_history
                )
                if evaluated:
                    ppl = self._evaluate(model, self.eval_token_groups["held_out"], device)
                    eval_history.append({
                        "step": current_step, "tokens_processed": tokens,
                        "held_out_ppl": ppl, "elapsed_seconds": cumulative,
                    })
                    if ppl < best_ppl:
                        best_ppl = ppl
                        with self.io_lock:
                            self._save_plain_state(self._model_dir(label) / "best.pt", model)

                due = (
                    current_step == self.target_steps
                    or current_step - last_checkpoint_step >= self.checkpoint_steps
                    or time.monotonic() - last_checkpoint_time >= self.checkpoint_seconds
                    or evaluated
                )
                if due:
                    reason = "completion" if current_step == self.target_steps else (
                        "evaluation" if evaluated else "interval"
                    )
                    last_checkpoint_step = self._save_checkpoint(
                        label, model, optimizer, scaler, step=current_step,
                        history=history, eval_history=eval_history,
                        cumulative_seconds=cumulative, device=device,
                        cpu_generator=cpu_generator,
                        sampler_state=sampler.state_dict(current_step),
                        last_checkpoint_step=current_step, reason=reason,
                    )
                    last_checkpoint_time = time.monotonic()
                    self.maybe_export_recovery_zip()

                recent = [row["loss"] for row in history[-20:]]
                rate = (current_step - start_step) / max(session_elapsed, 1e-9)
                progress.update(1)
                progress.set_postfix({
                    "epoch": sampler.state_dict(current_step)["epoch"],
                    "loss": f"{sum(recent) / len(recent):.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "tokens": tokens,
                    "step/s": f"{rate:.2f}",
                    "tok/s": f"{rate * self.batch_size * self.seq_len:.0f}",
                    "elapsed": f"{session_elapsed:.0f}s",
                    "cumulative": f"{cumulative:.0f}s",
                    "ckpt": last_checkpoint_step,
                    "state": "resumed" if resumed else "new",
                })

            cumulative = cumulative_base + (time.monotonic() - session_start)
            held_ppl = self._evaluate(model, self.eval_token_groups["held_out"], device)
            dialogue_ppl = self._evaluate(model, self.eval_token_groups["dialogue"], device)
            narration_ppl = self._evaluate(model, self.eval_token_groups["narration"], device)
            with self.io_lock:
                final_path = self._model_dir(label) / "final.pt"
                final_hash = self._save_plain_state(final_path, model)
                best_path = self._model_dir(label) / "best.pt"
                if not best_path.exists():
                    self._save_plain_state(best_path, model)
                if not verify_checksummed_file(best_path):
                    raise RuntimeError(f"Best checkpoint failed final validation: {label}")
                best_state_hash = state_hash(torch_load(best_path, "cpu"))
            kv_params, cache_bytes = self._kv_counts(label)
            params = int(sum(parameter.numel() for parameter in model.parameters()))
            definition = self.variant_definitions[label]
            result = {
                "attention_type": label,
                "n_kv_heads": definition.get("n_kv_heads"),
                "kv_latent_dim": definition.get("kv_latent_dim"),
                "held_out_ppl": float(held_ppl),
                "dialogue_ppl": float(dialogue_ppl),
                "narration_ppl": float(narration_ppl),
                "params": params,
                "kv_projection_params": int(kv_params),
                "kv_cache_bytes_per_token": int(cache_bytes),
            }
            metrics = {
                "result": result,
                "loss_curve": [float(row["loss"]) for row in history],
                "history": history,
                "eval_history": eval_history,
                "completed_steps": current_step,
                "tokens_processed": current_step * self.batch_size * self.seq_len,
                "cumulative_active_seconds": cumulative,
                "initialization": self.initialization[label],
                "final_state_sha256": final_hash,
                "final_file_sha256": sha256_file(final_path),
                "best_state_sha256": best_state_hash,
                "best_file_sha256": sha256_file(best_path),
                "trajectory_hash": trajectory_hash(history, eval_history),
                "spec_fingerprint": self._spec_fingerprint(label),
            }
            with self.io_lock:
                atomic_json(self._model_dir(label) / "metrics.json", metrics)
                metrics_sha256 = json_hash(metrics)
                atomic_json(status_path, {
                    "status": "complete", "experiment_id": self._experiment_id(label),
                    "completed_step": current_step, "assigned_device": str(device),
                    "final_held_out_ppl": held_ppl,
                    "final_state_sha256": final_hash,
                    "best_state_sha256": best_state_hash,
                    "trajectory_hash": metrics["trajectory_hash"],
                    "metrics_sha256": metrics_sha256,
                    "updated_utc": utc_now(),
                })
            print(f"[{label}] completed on {device}; this session={time.monotonic()-session_start:.1f}s, "
                  f"cumulative={cumulative:.1f}s, held-out PPL={held_ppl:.4f}")
            return result
        except BaseException as exc:
            emergency_error = None
            if safe_to_checkpoint:
                try:
                    cumulative = cumulative_base + (time.monotonic() - session_start)
                    self._save_checkpoint(
                        label, model, optimizer, scaler, step=current_step,
                        history=history, eval_history=eval_history,
                        cumulative_seconds=cumulative, device=device,
                        cpu_generator=cpu_generator,
                        sampler_state=sampler.state_dict(current_step),
                        last_checkpoint_step=current_step, reason="emergency",
                    )
                except BaseException as checkpoint_exc:
                    emergency_error = f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
            else:
                emergency_error = "unsafe mid-update boundary; retained the latest verified checkpoint"
            failure_status = "interrupted" if (
                isinstance(exc, (KeyboardInterrupt, InterruptedError)) or self.stop_event.is_set()
            ) else "failed"
            with self.io_lock:
                atomic_json(status_path, {
                    "status": failure_status, "experiment_id": self._experiment_id(label),
                    "completed_step": current_step, "assigned_device": str(device),
                    "error": f"{type(exc).__name__}: {exc}",
                    "emergency_checkpoint_error": emergency_error,
                    "traceback": traceback.format_exc(), "updated_utc": utc_now(),
                })
            raise
        finally:
            progress.close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _validate_final(self, label):
        status_path = self._model_dir(label) / "status.json"
        metrics_path = self._model_dir(label) / "metrics.json"
        final_path = self._model_dir(label) / "final.pt"
        best_path = self._model_dir(label) / "best.pt"
        if not all(path.is_file() for path in (status_path, metrics_path, final_path, best_path)):
            return False
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete" or status.get("completed_step") != self.target_steps:
                return False
            if metrics.get("completed_steps") != self.target_steps:
                return False
            if metrics.get("spec_fingerprint") != self._spec_fingerprint(label):
                return False
            if status.get("metrics_sha256") != json_hash(metrics):
                return False
            history = metrics.get("history")
            eval_history = metrics.get("eval_history")
            if not isinstance(history, list) or len(history) != self.target_steps:
                return False
            if [row.get("step") for row in history] != list(range(1, self.target_steps + 1)):
                return False
            if metrics.get("loss_curve") != [row.get("loss") for row in history]:
                return False
            if metrics.get("tokens_processed") != self.target_steps * self.batch_size * self.seq_len:
                return False
            if not isinstance(eval_history, list) or not eval_history or eval_history[-1].get("step") != self.target_steps:
                return False
            expected_trajectory = trajectory_hash(history, eval_history)
            if metrics.get("trajectory_hash") != expected_trajectory or status.get("trajectory_hash") != expected_trajectory:
                return False
            if not all(
                type(row.get("loss")) in (int, float) and math.isfinite(row["loss"])
                for row in history
            ):
                return False
            if not (verify_checksummed_file(final_path) and verify_checksummed_file(best_path)):
                return False
            state = torch_load(final_path, "cpu")
            best_state = torch_load(best_path, "cpu")
            if state_hash(state) != metrics.get("final_state_sha256") or state_hash(best_state) != metrics.get("best_state_sha256"):
                return False
            if sha256_file(final_path) != metrics.get("final_file_sha256") or sha256_file(best_path) != metrics.get("best_file_sha256"):
                return False
            if status.get("final_state_sha256") != metrics.get("final_state_sha256"):
                return False
            if status.get("best_state_sha256") != metrics.get("best_state_sha256"):
                return False
            initial = self._expected_initial_state(label)
            if set(state) != set(initial) or set(best_state) != set(initial):
                return False
            for name in state:
                if (tuple(state[name].shape) != tuple(initial[name].shape)
                        or tuple(best_state[name].shape) != tuple(initial[name].shape)):
                    return False
            row = metrics.get("result", {})
            definition = self.variant_definitions[label]
            if row.get("attention_type") != label:
                return False
            if row.get("n_kv_heads") != definition.get("n_kv_heads"):
                return False
            if row.get("kv_latent_dim") != definition.get("kv_latent_dim"):
                return False
            if not all(
                type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 1
                for key in ("held_out_ppl", "dialogue_ppl", "narration_ppl")
            ):
                return False
            if status.get("final_held_out_ppl") != row["held_out_ppl"]:
                return False
            expected_kv, expected_cache = self._kv_counts(label)
            if row.get("kv_projection_params") != expected_kv or row.get("kv_cache_bytes_per_token") != expected_cache:
                return False
            model = self._build_model(label)
            model.load_state_dict(state, strict=True)
            model.load_state_dict(best_state, strict=True)
            expected_params = int(sum(parameter.numel() for parameter in model.parameters()))
            if type(row.get("params")) is not int or row["params"] != expected_params or row["params"] > 50_000_000:
                return False
            if metrics.get("initialization") != self.initialization[label]:
                return False
            cumulative = metrics.get("cumulative_active_seconds")
            return type(cumulative) in (int, float) and math.isfinite(cumulative) and cumulative >= 0
        except Exception:
            return False

    def is_complete(self, label):
        return self._validate_final(label)

    def status_report(self):
        report = {}
        for label in VARIANT_ORDER:
            status_path = self._model_dir(label) / "status.json"
            if self.is_complete(label):
                metrics = json.loads((self._model_dir(label) / "metrics.json").read_text(encoding="utf-8"))
                report[label] = {
                    "status": "complete", "step": self.target_steps,
                    "held_out_ppl": metrics["result"]["held_out_ppl"],
                }
            else:
                checkpoint = self.load_checkpoint(label)
                if checkpoint is not None:
                    prior_status = (
                        json.loads(status_path.read_text(encoding="utf-8")).get("status")
                        if status_path.is_file() else None
                    )
                    report[label] = {
                        "status": "resumable", "step": int(checkpoint["completed_step"]),
                        "last_recorded_status": prior_status,
                    }
                elif status_path.is_file():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    report[label] = {
                        "status": status.get("status", "incomplete"),
                        "step": int(status.get("completed_step", 0)),
                    }
                else:
                    report[label] = {"status": "missing", "step": 0}
        return report

    def _persist_pairing(self, plan, evidence):
        flat = [label for pair in plan for label in pair]
        if len(plan) != 2 or any(len(pair) != 2 for pair in plan) or sorted(flat) != sorted(VARIANT_ORDER):
            raise ValueError("A Q2 pairing plan must cover every variant exactly once")
        path = self.root / "manifests" / "pairing_plan.json"
        record = {"pairs": [list(pair) for pair in plan], "evidence": evidence, "created_utc": utc_now()}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            return tuple(tuple(pair) for pair in existing["pairs"])
        atomic_json(path, record)
        return tuple(tuple(pair) for pair in plan)

    def load_pairing_plan(self):
        path = self.root / "manifests" / "pairing_plan.json"
        if path.is_file():
            return tuple(tuple(pair) for pair in json.loads(path.read_text(encoding="utf-8"))["pairs"])
        return DEFAULT_PAIRING

    def timing_probe(self, measured_steps=3):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sample_indices = list(range(self.batch_size))
        rows = [self._chunk(index) for index in sample_indices]
        x = torch.stack([row[0] for row in rows]).to(device)
        y = torch.stack([row[1] for row in rows]).to(device)
        timings = {}
        for label in VARIANT_ORDER:
            model = self._build_model(label)
            model.load_state_dict(self._expected_initial_state(label), strict=True)
            model = model.to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            scaler = self._make_scaler(device.type == "cuda")
            samples = []
            for iteration in range(2 + int(measured_steps)):
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if iteration >= 2:
                    samples.append(elapsed)
            timings[label] = float(statistics.median(samples))
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
        record = {"device": str(device), "median_step_seconds": timings, "measured_steps": measured_steps}
        atomic_json(self.root / "analysis" / "timing_probe.json", record)
        return record

    def choose_pairing(self, timing_record):
        existing = self.root / "manifests" / "pairing_plan.json"
        if existing.is_file():
            return self.load_pairing_plan()
        if any(self._checkpoint_paths(label)[0].exists() or self.is_complete(label) for label in VARIANT_ORDER):
            return self._persist_pairing(DEFAULT_PAIRING, {
                "reason": "Training state already exists; retained stable default pairing",
                "timing_probe": timing_record,
            })
        times = timing_record["median_step_seconds"]
        ordered = sorted(VARIANT_ORDER, key=lambda label: (times[label], label))
        measured_plan = ((ordered[0], ordered[1]), (ordered[2], ordered[3]))
        default_gap = sum(abs(times[a] - times[b]) for a, b in DEFAULT_PAIRING)
        measured_gap = sum(abs(times[a] - times[b]) for a, b in measured_plan)
        plan = measured_plan if measured_gap + 1e-12 < default_gap else DEFAULT_PAIRING
        reason = (
            "Adjacent measured step times minimize within-pair imbalance"
            if plan == measured_plan and plan != DEFAULT_PAIRING
            else "The requested MHA+MLA and GQA+MQA pairing is as balanced as the measured alternative"
        )
        return self._persist_pairing(plan, {
            "reason": reason, "timing_probe": timing_record,
            "default_total_abs_gap": default_gap, "selected_total_abs_gap": min(default_gap, measured_gap),
        })

    def _prepare_model_for_run(self, label, restored):
        model = self._build_model(label)
        state = restored["model_state"] if restored is not None else self._expected_initial_state(label)
        model.load_state_dict(state, strict=True)
        return model

    def run_pair(self, pair, position_offset=0, full_plan=None):
        self.stop_event.clear()
        pair = tuple(pair)
        if len(pair) != 2 or pair[0] == pair[1] or any(label not in VARIANT_ORDER for label in pair):
            raise ValueError(f"Invalid Q2 pair: {pair}")
        if full_plan is not None:
            persisted = self._persist_pairing(tuple(tuple(x) for x in full_plan), {
                "reason": "Notebook-selected plan", "source": "timing/default cell",
            })
            if pair not in persisted:
                raise ValueError(f"Pair {pair} is not in the persisted plan {persisted}")
        unfinished = [label for label in pair if not self.is_complete(label)]
        if not unfinished:
            print(f"Pair {pair} already complete; validated saved results and skipped training.")
            return {label: self.result(label) for label in pair}
        if not self.smoke_test and torch.cuda.device_count() < 2:
            raise RuntimeError(
                "Full Q2 pair training requires two visible GPUs. Enable Kaggle's 2 x T4 accelerator."
            )

        prepared = {}
        for position, label in enumerate(pair):
            if label not in unfinished:
                continue
            restored = self.load_checkpoint(label)
            prepared[label] = (restored, self._prepare_model_for_run(label, restored), position)

        errors, outputs = {}, {}
        try:
            if torch.cuda.device_count() >= 2:
                with ThreadPoolExecutor(max_workers=len(prepared), thread_name_prefix="q2") as pool:
                    futures = {}
                    for label, (restored, model, position) in prepared.items():
                        device_name = f"cuda:{position}"  # fixed by position even if its partner finished
                        future = pool.submit(
                            self._train_worker, label, device_name, position_offset + position, model, restored
                        )
                        futures[future] = label
                    try:
                        for future in as_completed(futures):
                            label = futures[future]
                            try:
                                outputs[label] = future.result()
                            except BaseException as exc:
                                errors[label] = exc
                    except BaseException:
                        # Main-thread interrupts request a safe-boundary stop on both workers.
                        self.stop_event.set()
                        raise
            else:
                fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
                print(f"SMOKE ONLY: two GPUs unavailable; executing {pair} sequentially on {fallback}.")
                for label, (restored, model, position) in prepared.items():
                    try:
                        outputs[label] = self._train_worker(
                            label, fallback, position_offset, model, restored
                        )
                    except (KeyboardInterrupt, SystemExit):
                        self.stop_event.set()
                        raise
                    except BaseException as exc:
                        errors[label] = exc
        finally:
            try:
                self.export_recovery_zip(force=True)
            except BaseException as archive_exc:
                print("Q2 emergency archive failed:", type(archive_exc).__name__, archive_exc)
        if errors:
            details = "; ".join(f"{label}: {type(exc).__name__}: {exc}" for label, exc in errors.items())
            raise RuntimeError(
                "One or more Q2 workers failed after every partner was allowed to finish: " + details
            )
        for label in pair:
            if self.is_complete(label):
                outputs[label] = self.result(label)
        return outputs

    def result(self, label):
        if not self.is_complete(label):
            raise RuntimeError(f"Q2 {label} is not complete and deeply validated")
        metrics = json.loads((self._model_dir(label) / "metrics.json").read_text(encoding="utf-8"))
        return metrics["result"]

    def results(self):
        return [self.result(label) for label in VARIANT_ORDER]

    def require_all_complete(self):
        missing = [label for label in VARIANT_ORDER if not self.is_complete(label)]
        if missing:
            raise RuntimeError(f"Complete/resume these Q2 models first: {missing}")
        return self.results()

    def load_final_model(self, label, device="cpu"):
        if not self.is_complete(label):
            raise RuntimeError(f"Q2 {label} has no valid completed final state")
        model = self._build_model(label)
        model.load_state_dict(torch_load(self._model_dir(label) / "final.pt", "cpu"), strict=True)
        return model.to(device)

    def parameter_accounting(self):
        rows = []
        for label in VARIANT_ORDER:
            model = self.load_final_model(label, "cpu")
            cfg = self.architecture(label)
            if label == "mla":
                names = ("kv_down_proj.weight", "k_up_proj.weight", "v_up_proj.weight")
            else:
                names = ("k_proj.weight", "v_proj.weight")
            actual = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if any(name.endswith("attn." + suffix) for suffix in names)
            )
            formula, cache = self._kv_counts(label)
            if int(actual) != int(formula):
                raise AssertionError(f"{label} actual KV tensors={actual}, formula={formula}")
            result = self.result(label)
            if result["kv_projection_params"] != formula or result["kv_cache_bytes_per_token"] != cache:
                raise AssertionError(f"{label} persisted accounting is inconsistent")
            rows.append({
                "variant": label, "total_params": int(sum(p.numel() for p in model.parameters())),
                "actual_kv_projection_params": int(actual), "formula_kv_projection_params": int(formula),
                "kv_cache_bytes_per_token": int(cache),
            })
        mha = next(row for row in rows if row["variant"] == "mha")
        for row in rows:
            row["kv_param_fraction_vs_mha"] = row["actual_kv_projection_params"] / mha["actual_kv_projection_params"]
            row["cache_fraction_vs_mha"] = row["kv_cache_bytes_per_token"] / mha["kv_cache_bytes_per_token"]
        atomic_json(self.root / "analysis" / "parameter_accounting.json", {"rows": rows})
        return rows

    def select_for_q3(self):
        eligible = [self.result(label) for label in ("mha", "gqa", "mqa")]
        eligible = [row for row in eligible if math.isfinite(float(row["held_out_ppl"]))]
        if len(eligible) != 3:
            raise RuntimeError("Q3 selection requires all finite MHA/GQA/MQA held-out results")
        winner = min(eligible, key=lambda row: (row["held_out_ppl"], row["attention_type"]))
        label = winner["attention_type"]
        record = {
            "provisional_smoke_only": self.smoke_test,
            "selection_rule": "lowest finite held_out_ppl among mha/gqa/mqa; mla excluded",
            "selected_label": label,
            "held_out_ppl": winner["held_out_ppl"],
            "n_heads": self.architecture(label)["n_heads"],
            "n_kv_heads": self.architecture(label)["n_kv_heads"],
            "architecture": self.architecture(label),
            "final_state_path": str(self._model_dir(label) / "final.pt"),
            "final_state_file_sha256": sha256_file(self._model_dir(label) / "final.pt"),
            "source_snapshot_dir": str(self.root / "source_snapshots"),
            "source_hashes": self.source_hashes,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "train_ids_hash": self.train_ids_hash,
            "held_out_hash": self.held_out_hash,
            "eligible_results": eligible,
        }
        atomic_json(self.root / "analysis" / "q3_selection.json", record)
        return record

    def _official_submission_valid(self):
        if self.smoke_test:
            return False
        submission = Path("submissions/q2")
        required = [submission / name for name in
                    ("attention.py", "model_arch.py", "model.pt", "config.json", "metrics.json")]
        if not all(path.is_file() for path in required):
            return False
        try:
            if not all(self.is_complete(label) for label in VARIANT_ORDER):
                return False
            config = json.loads((submission / "config.json").read_text(encoding="utf-8"))
            expected_config = self.architecture("gqa")
            if any(config.get(key) != value for key, value in expected_config.items()):
                return False
            metrics = json.loads((submission / "metrics.json").read_text(encoding="utf-8"))
            if metrics != {"per_config_results": self.results()}:
                return False
            submitted_state = torch_load(submission / "model.pt", "cpu")
            final_state = torch_load(self._model_dir("gqa") / "final.pt", "cpu")
            if state_hash(submitted_state) != state_hash(final_state):
                return False
            for name in ("a1_tokenizer.py", "attention.py", "model_arch.py"):
                if sha256_file(submission / name) != sha256_file(self.root / "source_snapshots" / name):
                    return False
            return True
        except Exception:
            return False

    def _archive_file_list(self):
        files = []
        if self.root.is_dir():
            files.extend(
                path for path in self.root.rglob("*")
                if path.is_file() and path != self.root / "recovery" / "latest_export.json"
            )
        submission = Path("submissions/q2")
        submission_valid = self._official_submission_valid()
        if submission_valid:
            files.extend(path for path in submission.rglob("*") if path.is_file())
        if submission_valid and Path("analysis_log.json").is_file():
            files.append(Path("analysis_log.json"))
        if self.notebook_path.is_file():
            files.append(self.notebook_path)
        unique = {}
        for path in files:
            resolved = path.resolve()
            unique[resolved] = path
        return [unique[key] for key in sorted(unique, key=str)]

    def _validate_archive(self, path):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"bad ZIP member: {bad}")
            manifest = json.loads(archive.read("recovery/integrity_manifest.json"))
            if manifest.get("mode") != self.mode:
                raise ValueError("recovery archive mode mismatch")
            for rel, info in manifest.get("files", {}).items():
                digest = hashlib.sha256()
                size = 0
                with archive.open(rel) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                if size != int(info["size"]) or digest.hexdigest() != info["sha256"]:
                    raise ValueError(f"archive manifest mismatch: {rel}")
        return manifest

    def export_recovery_zip(self, force=False):
        with self.io_lock:
            now = time.monotonic()
            if not force and now - self.last_archive_monotonic < self.archive_seconds:
                return self.archive_path if self.archive_path.exists() else None
            files = self._archive_file_list()
            manifest_files = {}
            for path in files:
                rel = path.as_posix()
                manifest_files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "question": "q2", "mode": self.mode, "created_utc": utc_now(),
                "files": manifest_files, "source_hashes": self.source_hashes,
                "q1_dependency": self.q1_dependency,
                "status": self.status_report(),
            }
            candidate = self.archive_path.with_name(
                self.archive_path.name + f".{uuid.uuid4().hex}.candidate"
            )
            with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in files:
                    archive.write(path, arcname=path.as_posix())
                archive.writestr(
                    "recovery/integrity_manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                )
            self._validate_archive(candidate)
            previous = self.archive_path.with_name(self.archive_path.name.replace(".zip", ".previous.zip"))
            if self.archive_path.is_file():
                try:
                    self._validate_archive(self.archive_path)
                    previous_candidate = previous.with_name(previous.name + f".{uuid.uuid4().hex}.tmp")
                    shutil.copy2(self.archive_path, previous_candidate)
                    self._validate_archive(previous_candidate)
                    os.replace(previous_candidate, previous)
                except Exception as exc:
                    print("Existing Q2 archive was not rotated because validation failed:", exc)
            os.replace(candidate, self.archive_path)
            archive_hash = sha256_file(self.archive_path)
            atomic_json(self.root / "recovery" / "latest_export.json", {
                "path": str(self.archive_path.resolve()), "sha256": archive_hash,
                "size": self.archive_path.stat().st_size, "created_utc": utc_now(),
            })
            self.last_archive_monotonic = now
            print(f"Validated Q2 recovery ZIP: {self.archive_path} ({archive_hash[:12]}..., "
                  f"{self.archive_path.stat().st_size / 2**20:.1f} MiB)")
            return self.archive_path

    def maybe_export_recovery_zip(self):
        return self.export_recovery_zip(force=False)

    def run_self_tests(self):
        from attention import CausalSelfAttention, MultiHeadLatentAttention
        tests = {}
        generator = torch.Generator(device="cpu").manual_seed(781)
        for kv in (8, 2, 1):
            module = CausalSelfAttention(64, 8, kv, 16, dropout=0.0).eval()
            for layer_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                layer = getattr(module, layer_name)
                with torch.no_grad():
                    layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.1)
            x = torch.randn(2, 7, 64, generator=generator)
            q = F.linear(x, module.q_proj.weight).view(2, 7, 8, 8).transpose(1, 2)
            k = F.linear(x, module.k_proj.weight).view(2, 7, kv, 8).transpose(1, 2)
            v = F.linear(x, module.v_proj.weight).view(2, 7, kv, 8).transpose(1, 2)
            index = torch.arange(8) // (8 // kv)
            k, v = k[:, index], v[:, index]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
            scores = scores.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
            expected = F.linear(
                (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 64),
                module.out_proj.weight,
            )
            torch.testing.assert_close(module(x), expected, atol=1e-5, rtol=1e-5)
            tests[f"standard_reference_kv{kv}"] = "passed"

        for latent in (5, 16):
            module = MultiHeadLatentAttention(64, 8, latent, 16, dropout=0.0, attention_scale=0.37)
            x = torch.randn(2, 7, 64, generator=generator, requires_grad=True)
            actual = module(x)
            q = F.linear(x, module.q_proj.weight).view(2, 7, 8, 8).transpose(1, 2)
            k = F.linear(x, module.k_up_proj.weight @ module.kv_down_proj.weight).view(2, 7, 8, 8).transpose(1, 2)
            v = F.linear(x, module.v_up_proj.weight @ module.kv_down_proj.weight).view(2, 7, 8, 8).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * 0.37
            scores = scores.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
            expected = F.linear(
                (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 64),
                module.out_proj.weight,
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            actual.square().mean().backward()
            if not torch.isfinite(x.grad).all():
                raise AssertionError("MLA gradient is non-finite")
            tests[f"mla_reference_r{latent}"] = "passed"

        for label in VARIANT_ORDER:
            model = self._build_model(label)
            model.load_state_dict(self._expected_initial_state(label), strict=True)
            ids = torch.randint(0, self.base_config["vocab_size"], (2, min(8, self.seq_len)), generator=generator)
            logits = model(ids)
            loss = logits.float().square().mean()
            loss.backward()
            if tuple(logits.shape) != (2, ids.size(1), self.base_config["vocab_size"]):
                raise AssertionError(f"bad {label} output shape")
            if not torch.isfinite(logits).all():
                raise AssertionError(f"non-finite {label} forward")
            tests[f"model_forward_backward_{label}"] = "passed"

        # Causality: perturb a future suffix and compare the unchanged prefix.
        causal = self._build_model("gqa").eval()
        causal.load_state_dict(self._expected_initial_state("gqa"), strict=True)
        ids = torch.randint(0, self.base_config["vocab_size"], (2, min(8, self.seq_len)), generator=generator)
        changed = ids.clone()
        cutoff = ids.size(1) // 2 - 1
        changed[:, cutoff + 1:] = (changed[:, cutoff + 1:] + 1) % self.base_config["vocab_size"]
        with torch.no_grad():
            torch.testing.assert_close(causal(ids)[:, :cutoff + 1], causal(changed)[:, :cutoff + 1])
        tests["causal_mask"] = "passed"

        sampler_a = list(ResumableBatchSampler(23, 4, 1337, 0, 6))
        sampler_b = list(ResumableBatchSampler(23, 4, 1337, 3, 6))
        if sampler_a[3:] != sampler_b:
            raise AssertionError("deterministic sampler continuation failed")
        tests["sampler_continuation"] = "passed"

        bad = {
            "schema_version": SCHEMA_VERSION, "experiment_id": self._experiment_id("mha"),
            "spec_fingerprint": "wrong",
        }
        try:
            self._validate_checkpoint_payload("mha", bad)
            raise AssertionError("incompatible checkpoint was accepted")
        except ValueError:
            tests["incompatible_checkpoint_rejected"] = "passed"

        with tempfile.TemporaryDirectory(prefix="q2_optimizer_test_") as temp:
            model = self._build_model("mha")
            model.load_state_dict(self._expected_initial_state("mha"), strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            ids = torch.randint(0, self.base_config["vocab_size"], (1, min(4, self.seq_len)), generator=generator)
            model(ids).float().mean().backward()
            optimizer.step()
            path = Path(temp) / "state.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            loaded = torch_load(path, "cpu")
            clone = self._build_model("mha")
            clone.load_state_dict(loaded["model"], strict=True)
            clone_optimizer = torch.optim.Adam(clone.parameters(), lr=self.learning_rate)
            clone_optimizer.load_state_dict(loaded["optimizer"])
            if len(clone_optimizer.state) != len(optimizer.state):
                raise AssertionError("optimizer restoration failed")
        tests["checkpoint_and_optimizer_roundtrip"] = "passed"

        with tempfile.TemporaryDirectory(prefix="q2_failure_test_") as temp:
            marker = Path(temp) / "partner.finished"
            def fail():
                raise RuntimeError("intentional isolated test failure")
            def finish():
                marker.write_text("ok", encoding="utf-8")
                return "ok"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(fail), pool.submit(finish)]
                failures = 0
                for future in futures:
                    try:
                        future.result()
                    except RuntimeError:
                        failures += 1
            if failures != 1 or marker.read_text(encoding="utf-8") != "ok":
                raise AssertionError("partner-failure isolation failed")
        tests["partner_failure_isolation"] = "passed"

        if torch.cuda.device_count() >= 2:
            def gpu_probe(index):
                tensor = torch.arange(8, device=f"cuda:{index}")
                return str(tensor.device), int(tensor.sum().item())
            with ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(gpu_probe, (0, 1)))
            if values != [("cuda:0", 28), ("cuda:1", 28)]:
                raise AssertionError(f"dual-GPU probe failed: {values}")
            tests["dual_gpu_concurrency"] = "passed"
        else:
            tests["dual_gpu_concurrency"] = "skipped (requires Kaggle 2 x T4)"

        archive = self.export_recovery_zip(force=True)
        self._validate_archive(archive)
        tests["recovery_zip_integrity"] = "passed"
        command = [sys.executable, "-c", "import attention, model_arch, q2_runtime; print('fresh-import-ok')"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or "fresh-import-ok" not in completed.stdout:
            raise AssertionError("fresh-process module import failed: " + completed.stderr)
        tests["fresh_process_reconstruction"] = "passed"
        atomic_json(self.root / "analysis" / "self_tests.json", {
            "mode": self.mode, "tests": tests, "completed_utc": utc_now(),
        })
        return tests

    def run_official_autograder(self, student_dir="."):
        if self.smoke_test:
            raise RuntimeError("Official grading is intentionally disabled for smoke artifacts")
        command = [sys.executable, "autograder_q2.py", "--student_dir", str(student_dir)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        report = {
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "completed_utc": utc_now(),
        }
        atomic_json(self.root / "analysis" / "official_autograder.json", report)
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"Official Q2 autograder failed with exit code {completed.returncode}")
        return report
'''),
    code_cell("q2-runtime-configure", r'''
# Matched Q2 experiment construction. Tokenization happens once, before worker threads.
from q2_runtime import Q2Runtime, DEFAULT_PAIRING, VARIANT_ORDER

Q2_VARIANT_DEFINITIONS = {
    "mha": {"attention_type": "standard", "n_kv_heads": 8, "kv_latent_dim": None},
    "gqa": {"attention_type": "standard", "n_kv_heads": 2, "kv_latent_dim": None},
    "mqa": {"attention_type": "standard", "n_kv_heads": 1, "kv_latent_dim": None},
    "mla": {"attention_type": "mla", "n_kv_heads": None, "kv_latent_dim": Q2_MLA_LATENT_DIM},
}
Q2_BASE_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "d_model": Q2_MODEL_DIM,
    "n_layers": Q2_MODEL_LAYERS,
    "n_heads": 8,
    "d_ff": Q2_MODEL_FF,
    "max_seq_len": Q2_MODEL_SEQ_LEN,
    "dropout": float(_q1_arch["dropout"]),
    "pos_emb_type": "sinusoidal",
    "attention_scale": None,
    "norm_first": bool(_q1_arch["norm_first"]),
}
Q2_EVAL_TOKEN_GROUPS = {
    "held_out": [tokenizer.encode(text) for text in held_out],
    "dialogue": [tokenizer.encode(text) for text in q2_dialogue_sentences],
    "narration": [tokenizer.encode(text) for text in q2_narration_sentences],
}

q2 = Q2Runtime(
    smoke_test=Q2_SMOKE_TEST,
    train_ids=train_ids,
    eval_token_groups=Q2_EVAL_TOKEN_GROUPS,
    base_config=Q2_BASE_CONFIG,
    variant_definitions=Q2_VARIANT_DEFINITIONS,
    target_steps=Q2_TRAIN_STEPS,
    batch_size=Q2_BATCH_SIZE,
    learning_rate=Q2_LEARNING_RATE,
    seed=int(_q1_spec["seed"]),
    tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
    train_ids_hash=_q1_spec["train_ids_hash"],
    held_out_hash=_q1_spec["held_out_hash"],
    q1_dependency=Q1_DEPENDENCY,
    checkpoint_steps=Q2_CHECKPOINT_STEPS,
    checkpoint_seconds=Q2_CHECKPOINT_SECONDS,
    archive_seconds=Q2_ARCHIVE_SECONDS,
)
q2_status = q2.configure()
q2_pair_plan = q2.load_pairing_plan()
print("Q2 status:", json.dumps(q2_status, indent=2))
print("Current pair plan:", q2_pair_plan)
q2.export_recovery_zip(force=True)  # first protected export exists before training starts
'''),
    code_cell("q2-smoke-tests", r'''
# Lightweight architecture/recovery tests only; this cell does not launch four full runs.
q2_self_tests = q2.run_self_tests()
print(json.dumps(q2_self_tests, indent=2))
'''),
    code_cell("q2-timing-probe", r'''
# Optional disposable probe. Run before either training pair for measured load balancing.
q2_timing = q2.timing_probe(measured_steps=3)
q2_pair_plan = q2.choose_pairing(q2_timing)
print("Median train-step seconds:", q2_timing["median_step_seconds"])
print("Persisted balanced pair plan:", q2_pair_plan)
'''),
    code_cell("q2-pair1-training", r'''
# Pair 1 only: first member -> cuda:0, second member -> cuda:1.
# Re-running resumes unfinished members and deeply validates/skips completed members.
q2_pair1_results = q2.run_pair(q2_pair_plan[0], position_offset=0, full_plan=q2_pair_plan)
print(json.dumps(q2_pair1_results, indent=2))
'''),
    code_cell("q2-pair2-training", r'''
# Pair 2 only: first member -> cuda:0, second member -> cuda:1.
# This cell never triggers Pair 1.
q2_pair2_results = q2.run_pair(q2_pair_plan[1], position_offset=0, full_plan=q2_pair_plan)
print(json.dumps(q2_pair2_results, indent=2))
'''),
])


replace("4a2cf918", r'''
# Aggregate only deeply validated completed experiments. No metric is invented or prefilled.
from q2_runtime import atomic_json

q2_results = q2.require_all_complete()
q2_results_by_name = {row["attention_type"]: row for row in q2_results}
q2_loss_curves = {
    label: json.loads((q2.root / label / "metrics.json").read_text(encoding="utf-8"))["loss_curve"]
    for label in VARIANT_ORDER
}
metrics_q2 = {"per_config_results": q2_results}
atomic_json(q2.root / "analysis" / "per_config_results.json", metrics_q2)
log_result("q2_per_config_results", metrics_q2)

print("Matched Q2 results (same tokenizer, split, seed/order, steps, batch, optimizer, LR, and PE):")
print(f"{'variant':<8} {'held PPL':>14} {'dialogue':>14} {'narration':>14} "
      f"{'params':>12} {'KV params':>12} {'cache B/tok':>12}")
for row in q2_results:
    print(f"{row['attention_type']:<8} {row['held_out_ppl']:>14.4f} "
          f"{row['dialogue_ppl']:>14.4f} {row['narration_ppl']:>14.4f} "
          f"{row['params']:>12,d} {row['kv_projection_params']:>12,d} "
          f"{row['kv_cache_bytes_per_token']:>12,d}")
''')


replace("1d6b31a0", r'''
# Q2.4: measured quality and cache trade-off plots.
plot_dir = q2.root / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
labels = [row["attention_type"].upper() for row in q2_results]
ppls = [row["held_out_ppl"] for row in q2_results]
caches = [row["kv_cache_bytes_per_token"] for row in q2_results]
colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2"]

fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(labels, ppls, color=colors)
ax.set_ylabel("Held-out perplexity (sentence mean)")
ax.set_title("Q2 attention variants: held-out quality")
if max(ppls) / max(min(ppls), 1e-12) > 20:
    ax.set_yscale("log")
for bar, value in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
quality_plot = plot_dir / "q2_quality_by_variant.png"
fig.savefig(quality_plot, dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5.2))
for label, cache, ppl, color in zip(labels, caches, ppls, colors):
    ax.scatter(cache, ppl, s=90, color=color)
    ax.annotate(label, (cache, ppl), xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("FP32 KV-cache bytes per token across all layers")
ax.set_ylabel("Held-out perplexity (sentence mean)")
ax.set_title("Q2 quality versus KV-cache footprint")
if max(ppls) / max(min(ppls), 1e-12) > 20:
    ax.set_yscale("log")
ax.grid(alpha=0.25)
fig.tight_layout()
cache_plot = plot_dir / "q2_quality_vs_cache.png"
fig.savefig(cache_plot, dpi=180, bbox_inches="tight")
plt.show()

best_quality = min(q2_results, key=lambda row: row["held_out_ppl"])
smallest_cache = min(row["kv_cache_bytes_per_token"] for row in q2_results)
small_cache_variants = [row["attention_type"] for row in q2_results
                        if row["kv_cache_bytes_per_token"] == smallest_cache]
q2_tradeoff = {
    "best_quality_variant": best_quality["attention_type"],
    "best_held_out_ppl": best_quality["held_out_ppl"],
    "smallest_cache_bytes_per_token": smallest_cache,
    "smallest_cache_variants": small_cache_variants,
    "measured_conclusion": (
        f"{best_quality['attention_type'].upper()} has the lowest measured held-out PPL "
        f"({best_quality['held_out_ppl']:.4f}); {', '.join(v.upper() for v in small_cache_variants)} "
        f"use the smallest measured cache ({smallest_cache} bytes/token)."
    ),
    "plots": [str(quality_plot), str(cache_plot)],
}
atomic_json(q2.root / "analysis" / "quality_cache_tradeoff.json", q2_tradeoff)
log_result("q2_quality_cache_tradeoff", q2_tradeoff)
print(q2_tradeoff["measured_conclusion"])
''')


replace("0b386613", r'''
# Q2.5: verify formula counts against the actual trained tensors, then quantify savings.
q2_accounting_rows = q2.parameter_accounting()
q2_accounting_by_name = {row["variant"]: row for row in q2_accounting_rows}
mha_accounting = q2_accounting_by_name["mha"]

for row in q2_accounting_rows:
    assert row["actual_kv_projection_params"] == row["formula_kv_projection_params"]
    print(
        f"{row['variant'].upper():<4}: total={row['total_params']:,}; "
        f"KV actual=formula={row['actual_kv_projection_params']:,}; "
        f"cache={row['kv_cache_bytes_per_token']:,} B/token; "
        f"KV-parameter fraction vs MHA={row['kv_param_fraction_vs_mha']:.3f}; "
        f"cache fraction vs MHA={row['cache_fraction_vs_mha']:.3f}"
    )

q2_savings = {
    label: {
        "kv_parameter_reduction_vs_mha": 1.0 - q2_accounting_by_name[label]["kv_param_fraction_vs_mha"],
        "cache_reduction_vs_mha": 1.0 - q2_accounting_by_name[label]["cache_fraction_vs_mha"],
    }
    for label in ("mqa", "mla")
}
atomic_json(q2.root / "analysis" / "mqa_mla_savings.json", q2_savings)
log_result("q2_parameter_cache_accounting", {
    "rows": q2_accounting_rows, "mqa_mla_savings": q2_savings,
})
''')


replace("6037632f", r'''
# Q2.6: the exact Q1 dialogue rule is reused; all conclusions come from persisted results.
mha_row = q2_results_by_name["mha"]
q2_interaction = {
    "dialogue_rule": Q2_DIALOGUE_RULE,
    "subset_sizes": {
        "dialogue": len(q2_dialogue_sentences),
        "narration": len(q2_narration_sentences),
    },
    "mqa_minus_mha": {
        "dialogue_ppl_gap": q2_results_by_name["mqa"]["dialogue_ppl"] - mha_row["dialogue_ppl"],
        "narration_ppl_gap": q2_results_by_name["mqa"]["narration_ppl"] - mha_row["narration_ppl"],
    },
    "mla_minus_mha": {
        "dialogue_ppl_gap": q2_results_by_name["mla"]["dialogue_ppl"] - mha_row["dialogue_ppl"],
        "narration_ppl_gap": q2_results_by_name["mla"]["narration_ppl"] - mha_row["narration_ppl"],
    },
}
for label, key in (("MQA", "mqa_minus_mha"), ("MLA", "mla_minus_mha")):
    dialogue_gap = q2_interaction[key]["dialogue_ppl_gap"]
    narration_gap = q2_interaction[key]["narration_ppl_gap"]
    if dialogue_gap > narration_gap:
        interpretation = "the measured compression penalty is larger on dialogue"
    elif dialogue_gap < narration_gap:
        interpretation = "the measured compression penalty is larger on narration"
    else:
        interpretation = "the measured dialogue and narration penalties are equal"
    q2_interaction[key]["measured_conclusion"] = interpretation
    print(f"{label} - MHA: dialogue gap={dialogue_gap:.4f}, narration gap={narration_gap:.4f}; {interpretation}.")

atomic_json(q2.root / "analysis" / "dialogue_interaction.json", q2_interaction)
log_result("q2_dialogue_interaction", q2_interaction)
''')


insert_after("6037632f", [
    markdown_cell("q2-q3-selection-heading", r'''
### Q2 model selection record for Question 3

Question 3 is allowed to select only among MHA, GQA, and MQA.  The record below applies
that rule to measured finite held-out perplexities and preserves the winning architecture,
checkpoint identity, source hashes, tokenizer identity, and split fingerprints.  MLA is
deliberately excluded even if it has the lowest Q2 perplexity.
'''),
    code_cell("q2-q3-selection", r'''
q2_q3_selection = q2.select_for_q3()
print(json.dumps(q2_q3_selection, indent=2))
if q2_q3_selection["provisional_smoke_only"]:
    print("SMOKE label: this selection is provisional and is isolated from artifacts/full/q2.")
else:
    log_result("q2_selection_for_q3", q2_q3_selection)
'''),
])


replace("62d69db9", r'''
# Q2.7: the official primary checkpoint is GQA (H=8, H_kv=2), independent of Q3's winner.
if Q2_SMOKE_TEST:
    raise RuntimeError(
        "Refusing to create an official submission from smoke metrics. Set Q2_SMOKE_TEST=False, "
        "restore/train all four full experiments, and rerun the Q2 analysis cells first."
    )

q2.require_all_complete()
official_rows = q2.results()
if [row["attention_type"] for row in official_rows] != ["mha", "gqa", "mqa", "mla"]:
    raise AssertionError("Q2 official evidence must contain exactly MHA/GQA/MQA/MLA in order")
metrics_q2_official = {"per_config_results": official_rows}
gqa_model = q2.load_final_model("gqa", device="cpu").eval()
gqa_config = q2.architecture("gqa")
assert gqa_config["n_heads"] == 8 and gqa_config["n_kv_heads"] == 2
assert gqa_config["attention_type"] == "standard"
assert gqa_config["pos_emb_type"] == "sinusoidal"

save_submission("submissions/q2", gqa_model, gqa_config, metrics_q2_official, VOCAB_SIZE)
submission_dir = Path("submissions/q2")

# save_submission copies active source files. Pin Q2 to its immutable snapshots and remove
# opportunistic later-question modules from this submission only (never from the workspace).
for _name in ("a1_tokenizer.py", "attention.py", "model_arch.py"):
    _snapshot = q2.root / "source_snapshots" / _name
    _target = submission_dir / _name
    shutil.copy2(_snapshot, _target)
    if _q2_sha256(_target) != _q2_sha256(_snapshot):
        raise RuntimeError(f"Q2 submission source-copy verification failed: {_name}")
for _stale in ("rope.py", "optim_muon.py", "model_builtin.py"):
    (submission_dir / _stale).unlink(missing_ok=True)

saved_config = json.loads((submission_dir / "config.json").read_text(encoding="utf-8"))
saved_metrics = json.loads((submission_dir / "metrics.json").read_text(encoding="utf-8"))
if saved_config["n_heads"] != 8 or saved_config["n_kv_heads"] != 2:
    raise AssertionError("Official Q2 config is not the required GQA checkpoint")
if saved_metrics != metrics_q2_official:
    raise AssertionError("Official Q2 metrics differ from the validated four-run evidence")
try:
    saved_state = torch.load(submission_dir / "model.pt", map_location="cpu", weights_only=True)
except TypeError:
    saved_state = torch.load(submission_dir / "model.pt", map_location="cpu")
if not isinstance(saved_state, dict) or not all(torch.is_tensor(value) for value in saved_state.values()):
    raise AssertionError("Official model.pt must be a plain state dictionary")

log_result("q2_submission", {
    "primary_variant": "gqa", "config": saved_config,
    "metrics": saved_metrics,
    "model_sha256": _q2_sha256(submission_dir / "model.pt"),
})
q2.export_recovery_zip(force=True)
print("Prepared official Q2 GQA submission with all four measured evidence rows.")
''')


insert_after("62d69db9", [
    markdown_cell("q2-final-validation-heading", r'''
### Q2 final validation and recovery export

Run this only after both **full** GPU pairs, Q2.3–Q2.6 analysis, Q3 selection, and the
official submission cell.  It runs the supplied Q2 autograder unchanged, saves its actual
output, and then creates one more validated recovery archive while preserving the previous
valid archive.
'''),
    code_cell("q2-final-validation", r'''
if Q2_SMOKE_TEST:
    raise RuntimeError("Final official validation requires Q2_SMOKE_TEST=False and full-run artifacts.")

q2.require_all_complete()
required_analysis = [
    q2.root / "analysis" / "per_config_results.json",
    q2.root / "analysis" / "quality_cache_tradeoff.json",
    q2.root / "analysis" / "parameter_accounting.json",
    q2.root / "analysis" / "dialogue_interaction.json",
    q2.root / "analysis" / "q3_selection.json",
    q2.root / "plots" / "q2_quality_by_variant.png",
    q2.root / "plots" / "q2_quality_vs_cache.png",
]
missing = [str(path) for path in required_analysis if not path.is_file()]
if missing:
    raise FileNotFoundError("Run all Q2 analysis/selection cells first; missing: " + str(missing))
required_submission = [Path("submissions/q2") / name for name in
                       ("attention.py", "model_arch.py", "model.pt", "config.json", "metrics.json")]
missing = [str(path) for path in required_submission if not path.is_file()]
if missing:
    raise FileNotFoundError("Run the Q2 submission cell first; missing: " + str(missing))

q2_autograder_report = q2.run_official_autograder(student_dir=".")
q2_archive_path = q2.export_recovery_zip(force=True)
q2_archive_manifest = q2._validate_archive(q2_archive_path)
print("Official autograder exit code:", q2_autograder_report["returncode"])
print("Final validated archive:", q2_archive_path.resolve())
print("Archived files:", len(q2_archive_manifest["files"]))
print("Q2 complete; Q1 was not rerun and Q3–Q5 remain untouched.")
'''),
])


# Hard preservation gates: abort before writing if any immutable cell moved or changed.
if nb["cells"][:q1_boundary] != q1_cells_before:
    raise RuntimeError("Q1/Section-0 preservation check failed; notebook will not be written")
if nb["cells"][index(q3_start_id):] != q3_cells_before:
    raise RuntimeError("Q3-Q5 preservation check failed; notebook will not be written")
cell_ids = [cell.get("id") for cell in nb["cells"]]
if any(cell_id is None for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
    raise RuntimeError("Notebook cell IDs are missing or duplicated")

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({
    "notebook": str(NOTEBOOK.resolve()),
    "cells": len(nb["cells"]),
    "q1_cells_preserved": q1_boundary,
    "q2_start_index_1_based": index("q2-intro-final") + 1,
    "q3_start_index_1_based": index(q3_start_id) + 1,
}, indent=2))
