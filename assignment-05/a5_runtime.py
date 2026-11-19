"""Persistent experiment state for all six CS781 A5 questions.

This module is written from a helper cell in the notebook on Run All.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch


AUTO_RESUME = True
SKIP_COMPLETED = True
CHECKPOINT_EVERY_UPDATES = 100
CHECKPOINT_EVERY_SECONDS = 900
KEEP_LAST_CHECKPOINTS = 2
SAVE_BEST_CHECKPOINT = True
RUN_SMOKE_TESTS = True
RUN_FULL_EXPERIMENTS = True


def artifact_root() -> Path:
    configured = os.environ.get("A5_ARTIFACT_ROOT")
    in_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    if configured:
        root = Path(configured).expanduser().resolve()
        if (in_kaggle and str(root).startswith("/kaggle/working") and
                os.environ.get("A5_KAGGLE_FILE_PERSISTENCE") != "1"):
            raise RuntimeError("Enable Kaggle Session persistence: Files and set A5_KAGGLE_FILE_PERSISTENCE=1, or use a persistent mounted root.")
        if in_colab and (str(root) == "/content" or
                         str(root).startswith("/content/sample_data")):
            raise RuntimeError("A5_ARTIFACT_ROOT points to temporary Colab storage")
    else:
        if in_kaggle or in_colab:
            raise RuntimeError("Set A5_ARTIFACT_ROOT to a persistent mounted directory; /kaggle/working and /content are temporary.")
        root = (Path.cwd() / "assignment5_artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    for category in ("checkpoints", "experiment_states", "training_histories", "figures", "metrics", "logs", "caches", "reports"):
        for q in ("q1", "q2", "q3", "q4", "q5", "q6"):
            (root / category / q).mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    (root / "caches").mkdir(exist_ok=True)
    return root


ROOT = artifact_root()


def atomic_replace(source, destination):
    """Retry a rename briefly when Windows sync software holds a file open."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 0.5))


def cuda_info() -> list[dict]:
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"index": i, "name": torch.cuda.get_device_name(i),
                        "vram_gb": round(p.total_memory / 2**30, 2),
                        "bf16": bool(torch.cuda.is_bf16_supported())})
    return out


def hash_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def atomic_torch_save(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        torch.load(tmp, map_location="cpu", weights_only=False)
        if path.exists():
            try:
                previous_payload = torch.load(path, map_location="cpu", weights_only=False)
                if (isinstance(previous_payload, dict) and
                        previous_payload.get("fingerprint") == payload.get("fingerprint")):
                    shutil.copy2(path, path.with_suffix(path.suffix + ".prev"))
            except (OSError, ValueError, RuntimeError, EOFError):
                pass  # retain the last valid recovery checkpoint
        atomic_replace(tmp, path)
        torch.load(path, map_location="cpu", weights_only=False)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json_save(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".",
                                     suffix=".tmp", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        json.load(open(tmp, encoding="utf-8"))
        atomic_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            warnings.warn("CUDA device count changed; restoring RNG for matching devices only")
        for gpu, gpu_state in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(gpu_state.cpu(), gpu)


def state_path(question: str, exp_id: str) -> Path:
    return ROOT / "experiment_states" / question / f"{exp_id}.pt"


def checkpoint_path(question: str, exp_id: str) -> Path:
    return ROOT / "checkpoints" / question / f"{exp_id}.pt"


def manifest_path(question: str, exp_id: str) -> Path:
    return ROOT / "manifests" / f"{question}_{exp_id}.json"


def load_completed(question: str, exp_id: str, fp: str, required=()):
    if not SKIP_COMPLETED:
        return None
    manifest = manifest_path(question, exp_id)
    state = state_path(question, exp_id)
    if not manifest.exists() or not state.exists():
        return None
    try:
        meta = json.load(open(manifest, encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("fingerprint") != fp:
            return None
        if not all(Path(p).exists() for p in required):
            return None
        recorded_hashes = meta.get("required_sha256", {})
        if any(recorded_hashes.get(str(Path(p))) != hash_file(p) for p in required):
            return None
        result = torch.load(state, map_location="cpu", weights_only=False)
        if result.get("fingerprint") != fp:
            return None
        return result
    except (OSError, ValueError, RuntimeError, EOFError, KeyError):
        return None


def mark_complete(question: str, exp_id: str, fp: str, result: dict, config: dict, required=()):
    result = dict(result)
    result["fingerprint"] = fp
    state = state_path(question, exp_id)
    atomic_torch_save(result, state)
    required = [str(Path(p)) for p in required]
    if not all(Path(p).exists() for p in required):
        raise FileNotFoundError(f"Cannot complete {exp_id}; missing: {[p for p in required if not Path(p).exists()]}")
    atomic_json_save({"status": "complete", "experiment_id": exp_id, "fingerprint": fp,
                      "config": config, "state": str(state), "required": required,
                      "required_sha256": {p: hash_file(p) for p in required}},
                     manifest_path(question, exp_id))


def load_checkpoint_compatible(path, fp):
    for candidate in (Path(path), Path(str(path) + ".prev"), Path(str(path) + ".previous")):
        if not candidate.exists():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            if payload.get("fingerprint") == fp:
                return payload
        except (OSError, ValueError, RuntimeError, EOFError):
            continue
    return None


def prepare_checkpoint(path, fp):
    """Retain incompatible partial states in an archive, then start this config clean."""
    path = Path(path)
    candidates = [p for p in (path, Path(str(path) + ".prev"),
                              Path(str(path) + ".previous")) if p.exists()]
    if not candidates or load_checkpoint_compatible(path, fp) is not None:
        return
    archive = path.parent / "archive"
    archive.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    for index, candidate in enumerate(candidates):
        candidate.rename(archive / f"{candidate.name}.{stamp}.{index}.incompatible")


def strict_reload(path, architecture_module):
    import common_a5 as C
    model, extra = C.load_decoder_model(str(path), architecture_module)
    return model, extra
