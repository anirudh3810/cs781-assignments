"""
common.py — shared infrastructure for Assignment A-4 (Using LLMs).

Imported by the questions notebook, by every qN_*.py module the notebook writes, and by
every autograder in this package. Keeping it in one place guarantees that students, the
reference solution and the autograders split the IMDb data, compute metrics, save
checkpoints and reload predictors in EXACTLY the same way.

You should not need to modify this file. Read it: understanding the data contract,
the metric definitions and the resumable training harness is part of the assignment.

Contents
  0. Runtime configuration (configure)            4. Measurement helpers
  1. IMDb data contract (load_imdb, build_splits)  5. Submission artifacts + predict contract
  2. Metrics + classifier evaluation               6. Q3 helpers (local chat generator, call cache, GSM8K)
  3. Resumable training harness (train_updates)    7. Q1 helpers (restore the A-3 Q5 model/tokenizer)
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------
# 0. Runtime configuration
# ----------------------------------------------------------------------------------
LABELS = ["negative", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
COURSE_DATASET_ID = "Exploration-Lab/CS781"
IMDB_FILES = {"train": "Assignment4/train.parquet", "test": "Assignment4/test.parquet"}
IMDB_SIZES = {"train": 25000, "test": 12500}
VALIDATION_POOL_PER_CLASS = 1250
BACKBONE_ID = "openai-community/gpt2"
STUDENT_ID = "distilbert/distilgpt2"
QUESTIONS = ("q1", "q2", "q3", "q4", "q5", "q6")
QUESTION_MODULES = {"q1": "q1_decoding.py", "q2": "q2_classifier.py", "q3": "q3_prompting.py",
                    "q4": "q4_peft.py", "q5": "q5_qlora.py", "q6": "q6_distill.py"}

PROFILES = {
    "core": dict(train_per_class=448, val_per_class=56, test_per_class=112,
                 small_updates=150, q5_pilot=25, q5_total=100,
                 prompt_per_class=21, context_per_class=7, checkpoint_every=25),
    "low_resource": dict(train_per_class=224, val_per_class=56, test_per_class=112,
                         small_updates=80, q5_pilot=12, q5_total=50,
                         prompt_per_class=10, context_per_class=3, checkpoint_every=12),
    "smoke": dict(train_per_class=2, val_per_class=1, test_per_class=1,
                  small_updates=2, q5_pilot=1, q5_total=2,
                  prompt_per_class=1, context_per_class=1, checkpoint_every=1),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
elif DEVICE.type == "cuda":
    AMP_DTYPE = torch.float16        # T4: fp16 autocast + GradScaler
else:
    AMP_DTYPE = None                 # CPU: plain fp32

# Set by configure(); defaults let autograders import this module without a notebook.
SEED = 7814
PROFILE = "core"
CFG = dict(PROFILES["core"])
EFFECTIVE_BATCH = 16
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
WORK_DIR = Path("work") / "core"
SUBMISSION_DIR = Path("submissions")
PERSIST_DIR = None


def configure(profile="core", cfg=None, seed=7814, work_dir=None, submission_dir=None,
              persistent_root=None, effective_batch=16, teacher_id=None):
    """Set the runtime contract used by every helper below. Call once after the config cell."""
    global SEED, PROFILE, CFG, EFFECTIVE_BATCH, TEACHER_ID, WORK_DIR, SUBMISSION_DIR, PERSIST_DIR
    assert profile in PROFILES, profile
    PROFILE, SEED, EFFECTIVE_BATCH = profile, seed, effective_batch
    CFG = dict(cfg or PROFILES[profile])
    TEACHER_ID = teacher_id or ("Qwen/Qwen2.5-1.5B-Instruct" if profile == "core" else "Qwen/Qwen2.5-0.5B-Instruct")
    WORK_DIR = Path(work_dir) if work_dir else Path("work") / profile
    SUBMISSION_DIR = Path(submission_dir) if submission_dir else Path(
        "smoke_submissions" if profile == "smoke" else "submissions")
    PERSIST_DIR = Path(persistent_root) / profile if persistent_root else None
    for d in (WORK_DIR, SUBMISSION_DIR) + ((PERSIST_DIR,) if PERSIST_DIR else ()):
        d.mkdir(parents=True, exist_ok=True)
    reset_seed(seed)
    return {"profile": PROFILE, "cfg": CFG, "device": str(DEVICE), "amp_dtype": str(AMP_DTYPE),
            "work_dir": str(WORK_DIR), "submission_dir": str(SUBMISSION_DIR),
            "persist_dir": str(PERSIST_DIR) if PERSIST_DIR else None, "teacher_id": TEACHER_ID}


def reset_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_to_persistent(path):
    """Copy a finished file/directory to PERSIST_DIR (no-op when not configured)."""
    if PERSIST_DIR is None:
        return None
    path = Path(path)
    target = PERSIST_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    elif path.exists():
        shutil.copy2(path, target)
    return target


def restore_from_persistent(path):
    """Copy `path` back from PERSIST_DIR if it is missing locally. Returns True if it exists."""
    path = Path(path)
    if path.exists() or PERSIST_DIR is None or not (PERSIST_DIR / path).exists():
        return path.exists()
    source = PERSIST_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, path, dirs_exist_ok=True)
    else:
        shutil.copy2(source, path)
    return True


def log_result(key, value):
    """Persist headline numbers so a restarted runtime can rebuild summary tables."""
    path = WORK_DIR / "results_log.json"
    restore_from_persistent(path)
    log = json.loads(path.read_text()) if path.exists() else {}
    log[key] = value
    path.write_text(json.dumps(log, indent=2, default=_json_default))
    sync_to_persistent(path)
    print(f"[logged] {key}")


# ----------------------------------------------------------------------------------
# 1. IMDb data contract
# ----------------------------------------------------------------------------------
def load_imdb(local_dir="data/imdb"):
    """Return {"train": DataFrame, "test": DataFrame} with columns review_id, text, label.

    Uses files prepared by `python download_data.py` when present, otherwise downloads the
    two course parquet files from the Hugging Face hub (HF_TOKEN / cached login).
    """
    local_dir = Path(local_dir)
    frames = {}
    for split, remote in IMDB_FILES.items():
        path = local_dir / Path(remote).name
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=COURSE_DATASET_ID, repo_type="dataset", filename=remote))
        frame = pd.read_parquet(path)
        missing = {"review_id", "text", "label"} - set(frame.columns)
        assert not missing, f"{split}: missing columns {missing}"
        frame = frame[["review_id", "text", "label"]].copy()
        frame["label"] = frame["label"].astype(int)
        assert len(frame) == IMDB_SIZES[split], f"{split}: expected {IMDB_SIZES[split]} rows, got {len(frame)}"
        assert set(frame.label.unique()) == {0, 1}
        frames[split] = frame.reset_index(drop=True)
    return frames


def build_splits(raw, seed=None, cfg=None, validation_pool_per_class=VALIDATION_POOL_PER_CLASS):
    """Fixed working subsets (see notebook Section 0.1). Returns (splits, manifest).

    1. per class c: sort train rows by review_id, shuffle with Random(seed + c), first 1,250 ->
       validation pool, rest -> training pool;
    2. take the first train_per_class / val_per_class rows of each pool;
    3. test: sort by review_id per class, shuffle with Random(seed + 100 + c), first test_per_class;
    4. concatenate class 0 then class 1.
    """
    seed = SEED if seed is None else seed
    cfg = CFG if cfg is None else cfg
    train_df, test_df = _as_frame(raw["train"]), _as_frame(raw["test"])
    parts = {"train": [], "validation": [], "test": []}
    pool_sizes = {}
    for c in (0, 1):
        ids = sorted(train_df.loc[train_df.label == c, "review_id"].tolist())
        random.Random(seed + c).shuffle(ids)
        val_pool, train_pool = ids[:validation_pool_per_class], ids[validation_pool_per_class:]
        pool_sizes[c] = {"validation_pool": len(val_pool), "train_pool": len(train_pool)}
        parts["train"].append(train_pool[:cfg["train_per_class"]])
        parts["validation"].append(val_pool[:cfg["val_per_class"]])
        test_ids = sorted(test_df.loc[test_df.label == c, "review_id"].tolist())
        random.Random(seed + 100 + c).shuffle(test_ids)
        parts["test"].append(test_ids[:cfg["test_per_class"]])

    by_id = {"train": train_df.set_index("review_id"), "test": test_df.set_index("review_id")}
    splits = {}
    for name, id_lists in parts.items():
        source = by_id["test" if name == "test" else "train"]
        ids = id_lists[0] + id_lists[1]
        splits[name] = source.loc[ids].reset_index()[["review_id", "text", "label"]].reset_index(drop=True)

    sets = {k: set(v.review_id) for k, v in splits.items()}
    for k, v in splits.items():
        assert len(sets[k]) == len(v), f"duplicate review_id inside {k}"
    assert not (sets["train"] & sets["validation"]), "train/validation overlap"
    assert not ((sets["train"] | sets["validation"]) & sets["test"]), "train/test ID overlap"
    manifest = {
        "seed": seed, "profile": PROFILE, "cfg": cfg, "labels": LABELS, "dataset": COURSE_DATASET_ID,
        "files": IMDB_FILES, "validation_pool_per_class": validation_pool_per_class, "pool_sizes": pool_sizes,
        "ids": {k: v.review_id.tolist() for k, v in splits.items()},
        "class_counts": {k: {int(a): int(b) for a, b in v.label.value_counts().sort_index().items()}
                         for k, v in splits.items()},
    }
    return splits, manifest


def make_prompt_subsets(splits, cfg=None):
    """prompt_dev / prompt_test / context_test in fixed within-class manifest order."""
    cfg = CFG if cfg is None else cfg

    def first_per_class(frame, n):
        return pd.concat([frame[frame.label == c].head(n) for c in (0, 1)]).reset_index(drop=True)
    prompt_dev = first_per_class(splits["validation"], 2)
    prompt_test = first_per_class(splits["test"], cfg["prompt_per_class"])
    context_test = first_per_class(prompt_test, cfg["context_per_class"])
    return prompt_dev, prompt_test, context_test


def _as_frame(obj):
    if isinstance(obj, pd.DataFrame):
        return obj[["review_id", "text", "label"]].copy()
    return obj.to_pandas()[["review_id", "text", "label"]]       # datasets.Dataset


# ----------------------------------------------------------------------------------
# 2. Metrics + classifier evaluation
# ----------------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, probs=None):
    """The ONLY metric function for IMDb results. Invalid predictions (-1) count as errors."""
    from sklearn.metrics import confusion_matrix, f1_score
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape and set(np.unique(y_true)) <= {0, 1}
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, -1])[:2]
    out = {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(per_class)),
        "f1_negative": float(per_class[0]),
        "f1_positive": float(per_class[1]),
        "support": [int((y_true == 0).sum()), int((y_true == 1).sum())],
        "invalid_rate": float(np.mean(y_pred == -1)),
        "confusion[true][pred0,pred1,invalid]": cm.tolist(),
    }
    if probs is not None:
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
        assert p.shape == (len(y_true), 2), "probs must be [N, 2]"
        onehot = np.eye(2)[y_true]
        out["nll"] = float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))
        out["brier"] = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return out


def metrics_table(results):
    """{method_name: metrics_dict} -> tidy DataFrame for display."""
    cols = ["n", "accuracy", "macro_f1", "f1_negative", "f1_positive", "invalid_rate", "nll", "brier"]
    return pd.DataFrame({k: {c: v.get(c) for c in cols} for k, v in results.items()}).T


def _autocast():
    return torch.autocast(DEVICE.type, dtype=AMP_DTYPE) if AMP_DTYPE else contextlib.nullcontext()


@torch.no_grad()
def evaluate_classifier(model, frame, collate, batch_size=16):
    """ids, labels, preds, probs, logits and metrics for a [B,2]-logit classifier."""
    model.eval()
    device = next(model.parameters()).device
    logits_all = []
    for start in range(0, len(frame), batch_size):
        batch = collate(list(range(start, min(start + batch_size, len(frame)))), frame)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with (_autocast() if device.type == "cuda" else contextlib.nullcontext()):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits_all.append(logits.float().cpu())
    logits = torch.cat(logits_all)
    probs = logits.softmax(-1).numpy()
    preds = probs.argmax(-1)
    labels = frame.label.to_numpy()
    return {"ids": frame.review_id.tolist(), "labels": labels, "preds": preds, "probs": probs,
            "logits": logits, "metrics": classification_metrics(labels, preds, probs)}


def make_classifier_collate(tokenizer, max_length=128):
    """Right-padded batches. GPT-2 has no PAD token: EOS is reused, so locate padding with
    attention_mask, never by token value."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def collate(indices, frame):
        rows = frame.iloc[list(indices)]
        enc = tokenizer(rows.text.tolist(), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(rows.label.to_numpy(), dtype=torch.long),
                "ids": rows.review_id.tolist()}
    return collate


class ClassifierPredictor:
    """Wraps a [B,2]-logit classifier + collate into the hidden-set predict(texts) contract."""

    def __init__(self, model, collate, batch_size=16):
        self.model, self.collate, self.batch_size = model.eval(), collate, batch_size

    def predict(self, texts):
        frame = pd.DataFrame({"review_id": range(len(texts)), "text": list(texts), "label": 0})
        out = evaluate_classifier(self.model, frame, self.collate, self.batch_size)
        return out["preds"].astype(int), out["probs"]


# ----------------------------------------------------------------------------------
# 3. Resumable training harness
# ----------------------------------------------------------------------------------
class BatchStream:
    """Deterministic infinite stream of example indices, reshuffled every epoch."""

    def __init__(self, num_examples, seed=None):
        self.n, self.seed, self.epoch, self.cursor = num_examples, SEED if seed is None else seed, 0, 0
        self._order = self._make_order()

    def _make_order(self):
        order = list(range(self.n))
        random.Random(self.seed * 1000 + self.epoch).shuffle(order)
        return order

    def next(self, size):
        out = []
        while len(out) < size:
            if self.cursor >= self.n:
                self.epoch, self.cursor = self.epoch + 1, 0
                self._order = self._make_order()
            out.append(self._order[self.cursor])
            self.cursor += 1
        return out

    def state_dict(self):
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        assert state["n"] == self.n, "stream size changed between runs"
        self.seed, self.epoch, self.cursor = state["seed"], state["epoch"], state["cursor"]
        self._order = self._make_order()


def get_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state_dict(model):
    """Only parameters being trained (PEFT modules, soft prompts, heads, ...)."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def _atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    sync_to_persistent(path)


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, run_state=None):
    _atomic_save({"model": trainable_state_dict(model),
                  "optimizer": optimizer.state_dict() if optimizer else None,
                  "scheduler": scheduler.state_dict() if scheduler else None,
                  "scaler": scaler.state_dict() if scaler else None,
                  "run_state": run_state or {}, "rng": get_rng_state()}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True):
    restore_from_persistent(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not (trainable & set(missing)), "checkpoint is missing trainable parameters"
    if optimizer is not None and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        set_rng_state(ckpt["rng"])
    return ckpt["run_state"]


def train_updates(model, train_frame, collate, loss_fn, optimizer, *, max_updates, run_dir,
                  eval_fn=None, scheduler=None, micro_batch=4, effective_batch=None,
                  eval_every=None, clip_norm=1.0, select_metric="macro_f1", config=None,
                  stop_after=None, log_every=10):
    """Train for max_updates optimizer steps; resume from run_dir/last.pt if it exists.

    loss_fn(model, batch) -> scalar loss;  eval_fn(model) -> metrics dict containing select_metric.
    Writes last.pt (full resume state) at every validation boundary and best.pt (trainable
    weights) whenever validation improves (ties keep the earlier step). stop_after simulates
    an interruption after that many updates.
    """
    effective_batch = effective_batch or EFFECTIVE_BATCH
    assert effective_batch % micro_batch == 0
    accum = effective_batch // micro_batch
    eval_every = eval_every or CFG["checkpoint_every"]
    run_dir = Path(run_dir)
    device = next(p for p in model.parameters()).device
    use_scaler = device.type == "cuda" and AMP_DTYPE == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    stream = BatchStream(len(train_frame))
    state = {"update": 0, "history": [], "best_score": -float("inf"), "best_update": None,
             "examples_seen": 0, "batch_ids": [], "config": config or {}, "train_seconds": 0.0}
    restore_from_persistent(run_dir / "last.pt")
    if (run_dir / "last.pt").exists():
        state = load_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler)
        stream.load_state_dict(state["stream"])
        print(f"[resume] {run_dir.name}: continuing from update {state['update']}")

    model.train()
    target = min(max_updates, stop_after) if stop_after else max_updates
    while state["update"] < target:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss, step_ids = 0.0, []
        for _ in range(accum):
            idx = stream.next(micro_batch)
            batch = collate(idx, train_frame)
            step_ids += batch.get("ids", idx)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with (_autocast() if device.type == "cuda" else contextlib.nullcontext()):
                loss = loss_fn(model, batch) / accum
            (scaler.scale(loss) if scaler else loss).backward()
            step_loss += loss.item()
        params = [p for p in model.parameters() if p.requires_grad]
        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_norm) if clip_norm else None
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        state["train_seconds"] += time.perf_counter() - t0
        state["update"] += 1
        state["examples_seen"] += effective_batch
        state["batch_ids"].append(step_ids)
        record = {"update": state["update"], "train_loss": step_loss,
                  "grad_norm": float(grad_norm) if grad_norm is not None else None,
                  "lr": optimizer.param_groups[0]["lr"]}
        boundary = state["update"] % eval_every == 0 or state["update"] == max_updates
        if boundary and eval_fn is not None:
            val = eval_fn(model)
            model.train()
            record.update({f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float))})
            score = val[select_metric]
            if score > state["best_score"]:
                state["best_score"], state["best_update"] = score, state["update"]
                _atomic_save({"model": trainable_state_dict(model), "update": state["update"],
                              "score": score, "config": state["config"]}, run_dir / "best.pt")
        state["history"].append(record)
        if state["update"] % log_every == 0 or boundary:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()})
        if boundary or state["update"] == target:
            state["stream"] = stream.state_dict()
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, state)
    return state


def load_best(model, run_dir):
    """Load best.pt trainable weights into a freshly constructed model of the same layout."""
    restore_from_persistent(Path(run_dir) / "best.pt")
    best = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=False)
    return best


def plot_history(histories, keys=("train_loss", "val_macro_f1"), title=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5))
    for ax, key in zip(np.atleast_1d(axes), keys):
        for name, hist in histories.items():
            pts = [(h["update"], h[key]) for h in hist if h.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o" if key.startswith("val") else None, ms=3, label=name)
        ax.set_xlabel("optimizer update"); ax.set_ylabel(key); ax.set_title(f"{title} {key}"); ax.legend()
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------
# 4. Measurement helpers
# ----------------------------------------------------------------------------------
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median_latency(fn, repeats=3, warmup=1):
    """Median wall time (s) of fn() after warm-up, with CUDA synchronisation."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        cuda_sync(); t0 = time.perf_counter()
        fn()
        cuda_sync(); times.append(time.perf_counter() - t0)
    return float(np.median(times))


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_report():
    if not torch.cuda.is_available():
        return {"allocated_mb": None, "peak_allocated_mb": None, "reserved_mb": None, "peak_reserved_mb": None}
    mib = 2 ** 20
    return {"allocated_mb": torch.cuda.memory_allocated() / mib,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / mib,
            "reserved_mb": torch.cuda.memory_reserved() / mib,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / mib}


def count_parameters(model):
    """(trainable, total). For 4-bit bitsandbytes layers numel() counts packed storage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def state_bytes(state_dict):
    return int(sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t)))


# ----------------------------------------------------------------------------------
# 5. Submission artifacts + the hidden-set predict contract
# ----------------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def reload_check_record(frame, probs_by_run, n=3):
    """Metrics-json entry used by the autograders to verify reloads:
    the first n review texts and the probabilities each saved run produced for them."""
    return {"review_ids": frame.review_id.head(n).tolist(), "texts": frame.text.head(n).tolist(),
            "probs": {run: np.asarray(p)[:n].tolist() for run, p in probs_by_run.items()}}


def save_submission(question, metrics, states=None, extra_sources=()):
    """Write submissions/<question>/{metrics.json, weights.pt, <module>.py}.

    states: {run_name: {"state": state_dict, "config": {...}}, ...} (None for Q1/Q3).
    The question's module file (e.g. q2_classifier.py) and common.py are copied next to the
    weights so each checkpoint stays paired with the exact source that created it.
    """
    assert question in QUESTIONS, question
    out = SUBMISSION_DIR / question
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    if states is not None:
        torch.save(states, out / "weights.pt")
    copied = []
    for name in (QUESTION_MODULES[question], "common.py", *extra_sources):
        if Path(name).is_file():
            shutil.copy2(name, out / Path(name).name)
            copied.append(Path(name).name)
    sync_to_persistent(out)
    size = (out / "weights.pt").stat().st_size / 2 ** 20 if states is not None else 0
    print(f"Saved {out}/ (metrics.json{', weights.pt %.1f MiB' % size if states is not None else ''}; "
          f"sources: {', '.join(copied)})")
    return out


def load_weights(question, root=None):
    path = Path(root or SUBMISSION_DIR) / question / "weights.pt"
    restore_from_persistent(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def check_predictor(predictor, texts, stored_probs, atol=1e-3):
    """Compare a freshly reloaded predictor with probabilities stored during evaluation."""
    labels, probs = predictor.predict(list(texts))
    stored = np.asarray(stored_probs)[:len(texts)]
    ok = np.allclose(probs, stored, atol=atol) and np.array_equal(np.asarray(labels), stored.argmax(-1))
    print("reload check:", "PASS" if ok else "FAIL", "| max |Δp| =", float(np.abs(probs - stored).max()))
    return ok


# ----------------------------------------------------------------------------------
# 6. Q3 helpers: local chat generation, resumable call cache, GSM8K subset
# ----------------------------------------------------------------------------------
class LocalChatGenerator:
    """Loads a chat model once; renders messages with its chat template; greedy when
    temperature == 0. The model's default sampling extras (top_k, repetition_penalty)
    are switched OFF so only the settings you pass are active."""

    def __init__(self, model_id=None, revision=None, dtype=torch.float16, model=None, tokenizer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id or TEACHER_ID
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.model_id, revision=revision)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=revision,
                torch_dtype=dtype if DEVICE.type == "cuda" else torch.float32,
                device_map={"": 0} if DEVICE.type == "cuda" else None)
        self.model = model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def __call__(self, messages, max_new_tokens=16, temperature=0.0, top_p=1.0, seed=None, n=1):
        prompt = self.render(messages)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = enc["input_ids"].shape[1]
        sampling = temperature > 0
        if n > 1 and not sampling:
            raise ValueError("n > 1 requires temperature > 0")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sampling, num_return_sequences=n,
                      top_k=None, repetition_penalty=1.0,
                      temperature=temperature if sampling else None, top_p=top_p if sampling else None,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        cuda_sync(); t0 = time.perf_counter()
        out = self.model.generate(**enc, **kwargs)
        cuda_sync(); latency = time.perf_counter() - t0
        records = []
        for row in out[:, input_len:]:
            ids = row.tolist()
            ended = [i for i, t in enumerate(ids) if t in self.eos_ids]
            ids_trim = ids[:ended[0] + 1] if ended else ids
            records.append({"text": self.tokenizer.decode(ids_trim, skip_special_tokens=True),
                            "input_tokens": input_len, "output_tokens": len(ids_trim),
                            "finish_reason": "eos" if ended else "length",
                            "hit_token_cap": not ended and len(ids_trim) >= max_new_tokens,
                            "latency_s": latency / n, "batch_latency_s": latency})
        return records[0] if n == 1 else records


class CallCache:
    """Append-only JSONL cache keyed by everything that affects the output."""

    def __init__(self, path):
        self.path = Path(path)
        restore_from_persistent(self.path)
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                self.records[rec["key"]] = rec

    @staticmethod
    def make_key(**fields):
        return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:24]

    def call(self, generator, messages, *, method, example_id, stage="final", **settings):
        key = self.make_key(model=generator.model_id, revision=generator.revision, messages=messages,
                            method=method, example_id=example_id, stage=stage, settings=settings)
        if key in self.records:
            return self.records[key]
        result = generator(messages, **settings)
        rec = {"key": key, "method": method, "example_id": example_id, "stage": stage,
               "messages": messages, "settings": settings, "model": generator.model_id,
               "revision": generator.revision, "result": result}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.records[key] = rec
        return rec

    def sync(self):
        sync_to_persistent(self.path)


def load_reasoning_set(n_total=40, n_dev=5, seed=None):
    """Fixed GSM8K test subset -> (reason_dev, reason_test) with float gold answers."""
    from datasets import load_dataset
    seed = SEED if seed is None else seed
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if PROFILE == "smoke":
        n_total, n_dev = 4, 2
    idx = sorted(random.Random(seed).sample(range(len(ds)), n_total))
    rows = [{"qid": f"gsm8k-test-{i}", "question": ds[i]["question"],
             "gold": float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))} for i in idx]
    frame = pd.DataFrame(rows)
    return frame.iloc[:n_dev].reset_index(drop=True), frame.iloc[n_dev:].reset_index(drop=True)


def extract_final_number(text):
    """Strict rule: the LAST line starting with 'ANSWER:' must contain exactly one number."""
    lines = [l.strip() for l in str(text).strip().splitlines() if l.strip().upper().startswith("ANSWER:")]
    if not lines:
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", lines[-1].split(":", 1)[1])
    if len(nums) != 1:
        return None
    try:
        return float(nums[0].replace(",", "").rstrip("."))
    except ValueError:
        return None


def run_python_snippet(code, timeout_s=5):
    """Run model-written arithmetic code (Program-of-Thought) in an isolated subprocess.
    Imports, file access and dunder names are rejected. Returns (stdout, error)."""
    if re.search(r"^\s*(import|from)\s", code, flags=re.M) or "open(" in code or "__" in code:
        return None, "rejected: imports / file access / dunder are not allowed"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code)
        try:
            proc = subprocess.run([sys.executable, "-I", "-S", str(script)], cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout_s, env={})
        except subprocess.TimeoutExpired:
            return None, "timeout"
    return proc.stdout.strip(), (proc.stderr.strip() or None)


# ----------------------------------------------------------------------------------
# 7. Q1 helpers: restore the A-3 Q5 model and its exact tokenizer
# ----------------------------------------------------------------------------------
def import_a3_common(a3_dir):
    """Import the A-3 package's common.py under the name `a3_common` (no clash with this file)."""
    path = Path(a3_dir) / "common.py"
    if not path.is_file():
        raise FileNotFoundError(f"A-3 common.py not found at {path}")
    spec = importlib.util.spec_from_file_location("a3_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["a3_common"] = module
    spec.loader.exec_module(module)
    reset_seed()                     # A-3 common seeds its own RNGs on import
    return module


def prepare_a3_corpus(a3_dir):
    """Recreate the A-3 corpus, held-out sentences (A-3 defaults) and training text."""
    a3_dir = Path(a3_dir).resolve()
    raw = a3_dir / "data" / "raw"
    if len(list(raw.glob("book*.txt"))) != 7:
        subprocess.run([sys.executable, "download_data.py", "--raw-dir", str(raw)], cwd=a3_dir, check=True)
    a3c = import_a3_common(a3_dir)
    corpus = a3c.load_corpus(str(raw))
    held_out = a3c.held_out_sentences(corpus, n_chapters=5, sentences_per_chapter=6)
    train_text = a3c.remove_sentences_from_text(corpus.raw_text, held_out)
    return corpus, held_out, train_text, a3c


def restore_a3_model(q5_dir):
    """Rebuild BuiltinDecoderLM from submissions/q5 (model_builtin.py snapshot + config.json + model.pt)."""
    q5_dir = Path(q5_dir)
    if str(q5_dir) not in sys.path:
        sys.path.insert(0, str(q5_dir))
    for name in ("model_builtin", "model_arch", "attention", "rope"):
        sys.modules.pop(name, None)
    mb = importlib.import_module("model_builtin")
    cfg_raw = json.loads((q5_dir / "config.json").read_text())
    fields = mb.BuiltinLMConfig.__dataclass_fields__
    cfg = mb.BuiltinLMConfig(**{k: v for k, v in cfg_raw.items() if k in fields})
    model = mb.BuiltinDecoderLM(cfg)
    state = torch.load(q5_dir / "model.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval(), cfg, cfg_raw


def restore_a3_tokenizer(a3_dir, train_text, vocab_size, a3c=None):
    """Re-create the A-3 tokenizer exactly as A-3 did: retrain your A-1 BPE hook
    (a1_tokenizer.train_tokenizer) on the same training text and vocabulary size, falling
    back to the course BPE when the hook is unavailable — the same rule A-3's grader used."""
    a3c = a3c or import_a3_common(a3_dir)
    module_path = Path(a3_dir) / "a1_tokenizer.py"
    if not module_path.is_file():
        module_path = Path(a3_dir) / "submissions" / "q1" / "a1_tokenizer.py"
    tok, used_student, diagnostics = a3c.load_a1_tokenizer_or_fallback(
        train_text, vocab_size=vocab_size, module_path=str(module_path), verbose=True)
    return tok, used_student, diagnostics
