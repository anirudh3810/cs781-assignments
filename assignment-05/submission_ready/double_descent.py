"""Question 2: random features and DecoderLM width experiments.

The four assignment functions retain their original positional signatures. Optional
keyword arguments provide persistent progress for the notebook's long sweeps.
"""

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def one_hot_contexts(ids, context: int, n: int, vocab_size: int, rng: np.random.RandomState):
    """GIVEN. Sample n positions from a 1-D id array."""
    ids = np.asarray(ids)
    starts = rng.choice(len(ids) - context - 1, n, replace=False)
    X = np.zeros((n, context * vocab_size), dtype=np.float32)
    for j in range(context):
        X[np.arange(n), j * vocab_size + ids[starts + j]] = 1.0
    y = ids[starts + context]
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))


def random_feature_experiment(train_ids, test_ids, vocab_size, widths, n_train=200,
                              n_test=2000, context=3, seed=0, *, cached_results=None,
                              on_width_complete=None) -> dict:
    """Fit minimum-norm random-feature regression for every width.

    Cached entries may skip the expensive solve, but the random matrix is still
    drawn for each width so later widths use the assignment's exact RNG stream.
    """
    rng = np.random.RandomState(seed)
    Xtr, ytr = one_hot_contexts(train_ids, context, n_train, vocab_size, rng)
    Xte, yte = one_hot_contexts(test_ids, context, n_test, vocab_size, rng)
    Ytr = F.one_hot(ytr, num_classes=vocab_size).to(torch.float32)
    Yte = F.one_hot(yte, num_classes=vocab_size).to(torch.float32)
    n_unique = int(torch.unique(Xtr, dim=0).shape[0])
    result = {"widths": [], "train_mse": [], "test_mse": [], "test_acc": [],
              "n_train": int(n_train), "n_unique": n_unique, "beta_norm": []}
    cached_results = cached_results or {}

    for P in widths:
        P = int(P)
        W = torch.from_numpy(rng.randn(context * vocab_size, P) / math.sqrt(context))
        row = cached_results.get(P, cached_results.get(str(P)))
        if row is None or not all(k in row for k in ("train_mse", "test_mse", "test_acc", "beta_norm")):
            # Double precision near the interpolation peak avoids changing the
            # numerical rank of the feature matrix through float32 roundoff.
            train_features = torch.relu(Xtr.to(torch.float64) @ W)
            test_features = torch.relu(Xte.to(torch.float64) @ W)
            beta = torch.linalg.pinv(train_features) @ Ytr.to(torch.float64)
            train_scores = train_features @ beta
            test_scores = test_features @ beta
            row = {
                "width": P,
                "train_mse": float((train_scores - Ytr).square().sum(dim=1).mean()),
                "test_mse": float((test_scores - Yte).square().sum(dim=1).mean()),
                "test_acc": float((test_scores.argmax(dim=1) == yte).float().mean()),
                "beta_norm": float(torch.linalg.vector_norm(beta)),
            }
            if on_width_complete is not None:
                on_width_complete(P, row)
        result["widths"].append(P)
        for key in ("train_mse", "test_mse", "test_acc", "beta_norm"):
            result[key].append(float(row[key]))
    return result


def interpolation_threshold(result: dict) -> int:
    """Width with the largest observed held-out MSE, as the assignment specifies."""
    if len(result["widths"]) != len(result["test_mse"]) or not result["widths"]:
        raise ValueError("widths and test_mse must be nonempty and have equal length")
    return int(result["widths"][int(np.argmax(result["test_mse"]))])


def add_label_noise(targets: torch.Tensor, vocab_size: int, frac: float,
                    generator: torch.Generator) -> torch.Tensor:
    """Copy targets and independently replace selected labels by uniform IDs."""
    if not 0 <= frac <= 1:
        raise ValueError("frac must be between zero and one")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    # The notebook supplies a CPU generator. Draw there even if targets are on
    # a GPU so a saved seed has the same meaning on CPU and CUDA.
    mask = torch.rand(targets.shape, generator=generator) < frac
    replacement = torch.randint(0, vocab_size, targets.shape, generator=generator)
    result = targets.clone()
    result[mask.to(targets.device)] = replacement[mask].to(targets.device, dtype=targets.dtype)
    return result


def _tensor_identity(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        for gpu, gpu_state in enumerate(state["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(gpu_state.cpu(), gpu)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists():
            # Preserve only a verified recovery point. A damaged primary must
            # never replace a valid previous checkpoint.
            previous = path.with_name(path.name + ".previous")
            try:
                old = _torch_load(path)
                if all(key in old for key in ("fingerprint", "next_step", "model", "optimizer", "rng")):
                    shutil.copy2(path, previous)
            except Exception:
                pass
        os.replace(temporary, path)
        restored = _torch_load(path)
        if restored.get("fingerprint") != payload.get("fingerprint") or restored.get("next_step") != payload.get("next_step"):
            raise IOError(f"checkpoint verification failed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _load_recovery_checkpoint(path, expected_fingerprint):
    previous = path.with_name(path.name + ".previous")
    candidates = [candidate for candidate in (path, previous) if candidate.exists()]
    problems = []
    for candidate in candidates:
        try:
            saved = _torch_load(candidate)
            if saved.get("fingerprint") != expected_fingerprint:
                problems.append(f"{candidate}: configuration fingerprint mismatch")
                continue
            for key in ("next_step", "model", "optimizer", "history", "rng"):
                if key not in saved:
                    raise ValueError(f"missing {key}")
            return saved
        except Exception as exc:
            problems.append(f"{candidate}: {exc}")
    if candidates:
        raise ValueError("no compatible checkpoint; " + "; ".join(problems))
    return None


def train_one_width(d_model, Xtr, Ytr_noisy, Xte, vocab_size, model_arch, steps=3000,
                    lr=3e-3, n_layers=1, n_heads=4, device="cpu", seed=0, *,
                    checkpoint_path=None, checkpoint_every=100,
                    checkpoint_every_seconds=900, log_every=100, resume=True,
                    return_details=False, fingerprint=None, save_best_checkpoint=True):
    """Train one width with exact Adam continuation when a checkpoint is supplied."""
    if tuple(Ytr_noisy.shape) != (Xtr.shape[0], Xtr.shape[1] - 1):
        raise ValueError("Ytr_noisy must match Xtr[:, 1:] in shape")
    config = {"d_model": int(d_model), "vocab_size": int(vocab_size),
              "n_layers": int(n_layers), "n_heads": int(n_heads),
              "n_kv_heads": int(n_heads), "d_ff": int(4 * d_model),
              "max_seq_len": int(Xtr.shape[1] - 1), "dropout": 0.0,
              "steps": int(steps), "lr": float(lr), "seed": int(seed),
              "data_sha256": _tensor_identity(Xtr, Ytr_noisy, Xte),
              "architecture": getattr(model_arch, "__name__", type(model_arch).__name__),
              "external_fingerprint": fingerprint}
    expected_fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    torch.manual_seed(seed)
    cfg = model_arch.LMConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                              n_heads=n_heads, n_kv_heads=n_heads, d_ff=4*d_model,
                              max_seq_len=Xtr.shape[1] - 1, dropout=0.0)
    model = model_arch.DecoderLM(cfg)
    with torch.no_grad():
        model.tok_emb.weight.mul_(0.1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_dev, Ytr_dev, Xte_dev = Xtr.to(device), Ytr_noisy.to(device), Xte.to(device)
    history = {"step": [], "train_acc_noisy": [], "test_err": [], "test_loss": [], "train_loss": []}
    next_step, elapsed_before, best_test_err = 0, 0.0, float("inf")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint_path is not None:
        saved = _load_recovery_checkpoint(checkpoint_path, expected_fingerprint)
    else:
        saved = None
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        next_step = int(saved["next_step"])
        history = saved["history"]
        elapsed_before = float(saved.get("elapsed_seconds", 0.0))
        best_test_err = float(saved.get("best_test_err", float("inf")))
        _restore_rng(saved["rng"])

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.monotonic()
    last_saved = started

    def measure(step, train_loss=None):
        nonlocal best_test_err
        model.eval()
        with torch.inference_mode():
            tr_logits = model(Xtr_dev[:, :-1])
            te_logits = model(Xte_dev[:, :-1])
            tr_acc = float((tr_logits.argmax(-1) == Ytr_dev).float().mean())
            te_target = Xte_dev[:, 1:]
            test_err = float((te_logits.argmax(-1) != te_target).float().mean())
            test_loss = float(F.cross_entropy(te_logits.reshape(-1, te_logits.shape[-1]),
                                              te_target.reshape(-1)))
            if train_loss is None:
                train_loss = float(F.cross_entropy(tr_logits.reshape(-1, tr_logits.shape[-1]),
                                                    Ytr_dev.reshape(-1)))
        model.train()
        if not history["step"] or history["step"][-1] != step:
            for key, value in (("step", int(step)), ("train_acc_noisy", tr_acc),
                               ("test_err", test_err), ("test_loss", test_loss),
                               ("train_loss", float(train_loss))):
                history[key].append(value)
        improved = test_err < best_test_err
        best_test_err = min(best_test_err, test_err)
        if improved and checkpoint_path is not None and save_best_checkpoint:
            save_state(step, checkpoint_path.with_name(checkpoint_path.name + ".best"))

    def save_state(step, destination=None):
        if checkpoint_path is None:
            return
        payload = {"fingerprint": expected_fingerprint, "config": config, "next_step": int(step),
                   "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   "optimizer": opt.state_dict(), "history": history,
                   "rng": _capture_rng(), "elapsed_seconds": elapsed_before + time.monotonic() - started,
                   "best_test_err": best_test_err}
        _atomic_torch_save(payload, destination or checkpoint_path)

    completed_updates = next_step
    try:
        for step in range(next_step, steps):
            model.train()
            logits = model(Xtr_dev[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Ytr_dev.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            update = step + 1
            completed_updates = update
            if update % log_every == 0 or update == steps:
                measure(update, float(loss.detach()))
            now = time.monotonic()
            if checkpoint_path is not None and ((checkpoint_every and update % checkpoint_every == 0)
                                               or now - last_saved >= checkpoint_every_seconds
                                               or update == steps):
                save_state(update)
                last_saved = now
    except BaseException:
        save_state(completed_updates)
        raise

    if not history["step"] or history["step"][-1] != steps:
        measure(steps)
    if checkpoint_path is not None:
        try:
            primary = _torch_load(checkpoint_path)
            primary_complete = (primary.get("fingerprint") == expected_fingerprint
                                and primary.get("next_step") == steps)
        except (FileNotFoundError, OSError, ValueError, RuntimeError):
            primary_complete = False
        if not primary_complete:
            save_state(steps)
    row = {"d_model": int(d_model),
           "params": int(sum(p.numel() for p in model.parameters())),
           "train_acc_noisy": float(history["train_acc_noisy"][-1]),
           "test_err": float(history["test_err"][-1]),
           "test_loss": float(history["test_loss"][-1])}
    if return_details:
        row.update(history=history, config=config,
                   training_seconds=elapsed_before + time.monotonic() - started,
                   peak_gpu_bytes=(int(torch.cuda.max_memory_allocated(torch.device(device)))
                                   if torch.cuda.is_available() and torch.device(device).type == "cuda" else None))
    model.eval().cpu()
    return model, row
