"""grokking.py — Q1: grokking on Harry Potter text.

Part A: plain next-character LM on a tiny HP subset -> memorisation without generalisation.
Part B: next-character LM on HP passages shown twice ([p ; <SEP> ; p]). The repeated half
        can be predicted by memorising the training passages OR by a general copy rule.
        With a small training set, scaled-up init and weight decay, the model memorises
        first and generalises much later: grokking.

Implement the four TODO functions. The training loops and `grokking_gap` are given.
"""
import copy
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- data
def make_repeated_passages(ids: torch.Tensor, n: int, passage_len: int, sep_id: int,
                           generator: torch.Generator) -> torch.Tensor:
    """Sample n random contiguous passages of `passage_len` characters from the 1-D id tensor
    `ids` (use torch.randperm(..., generator=generator) for the start offsets) and return them
    doubled with a separator: shape (n, 2*passage_len + 1), each row
    [p_0 .. p_{P-1}, sep_id, p_0 .. p_{P-1}]."""
    if ids.ndim != 1 or passage_len < 1 or n < 0:
        raise ValueError("expected 1-D ids, positive passage_len and nonnegative n")
    starts_available = ids.numel() - passage_len + 1
    if n > starts_available:
        raise ValueError("cannot sample more distinct starts than the text contains")
    starts = torch.randperm(starts_available, generator=generator)[:n].to(ids.device)
    passages = ids[starts[:, None] + torch.arange(passage_len, device=ids.device)]
    separator = torch.full((n, 1), sep_id, dtype=ids.dtype, device=ids.device)
    return torch.cat((passages, separator, passages), dim=1)


# ----------------------------------------------------------------------------- init
@torch.no_grad()
def scale_init_(model, alpha: float):
    """Omnigrok-style initialisation (Liu et al., 2023): multiply every parameter EXCEPT the
    LayerNorm weights/biases (names containing '.ln' or starting with 'ln_') by `alpha`, in
    place. Return the model."""
    for name, param in model.named_parameters():
        if ".ln" not in name and not name.startswith("ln_"):
            param.mul_(alpha)
    return model


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate_lm(model, X: torch.Tensor) -> dict:
    """Plain next-token metrics over every position of windows X (B, T+1):
    inputs X[:, :-1], targets X[:, 1:]. Return {"acc": float, "loss": float}."""
    logits = model(X[:, :-1])
    targets = X[:, 1:]
    return {
        "acc": (logits.argmax(dim=-1) == targets).float().mean().item(),
        "loss": F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item(),
    }


@torch.no_grad()
def evaluate_repeat(model, X: torch.Tensor, P: int) -> dict:
    """Metrics on doubled passages X (B, 2P+1). Inputs X[:, :-1], targets X[:, 1:].
      * acc_first   : accuracy on target indices 0 .. P-2    (copy 1, not copyable)
      * acc_repeat  : accuracy on target indices P+1 .. 2P-1 (copy 2, copyable)
      * loss_repeat : cross-entropy on the repeated-half targets
    Return {"acc_first", "acc_repeat", "loss_repeat"} as floats."""
    logits = model(X[:, :-1])
    targets = X[:, 1:]
    correct = logits.argmax(dim=-1) == targets
    repeat_logits = logits[:, P + 1:2 * P]
    repeat_targets = targets[:, P + 1:2 * P]
    return {
        "acc_first": correct[:, :P - 1].float().mean().item(),
        "acc_repeat": correct[:, P + 1:2 * P].float().mean().item(),
        "loss_repeat": F.cross_entropy(repeat_logits.reshape(-1, repeat_logits.size(-1)),
                                       repeat_targets.reshape(-1)).item(),
    }


def _save_training_checkpoint(path, payload):
    """Write, verify and rotate a resumable state before replacing the live file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False)
    if (check.get("kind"), check.get("next_step"), check.get("fingerprint")) != (
            payload.get("kind"), payload.get("next_step"), payload.get("fingerprint")):
        raise RuntimeError(f"checkpoint verification failed: {temporary}")
    if os.path.exists(path):
        previous_temporary = str(path) + ".prev.tmp"
        shutil.copy2(path, previous_temporary)
        try:
            torch.load(previous_temporary, map_location="cpu", weights_only=False)
        except Exception:
            os.remove(previous_temporary)
        else:
            _replace_with_retry(previous_temporary, str(path) + ".prev")
    _replace_with_retry(temporary, path)


def _replace_with_retry(source, destination):
    """Windows sync/antivirus tools can briefly hold a freshly written file open."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def _training_state(kind, model, optimizer, history, next_step, fingerprint=None, **extra):
    state = {
        "kind": kind,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "next_step": next_step,
        "fingerprint": fingerprint,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    state.update(extra)
    return state


def _restore_training_checkpoint(path, kind, model, optimizer, device, fingerprint=None):
    if not path:
        return None
    candidates = (path, str(path) + ".prev", str(path) + ".tmp")
    state = None
    problems = []
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                candidate_state = torch.load(candidate, map_location=device, weights_only=False)
                if candidate_state.get("kind") != kind:
                    problems.append(f"{candidate}: wrong training kind")
                    continue
                if fingerprint is not None and candidate_state.get("fingerprint") != fingerprint:
                    problems.append(f"{candidate}: configuration fingerprint mismatch")
                    continue
                state = candidate_state
                break
            except Exception as exc:
                warnings.warn(f"unreadable checkpoint {candidate}: {exc}")
    if state is None:
        if problems:
            raise ValueError("No compatible checkpoint: " + "; ".join(problems))
        return None
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
        saved_states = state["cuda_rng_states"]
        current_count = torch.cuda.device_count()
        if len(saved_states) != current_count:
            warnings.warn("GPU count changed since checkpoint; restoring RNG states for matching devices")
        for gpu_index in range(min(len(saved_states), current_count)):
            torch.cuda.set_rng_state(saved_states[gpu_index].cpu(), gpu_index)
    return state


def _maybe_checkpoint(path, every, callback, step, steps, kind, model, optimizer,
                      history, force=False, fingerprint=None, **extra):
    if not (path or callback):
        return False
    if force or (every and (step + 1) % every == 0) or step == steps:
        state = _training_state(kind, model, optimizer, history, step + 1,
                                fingerprint=fingerprint, **extra)
        if path:
            _save_training_checkpoint(path, state)
        if callback:
            callback(state)
        return True
    return False


# ----------------------------------------------------------------------------- training (GIVEN)
def train_plain_lm(model, Xtr, Xte, steps=3000, lr=1e-3, weight_decay=1.0,
                   log_every=100, device="cpu", verbose=True,
                   checkpoint_path=None, checkpoint_every=100, checkpoint_callback=None,
                   checkpoint_every_seconds=None, fingerprint=None, checkpoint_metadata=None,
                   save_best_checkpoint=True):
    """Part A. Full-batch AdamW next-token training on windows. Returns history."""
    model.to(device); Xtr, Xte = Xtr.to(device), Xte.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    hist = {"step": [], "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    resumed = _restore_training_checkpoint(checkpoint_path, "plain_lm", model, opt, device,
                                           fingerprint=fingerprint)
    start_step = resumed["next_step"] if resumed else 0
    if resumed:
        hist = resumed["history"]
    best_test_loss = min(hist["test_loss"], default=float("inf"))
    last_checkpoint_at = time.monotonic()
    for step in range(start_step, steps + 1):
        if step % log_every == 0:
            model.eval()
            tr, te = evaluate_lm(model, Xtr), evaluate_lm(model, Xte)
            hist["step"].append(step)
            hist["train_acc"].append(tr["acc"]); hist["test_acc"].append(te["acc"])
            hist["train_loss"].append(tr["loss"]); hist["test_loss"].append(te["loss"])
            if checkpoint_path and save_best_checkpoint and te["loss"] < best_test_loss:
                best_test_loss = te["loss"]
                _save_training_checkpoint(str(checkpoint_path) + ".best",
                                          _training_state("plain_lm", model, opt, hist, step,
                                                          fingerprint=fingerprint,
                                                          metadata=checkpoint_metadata,
                                                          best_test_loss=best_test_loss,
                                                          best_checkpoint_step=step))
            if verbose and step % (log_every * 10) == 0:
                print(f"step {step:6d} | train acc {tr['acc']:.3f} loss {tr['loss']:.3f} | "
                      f"held-out acc {te['acc']:.3f} loss {te['loss']:.3f}")
            model.train()
        logits = model(Xtr[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Xtr[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        time_due = bool(checkpoint_every_seconds and
                        time.monotonic() - last_checkpoint_at >= checkpoint_every_seconds)
        saved = _maybe_checkpoint(checkpoint_path, checkpoint_every, checkpoint_callback,
                                  step, steps, "plain_lm", model, opt, hist,
                                  force=time_due, fingerprint=fingerprint,
                                  metadata=checkpoint_metadata)
        if saved:
            last_checkpoint_at = time.monotonic()
    model.eval()
    return hist


def train_grokking(model, Xtr, Xte, P, steps=5000, lr=1e-3, weight_decay=1.0,
                   log_every=100, device="cpu", mem_thresh=0.9, verbose=True,
                   checkpoint_path=None, checkpoint_every=100, checkpoint_callback=None,
                   checkpoint_every_seconds=None, fingerprint=None, stop_on_success=False,
                   checkpoint_metadata=None, save_best_checkpoint=True):
    """Part B. Full-batch AdamW on doubled passages. Logs train/held-out acc_first,
    acc_repeat and the parameter norm, and snapshots the model the FIRST time the training
    repeated-half accuracy reaches `mem_thresh` (the memorised, pre-grok model).
    Returns (history, memorised_state_dict or None)."""
    model.to(device); Xtr, Xte = Xtr.to(device), Xte.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    keys = ["train_acc_first", "train_acc_repeat", "test_acc_first", "test_acc_repeat",
            "test_loss_repeat", "param_norm"]
    hist = {"step": []} | {k: [] for k in keys}
    mem_state = None
    resumed = _restore_training_checkpoint(checkpoint_path, "grokking", model, opt, device,
                                           fingerprint=fingerprint)
    start_step = resumed["next_step"] if resumed else 0
    if resumed:
        hist = resumed["history"]
        mem_state = resumed["mem_state"]
    best_repeat_acc = max(hist["test_acc_repeat"], default=float("-inf"))
    last_checkpoint_at = time.monotonic()
    for step in range(start_step, steps + 1):
        if step % log_every == 0:
            model.eval()
            tr, te = evaluate_repeat(model, Xtr, P), evaluate_repeat(model, Xte, P)
            norm = torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).item()
            hist["step"].append(step)
            hist["train_acc_first"].append(tr["acc_first"]); hist["train_acc_repeat"].append(tr["acc_repeat"])
            hist["test_acc_first"].append(te["acc_first"]); hist["test_acc_repeat"].append(te["acc_repeat"])
            hist["test_loss_repeat"].append(te["loss_repeat"]); hist["param_norm"].append(norm)
            if mem_state is None and tr["acc_repeat"] >= mem_thresh:
                mem_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mem_state["__step__"] = step
            if checkpoint_path and save_best_checkpoint and te["acc_repeat"] > best_repeat_acc:
                best_repeat_acc = te["acc_repeat"]
                _save_training_checkpoint(str(checkpoint_path) + ".best",
                                          _training_state("grokking", model, opt, hist, step,
                                                          fingerprint=fingerprint,
                                                          mem_state=mem_state,
                                                          metadata=checkpoint_metadata,
                                                          best_test_acc_repeat=best_repeat_acc,
                                                          best_checkpoint_step=step))
            observed_success = (mem_state is not None and tr["acc_repeat"] >= 0.95
                                and te["acc_repeat"] >= 0.80 and
                                (grokking_gap(hist)["gap"] or 0) > 0)
            successful_path = str(checkpoint_path) + ".successful" if checkpoint_path else None
            success_saved = False
            if successful_path and observed_success and os.path.exists(successful_path):
                try:
                    success_saved = (torch.load(successful_path, map_location="cpu",
                                                weights_only=False).get("fingerprint") == fingerprint)
                except Exception:
                    pass
            if successful_path and observed_success and not success_saved:
                _save_training_checkpoint(successful_path,
                                          _training_state("grokking", model, opt, hist, step,
                                                          fingerprint=fingerprint,
                                                          mem_state=mem_state,
                                                          metadata=checkpoint_metadata))
            if verbose and step % (log_every * 5) == 0:
                print(f"step {step:6d} | train first {tr['acc_first']:.3f} repeat {tr['acc_repeat']:.3f} | "
                      f"held-out first {te['acc_first']:.3f} repeat {te['acc_repeat']:.3f} | norm {norm:.1f}")
            if stop_on_success and observed_success:
                state = _training_state("grokking", model, opt, hist, step,
                                        fingerprint=fingerprint, mem_state=mem_state,
                                        metadata=checkpoint_metadata)
                if checkpoint_path:
                    _save_training_checkpoint(checkpoint_path, state)
                if checkpoint_callback:
                    checkpoint_callback(state)
                return hist, mem_state
            model.train()
        logits = model(Xtr[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Xtr[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        time_due = bool(checkpoint_every_seconds and
                        time.monotonic() - last_checkpoint_at >= checkpoint_every_seconds)
        saved = _maybe_checkpoint(checkpoint_path, checkpoint_every, checkpoint_callback,
                                  step, steps, "grokking", model, opt, hist,
                                  force=time_due, fingerprint=fingerprint,
                                  mem_state=mem_state, metadata=checkpoint_metadata)
        if saved:
            last_checkpoint_at = time.monotonic()
    model.eval()
    return hist, mem_state


def grokking_gap(history, mem_key="train_acc_repeat", gen_key="test_acc_repeat",
                 mem_thresh=0.9, gen_thresh=0.8) -> dict:
    """GIVEN. memorised_step = first step history[mem_key] >= mem_thresh,
    grokked_step = first step history[gen_key] >= gen_thresh, gap = difference."""
    def first(key, thr):
        for s, v in zip(history["step"], history[key]):
            if v >= thr:
                return s
        return None
    m, g = first(mem_key, mem_thresh), first(gen_key, gen_thresh)
    return {"memorised_step": m, "grokked_step": g,
            "gap": (g - m) if (m is not None and g is not None) else None}
