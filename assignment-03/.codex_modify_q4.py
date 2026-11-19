"""Temporary Q4-only notebook mutator; the rollback is authoritative."""
import hashlib
import json
import textwrap
from pathlib import Path

NOTEBOOK = Path("copy_assignment3.ipynb")
ROLLBACK = Path(".codex_backups/copy_assignment3.before_q4_implementation.ipynb")
EXPECTED_SHA = "c4feb163b4e28ed509683a0f44b0f4c40ea9114f6d563d3c394c56a0bba6e2df"
raw = ROLLBACK.read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
    raise RuntimeError("Q4 rollback hash mismatch")
nb = json.loads(raw.decode("utf-8"))

def source(value):
    value = textwrap.dedent(value).lstrip("\n")
    if not value.endswith("\n"):
        value += "\n"
    return value.splitlines(keepends=True)

def at(cell_id):
    matches = [i for i, cell in enumerate(nb["cells"]) if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise KeyError((cell_id, matches))
    return matches[0]

def code(cell_id, value):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source(value)}

def markdown(cell_id, value):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source(value)}

def replace(cell_id, value):
    cell = nb["cells"][at(cell_id)]
    if cell["cell_type"] != "code":
        raise TypeError(cell_id)
    cell["source"] = source(value)
    cell["execution_count"] = None
    cell["outputs"] = []

def insert_after(cell_id, cells):
    position = at(cell_id) + 1
    nb["cells"][position:position] = cells

q4_start = at("q4-intro-final")
q5_start = at("q5-intro-final")
frozen_prior = json.loads(json.dumps(nb["cells"][:q4_start + 1], ensure_ascii=False))
frozen_later = json.loads(json.dumps(nb["cells"][q5_start:], ensure_ascii=False))

q3_bootstrap = "".join(nb["cells"][at("q3-bootstrap")]["source"])
q3_bootstrap = q3_bootstrap.replace("Q3_PROJECT_ROOT = None", "Q3_PROJECT_ROOT = Q4_PROJECT_ROOT")
q3_bootstrap = q3_bootstrap.replace("Q3_SUPPORT_ROOT = None", "Q3_SUPPORT_ROOT = Q4_SUPPORT_ROOT")
q3_bootstrap = q3_bootstrap.replace(
    'if not _source.is_file() or _source.name == "a1_tokenizer.py":',
    'if not _source.is_file() or _source.name in ("a1_tokenizer.py", "attention.py", "model_arch.py", "rope.py", "q3_runtime.py", "optim_muon.py", "q4_runtime.py", "model_builtin.py"):'
)
for q in ("Q1", "Q2", "RECOVERY"):
    q3_bootstrap = q3_bootstrap.replace(f"Q3_{q}_ARCHIVE = None", f"Q3_{q}_ARCHIVE = Q4_{q}_ARCHIVE")

q4_bootstrap = r'''
# Q4 fresh-session controls. Queue every Q4 code cell once; no smoke/full switch.
Q4_PROJECT_ROOT = None
Q4_SUPPORT_ROOT = None
Q4_Q1_ARCHIVE = None
Q4_Q2_ARCHIVE = None
Q4_RECOVERY_ARCHIVE = None  # full q3_all_artifacts.zip
Q4_OWN_ARCHIVE = None       # optional exact q4_all_artifacts.zip
'''+q3_bootstrap+r'''

# Q3 bootstrap above is read-only with respect to prior questions: it validates and
# restores required full archives without running any prior training cell.
_q3_required = [Path(name) for name in (
    "submissions/q3/config.json", "submissions/q3/metrics.json",
    "submissions/q3/model.pt", "submissions/q3/rope.py",
    "submissions/q3/attention.py", "submissions/q3/model_arch.py",
    "artifacts/full/q3/manifests/q2_selection.json",
    "artifacts/full/q3/manifests/dependencies.json",
    "artifacts/full/q3/manifests/submission_identity.json",
    "artifacts/full/q3/source_snapshots/a1_tokenizer.py",
    "artifacts/full/q3/source_snapshots/rope.py",
    "artifacts/full/q3/source_snapshots/attention.py",
    "artifacts/full/q3/source_snapshots/model_arch.py",
    "artifacts/full/q3/source_snapshots/q3_runtime.py",
    "artifacts/full/q3/rope/final.pt", "artifacts/full/q3/sinusoidal/final.pt",
)]
if not all(path.is_file() for path in _q3_required):
    raise FileNotFoundError("Q4 needs a completed Q3 full recovery ZIP or exact restored files")
if _q3_manifest is not None and (_q3_manifest.get("question") != "q3" or _q3_manifest.get("mode") != "full"):
    raise RuntimeError("Attached Q3 archive is not a full Question 3 recovery archive")

_q3_metrics = json.loads(Path("submissions/q3/metrics.json").read_text(encoding="utf-8"))
_q3_submission_config = json.loads(Path("submissions/q3/config.json").read_text(encoding="utf-8"))
_q3_identity = json.loads(Path("artifacts/full/q3/manifests/submission_identity.json").read_text(encoding="utf-8"))
_q3_selection = json.loads(Path("artifacts/full/q3/manifests/q2_selection.json").read_text(encoding="utf-8"))
if _q3_selection != Q3_SELECTION or _q3_identity["q2_selected_label"] != Q3_SELECTED_LABEL:
    raise RuntimeError("Q3 selected a different Q2 standard-attention architecture")
if _q3_submission_config["pos_emb_type"] != "rope" or _q3_identity["submitted_scheme"] != "rope":
    raise RuntimeError("Q3 official submission identity is not the measured RoPE model")
if q3_sha256("submissions/q3/model.pt") != _q3_identity["submission_model_sha256"]:
    raise RuntimeError("Q3 official model checkpoint differs from its identity record")
_q3_ppls = {"sinusoidal": float(_q3_metrics["sinusoidal_held_out_ppl"]),
            "rope": float(_q3_metrics["rope_held_out_ppl"])}
if not all(math.isfinite(value) and value >= 1 for value in _q3_ppls.values()):
    raise RuntimeError("Q3 full held-out perplexities are not finite")
for _scheme in ("sinusoidal", "rope"):
    _internal = json.loads((Path("artifacts/full/q3") / _scheme / "metrics.json").read_text())
    _status = json.loads((Path("artifacts/full/q3") / _scheme / "status.json").read_text())
    if (_status.get("status") != "complete" or _internal.get("completed_steps") != 500 or
        not math.isclose(float(_internal["held_out_ppl"]), _q3_ppls[_scheme], rel_tol=1e-5)):
        raise RuntimeError(f"Q3 {_scheme} is not a matching completed full result")
Q4_POSITION = min(_q3_ppls, key=lambda name: (_q3_ppls[name], name))
_q3_chosen_config = json.loads((Path("artifacts/full/q3") / Q4_POSITION / "config.json").read_text())
Q4_ARCH = dict(_q3_chosen_config["model_config"])
if (Q4_ARCH["pos_emb_type"] != Q4_POSITION or
    Q4_ARCH["attention_type"] != "standard" or
    Q4_ARCH["n_heads"] != Q3_SELECTED_CONFIG["n_heads"] or
    Q4_ARCH["n_kv_heads"] != Q3_SELECTED_CONFIG["n_kv_heads"] or
    Q4_ARCH["vocab_size"] != VOCAB_SIZE):
    raise RuntimeError("Q4 architecture conflicts with measured Q2/Q3 selection")
Q4_OPT_STEPS = int(_q3_chosen_config["contract"]["steps"])
Q4_BATCH_SIZE = int(_q3_chosen_config["contract"]["batch_size"])
Q4_SEQ_LEN = int(_q3_chosen_config["contract"]["training_window"])
Q4_SEED = int(_q3_chosen_config["contract"]["seed"])
if (Q4_OPT_STEPS != 500 or Q4_SEQ_LEN != 64 or Q4_ARCH["max_seq_len"] != 128):
    raise RuntimeError("Q4 full budget/context differ from the completed Q3 controls")

# Verify immutable Q3 snapshots, then activate that exact source version at root.
for _name in ("a1_tokenizer.py", "rope.py", "attention.py", "model_arch.py", "q3_runtime.py"):
    _snapshot = Path("artifacts/full/q3/source_snapshots") / _name
    _expected = _q3_chosen_config["source_hashes"][_name]
    if q3_sha256(_snapshot) != _expected:
        raise RuntimeError(f"Q3 source snapshot mismatch: {_name}")
    _active = Path(_name)
    if not _active.is_file() or q3_sha256(_active) != _expected:
        _staged = _active.with_name(_active.name + ".q4restore")
        shutil.copy2(_snapshot, _staged)
        os.replace(_staged, _active)
    if q3_sha256(_active) != _expected:
        raise RuntimeError(f"Could not activate exact Q3 source: {_name}")
for _name in ("a1_tokenizer.py", "rope.py", "attention.py", "model_arch.py"):
    if q3_sha256(Path("submissions/q3") / _name) != _q3_identity["source_hashes"][_name]:
        raise RuntimeError(f"Q3 submitted source mismatch: {_name}")
for _module in ("q3_runtime", "model_arch", "attention", "rope"):
    sys.modules.pop(_module, None)
from model_arch import LMConfig, DecoderLM
from q3_runtime import (atomic_json, atomic_torch, cpu_weights, file_sha256,
                        load_torch, stable_hash, verify_file, weights_hash, StepBatchSampler)
Q4_ARCH_FINGERPRINT = stable_hash(Q4_ARCH)
Q4_DEPENDENCIES = {
    "q1_archive_sha256": q3_sha256(_q1_path) if _q1_path else None,
    "q2_archive_sha256": q3_sha256(_q2_path) if _q2_path else None,
    "q3_archive_sha256": q3_sha256(_q3_path) if _q3_path else None,
    "q2_selection_sha256": q3_sha256("artifacts/full/q2/analysis/q3_selection.json"),
    "q3_metrics_sha256": q3_sha256("submissions/q3/metrics.json"),
    "q3_chosen_final_sha256": q3_sha256(Path("artifacts/full/q3") / Q4_POSITION / "final.pt"),
    "tokenizer_fingerprint": Q3_TOKENIZER_FINGERPRINT,
    "train_ids_hash": Q3_TRAIN_IDS_HASH, "held_out_hash": Q3_HELD_OUT_HASH,
    "training_corpus_sha256": Q3_CORPUS_HASH,
    "q4_architecture_fingerprint": Q4_ARCH_FINGERPRINT,
}

_q4_path, _q4_manifest = q3_find_archive("q4_all_artifacts.zip", Q4_OWN_ARCHIVE, "full")
if _q4_path:
    if _q4_manifest.get("question") != "q4":
        raise RuntimeError("Q4 recovery archive manifest has the wrong question")
    q3_restore_selected(_q4_path, _q4_manifest,
                        ("artifacts/full/q4", "submissions/q4"),
                        "Q4 recovery", allow_local_newer=True)
print("Q4 fixed architecture from measured Q2/Q3:", Q3_SELECTED_LABEL, Q4_POSITION,
      "H=", Q4_ARCH["n_heads"], "Hkv=", Q4_ARCH["n_kv_heads"],
      "Q3 PPLs=", _q3_ppls, "steps=", Q4_OPT_STEPS)
'''

insert_after("q4-intro-final", [
    markdown("q4-bootstrap-heading", r'''
### Q4 fresh-session restoration and automatic pipeline

Queue the Q4 code cells in order. They restore the completed Q1–Q3 artifacts, run
isolated smoke tests, then automatically start or resume full HPO on two GPUs and
continue through analysis, submission, autograding, and recovery export.
'''),
    code("q4-bootstrap", q4_bootstrap),
])

replace("62d70bd7", r'''
%%writefile optim_muon.py
"""Assignment Q4 simplified Muon: matrix momentum + quintic Newton–Schulz."""
import math
import torch


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("Newton–Schulz requires a 2D matrix")
    if steps < 0 or eps <= 0:
        raise ValueError("steps must be nonnegative and eps positive")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5):
        if lr <= 0 or not 0 <= momentum < 1 or ns_steps < 0:
            raise ValueError("Invalid Muon hyperparameters")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon is defined only for 2D weight matrices")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = newton_schulz_orthogonalize(buf, steps=group["ns_steps"])
                update = update * math.sqrt(max(1.0, p.size(0) / p.size(1)))
                p.add_(update, alpha=-group["lr"])
        return loss
''')

insert_after("4c7802b6", [
    code("q4-muon-unit-tests", r'''
# Critical Q4.1 preflight. A failure here stops the queued pipeline before HPO.
import io, math, torch
for shape in ((5, 5), (9, 4), (4, 9), (4, 1)):
    gradient = torch.randn(shape, generator=torch.Generator().manual_seed(781))
    for steps in (0, 1, 3, 5):
        x = gradient / (gradient.norm() + 1e-7)
        if shape[0] > shape[1]: x = x.T
        for _ in range(steps):
            gram = x @ x.T
            x = 3.4445*x - 4.7750*(gram@x) + 2.0315*(gram@gram@x)
        expected = x.T if shape[0] > shape[1] else x
        actual = newton_schulz_orthogonalize(gradient, steps=steps)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert actual.shape == gradient.shape and torch.isfinite(actual).all()
assert torch.equal(newton_schulz_orthogonalize(torch.zeros(4, 9)), torch.zeros(4, 9))
assert torch.isfinite(newton_schulz_orthogonalize(torch.full((9, 4), 1e-12))).all()
_muon_source = Path("optim_muon.py").read_text(encoding="utf-8").lower()
assert "torch.linalg.svd" not in _muon_source and "torch.inverse" not in _muon_source
try:
    newton_schulz_orthogonalize(torch.ones(4))
    raise AssertionError("Muon accepted a 1D matrix")
except ValueError:
    pass
parameter = torch.nn.Parameter(torch.ones(4, 9))
optimizer = Muon([parameter], lr=.01, momentum=.8)
parameter.grad = torch.ones_like(parameter)
optimizer.step()
assert torch.isfinite(parameter).all() and not torch.equal(parameter, torch.ones_like(parameter))
assert "momentum_buffer" in optimizer.state[parameter]
first_buffer = optimizer.state[parameter]["momentum_buffer"].clone()
parameter.grad = torch.ones_like(parameter)
optimizer.step()
torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"],
                           first_buffer*.8 + torch.ones_like(first_buffer))
buffer = io.BytesIO(); torch.save(optimizer.state_dict(), buffer); buffer.seek(0)
clone = torch.nn.Parameter(parameter.detach().clone())
restored = Muon([clone], lr=.01, momentum=.8)
restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=False))
assert restored.state_dict()["state"]
print("Q4.1 Muon/NS unit tests passed (shapes, recurrence, momentum, state restore)")
'''),
])

replace("7720607f", r'''
# Q4.2: every unique trainable tensor belongs to exactly one optimizer.
OPT_STEPS = Q4_OPT_STEPS
Q4_OPTIMIZERS = ("sgd", "rmsprop", "adamw", "muon")

def build_param_groups_for_muon(model):
    hidden_fragments = (".attn.q_proj.weight", ".attn.k_proj.weight",
                        ".attn.v_proj.weight", ".attn.out_proj.weight",
                        ".ffn.net.0.weight", ".ffn.net.3.weight")
    muon_named, adamw_named = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and any(fragment in name for fragment in hidden_fragments):
            muon_named.append((name, parameter))
        else:
            adamw_named.append((name, parameter))
    muon_ids = {id(parameter) for _, parameter in muon_named}
    adamw_ids = {id(parameter) for _, parameter in adamw_named}
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert muon_ids and adamw_ids and not muon_ids & adamw_ids
    assert muon_ids | adamw_ids == trainable_ids
    assert all(parameter.ndim == 2 for _, parameter in muon_named)
    assert not any("tok_emb" in name or "head" in name or "ln" in name
                   for name, _ in muon_named)
    return {
        "muon_params": [parameter for _, parameter in muon_named],
        "adamw_params": [parameter for _, parameter in adamw_named],
        "muon_names": [name for name, _ in muon_named],
        "adamw_names": [name for name, _ in adamw_named],
        "muon_count": sum(parameter.numel() for _, parameter in muon_named),
        "adamw_count": sum(parameter.numel() for _, parameter in adamw_named),
    }

def train_with_optimizer(optimizer_name, lr, steps=OPT_STEPS, *, phase="full",
                         device="cuda:0", position=0, variant="recommended"):
    """Fresh or resumed Q4 trial; returns saved measured record, not a live GPU model."""
    return q4.run_trial(optimizer_name, float(lr), int(steps), phase=phase,
                        device_name=device, position=position, variant=variant)

_q4_group_probe = DecoderLM(LMConfig(**Q4_ARCH))
_q4_groups = build_param_groups_for_muon(_q4_group_probe)
print("Recommended Muon split:", {key: _q4_groups[key] for key in
      ("muon_names", "adamw_names", "muon_count", "adamw_count")})
del _q4_group_probe
''')

insert_after("7720607f", [
    markdown("q4-runtime-heading", r'''
### Q4 resumable two-GPU trial engine

This Q4-specific helper keeps independent trial checkpoints, runs one complete model
per T4, resumes validated partial trials, and periodically exports a verified ZIP.
'''),
    code("q4-runtime-writefile", r'''
%%writefile q4_runtime.py
"""Q4 controlled optimizer experiments, two-GPU scheduling, and recovery."""
import gc
import json
import math
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import TokenDataset, perplexity_of_texts
from model_arch import LMConfig, DecoderLM
from optim_muon import Muon
from q3_runtime import (atomic_json, atomic_torch, cpu_weights, file_sha256,
                        load_torch, stable_hash, verify_file, weights_hash, StepBatchSampler)

OPTIMIZERS = ("sgd", "rmsprop", "adamw", "muon")
SCHEMA = 1


class BroadMomentum1D(torch.optim.Optimizer):
    """Experimental 1D Muon-style momentum, intentionally without NS."""
    def __init__(self, params, lr, momentum=.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim == 2:
                    raise ValueError("BroadMomentum1D only handles non-2D tensors")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(parameter.grad)
                parameter.add_(buf, alpha=-group["lr"])


class Q4Runtime:
    def __init__(self, *, arch, selected_attention, selected_position, dependencies,
                 train_ids, held_out, tokenizer, grouper, seed, batch_size, seq_len,
                 full_steps, smoke=False):
        self.arch = dict(arch)
        self.selected_attention = selected_attention
        self.selected_position = selected_position
        self.dependencies = dict(dependencies)
        self.semantic_dependencies = {key:value for key,value in dependencies.items()
                                      if not key.endswith("archive_sha256")}
        self.train_ids = [int(value) for value in train_ids]
        self.held_out = list(held_out)
        self.tokenizer = tokenizer
        self.grouper = grouper
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.full_steps = int(full_steps)
        self.smoke = bool(smoke)
        self.mode = "smoke" if self.smoke else "full"
        self.root = Path("artifacts") / self.mode / "q4"
        self.archive = Path("q4_smoke_all_artifacts.zip" if self.smoke else "q4_all_artifacts.zip")
        self.io_lock = threading.RLock()
        self.build_lock = threading.Lock()
        self.last_export = 0.0
        self.archive_seconds = 600.0
        self.initial_hash = None
        self.source_hashes = {}
        self.dataset_size = len(TokenDataset(self.train_ids, self.seq_len))
        if self.dataset_size < self.batch_size:
            raise ValueError("Q4 corpus has fewer than one full training batch")
        if self.selected_attention not in ("mha", "gqa", "mqa") or self.arch["attention_type"] != "standard":
            raise ValueError("Q4 requires the measured standard-attention Q2 winner")
        if self.arch["pos_emb_type"] != self.selected_position:
            raise ValueError("Q4 positional mode differs from measured Q3 selection")

    def build_model(self):
        # No unsafe global RNG reset in parallel workers: CPU fork is serialized.
        with self.build_lock, torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            return DecoderLM(LMConfig(**self.arch)).cpu()

    def setup(self):
        for sub in ("trials", "best", "muon_split_ablation", "analysis", "plots",
                    "source_snapshots", "manifests", "recovery"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for name in ("a1_tokenizer.py", "rope.py", "attention.py", "model_arch.py",
                     "optim_muon.py", "q3_runtime.py", "q4_runtime.py"):
            active = Path(name)
            if not active.is_file():
                raise FileNotFoundError(f"Q4 source missing: {name}")
            self.source_hashes[name] = file_sha256(active)
            snapshot = self.root / "source_snapshots" / name
            if snapshot.is_file() and file_sha256(snapshot) != self.source_hashes[name]:
                raise RuntimeError(f"Immutable Q4 source snapshot changed: {name}")
            if not snapshot.is_file():
                shutil.copy2(active, snapshot)
        manifest = {"schema": SCHEMA, "mode": self.mode, "architecture": self.arch,
                    "attention": self.selected_attention, "position": self.selected_position,
                    "dependencies": self.semantic_dependencies, "source_hashes": self.source_hashes,
                    "seed": self.seed, "batch_size": self.batch_size,
                    "seq_len": self.seq_len, "full_steps": self.full_steps}
        target = self.root / "manifests" / "experiment.json"
        if target.is_file() and json.loads(target.read_text()) != manifest:
            raise RuntimeError("Existing Q4 experiments have incompatible architecture/data/source")
        if not target.is_file(): atomic_json(target, manifest)
        status_path=self.root/"pipeline_status.json"
        if not status_path.is_file():
            names=("bootstrap","smoke_tests","hpo","hpo_screening","hpo_sgd",
                   "hpo_rmsprop","hpo_adamw","hpo_muon","best_run_selection",
                   "convergence_analysis","lr_sensitivity","muon_split_ablation",
                   "submission","autograder","recovery_export","complete")
            atomic_json(status_path,{"schema":SCHEMA,
                                     "stages":{name:{"status":"pending"} for name in names}})
        initial = self.root / "manifests" / "initial.pt"
        if initial.is_file():
            if not verify_file(initial): raise RuntimeError("Q4 initial state checksum failed")
            state = load_torch(initial)
        else:
            state = cpu_weights(self.build_model())
            atomic_torch(initial, state)
        self.initial_hash = weights_hash(state)
        model = self.build_model(); model.load_state_dict(state, strict=True)
        del model
        return manifest

    def stage(self, name, status, **details):
        path = self.root / "pipeline_status.json"
        with self.io_lock:
            report = json.loads(path.read_text()) if path.is_file() else {"schema":SCHEMA,"stages":{}}
            report["stages"][name] = {"status":status, "updated_utc":datetime.now(timezone.utc).isoformat(), **details}
            atomic_json(path, report)

    def trial_id(self, phase, optimizer, lr, variant):
        rate = f"{lr:.8g}".replace(".", "p").replace("-", "m").replace("+", "")
        return f"{phase}-{optimizer}-{variant}-{rate}"

    def trial_dir(self, phase, optimizer, lr, variant="recommended"):
        if variant == "broad":
            return self.root / "muon_split_ablation" / self.trial_id(phase,optimizer,lr,variant)
        return self.root / "trials" / optimizer / self.trial_id(phase,optimizer,lr,variant)

    def spec(self, phase, optimizer, lr, steps, variant):
        companion_lr = 3e-4 if optimizer == "muon" and variant == "recommended" else None
        hyper = {"sgd":{"momentum":.9}, "rmsprop":{"alpha":.99,"eps":1e-8},
                 "adamw":{"weight_decay":.01},
                 "muon":{"momentum":.95,"ns_steps":5,"companion_adamw_lr":companion_lr}}[optimizer]
        return {"schema":SCHEMA, "id":self.trial_id(phase,optimizer,lr,variant),
                "mode":self.mode, "phase":phase, "optimizer":optimizer, "variant":variant,
                "lr":float(lr), "hyperparameters":hyper, "steps":int(steps),
                "batch_size":self.batch_size, "seq_len":self.seq_len,
                "architecture":self.arch, "initial_state_hash":self.initial_hash,
                "seed":self.seed, "data_seed":self.seed+10000,
                "dependencies":self.semantic_dependencies,
                "source_hashes":self.source_hashes}

    def _prepare(self, spec):
        path = self.trial_dir(spec["phase"],spec["optimizer"],spec["lr"],spec["variant"])
        path.mkdir(parents=True, exist_ok=True)
        config = path / "config.json"
        expected = dict(spec, fingerprint=stable_hash(spec))
        if config.is_file() and json.loads(config.read_text()) != expected:
            raise RuntimeError(f"Incompatible Q4 trial at {path}")
        if not config.is_file():
            with self.io_lock: atomic_json(config, expected)
        return path, expected

    def _optimizers(self, model, spec):
        name, lr, variant = spec["optimizer"], spec["lr"], spec["variant"]
        if name == "sgd": return [torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)], {}
        if name == "rmsprop": return [torch.optim.RMSprop(model.parameters(),lr=lr,alpha=.99,eps=1e-8)], {}
        if name == "adamw": return [torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)], {}
        if variant == "recommended":
            groups = self.grouper(model)
            optimizers = [Muon(groups["muon_params"],lr=lr,momentum=.95,ns_steps=5),
                          torch.optim.AdamW(groups["adamw_params"],lr=3e-4,weight_decay=.01)]
            group_record = {key:groups[key] for key in
                            ("muon_names","adamw_names","muon_count","adamw_count")}
            return optimizers, group_record
        if variant == "broad":
            matrix = [(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2]
            other = [(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim!=2]
            ids = {id(p) for _,p in matrix} | {id(p) for _,p in other}
            assert ids == {id(p) for p in model.parameters() if p.requires_grad}
            assert not {id(p) for _,p in matrix} & {id(p) for _,p in other}
            optimizers = [Muon([p for _,p in matrix],lr=lr,momentum=.95,ns_steps=5),
                          BroadMomentum1D([p for _,p in other],lr=lr,momentum=.95)]
            return optimizers, {"broad_2d_names":[n for n,_ in matrix],
                                "broad_non2d_names":[n for n,_ in other]}
        raise ValueError(variant)

    @staticmethod
    def _state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key,value in list(state.items()):
                if torch.is_tensor(value): state[key] = value.to(device)

    def _payload(self, spec, model, optimizers, scaler, history, step, elapsed,
                 device, cpu_generator, group_record):
        state = cpu_weights(model)
        sampler = StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,step)
        return {"schema":SCHEMA, "id":spec["id"], "fingerprint":stable_hash(spec),
                "completed_steps":step, "tokens_processed":step*self.batch_size*self.seq_len,
                "model_state":state, "model_state_hash":weights_hash(state),
                "optimizer_states":[opt.state_dict() for opt in optimizers],
                "scheduler_state":None, "amp_scaler_state":scaler.state_dict(),
                "cpu_rng_state":cpu_generator.get_state(),
                "device_rng_state":torch.cuda.get_rng_state(device).cpu() if device.type=="cuda" else torch.get_rng_state().cpu(),
                "sampler_state":sampler.state(step), "history":history,
                "history_hash":stable_hash(history), "cumulative_seconds":elapsed,
                "group_record":group_record, "spec":spec,
                "saved_utc":datetime.now(timezone.utc).isoformat()}

    def _validate_payload(self, spec, payload):
        if payload.get("schema")!=SCHEMA or payload.get("id")!=spec["id"] or payload.get("fingerprint")!=stable_hash(spec):
            raise ValueError("Incompatible Q4 checkpoint identity/fingerprint")
        step=payload.get("completed_steps")
        if type(step) is not int or not 0<=step<=spec["steps"]: raise ValueError("Q4 checkpoint step invalid")
        hist=payload.get("history")
        if not isinstance(hist,list) or len(hist)!=step or [row["step"] for row in hist]!=list(range(1,step+1)):
            raise ValueError("Q4 checkpoint history discontinuity")
        if payload.get("history_hash")!=stable_hash(hist): raise ValueError("Q4 history digest mismatch")
        if payload.get("tokens_processed")!=step*self.batch_size*self.seq_len: raise ValueError("Q4 token count mismatch")
        sampler=StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,step)
        if payload.get("sampler_state")!=sampler.state(step): raise ValueError("Q4 sampler cursor mismatch")
        if weights_hash(payload["model_state"])!=payload.get("model_state_hash"):
            raise ValueError("Q4 checkpoint tensor digest mismatch")
        if len(payload.get("optimizer_states",[]))!=(2 if spec["optimizer"]=="muon" else 1):
            raise ValueError("Q4 optimizer-state count mismatch")
        if not isinstance(payload.get("cumulative_seconds"),(int,float)) or not math.isfinite(payload["cumulative_seconds"]):
            raise ValueError("Q4 elapsed time invalid")
        model=self.build_model(); model.load_state_dict(payload["model_state"],strict=True)
        opts,_=self._optimizers(model,spec)
        for opt,state in zip(opts,payload["optimizer_states"]): opt.load_state_dict(state)
        del model,opts
        return payload

    def _load_checkpoint(self, path, spec):
        errors=[]
        for candidate in (path/"checkpoint.latest.pt",path/"checkpoint.previous.pt"):
            if not candidate.is_file(): continue
            try:
                if not verify_file(candidate): raise ValueError("checksum mismatch")
                result=self._validate_payload(spec,load_torch(candidate))
                if candidate.name.endswith("previous.pt"):
                    print(f"[{spec['id']}] restored previous-known-good checkpoint")
                return result
            except Exception as exc: errors.append(f"{candidate.name}: {exc}")
        if errors: raise RuntimeError("No valid Q4 checkpoint: "+"; ".join(errors))
        return None

    def _save_checkpoint(self, path, spec, model, optimizers, scaler, history,
                         step, elapsed, device, cpu_generator, group_record):
        payload=self._payload(spec,model,optimizers,scaler,history,step,elapsed,
                              device,cpu_generator,group_record)
        latest,previous=path/"checkpoint.latest.pt",path/"checkpoint.previous.pt"
        with self.io_lock:
            candidate=latest.with_name(latest.name+"."+uuid.uuid4().hex+".candidate")
            torch.save(payload,candidate)
            self._validate_payload(spec,load_torch(candidate))
            latest_valid = False
            if latest.is_file() and verify_file(latest):
                try:
                    self._validate_payload(spec,load_torch(latest))
                    latest_valid = True
                except Exception:
                    latest_valid = False
            if latest_valid:
                old=previous.with_name(previous.name+"."+uuid.uuid4().hex+".tmp")
                shutil.copy2(latest,old)
                if file_sha256(old)!=file_sha256(latest): raise IOError("Q4 previous-checkpoint copy changed")
                os.replace(old,previous)
                atomic_json(previous.with_name(previous.name+".sha256.json"),
                            {"sha256":file_sha256(previous),"size":previous.stat().st_size})
            digest=file_sha256(candidate)
            os.replace(candidate,latest)
            atomic_json(latest.with_name(latest.name+".sha256.json"),
                        {"sha256":digest,"size":latest.stat().st_size})
            atomic_json(path/"status.json",{"status":"running","completed_steps":step,
                       "checkpoint_sha256":digest,"device":str(device),"id":spec["id"]})

    def _terminal(self, path, spec):
        metrics_path,status_path=path/"metrics.json",path/"status.json"
        if not (metrics_path.is_file() and status_path.is_file()): return None
        metrics=json.loads(metrics_path.read_text()); status=json.loads(status_path.read_text())
        if metrics.get("fingerprint")!=stable_hash(spec) or status.get("id")!=spec["id"]:
            raise RuntimeError(f"Q4 trial has incompatible existing result: {path}")
        if status.get("status")=="diverged" and metrics.get("state")=="diverged": return metrics
        if status.get("status")!="complete" or metrics.get("state")!="complete": return None
        if (metrics.get("completed_steps")!=spec["steps"] or
            len(metrics.get("loss_curve",[]))!=spec["steps"] or
            len(metrics.get("time_curve",[]))!=spec["steps"]): return None
        final=path/"final.pt"
        if not verify_file(final) or file_sha256(final)!=metrics.get("final_file_sha256"):
            print(f"Q4 completed final file invalid at {path}; reconstructing from verified checkpoint")
            return None
        if weights_hash(load_torch(final))!=metrics.get("final_state_hash"):
            print(f"Q4 completed final tensors invalid at {path}; reconstructing from verified checkpoint")
            return None
        return metrics

    def run_trial(self, optimizer_name, lr, steps, *, phase, device_name, position,
                  variant="recommended"):
        if optimizer_name not in OPTIMIZERS or variant not in ("recommended","broad"):
            raise ValueError("Unknown Q4 optimizer/variant")
        if variant=="broad" and optimizer_name!="muon": raise ValueError("Only Muon has a broad ablation")
        spec=self.spec(phase,optimizer_name,float(lr),int(steps),variant)
        path,_=self._prepare(spec)
        result=self._terminal(path,spec)
        if result is not None:
            print(f"Q4 validated completed trial skipped: {spec['id']}")
            return result
        prior=self._load_checkpoint(path,spec)
        device=torch.device(device_name)
        model=self.build_model()
        model.load_state_dict(prior["model_state"] if prior else load_torch(self.root/"manifests"/"initial.pt"),strict=True)
        model=model.to(device)
        optimizers,group_record=self._optimizers(model,spec)
        try: scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        except (AttributeError,TypeError): scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
        generator=torch.Generator(device="cpu").manual_seed(self.seed+20000)
        history,step,elapsed_base=[],0,0.0
        if prior:
            for opt,state in zip(optimizers,prior["optimizer_states"]):
                opt.load_state_dict(state); self._state_to_device(opt,device)
            scaler.load_state_dict(prior["amp_scaler_state"])
            generator.set_state(prior["cpu_rng_state"])
            if device.type=="cuda": torch.cuda.set_rng_state(prior["device_rng_state"],device)
            else: torch.set_rng_state(prior["device_rng_state"])
            history=list(prior["history"]); step=prior["completed_steps"]
            elapsed_base=float(prior["cumulative_seconds"])
        elif device.type=="cuda":
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(self.seed+30000).get_state(),device)
        else:
            torch.manual_seed(self.seed+30000)
        dataset=TokenDataset(self.train_ids,self.seq_len)
        sampler=StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,steps)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,
                          pin_memory=device.type=="cuda",generator=generator)
        started=time.monotonic(); last_ckpt=step; last_ckpt_time=started
        last_safe_step=step
        bar=tqdm(total=steps,initial=step,position=position,leave=True,
                 desc=f"Q4 {optimizer_name} lr={lr:g} {phase} {device}",dynamic_ncols=True)
        try:
            with self.io_lock:
                atomic_json(path/"status.json",{"status":"running","completed_steps":step,
                           "device":str(device),"id":spec["id"],"resumed":bool(prior)})
            for x,y in loader:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    logits=model(x)
                    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite Q4 training loss")
                scaler.scale(loss).backward()
                last_safe_step=-1  # mid-update must never be checkpointed
                for opt in optimizers: scaler.step(opt)
                scaler.update()
                if device.type=="cuda": torch.cuda.synchronize(device)
                step+=1; last_safe_step=step
                elapsed=elapsed_base+time.monotonic()-started
                history.append({"step":step,"loss":float(loss.item()),
                                "cumulative_seconds":elapsed,
                                "tokens_processed":step*self.batch_size*self.seq_len,
                                "epoch":sampler.state(step)["epoch"]})
                with self.io_lock:
                    atomic_json(path/"history.json",{"history":history,"completed_steps":step,
                                                   "cumulative_seconds":elapsed})
                due=(step==steps or step-last_ckpt>=50 or time.monotonic()-last_ckpt_time>=300 or
                     step==max(1,steps//2))
                if due:
                    self._save_checkpoint(path,spec,model,optimizers,scaler,history,step,
                                          elapsed,device,generator,group_record)
                    last_ckpt=step; last_ckpt_time=time.monotonic()
                    self.export_archive(force=False)
                session=max(time.monotonic()-started,1e-9)
                speed=(step-(prior["completed_steps"] if prior else 0))/session
                bar.update(1)
                bar.set_postfix({"state":"RESUMED" if prior else "NEW", "loss":f"{sum(r['loss'] for r in history[-20:])/min(20,len(history)):.2f}",
                                 "lr":f"{lr:.1e}","tokens":step*self.batch_size*self.seq_len,
                                 "step/s":f"{speed:.2f}","tok/s":f"{speed*self.batch_size*self.seq_len:.0f}",
                                 "session":f"{session:.0f}s","total":f"{elapsed:.0f}s",
                                 "eta":f"{(steps-step)/max(speed,1e-9):.0f}s","ckpt":last_ckpt})
            if device.type=="cuda": torch.cuda.synchronize(device)
            model.eval()
            ppl=float(perplexity_of_texts(model,self.tokenizer,self.held_out,
                                          device=str(device),max_len=self.arch["max_seq_len"]))
            if not math.isfinite(ppl) or ppl<1: raise FloatingPointError("nonfinite held-out PPL")
            elapsed=elapsed_base+time.monotonic()-started
            with self.io_lock:
                final=path/"final.pt"; atomic_torch(final,cpu_weights(model))
                metrics={"schema":SCHEMA,"id":spec["id"],"state":"complete",
                         "mode":self.mode,"phase":phase,"optimizer":optimizer_name,
                         "variant":variant,"lr":lr,"hyperparameters":spec["hyperparameters"],
                         "seed":self.seed,"data_seed":self.seed+10000,"steps_requested":steps,
                         "completed_steps":step,"held_out_ppl":ppl,
                         "loss_curve":[r["loss"] for r in history],
                         "time_curve":[r["cumulative_seconds"] for r in history],
                         "history":history,"total_seconds":elapsed,
                         "mean_seconds_per_step":history[-1]["cumulative_seconds"]/step,
                         "training_seconds":history[-1]["cumulative_seconds"],
                         "evaluation_seconds":elapsed-history[-1]["cumulative_seconds"],
                         "tokens_processed":step*self.batch_size*self.seq_len,
                         "device":str(device),"checkpoint_path":str(path/"checkpoint.latest.pt"),
                         "final_path":str(final),"final_file_sha256":file_sha256(final),
                         "final_state_hash":weights_hash(load_torch(final)),
                         "fingerprint":stable_hash(spec),"initial_state_hash":self.initial_hash,
                         "group_record":group_record}
                atomic_json(path/"metrics.json",metrics)
                atomic_json(path/"status.json",{"status":"complete","id":spec["id"],
                                                  "completed_steps":step,"held_out_ppl":ppl})
            print(f"Q4 {spec['id']} complete on {device}; PPL={ppl:.4f}, duration={elapsed:.1f}s")
            return metrics
        except FloatingPointError as exc:
            elapsed=elapsed_base+time.monotonic()-started
            metrics={"schema":SCHEMA,"id":spec["id"],"state":"diverged","mode":self.mode,
                     "phase":phase,"optimizer":optimizer_name,"variant":variant,"lr":lr,
                     "hyperparameters":spec["hyperparameters"],"seed":self.seed,
                     "steps_requested":steps,"completed_steps":last_safe_step,
                     "held_out_ppl":float("inf"),"loss_curve":[r["loss"] for r in history],
                     "time_curve":[r["cumulative_seconds"] for r in history],
                     "history":history,"total_seconds":elapsed,
                     "tokens_processed":last_safe_step*self.batch_size*self.seq_len,
                     "device":str(device),"fingerprint":stable_hash(spec),"error":str(exc)}
            with self.io_lock:
                atomic_json(path/"metrics.json",metrics)
                atomic_json(path/"status.json",{"status":"diverged","id":spec["id"],
                                                      "error":str(exc),"completed_steps":last_safe_step})
            print(f"Q4 {spec['id']} diverged; excluded from best selection: {exc}")
            return metrics
        except BaseException as exc:
            try:
                if last_safe_step==step and step>0:
                    self._save_checkpoint(path,spec,model,optimizers,scaler,history,step,
                        elapsed_base+time.monotonic()-started,device,generator,group_record)
            except Exception as checkpoint_error:
                print(f"Q4 emergency checkpoint failed: {checkpoint_error}")
            with self.io_lock:
                atomic_json(path/"status.json",{"status":"failed","id":spec["id"],
                            "completed_steps":step,"error":f"{type(exc).__name__}: {exc}",
                            "traceback":traceback.format_exc()})
            raise
        finally:
            bar.close(); del model,optimizers,loader,dataset
            gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()

    def run_jobs(self, jobs):
        jobs=list(jobs)
        if not jobs: return []
        if not self.smoke and torch.cuda.device_count()<2:
            raise RuntimeError("Full Q4 HPO requires Kaggle 2 x T4 GPUs")
        devices=["cuda:0","cuda:1"] if torch.cuda.device_count()>=2 else ["cpu"]
        pending=queue.Queue()
        for position,job in enumerate(jobs): pending.put((position,job))
        results={}; failures={}; lock=threading.Lock()
        start_barrier=threading.Barrier(2) if len(devices)==2 and len(jobs)>=2 else None
        def worker(slot,device):
            first=True
            while True:
                try: position,job=pending.get_nowait()
                except queue.Empty: break
                try:
                    if first and start_barrier is not None: start_barrier.wait(timeout=60)
                    first=False
                    result=self.run_trial(job["optimizer"],job["lr"],job["steps"],
                        phase=job["phase"],device_name=device,position=slot,
                        variant=job.get("variant","recommended"))
                    with lock: results[position]=result
                except BaseException as exc:
                    with lock: failures[position]=f"{type(exc).__name__}: {exc}"
                finally: pending.task_done()
        with ThreadPoolExecutor(max_workers=len(devices),thread_name_prefix="q4gpu") as pool:
            futures=[pool.submit(worker,i,device) for i,device in enumerate(devices)]
            for future in futures: future.result()
        self.export_archive(force=True)
        if failures:
            raise RuntimeError(f"Q4 trial failures (other GPU progress retained): {failures}")
        return [results[i] for i in range(len(jobs))]

    def select_best(self, optimizer, trial_results):
        eligible=[result for result in trial_results if result["optimizer"]==optimizer and
                  result["phase"]=="full" and result["state"]=="complete" and
                  math.isfinite(result["held_out_ppl"])]
        if len(eligible)<2: raise RuntimeError(f"Q4 {optimizer} has fewer than two successful full LR trials")
        winner=min(eligible,key=lambda row:(row["held_out_ppl"],row["total_seconds"],row["lr"]))
        bestdir=self.root/"best"/optimizer; bestdir.mkdir(parents=True,exist_ok=True)
        source=Path(winner["final_path"])
        if not verify_file(source): raise RuntimeError(f"Selected Q4 model invalid: {source}")
        destination=bestdir/"model.pt"
        if not verify_file(destination) or weights_hash(load_torch(destination))!=winner["final_state_hash"]:
            atomic_torch(destination,load_torch(source))
        record={"optimizer":optimizer,"trial_id":winner["id"],"lr":winner["lr"],
                "held_out_ppl":winner["held_out_ppl"],"model_path":str(destination),
                "model_sha256":file_sha256(destination),"trial_final_path":str(source),
                "steps":winner["completed_steps"],"total_seconds":winner["total_seconds"],
                "fingerprint":winner["fingerprint"]}
        atomic_json(bestdir/"selection.json",record)
        return record

    def validate_archive(self,path):
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read("recovery/integrity_manifest.json"))
            if manifest.get("question")!="q4" or manifest.get("mode")!=self.mode:
                raise RuntimeError("Q4 recovery ZIP identity mismatch")
            for name,info in manifest["files"].items():
                digest,size=__import__("hashlib").sha256(),0
                with archive.open(name) as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b""):
                        digest.update(chunk);size+=len(chunk)
                if digest.hexdigest()!=info["sha256"] or size!=info["size"]:
                    raise RuntimeError(f"Q4 recovery member mismatch: {name}")
        return manifest

    def export_archive(self,force=False):
        with self.io_lock:
            now=time.monotonic()
            if not force and now-self.last_export<self.archive_seconds:
                return self.archive if self.archive.is_file() else None
            files=[path for path in self.root.rglob("*") if path.is_file() and
                   ".candidate" not in path.name and ".tmp" not in path.name and
                   path!=self.root/"recovery"/"latest_export.json"]
            submission=Path("submissions/q4")
            if not self.smoke and submission.is_dir():
                files += [path for path in submission.rglob("*") if path.is_file()]
            files=sorted(set(files),key=str)
            manifest={"schema":SCHEMA,"question":"q4","mode":self.mode,
                      "created_utc":datetime.now(timezone.utc).isoformat(),
                      "architecture_fingerprint":stable_hash(self.arch),
                      "dependencies":self.dependencies,"source_hashes":self.source_hashes,
                      "files":{path.as_posix():{"sha256":file_sha256(path),
                                                "size":path.stat().st_size} for path in files}}
            candidate=self.archive.with_name(self.archive.name+"."+uuid.uuid4().hex+".candidate")
            with zipfile.ZipFile(candidate,"w",compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
                for path in files: archive.write(path,arcname=path.as_posix())
                archive.writestr("recovery/integrity_manifest.json",json.dumps(manifest,indent=2))
            self.validate_archive(candidate)
            previous=self.archive.with_name(self.archive.name.replace(".zip",".previous.zip"))
            if self.archive.is_file():
                try:
                    self.validate_archive(self.archive)
                    staged=previous.with_name(previous.name+"."+uuid.uuid4().hex+".tmp")
                    shutil.copy2(self.archive,staged)
                    self.validate_archive(staged)
                    os.replace(staged,previous)
                except Exception as exc:
                    print("Existing Q4 ZIP invalid; preserving previous valid ZIP:",exc)
            os.replace(candidate,self.archive)
            digest=file_sha256(self.archive)
            atomic_json(self.root/"recovery"/"latest_export.json",
                        {"archive":str(self.archive.resolve()),"sha256":digest,
                         "size":self.archive.stat().st_size})
            self.last_export=now
            print(f"Validated Q4 recovery ZIP: {self.archive} ({digest[:12]}..., {self.archive.stat().st_size/2**20:.1f} MiB)")
            return self.archive
'''),
    code("q4-runtime-configure", r'''
import importlib, sys
sys.modules.pop("q4_runtime", None)
from q4_runtime import Q4Runtime, OPTIMIZERS
q4 = Q4Runtime(
    arch=Q4_ARCH, selected_attention=Q3_SELECTED_LABEL,
    selected_position=Q4_POSITION, dependencies=Q4_DEPENDENCIES,
    train_ids=train_ids, held_out=held_out, tokenizer=tokenizer,
    grouper=build_param_groups_for_muon, seed=Q4_SEED,
    batch_size=Q4_BATCH_SIZE, seq_len=Q4_SEQ_LEN,
    full_steps=Q4_OPT_STEPS, smoke=False,
)
q4.setup()
q4.stage("bootstrap", "completed", architecture=Q4_ARCH,
         architecture_fingerprint=Q4_ARCH_FINGERPRINT,
         q2_attention=Q3_SELECTED_LABEL, q3_position=Q4_POSITION)
q4.export_archive(force=True)
print("Q4 full runtime configured and initial recovery ZIP verified")
'''),
])

insert_after("q4-runtime-configure", [
    code("q4-automatic-smoke", r'''
# One-pass gate: complete all isolated smoke runs before the next cell starts full HPO.
import tempfile
q4.stage("smoke_tests", "running")
try:
    from common_grading import GradeReport
    from autograder_q4 import check_muon, check_q4_experiment_evidence
    _muon_grade = GradeReport("Q4 Muon structural preflight")
    check_muon(optim_muon, _muon_grade)
    if not _muon_grade.summary():
        raise RuntimeError("Official Q4 Muon numerical checks failed before training")
    smoke_arch = dict(Q4_ARCH)
    smoke_arch.update(d_model=64, n_layers=1, d_ff=128)
    q4_smoke = Q4Runtime(
        arch=smoke_arch, selected_attention=Q3_SELECTED_LABEL,
        selected_position=Q4_POSITION, dependencies=Q4_DEPENDENCIES,
        train_ids=train_ids, held_out=held_out, tokenizer=tokenizer,
        grouper=build_param_groups_for_muon, seed=Q4_SEED,
        batch_size=2, seq_len=8, full_steps=3, smoke=True,
    )
    q4_smoke.setup()
    smoke_probe = q4_smoke.build_model()
    smoke_probe.load_state_dict(load_torch(q4_smoke.root / "manifests" / "initial.pt"), strict=True)
    probe_ids = torch.randint(0, VOCAB_SIZE, (2, 8),
                              generator=torch.Generator().manual_seed(781))
    probe_logits = smoke_probe(probe_ids)
    assert probe_logits.shape == (2, 8, VOCAB_SIZE) and torch.isfinite(probe_logits).all()
    probe_logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in smoke_probe.parameters())
    smoke_groups = build_param_groups_for_muon(smoke_probe)
    assert smoke_groups["muon_count"] > 0 and smoke_groups["adamw_count"] > 0
    del smoke_probe
    smoke_lrs = {"sgd":.01, "rmsprop":.0001, "adamw":.0001, "muon":.005}
    smoke_jobs = [{"phase":"smoke", "optimizer":name, "lr":rate, "steps":3}
                  for name,rate in smoke_lrs.items()]
    smoke_results = q4_smoke.run_jobs(smoke_jobs)
    assert len(smoke_results) == 4 and all(result["state"] == "complete" for result in smoke_results)
    _smoke_by_name = {result["optimizer"]:result for result in smoke_results}
    _smoke_metrics = {
        "per_optimizer_best_ppl": {name:_smoke_by_name[name]["held_out_ppl"] for name in OPTIMIZERS},
        "per_optimizer_loss_curves": {name:_smoke_by_name[name]["loss_curve"] for name in OPTIMIZERS},
        "per_optimizer_time_curves": {name:_smoke_by_name[name]["time_curve"] for name in OPTIMIZERS},
        "lr_sweep_results": {name:[{"lr":_smoke_by_name[name]["lr"],
                                    "held_out_ppl":_smoke_by_name[name]["held_out_ppl"]}]
                             for name in OPTIMIZERS},
    }
    _evidence_grade = GradeReport("Q4 actual-smoke metrics schema")
    check_q4_experiment_evidence(_smoke_metrics,_evidence_grade,smoke_test=True)
    if not _evidence_grade.summary():
        raise RuntimeError("Q4 smoke metrics do not meet the official structural contract")
    if torch.cuda.device_count() >= 2:
        assert {result["device"] for result in smoke_results} == {"cuda:0","cuda:1"}
    # Verify actual latest/previous checkpoint fallback without damaging live smoke runs.
    first = smoke_results[0]
    first_spec = q4_smoke.spec("smoke", first["optimizer"],first["lr"],3,"recommended")
    first_path = q4_smoke.trial_dir("smoke",first["optimizer"],first["lr"])
    assert verify_file(first_path / "checkpoint.latest.pt")
    assert verify_file(first_path / "checkpoint.previous.pt")
    with tempfile.TemporaryDirectory(prefix="q4_checkpoint_test_") as scratch:
        scratch_path = Path(scratch)
        for filename in ("checkpoint.latest.pt","checkpoint.previous.pt"):
            shutil.copy2(first_path / filename, scratch_path / filename)
            shutil.copy2(first_path / (filename + ".sha256.json"),
                         scratch_path / (filename + ".sha256.json"))
        (scratch_path / "checkpoint.latest.pt").write_bytes(b"simulated corrupt latest")
        recovered = q4_smoke._load_checkpoint(scratch_path,first_spec)
        assert recovered is not None and recovered["completed_steps"] >= 1
        assert recovered["optimizer_states"] and recovered["history"]
        invalid = dict(recovered, fingerprint="wrong")
        try:
            q4_smoke._validate_payload(first_spec, invalid)
            raise AssertionError("Incompatible Q4 checkpoint was accepted")
        except ValueError:
            pass
    q4_smoke_zip = q4_smoke.export_archive(force=True)
    q4_smoke.validate_archive(q4_smoke_zip)
    smoke_report = {"passed":True, "source":"real isolated 3-step runs",
                    "results":{result["optimizer"]:{"ppl":result["held_out_ppl"],
                                                    "device":result["device"],
                                                    "steps":result["completed_steps"]}
                               for result in smoke_results},
                    "zip":str(q4_smoke_zip)}
    atomic_json(q4.root / "analysis" / "smoke_report.json",smoke_report)
    q4.stage("smoke_tests", "completed", report=smoke_report)
    q4.export_archive(force=True)
    print("Q4 smoke tests passed; queued full HPO starts in the next cell.")
except BaseException as exc:
    q4.stage("smoke_tests", "failed", error=f"{type(exc).__name__}: {exc}")
    q4.export_archive(force=True)
    raise
'''),
])

replace("eee9b90c", r'''
# Q4.3: persisted staged HPO. Screening runs are separate from comparable 500-step finals.
if not json.loads((q4.root / "analysis" / "smoke_report.json").read_text())["passed"]:
    raise RuntimeError("Q4 smoke gate has not passed")
Q4_HPO_PLAN = {
    "selection_rule": "lowest finite held-out PPL among successful full-budget runs",
    "screening_steps": 100, "full_steps": OPT_STEPS,
    "fresh_initialization": "same fixed 1337-seed initial state for every trial",
    "lr_grid": {
        "sgd": [.001, .005, .02, .08],
        "rmsprop": [3e-5, 1e-4, 5e-4, .001],
        "adamw": [3e-5, 1e-4, 3e-4, .001],
        "muon": [.002, .005, .02, .05],
    },
    "optimizer_hyperparameters": {
        "sgd": {"momentum": .9}, "rmsprop": {"alpha": .99, "eps": 1e-8},
        "adamw": {"weight_decay": .01},
        "muon": {"momentum": .95, "ns_steps": 5, "companion_adamw_lr": 3e-4},
    },
    "rationale": "four log-spaced, optimizer-specific coarse rates; carry two strongest successful rates to independent full-budget runs; automatically try the next screened rate if a full trial diverges",
}
_plan_path = q4.root / "manifests" / "hpo_plan.json"
if _plan_path.is_file() and json.loads(_plan_path.read_text()) != Q4_HPO_PLAN:
    raise RuntimeError("Existing Q4 HPO plan differs from current source; refusing to mix trials")
if not _plan_path.is_file(): atomic_json(_plan_path,Q4_HPO_PLAN)
q4.stage("hpo", "running", plan=str(_plan_path))
Q4_SCREEN_JOBS = [{"phase":"screening", "optimizer":name, "lr":lr,
                   "steps":Q4_HPO_PLAN["screening_steps"]}
                  for name in OPTIMIZERS for lr in Q4_HPO_PLAN["lr_grid"][name]]
Q4_SCREEN_RESULTS = q4.run_jobs(Q4_SCREEN_JOBS)
q4.stage("hpo_screening", "completed", trials=len(Q4_SCREEN_RESULTS))
q4.export_archive(force=True)

Q4_SCREEN_RANK = {}
for name in OPTIMIZERS:
    rows = [row for row in Q4_SCREEN_RESULTS if row["optimizer"]==name and
            row["state"]=="complete" and math.isfinite(row["held_out_ppl"])]
    rows.sort(key=lambda row:(row["held_out_ppl"],row["lr"]))
    if len(rows)<2:
        raise RuntimeError(f"Q4 screening left fewer than two viable LRs for {name}; see trial errors")
    Q4_SCREEN_RANK[name] = [row["lr"] for row in rows]
_rank_path=q4.root / "manifests" / "screening_rank.json"
if _rank_path.is_file() and json.loads(_rank_path.read_text())!=Q4_SCREEN_RANK:
    raise RuntimeError("Q4 screened LR ranking changed since the saved full-trial plan")
if not _rank_path.is_file(): atomic_json(_rank_path,Q4_SCREEN_RANK)
Q4_FULL_JOBS = [{"phase":"full", "optimizer":name, "lr":lr, "steps":OPT_STEPS}
                for name in OPTIMIZERS for lr in Q4_SCREEN_RANK[name][:2]]
Q4_FULL_RESULTS = q4.run_jobs(Q4_FULL_JOBS)

# If one of the first two finalists diverges, try the next genuinely screened LR.
for rank in range(2,max(len(values) for values in Q4_SCREEN_RANK.values())):
    missing = [name for name in OPTIMIZERS if sum(row["optimizer"]==name and
               row["state"]=="complete" and math.isfinite(row["held_out_ppl"])
               for row in Q4_FULL_RESULTS)<2]
    jobs = [{"phase":"full", "optimizer":name, "lr":Q4_SCREEN_RANK[name][rank],
             "steps":OPT_STEPS} for name in missing if rank<len(Q4_SCREEN_RANK[name])]
    if jobs: Q4_FULL_RESULTS.extend(q4.run_jobs(jobs))
    if not jobs: break

Q4_BEST = {name:q4.select_best(name,Q4_FULL_RESULTS) for name in OPTIMIZERS}
for name in OPTIMIZERS:
    q4.stage("hpo_"+name, "completed", best=Q4_BEST[name])
atomic_json(q4.root / "analysis" / "hpo_summary.json",{
    "screening":Q4_SCREEN_RESULTS,"full":Q4_FULL_RESULTS,"best":Q4_BEST,
    "selection_rule":Q4_HPO_PLAN["selection_rule"]})
q4.stage("best_run_selection", "completed", best=Q4_BEST)
q4.stage("hpo", "completed", best={name:Q4_BEST[name]["held_out_ppl"] for name in OPTIMIZERS})
q4.export_archive(force=True)
print("Q4 full-budget best PPL by optimizer:",
      {name:Q4_BEST[name]["held_out_ppl"] for name in OPTIMIZERS})
''')

replace("a0f97876", r'''
# Q4.4: actual synchronized cumulative training times from the best full trials.
import statistics
Q4_BEST_METRICS = {}
for name in OPTIMIZERS:
    record = Q4_BEST[name]
    path = Path(record["trial_final_path"]).parent / "metrics.json"
    result = json.loads(path.read_text())
    assert result["mode"] == "full" and result["phase"] == "full"
    assert result["state"] == "complete" and result["completed_steps"] == OPT_STEPS
    assert len(result["loss_curve"]) == len(result["time_curve"]) == OPT_STEPS
    assert all(math.isfinite(value) for value in result["loss_curve"]+result["time_curve"])
    assert all(b>=a for a,b in zip(result["time_curve"],result["time_curve"][1:]))
    Q4_BEST_METRICS[name] = result
_mins = sorted(min(result["loss_curve"]) for result in Q4_BEST_METRICS.values())
Q4_COMMON_THRESHOLD = float(statistics.median(_mins))
Q4_THRESHOLD = {}
for name,result in Q4_BEST_METRICS.items():
    first = next((i for i,value in enumerate(result["loss_curve"])
                  if value <= Q4_COMMON_THRESHOLD), None)
    Q4_THRESHOLD[name] = {
        "reached":first is not None,
        "first_step":first+1 if first is not None else None,
        "first_cumulative_seconds":result["time_curve"][first] if first is not None else None,
        "mean_seconds_per_step":result["mean_seconds_per_step"],
        "final_held_out_ppl":result["held_out_ppl"],
    }
q4_convergence = {"threshold":Q4_COMMON_THRESHOLD, "criterion":"median of four best-run minimum training losses",
                  "crossings":Q4_THRESHOLD,
                  "step_ranking":[name for name in sorted(OPTIMIZERS,key=lambda key:(Q4_THRESHOLD[key]["first_step"] is None,Q4_THRESHOLD[key]["first_step"] or float("inf")))],
                  "time_ranking":[name for name in sorted(OPTIMIZERS,key=lambda key:(Q4_THRESHOLD[key]["first_cumulative_seconds"] is None,Q4_THRESHOLD[key]["first_cumulative_seconds"] or float("inf")))],
                  "best_trial_ids":{name:Q4_BEST[name]["trial_id"] for name in OPTIMIZERS}}
atomic_json(q4.root / "analysis" / "convergence.json",q4_convergence)
log_result("q4_convergence",q4_convergence)
fig,ax=plt.subplots(figsize=(9,5))
for name,result in Q4_BEST_METRICS.items():
    ax.plot(range(1,OPT_STEPS+1),result["loss_curve"],label=name)
ax.axhline(Q4_COMMON_THRESHOLD,color="black",linestyle="--",alpha=.5,label="common threshold")
ax.set(xlabel="Optimization step",ylabel="Training loss",title="Q4 convergence by step")
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(q4.root/"plots"/"convergence_by_step.png",dpi=160)
plt.show();plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for name,result in Q4_BEST_METRICS.items():
    ax.plot(result["time_curve"],result["loss_curve"],label=name)
ax.axhline(Q4_COMMON_THRESHOLD,color="black",linestyle="--",alpha=.5,label="common threshold")
ax.set(xlabel="Cumulative synchronized training seconds",ylabel="Training loss",
       title="Q4 convergence by wall-clock time")
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(q4.root/"plots"/"convergence_by_time.png",dpi=160)
plt.show();plt.close(fig)
print("Common threshold:",Q4_COMMON_THRESHOLD)
print("First crossings (None = not reached):",json.dumps(Q4_THRESHOLD,indent=2))
print("Step ranking:",q4_convergence["step_ranking"],
      "| wall-clock ranking:",q4_convergence["time_ranking"])
q4.stage("convergence_analysis","completed",threshold=Q4_COMMON_THRESHOLD)
q4.export_archive(force=True)
''')

replace("33951eee", r'''
# Q4.5: real screened and full-budget LR evidence, never illustrative numbers.
q4_lr_sensitivity = {}
for name in OPTIMIZERS:
    rows = []
    for phase,results in (("screening",Q4_SCREEN_RESULTS),("full",Q4_FULL_RESULTS)):
        for result in results:
            if result["optimizer"] != name: continue
            rows.append({"phase":phase,"lr":result["lr"],
                         "held_out_ppl":result["held_out_ppl"],
                         "status":result["state"],
                         "steps_requested":result["steps_requested"],
                         "steps_completed":result["completed_steps"],
                         "final_training_loss":result["loss_curve"][-1] if result["loss_curve"] else None,
                         "trial_id":result["id"]})
    full_finite = [row["held_out_ppl"] for row in rows if row["phase"]=="full" and
                   row["status"]=="complete" and math.isfinite(row["held_out_ppl"])]
    assert len({row["lr"] for row in rows if row["phase"]=="full"})>=2
    q4_lr_sensitivity[name]={"trials":rows,
                             "full_ppl_worst_best_ratio":max(full_finite)/min(full_finite)}
atomic_json(q4.root/"analysis"/"lr_sensitivity.json",q4_lr_sensitivity)
log_result("q4_lr_sensitivity",q4_lr_sensitivity)
fig,ax=plt.subplots(figsize=(9,5))
for name in OPTIMIZERS:
    rows=[row for row in q4_lr_sensitivity[name]["trials"] if row["phase"]=="full" and
          row["status"]=="complete" and math.isfinite(row["held_out_ppl"])]
    rows.sort(key=lambda row:row["lr"])
    ax.plot([row["lr"] for row in rows],[row["held_out_ppl"] for row in rows],marker="o",label=name)
ax.set_xscale("log")
ax.set(xlabel="Full-trial learning rate",ylabel="Held-out perplexity",
       title="Q4 learning-rate sensitivity (500-step trials only)")
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(q4.root/"plots"/"lr_sensitivity.png",dpi=160)
plt.show();plt.close(fig)
print("Measured full-trial worst/best PPL ratios:",
      {name:value["full_ppl_worst_best_ratio"] for name,value in q4_lr_sensitivity.items()})
q4.stage("lr_sensitivity","completed")
q4.export_archive(force=True)
''')

replace("c65dda4d", r'''
# Q4.6: recommended best full Muon vs every-parameter Muon-style variant.
# Both use the same initial state, seed, data order, chosen LR and 500-step budget.
q4.stage("muon_split_ablation","running")
Q4_MUON_LR = float(Q4_BEST["muon"]["lr"])
_broad_job = [{"phase":"full","optimizer":"muon","lr":Q4_MUON_LR,
               "steps":OPT_STEPS,"variant":"broad"}]
_broad = q4.run_jobs(_broad_job)[0]
_recommended = Q4_BEST_METRICS["muon"]
assert _recommended["state"]=="complete" and _recommended["mode"]=="full"
q4_muon_ablation = {
    "matched_lr":Q4_MUON_LR,"matched_steps":OPT_STEPS,
    "recommended":{"status":_recommended["state"],
                   "held_out_ppl":_recommended["held_out_ppl"],
                   "total_seconds":_recommended["total_seconds"],
                   "loss_curve":_recommended["loss_curve"],
                   "time_curve":_recommended["time_curve"],
                   "parameter_groups":_recommended["group_record"]},
    "broad":{"status":_broad["state"],"held_out_ppl":_broad["held_out_ppl"],
             "total_seconds":_broad["total_seconds"],
             "loss_curve":_broad["loss_curve"],"time_curve":_broad["time_curve"],
             "parameter_groups":_broad.get("group_record"),
             "error":_broad.get("error")},
    "interpretation":("recommended lower PPL" if _recommended["held_out_ppl"] < _broad["held_out_ppl"]
                      else "broad lower PPL" if _broad["held_out_ppl"] < _recommended["held_out_ppl"]
                      else "equal measured PPL"),
}
atomic_json(q4.root/"analysis"/"muon_split_ablation.json",q4_muon_ablation)
log_result("q4_muon_split_ablation",q4_muon_ablation)
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(range(1,len(_recommended["loss_curve"])+1),_recommended["loss_curve"],label="recommended Muon+AdamW")
ax.plot(range(1,len(_broad["loss_curve"])+1),_broad["loss_curve"],label="broad Muon-style")
ax.set(xlabel="Optimization step",ylabel="Training loss",title="Q4 Muon parameter-split ablation")
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(q4.root/"plots"/"muon_split_ablation.png",dpi=160)
plt.show();plt.close(fig)
print("Muon split experiment:",q4_muon_ablation["interpretation"],
      "recommended PPL=",_recommended["held_out_ppl"],"broad PPL=",_broad["held_out_ppl"])
q4.stage("muon_split_ablation","completed",summary=q4_muon_ablation["interpretation"])
q4.export_archive(force=True)
''')

replace("7fbbae03", r'''
# Q4.7: official plain state_dict from the genuine best recommended full trial.
Q4_OVERALL = min((Q4_BEST[name] for name in OPTIMIZERS),
                 key=lambda row:(row["held_out_ppl"],row["lr"],row["optimizer"]))
assert all(math.isfinite(Q4_BEST[name]["held_out_ppl"]) for name in OPTIMIZERS)
metrics_q4 = {
    "per_optimizer_best_ppl": {name:Q4_BEST[name]["held_out_ppl"] for name in OPTIMIZERS},
    "per_optimizer_loss_curves": {name:Q4_BEST_METRICS[name]["loss_curve"] for name in OPTIMIZERS},
    "per_optimizer_time_curves": {name:Q4_BEST_METRICS[name]["time_curve"] for name in OPTIMIZERS},
    "lr_sweep_results": {
        name:[{"lr":row["lr"],"held_out_ppl":row["held_out_ppl"],
               "phase":"full","status":row["state"],"steps":row["completed_steps"],
               "trial_id":row["id"]} for row in Q4_FULL_RESULTS if row["optimizer"]==name]
        for name in OPTIMIZERS},
}
for name in OPTIMIZERS:
    assert Q4_BEST_METRICS[name]["mode"]=="full" and Q4_BEST_METRICS[name]["phase"]=="full"
    assert len(metrics_q4["per_optimizer_loss_curves"][name])==OPT_STEPS
    assert len(metrics_q4["per_optimizer_time_curves"][name])==OPT_STEPS
    finite_trials=[row["held_out_ppl"] for row in metrics_q4["lr_sweep_results"][name]
                   if math.isfinite(row["held_out_ppl"])]
    assert len(finite_trials)>=2 and math.isclose(min(finite_trials),
           metrics_q4["per_optimizer_best_ppl"][name],rel_tol=1e-8)
q4.stage("submission","running",overall_best=Q4_OVERALL)
best_model_q4 = DecoderLM(LMConfig(**Q4_ARCH))
best_model_q4.load_state_dict(load_torch(Q4_OVERALL["model_path"]),strict=True)
best_model_q4.eval()
q4_submission_config = dict(Q4_ARCH,question="q4",optimizer=Q4_OVERALL["optimizer"],
                            lr=Q4_OVERALL["lr"],trial_id=Q4_OVERALL["trial_id"],
                            smoke_test=False,architecture_fingerprint=Q4_ARCH_FINGERPRINT)
save_submission("submissions/q4",best_model_q4,q4_submission_config,metrics_q4,VOCAB_SIZE)
q4_submission = Path("submissions/q4")
shutil.copy2(q4.root/"source_snapshots"/"q4_runtime.py",q4_submission/"q4_runtime.py")
for name in ("a1_tokenizer.py","rope.py","attention.py","model_arch.py","optim_muon.py","q4_runtime.py"):
    if file_sha256(q4_submission/name)!=file_sha256(q4.root/"source_snapshots"/name):
        raise RuntimeError(f"Q4 submission source differs from immutable snapshot: {name}")
if weights_hash(load_torch(q4_submission/"model.pt"))!=Q4_BEST_METRICS[Q4_OVERALL["optimizer"]]["final_state_hash"]:
    raise RuntimeError("Q4 official model does not match the selected full trial")
assert json.loads((q4_submission/"metrics.json").read_text())==metrics_q4
assert json.loads((q4_submission/"config.json").read_text())["pos_emb_type"]==Q4_POSITION
atomic_json(q4.root/"analysis"/"metrics_q4.json",metrics_q4)
atomic_json(q4.root/"manifests"/"overall_best.json",Q4_OVERALL)
log_result("q4_submission",{"best_optimizer":Q4_OVERALL["optimizer"],
                            "best_lr":Q4_OVERALL["lr"],
                            "best_held_out_ppl":Q4_OVERALL["held_out_ppl"]})
q4.stage("submission","completed",overall_best=Q4_OVERALL)
q4.export_archive(force=True)
print("Q4 official best-model submission:",Q4_OVERALL)
''')

insert_after("7fbbae03", [
    markdown("q4-final-validation-heading", r'''
### Q4 official autograder, final integrity checks, and recovery

This final queued cell validates the full Q4 submission and archive, then writes the
completion marker only after the official autograder and recovery checks succeed.
'''),
    code("q4-autograder-final-export", r'''
q4.stage("autograder","running")
_grade = subprocess.run([sys.executable,"autograder_q4.py","--student_dir","."],
                        cwd=Path.cwd(),text=True,capture_output=True)
print(_grade.stdout)
if _grade.stderr: print(_grade.stderr)
atomic_json(q4.root/"analysis"/"autograder_result.json",
            {"returncode":_grade.returncode,"stdout":_grade.stdout,"stderr":_grade.stderr})
if _grade.returncode!=0:
    q4.stage("autograder","failed",returncode=_grade.returncode)
    q4.export_archive(force=True)
    raise RuntimeError("Official Q4 autograder failed; see saved output")
q4.stage("autograder","completed",returncode=0)
for filename in ("convergence_by_step.png","convergence_by_time.png",
                 "lr_sensitivity.png","muon_split_ablation.png"):
    assert (q4.root/"plots"/filename).is_file(),filename
for name in OPTIMIZERS:
    assert Q4_BEST_METRICS[name]["state"]=="complete"
    assert verify_file(Path(Q4_BEST[name]["model_path"]))
assert q4_muon_ablation["recommended"]["status"]=="complete"
assert q4_muon_ablation["broad"]["status"] in ("complete","diverged")
assert all((Path("submissions/q4")/filename).is_file() for filename in
           ("model.pt","config.json","metrics.json","optim_muon.py","attention.py","model_arch.py"))
q4.stage("recovery_export","running")
_q4_final_zip=q4.export_archive(force=True)
_q4_final_manifest=q4.validate_archive(_q4_final_zip)
assert len(_q4_final_manifest["files"])>20
q4.stage("recovery_export","completed",archive=str(_q4_final_zip.resolve()))
q4.stage("complete","completed",best_optimizer=Q4_OVERALL["optimizer"])
Q4_COMPLETE = {"complete":True,"autograder_passed":True,
               "best_optimizer":Q4_OVERALL["optimizer"],"best_lr":Q4_OVERALL["lr"],
               "best_held_out_ppl":Q4_OVERALL["held_out_ppl"],
               "submission_path":"submissions/q4",
               "recovery_archive":str(_q4_final_zip.resolve()),
               "architecture_fingerprint":Q4_ARCH_FINGERPRINT}
_q4_marker=q4.root/"Q4_COMPLETE.json"
atomic_json(_q4_marker,Q4_COMPLETE)
try:
    _q4_final_zip=q4.export_archive(force=True)
    q4.validate_archive(_q4_final_zip)
except BaseException:
    _q4_marker.unlink(missing_ok=True)
    q4.stage("complete","failed",reason="final ZIP including completion marker failed validation")
    raise
print("FINAL Q4 COMPLETION SUMMARY")
for name in OPTIMIZERS:
    print(f"{name:8s} LR={Q4_BEST[name]['lr']:.6g} PPL={Q4_BEST[name]['held_out_ppl']:.4f} "
          f"duration={Q4_BEST_METRICS[name]['total_seconds']:.1f}s")
print("Overall best:",Q4_OVERALL["optimizer"],"| official autograder: PASS")
print("Recovery ZIP:",_q4_final_zip.resolve(),"SHA256:",file_sha256(_q4_final_zip))
'''),
])

if nb["cells"][:q4_start+1]!=frozen_prior:
    raise RuntimeError("Q1/Q2/Q3 or original Q4 intro cells changed")
if nb["cells"][at("q5-intro-final"): ]!=frozen_later:
    raise RuntimeError("Q5/final cells changed")
old_q4_markdown={cell["id"]:cell for cell in json.loads(raw.decode("utf-8"))["cells"][q4_start:q5_start]
                 if cell["cell_type"]=="markdown"}
for cell in nb["cells"]:
    if cell.get("id") in old_q4_markdown and cell!=old_q4_markdown[cell["id"]]:
        raise RuntimeError(f"Original Q4 markdown changed: {cell['id']}")
ids=[cell.get("id") for cell in nb["cells"]]
if len(ids)!=len(set(ids)):
    raise RuntimeError("Duplicate notebook cell IDs")
NOTEBOOK.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print("Q4 notebook written; Q1–Q3 and Q5 preserved")
