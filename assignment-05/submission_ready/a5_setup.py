"""Portable preflight for Anirudh Bankhede's CS781 A5 Run All."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def configure(identity, *, artifact_root=None, restore_from=None, require_two_gpus=True):
    in_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    if artifact_root:
        root = Path(artifact_root).expanduser().resolve()
    elif in_kaggle:
        root = Path("/kaggle/working/assignment5_artifacts")
    elif in_colab:
        drive = Path("/content/drive/MyDrive")
        if not drive.is_dir():
            raise RuntimeError("Mount Google Drive before Run All, or set ARTIFACT_ROOT to persistent mounted storage.")
        root = drive / "cs781_assignment5" / "assignment5_artifacts"
    else:
        root = Path.cwd() / "assignment5_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["A5_ARTIFACT_ROOT"] = str(root)
    if in_kaggle:
        os.environ["A5_KAGGLE_FILE_PERSISTENCE"] = "1"
        print("Kaggle: enable Settings > Session persistence > Files before starting.")
        print("Checkpoints are in /kaggle/working; Save Version includes these outputs.")
    os.environ["A5_MULTI_GPU"] = "1"
    os.environ.setdefault("A5_WORKER_CPU_THREADS", "2")
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    if restore_from:
        source = Path(restore_from).expanduser().resolve()
        if (not source.is_dir() or source == root.resolve() or
                root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve())):
            raise ValueError("RESTORE_ARTIFACTS_FROM must be a different, existing artifact directory")
        if (any((root / "manifests").glob("*.json")) or
                any((root / "checkpoints").rglob("*.pt")) or
                any((root / "experiment_states").rglob("*.pt"))):
            print("Existing local progress found; skipping input restore.")
        else:
            shutil.copytree(source, root, dirs_exist_ok=True)
            print("Restored saved progress from", source)
    config = {"identity": dict(identity), "artifact_root": str(root),
              "require_two_gpus": bool(require_two_gpus), "kaggle": in_kaggle}
    (root / "student_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return config


def preflight(config, *, install=True):
    """Check the actual kernel, both accelerators, dependencies, and writable storage."""
    import torch
    version = torch.__version__.split("+")[0].split(".")
    if tuple(map(int, version[:2])) < (2, 6):
        raise RuntimeError("TransformerLens 3.9 needs torch>=2.6. Select the latest Kaggle GPU image and restart the session.")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config["require_two_gpus"] and count < 2:
        raise RuntimeError(f"Expected 2 GPUs, found {count}. In Kaggle select Settings > Accelerator > GPU T4 x2, then restart and Run All.")
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    print("Kernel:", sys.executable, "| PyTorch:", torch.__version__)
    hardware = []
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        # Exercise allocation and a real kernel on EACH GPU before long training.
        value = torch.ones((16, 16), device=device)
        assert float((value @ value).sum()) == 4096.0
        torch.cuda.synchronize(device)
        p = torch.cuda.get_device_properties(index)
        hardware.append({"index": index, "name": p.name, "memory_gib": p.total_memory / 2**30})
        print(f"GPU {index}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    required = ["common_a5.py", "model_arch.py", "attention.py", "rope.py", "data/preferences.jsonl",
                "port_contract.py", "tl_analysis_support.py", "autograder/common_grading.py",
                "autograder/expanded_grader_common.py"] + [f"autograder/autograder_q{i}.py" for i in range(1, 7)]
    missing = [name for name in required if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError("Bootstrap assets missing: " + ", ".join(missing))
    packages = []
    for name in ("numpy", "matplotlib", "datasets"):
        if importlib.util.find_spec(name) is None:
            packages.append(name)
    try:
        tl_version = importlib.metadata.version("transformer-lens")
    except importlib.metadata.PackageNotFoundError:
        tl_version = None
    if tl_version != "3.9.0":
        packages.append("transformer_lens==3.9.0")
    if packages:
        if not install:
            raise ImportError("Missing dependencies: " + ", ".join(packages))
        # Preserve the CUDA-enabled torch installed in the Kaggle image.
        with tempfile.TemporaryDirectory() as tmp:
            constraints = Path(tmp) / "constraints.txt"
            constraints.write_text("torch==" + torch.__version__ + "\n", encoding="utf-8")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                                   "--constraint", str(constraints), *packages])
        importlib.invalidate_caches()
    for name in ("numpy", "matplotlib", "datasets", "transformer_lens"):
        importlib.import_module(name)
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    if importlib.metadata.version("transformer-lens") != "3.9.0":
        raise RuntimeError("Pinned TransformerLens installation is not active; restart the kernel before Run All")
    root = Path(config["artifact_root"])
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
        stream.write(b"checkpoint-write-check")
        probe = Path(stream.name)
    renamed = probe.with_suffix(".verified")
    try:
        os.replace(probe, renamed)
        assert renamed.read_bytes() == b"checkpoint-write-check"
    finally:
        probe.unlink(missing_ok=True)
        renamed.unlink(missing_ok=True)
    environment = {**config, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "hardware": hardware, "transformer_lens": "3.9.0",
                   "free_disk_gib": shutil.disk_usage(root).free / 2**30}
    (root / "run_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print("Preflight passed | artifacts:", root)
    print("Q1/Q2/Q3 independent experiments and Q5 head sweeps use both GPUs; parity checks use CPU.")
    return environment
