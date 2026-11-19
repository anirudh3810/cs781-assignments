"""Shared utilities for expanded checks embedded in the combined Q4-Q6 CPU autograders."""
import argparse
import importlib.util
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils as tl_utils


def load_student(student_dir):
    path = Path(student_dir) / "interp.py"
    if not path.is_file():
        raise FileNotFoundError(f"expected student module at {path}")
    spec = importlib.util.spec_from_file_location("student_interp_expanded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(seed=17, gqa=False):
    torch.manual_seed(seed)
    kwargs = {"n_key_value_heads": 2} if gqa else {}
    cfg = HookedTransformerConfig(n_layers=2 if not gqa else 1, d_model=16, n_ctx=12,
                                  d_head=4 if not gqa else 4, n_heads=4,
                                  d_vocab=23, d_mlp=32, act_fn="gelu", normalization_type="LN",
                                  positional_embedding_type="standard", default_prepend_bos=False,
                                  device="cpu", **kwargs)
    return HookedTransformer(cfg).eval()


def check(name, fn):
    try:
        ok, detail = fn()
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
        return bool(ok)
    except Exception as exc:  # continue so one failure does not hide the other feedback
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return False


def main_for(run, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--student_dir", default=".", help="directory containing interp.py")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    try:
        student = load_student(args.student_dir)
    except Exception as exc:
        print(f"FAIL load student interp.py: {type(exc).__name__}: {exc}")
        return 1
    results = run(student)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


__all__ = ["HookedTransformer", "HookedTransformerConfig", "torch", "tl_utils", "make_model", "check", "main_for"]
