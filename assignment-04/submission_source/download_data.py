#!/usr/bin/env python3
"""Download the course IMDb files for Assignment 4.

Only the two files under Assignment4/ of the Exploration-Lab/CS781 Hugging Face dataset are
fetched (never the original IMDb test set, which contains the course's hidden records):

    data/imdb/train.parquet   25,000 labelled reviews (12,500 per class)
    data/imdb/test.parquet    12,500 public test reviews (6,250 per class)

Columns: review_id, text, label (0 = negative, 1 = positive).

Usage:
    python download_data.py                    # uses $HF_TOKEN or the cached `hf auth login` token
    python download_data.py --token hf_xxx     # never commit a token to the notebook
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ID = "Exploration-Lab/CS781"
FILES = {"train": ("Assignment4/train.parquet", 25000), "test": ("Assignment4/test.parquet", 12500)}


def download(out_dir: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (remote, expected) in FILES.items():
        cached = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=remote, token=token)
        target = out_dir / Path(remote).name
        shutil.copy2(cached, target)
        frame = pd.read_parquet(target)
        missing = {"review_id", "text", "label"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{target}: missing columns {missing}")
        if len(frame) != expected:
            raise RuntimeError(f"{target}: expected {expected} rows, found {len(frame)}")
        counts = frame["label"].value_counts().sort_index().to_dict()
        print(f"{target}: {len(frame):,} rows, class counts {counts}")
    print(f"IMDb files ready in {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/imdb"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face read token (defaults to $HF_TOKEN, then the cached login).")
    args = parser.parse_args()
    try:
        download(args.out_dir, args.token or None)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Download failed: {exc}\n"
            "If the dataset is gated: open https://huggingface.co/datasets/Exploration-Lab/CS781 while logged in,\n"
            "accept the access conditions, then set HF_TOKEN (Colab: Secrets panel) or run `hf auth login`."
        )


if __name__ == "__main__":
    main()
