#!/usr/bin/env python3
"""Download and normalize the course-approved Harry Potter corpus.

The corpus is hosted under the Assignment-3/ folder of the Exploration-Lab/CS781
Hugging Face dataset repository. It stores the seven books across
train/validation/test, misnumbers Prisoner of Azkaban and Goblet of Fire in the
filenames, and contains inconsistent/repeated chapter headers. This script
writes seven chronologically ordered, normalized files whose chapter boundaries
are compatible with common.py.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


REPO_ID = "Exploration-Lab/CS781"
REPO_SUBDIR = "Assignment-3"

# Match by title, not by the repository's book number: its books 3 and 4 are swapped.
BOOKS = [
    ("Sorcerers Stone", "book1.txt", 17),
    ("Chamber of Secrets", "book2.txt", 18),
    ("Prisoner of Azkaban", "book3.txt", 22),
    ("Goblet of Fire", "book4.txt", 37),
    ("Order of the Phoenix", "book5.txt", 38),
    ("Half-Blood Prince", "book6.txt", 30),
    ("Deathly Hallows", "book7.txt", 36),
]

CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([A-Z0-9\-]+)\s*$", re.IGNORECASE)
PAGE_RE = re.compile(r"P\s*a\s*g\s*e\s*\|\s*\d+", re.IGNORECASE)


def _is_uppercase_title(line: str) -> bool:
    letters = "".join(ch for ch in line if ch.isalpha())
    return (
        bool(letters)
        and letters == letters.upper()
        and 1 <= len(line.split()) <= 10
        and len(line) <= 100
    )


def _infer_title_only_chapters(text: str) -> list[str]:
    """Infer chapter-title lines in the two PDF-derived books lacking CHAPTER labels."""
    titles: list[str] = []
    for page in PAGE_RE.split(text):
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        while lines and (
            "Harry Potter and" in lines[0]
            or "J.K. Rowling" in lines[0]
            or "J. K. Rowling" in lines[0]
        ):
            lines.pop(0)
        if lines and _is_uppercase_title(lines[0]) and lines[0] not in titles:
            titles.append(lines[0])
    return titles


def normalize_chapters(text: str, expected_count: int) -> str:
    """Return text containing exactly one canonical CHAPTER N marker per chapter."""
    lines = text.splitlines()
    header_labels: list[str] = []
    for line in lines:
        match = CHAPTER_RE.match(line)
        if match:
            label = match.group(1).upper()
            if not header_labels or label != header_labels[-1]:
                header_labels.append(label)

    output: list[str] = []
    if len(header_labels) == expected_count:
        current_label = None
        chapter_number = 0
        for line in lines:
            match = CHAPTER_RE.match(line)
            if not match:
                output.append(line)
                continue
            label = match.group(1).upper()
            if label != current_label:
                chapter_number += 1
                current_label = label
                output.append(f"CHAPTER {chapter_number}")
            # Repeated page-header occurrences of the current chapter are discarded.
    elif not header_labels:
        titles = _infer_title_only_chapters(text)
        if len(titles) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} inferred chapter titles, found {len(titles)}. "
                "The upstream dataset format may have changed."
            )
        next_title = 0
        for line in lines:
            if next_title < len(titles) and line.strip() == titles[next_title]:
                output.append(f"CHAPTER {next_title + 1}")
                next_title += 1
            output.append(line)
    else:
        raise RuntimeError(
            f"Expected {expected_count} chapter transitions, found {len(header_labels)}. "
            "The upstream dataset format may have changed."
        )

    normalized = "\n".join(output).strip() + "\n"
    final_count = len(re.findall(r"^CHAPTER\s+\d+\s*$", normalized, re.MULTILINE))
    if final_count != expected_count:
        raise RuntimeError(f"Normalization produced {final_count} chapters; expected {expected_count}.")
    return normalized


def download_source(source_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the dependency first: pip install huggingface_hub") from exc

    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{REPO_SUBDIR}/{split}/*.txt" for split in ("train", "validation", "test")],
            local_dir=str(source_dir),
            token=token,
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not download {REPO_ID}/{REPO_SUBDIR}. The repository is access-restricted: "
            "run `hf auth login`, set HF_TOKEN, or pass --token with a token that has access."
        ) from exc

    subdir = Path(path) / REPO_SUBDIR
    if not any(subdir.rglob("*.txt")):
        raise RuntimeError(f"No .txt files found under {subdir}. Check the repository layout.")
    return subdir


def prepare_dataset(source_dir: Path, raw_dir: Path) -> None:
    source_files = list(source_dir.rglob("*.txt"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_chapters = 0
    for title, output_name, expected_count in BOOKS:
        matches = [path for path in source_files if title.lower() in path.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source file containing {title!r}; found {matches}")

        text = matches[0].read_text(encoding="utf-8", errors="ignore")
        normalized = normalize_chapters(text, expected_count)
        destination = raw_dir / output_name
        destination.write_text(normalized, encoding="utf-8")
        total_chapters += expected_count
        print(f"{output_name}: {expected_count:2d} chapters, {len(normalized):,} characters")

    print(f"Prepared 7 books with {total_chapters} chapters in {raw_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/source"))
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to $HF_TOKEN, then the cached `hf auth login` token).",
    )
    parser.add_argument(
        "--use-existing-source",
        action="store_true",
        help="Normalize files already present under --source-dir without downloading.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir if args.use_existing_source else download_source(args.source_dir, args.token)
    prepare_dataset(source_dir, args.raw_dir)


if __name__ == "__main__":
    main()