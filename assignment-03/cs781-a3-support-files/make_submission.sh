#!/usr/bin/env bash
# Usage: bash make_submission.sh Name RollNo [project_dir]
# Follow the naming convention from discord. Here are a few examples.
# Example: bash make_submission.sh "John" 1234567890
# Example: bash make_submission.sh "John-Doe" 1234567890
# Example: bash make_submission.sh "John-Barry-Doe" 1234567890
# Builds CS781-A3-Name-RollNo.zip with files at the zip root.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 Name RollNo [project_dir]" >&2
    exit 1
fi

NAME="$1"; ROLL="$2"
SRC="$(cd "${3:-.}" && pwd)"
ZIP_NAME="CS781-A3-${NAME}-${ROLL}.zip"
OUT="${SRC}/${ZIP_NAME}"

TOP_FILES=(
    notebook.ipynb a1_tokenizer.py attention.py model_arch.py rope.py
    optim_muon.py model_builtin.py common.py analysis_log.json
    requirements.txt download_data.py decoder.svg
)
REQUIRED_PER_Q=(model.pt config.json metrics.json)
QUESTIONS=(q1 q2 q3 q4 q5)

command -v zip >/dev/null || { echo "Error: 'zip' not installed." >&2; exit 1; }

missing=0

# --- Validate top-level files ---
for f in "${TOP_FILES[@]}"; do
    [[ -f "${SRC}/${f}" ]] || { echo "MISSING: ${f}"; missing=1; }
done

# --- Validate per-question files ---
for q in "${QUESTIONS[@]}"; do
    qdir="${SRC}/submissions/${q}"
    if [[ ! -d "$qdir" ]]; then
        echo "MISSING: submissions/${q}/"; missing=1; continue
    fi
    for f in "${REQUIRED_PER_Q[@]}"; do
        [[ -f "${qdir}/${f}" ]] || { echo "MISSING: submissions/${q}/${f}"; missing=1; }
    done
    if ! compgen -G "${qdir}/*.py" >/dev/null; then
        echo "WARNING: no saved .py source files in submissions/${q}/"
    fi
done

if [[ $missing -ne 0 ]]; then
    echo "Aborting: fix missing files above." >&2
    exit 1
fi

# --- Stage into a temp dir so the zip has a clean layout ---
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for f in "${TOP_FILES[@]}"; do
    cp "${SRC}/${f}" "${STAGE}/"
done

for q in "${QUESTIONS[@]}"; do
    dst="${STAGE}/submissions/${q}"
    mkdir -p "$dst"
    for f in "${REQUIRED_PER_Q[@]}"; do
        cp "${SRC}/submissions/${q}/${f}" "$dst/"
    done
    # Saved Python source files (recursive, preserves subdirs, skips caches)
    ( cd "${SRC}/submissions/${q}" && \
      find . -name '*.py' -not -path '*/__pycache__/*' -not -path '*/.ipynb_checkpoints/*' \
        -exec cp --parents {} "$dst/" \; )
done

# --- Zip ---
rm -f "$OUT"
( cd "$STAGE" && zip -rq -X "$OUT" . )

echo "Created: ${OUT} ($(du -h "$OUT" | cut -f1))"
echo "--- Contents ---"
unzip -l "$OUT"