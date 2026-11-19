#!/usr/bin/env bash
# Usage: bash make_submission.sh Name RollNo [project_dir]
# Follow the naming convention from discord. Here are a few examples.
# Example: bash make_submission.sh "John" 1234567890
# Example: bash make_submission.sh "John-Doe" 1234567890
# Builds CS781-A4-Name-RollNo.zip with files at the zip root.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 Name RollNo [project_dir]" >&2
    exit 1
fi

NAME="$1"; ROLL="$2"
SRC="$(cd "${3:-.}" && pwd)"
ZIP_NAME="CS781-A4-${NAME}-${ROLL}.zip"
OUT="${SRC}/${ZIP_NAME}"

TOP_FILES=(
    notebook.ipynb common.py requirements.txt download_data.py
    q1_decoding.py q2_classifier.py q3_prompting.py q4_peft.py q5_qlora.py q6_distill.py
)
QUESTIONS=(q1 q2 q3 q4 q5 q6)
WEIGHTS_REQUIRED=(q2 q4 q5 q6)

command -v zip >/dev/null || { echo "Error: 'zip' not installed." >&2; exit 1; }

missing=0
for f in "${TOP_FILES[@]}"; do
    [[ -f "${SRC}/${f}" ]] || { echo "MISSING: ${f}"; missing=1; }
done
for q in "${QUESTIONS[@]}"; do
    qdir="${SRC}/submissions/${q}"
    if [[ ! -d "$qdir" ]]; then echo "MISSING: submissions/${q}/"; missing=1; continue; fi
    [[ -f "${qdir}/metrics.json" ]] || { echo "MISSING: submissions/${q}/metrics.json"; missing=1; }
    compgen -G "${qdir}/*.py" >/dev/null || echo "WARNING: no saved .py source files in submissions/${q}/"
done
for q in "${WEIGHTS_REQUIRED[@]}"; do
    [[ -f "${SRC}/submissions/${q}/weights.pt" ]] || { echo "MISSING: submissions/${q}/weights.pt"; missing=1; }
done

if [[ $missing -ne 0 ]]; then
    echo "Aborting: fix missing files above (partial submission? zip the files you have by hand, same layout)." >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for f in "${TOP_FILES[@]}"; do cp "${SRC}/${f}" "${STAGE}/"; done
for q in "${QUESTIONS[@]}"; do
    dst="${STAGE}/submissions/${q}"; mkdir -p "$dst"
    ( cd "${SRC}/submissions/${q}" && \
      find . \( -name '*.py' -o -name 'metrics.json' -o -name 'weights.pt' \) \
           -not -path '*/__pycache__/*' -not -path '*/.ipynb_checkpoints/*' -exec cp --parents {} "$dst/" \; )
done
# Optional: A-3 Q5 checkpoint used by Q1 (only if you were told to include it)
ZIP_INPUTS=("${TOP_FILES[@]}" submissions)
if [[ -d "${SRC}/a3_q5" ]]; then
    cp -r "${SRC}/a3_q5" "${STAGE}/a3_q5"
    ZIP_INPUTS+=(a3_q5)
fi

rm -f "$OUT"
( cd "$STAGE" && zip -rq -X "$OUT" "${ZIP_INPUTS[@]}" )
echo "Created: ${OUT} ($(du -h "$OUT" | cut -f1))"
echo "--- Contents ---"
unzip -l "$OUT"
