from __future__ import annotations

import json
from pathlib import Path


RUNTIME_PATH = Path("q5_parallel_runtime.py")


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def notebook_cells(lang: str, device_id: int):
    lang_upper = lang.upper()
    language = "English" if lang == "en" else "Hindi"
    runtime_text = RUNTIME_PATH.read_text(encoding="utf-8")

    if lang == "en":
        upload_lines = [
            "- `shared_english_corpus.csv` is recommended so the notebook can reuse the exact Q3/Q4 corpus cache immediately.",
            "- `nnlm_en.pt` is recommended for Question 5 embedding and parameter comparisons.",
            "- `q4_metrics_summary_en.json` is recommended for the word-level NNLM / best n-gram comparison.",
            "- `q4_perplexity_table_en.csv` is an acceptable fallback if you do not have `q4_metrics_summary_en.json`.",
            "- `q4_en_artifacts.zip` or `question_4_bundle.zip` can be uploaded instead of the individual Q4 English files above.",
            "- `bpe_en.json` and `checkpoint_bpe_nnlm_en.pt` can be uploaded later if you need to resume this Q5 run after a crash.",
            "- Nothing from Question 3 is required for Question 5.",
        ]
    else:
        upload_lines = [
            "- `shared_hindi_corpus.csv` is recommended so the notebook can reuse the exact Q3/Q4 corpus cache immediately.",
            "- `nnlm_hi.pt` is recommended for Question 5 embedding and parameter comparisons.",
            "- `q4_metrics_summary_hi.json` is recommended for the word-level NNLM / best n-gram comparison.",
            "- `q4_perplexity_table_hi.csv` is an acceptable fallback if you do not have `q4_metrics_summary_hi.json`.",
            "- `q4_hi_artifacts.zip` or `question_4_bundle.zip` can be uploaded instead of the individual Q4 Hindi files above.",
            "- `bpe_hi.json` and `checkpoint_bpe_nnlm_hi.pt` can be uploaded later if you need to resume this Q5 run after a crash.",
            "- Nothing from Question 3 is required for Question 5.",
        ]

    intro = f"""# Q5 Training {lang_upper}

This notebook runs the full **Question 5** pipeline for **{language}** only.

It does all of the following in one place:
- trains the {language} byte-level BPE tokenizer
- tokenizes the {language} corpus into BPE subword pieces
- trains the {language} subword NNLM
- saves a resumable checkpoint every 5 epochs to `checkpoint_bpe_nnlm_{lang}.pt`
- resumes automatically if that checkpoint already exists
- writes the exact autograder files for this language
- writes the Question 5 comparison tables and draft answers for this language
- packages everything into `q5_{lang}_artifacts.zip`

This notebook is a standalone Question 5 notebook.
It does **not** require running the Question 3 or Question 4 notebooks inside the same session first.
"""

    install = """## Install Packages"""
    install_code = """import subprocess
import sys

packages = [
    "datasets",
    "matplotlib",
    "nltk",
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
"""

    bootstrap = """## Bootstrap Shared Runtime

This cell recreates `q5_parallel_runtime.py` automatically inside Kaggle,
so the notebook works even if you uploaded only the `.ipynb` file.
"""
    bootstrap_code = f"""from pathlib import Path

runtime_path = Path("q5_parallel_runtime.py")
runtime_path.write_text({runtime_text!r}, encoding="utf-8")
print("Prepared q5_parallel_runtime.py in the current session.")
"""

    config_md = "## Runtime Config\n\nRecommended uploads from earlier work:\n" + "\n".join(upload_lines) + "\n\nThis notebook defaults to `target_vocab_size = 5000`, because that is the only explicit BPE vocabulary-size example given in the assignment prompt. If your instructor told you to use a different size, change that one line below before training.\n\nIf only one GPU is visible, the runtime automatically falls back to that visible GPU."
    config_code = f"""import os
from q5_parallel_runtime import build_default_config

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:
        HF_TOKEN = ""

CONFIG = build_default_config()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    CONFIG["hf_token"] = HF_TOKEN

CONFIG["max_pairs"] = None
CONFIG["target_vocab_size"] = 5000
CONFIG["n"] = 4
CONFIG["m"] = 100
CONFIG["h"] = 100
CONFIG["epochs"] = 30
CONFIG["batch_size"] = 512
CONFIG["lr"] = 1e-3
CONFIG["weight_decay"] = 1e-4
CONFIG["patience"] = 3
CONFIG["direct"] = False

LANG = "{lang}"
DEVICE_ID = {device_id}

CONFIG
"""

    run_md = f"""## Run Full {language} Pipeline

This produces the Question 5 outputs for **{language}**, including:
- `bpe_{lang}.json`
- `bpe_nnlm_{lang}.pt`
- `bpe_train_log_{lang}.json`
- `q5_{lang}_artifacts.zip`

If the other language's three autograder files are already present in the same session,
the runtime also creates `problem_5.zip` and `question_5.zip`.
"""
    run_code = """from q5_parallel_runtime import run_full_language_pipeline

run_full_language_pipeline(LANG, CONFIG, device_id=DEVICE_ID)
"""

    return [
        md_cell(intro),
        md_cell(install),
        code_cell(install_code),
        md_cell(bootstrap),
        code_cell(bootstrap_code),
        md_cell(config_md),
        code_cell(config_code),
        md_cell(run_md),
        code_cell(run_code),
    ]


def write_notebook(path: Path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    write_notebook(Path("Q5_Training_EN.ipynb"), notebook_cells("en", 0))
    write_notebook(Path("Q5_Training_HI.ipynb"), notebook_cells("hi", 1))
    print("Wrote Q5_Training_EN.ipynb and Q5_Training_HI.ipynb")


if __name__ == "__main__":
    main()
