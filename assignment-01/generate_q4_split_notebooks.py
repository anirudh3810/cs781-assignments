from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INSTALL_CODE = """
import subprocess
import sys

packages = [
    "datasets",
    "indic-nlp-library",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "nltk",
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
"""


def runtime_bootstrap_code() -> str:
    runtime_source = (ROOT / "q4_parallel_runtime.py").read_text(encoding="utf-8")
    return f"""
from pathlib import Path

runtime_path = Path("q4_parallel_runtime.py")
runtime_path.write_text({runtime_source!r}, encoding="utf-8")
print("Prepared q4_parallel_runtime.py in the current session.")
"""


def config_code(lang: str) -> str:
    device_id = 0 if lang == "en" else None
    return f"""
import os
from q4_parallel_runtime import build_default_config

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
CONFIG["n"] = 4
CONFIG["m"] = 100
CONFIG["h"] = 100
CONFIG["epochs"] = 30
CONFIG["batch_size"] = 512
CONFIG["lr"] = 1e-3
CONFIG["weight_decay"] = 1e-4
CONFIG["patience"] = 3
CONFIG["min_freq"] = 2
CONFIG["direct"] = False
CONFIG["preferred_q3_method"] = {{"en": None, "hi": None}}

LANG = "{lang}"
DEVICE_ID = {device_id}

CONFIG
"""


RUN_CODE = """
from q4_parallel_runtime import run_full_language_pipeline

run_full_language_pipeline(LANG, CONFIG, device_id=DEVICE_ID)
"""


README_TEXT = dedent(
    """
    # Q4 Parallel Workflow

    This Q4 split is now aligned with the actual number of trainable models in Question 4:
    one English NNLM and one Hindi NNLM.

    Files to run in parallel:
    - `Q4_Training_EN.ipynb`
    - `Q4_Training_HI.ipynb`

    How the split works:
    1. `Q4_Training_EN.ipynb`
       - trains only the English NNLM
       - saves a resumable checkpoint every 5 epochs to `checkpoint_nnlm_en.pt`
       - resumes automatically if that checkpoint is already present
       - computes all English-side Q4 outputs
       - creates `q4_en_artifacts.zip` for download

    2. `Q4_Training_HI.ipynb`
       - trains only the Hindi NNLM
       - saves a resumable checkpoint every 5 epochs to `checkpoint_nnlm_hi.pt`
       - resumes automatically if that checkpoint is already present
       - computes all Hindi-side Q4 outputs
       - creates `q4_hi_artifacts.zip` for download

    Kaggle notes:
    - Open the English notebook in one Kaggle session and the Hindi notebook in another Kaggle session.
    - Set accelerator to GPU in both sessions.
    - The English notebook is pinned to `DEVICE_ID = 0`.
    - The Hindi notebook is pinned to `DEVICE_ID = 1`, but if that session only exposes one GPU it will still run on the visible GPU.
    - You only need to upload the notebook itself. It recreates `q4_parallel_runtime.py` automatically inside the session before importing it.
    - Files added through Kaggle's Input pane are copied automatically into the working directory before the run starts.

    Files produced per language notebook:
    - `nnlm_<lang>.pt`
    - `checkpoint_nnlm_<lang>.pt`
    - `q4_train_state_<lang>.pkl`
    - `q4_train_summary_<lang>.json`
    - `q4_convergence_<lang>.png`
    - `q4_perplexity_table_<lang>.csv`
    - `q4_parameter_table_<lang>.csv`
    - `q4_sparsity_table_<lang>.csv`
    - `q4_mixture_grid_<lang>.csv`
    - `q4_metrics_summary_<lang>.json`
    - `analogy_results_nnlm_<lang>.csv`
    - `q4_embedding_pca_<lang>.png`
    - `q4_word2vec_compare_<lang>.csv`
    - `q4_word2vec_proxy_<lang>.csv`
    - `q4_embedding_compare_<lang>.png`
    - `q4_analysis_summary_<lang>.json`
    - `q4_<lang>_artifacts.zip`

    Final combination:
    - After both notebooks finish, combine the English and Hindi outputs later for the final submission tables or write-up if needed.
    - This is the correct split because Q4 has only two actual training runs.
    """
).strip() + "\n"


def language_notebook(lang: str) -> dict:
    title = "English" if lang == "en" else "Hindi"
    file_tag = "EN" if lang == "en" else "HI"
    checkpoint_name = f"checkpoint_nnlm_{lang}.pt"
    bundle_name = f"q4_{lang}_artifacts.zip"
    gpu_name = "cuda:0" if lang == "en" else "the default visible GPU"
    return notebook(
        [
            md(
                f"""
                # Q4 Training {file_tag}

                This notebook runs the full Question 4 pipeline for **{title}** only.

                It does all of the following in one place:
                - trains the {title} NNLM
                - saves a resumable checkpoint every 5 epochs
                - resumes automatically from `{checkpoint_name}` after a crash
                - computes the {title} Q4 tables, plots, analogy output, and word2vec comparison
                - packages everything into `{bundle_name}`

                You do **not** need to run the Question 3 notebook first inside this session.
                The shared Q3/Q4 setup is recreated internally by `q4_parallel_runtime.py`
                using the same seed, tokenizer, corpus-cache names, and preprocessing rules.
                """
            ),
            md("## Install Packages"),
            code(INSTALL_CODE),
            md(
                f"""
                ## Bootstrap Shared Runtime

                This cell recreates `q4_parallel_runtime.py` automatically inside Kaggle,
                so the notebook works even if you uploaded only the `.ipynb` file.
                """
            ),
            code(runtime_bootstrap_code()),
            md(
                f"""
                ## Runtime Config

                This notebook is pinned to `{gpu_name}` when that device is visible.

                For the Question 4 word2vec-comparison deliverables, also upload the relevant
                Q3 embedding pickle files for this language into the same Kaggle session
                through Kaggle's Input pane:
                - English notebook: `en_embeddings_softmax.pkl`, `en_embeddings_ns.pkl`, `en_embeddings_hs.pkl`
                - Hindi notebook: `hi_embeddings_softmax.pkl`, `hi_embeddings_ns.pkl`, `hi_embeddings_hs.pkl`
                """
            ),
            code(config_code(lang)),
            md(
                f"""
                ## Run Full {title} Pipeline

                This produces all per-language Q4 artifacts and the download zip.
                """
            ),
            code(RUN_CODE),
        ]
    )


def write_notebook(path: Path, nb: dict):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    write_notebook(ROOT / "Q4_Training_EN.ipynb", language_notebook("en"))
    write_notebook(ROOT / "Q4_Training_HI.ipynb", language_notebook("hi"))
    (ROOT / "Q4_SPLIT_README.md").write_text(README_TEXT, encoding="utf-8")
    print("Generated:")
    print(" - Q4_Training_EN.ipynb")
    print(" - Q4_Training_HI.ipynb")
    print(" - Q4_SPLIT_README.md")


if __name__ == "__main__":
    main()
