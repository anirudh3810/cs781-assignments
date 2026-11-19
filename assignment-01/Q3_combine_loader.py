from pathlib import Path
import pickle
import numpy as np
import pandas as pd


ARTIFACT_PATHS = {
    "en_softmax": Path("en_embeddings_softmax.pkl"),
    "en_ns": Path("en_embeddings_ns.pkl"),
    "en_hs": Path("en_embeddings_hs.pkl"),
    "hi_softmax": Path("hi_embeddings_softmax.pkl"),
    "hi_ns": Path("hi_embeddings_ns.pkl"),
    "hi_hs": Path("hi_embeddings_hs.pkl"),
}


def _normalize_itos(raw_itos):
    if isinstance(raw_itos, list):
        return raw_itos
    if isinstance(raw_itos, dict):
        return [raw_itos[i] for i in range(len(raw_itos))]
    raise TypeError("Artifact vocabulary must be stored as a list or an index-keyed dict.")


def _load_q3_artifact(path):
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)

    if "embeddings" not in data or "stoi" not in data:
        raise KeyError(f"{path} is missing required keys.")

    raw_itos = data.get("itos", data.get("vocab"))
    if raw_itos is None:
        raise KeyError(f"{path} is missing both 'itos' and 'vocab'.")

    itos = _normalize_itos(raw_itos)
    stoi = data["stoi"]
    embeddings = np.asarray(data["embeddings"])

    if embeddings.ndim != 2:
        raise ValueError(f"{path} embeddings must be a 2-D matrix.")
    if embeddings.shape[0] != len(itos):
        raise ValueError(
            f"{path} has {embeddings.shape[0]} embedding rows but {len(itos)} vocabulary entries."
        )

    return {
        "path": path,
        "embeddings": embeddings,
        "stoi": stoi,
        "itos": itos,
        "epoch_losses": data.get("epoch_losses", []),
        "total_time": data.get("total_time"),
        "label": data.get("label"),
    }


def _assert_same_vocab(name, artifacts):
    base = artifacts[0]
    for other in artifacts[1:]:
        if base["itos"] != other["itos"] or base["stoi"] != other["stoi"]:
            raise ValueError(
                f"{name} vocabulary mismatch between {base['path'].name} and {other['path'].name}."
            )


artifacts = {name: _load_q3_artifact(path) for name, path in ARTIFACT_PATHS.items()}

_assert_same_vocab("English", [
    artifacts["en_softmax"],
    artifacts["en_ns"],
    artifacts["en_hs"],
])
_assert_same_vocab("Hindi", [
    artifacts["hi_softmax"],
    artifacts["hi_ns"],
    artifacts["hi_hs"],
])

en_emb_softmax = artifacts["en_softmax"]["embeddings"]
en_emb_ns = artifacts["en_ns"]["embeddings"]
en_emb_hs = artifacts["en_hs"]["embeddings"]

hi_emb_softmax = artifacts["hi_softmax"]["embeddings"]
hi_emb_ns = artifacts["hi_ns"]["embeddings"]
hi_emb_hs = artifacts["hi_hs"]["embeddings"]

en_stoi = artifacts["en_softmax"]["stoi"]
en_itos = artifacts["en_softmax"]["itos"]
hi_stoi = artifacts["hi_softmax"]["stoi"]
hi_itos = artifacts["hi_softmax"]["itos"]

en_vocab_list = en_itos
hi_vocab_list = hi_itos

en_loss_sm = artifacts["en_softmax"]["epoch_losses"]
en_loss_ns = artifacts["en_ns"]["epoch_losses"]
en_loss_hs = artifacts["en_hs"]["epoch_losses"]

hi_loss_sm = artifacts["hi_softmax"]["epoch_losses"]
hi_loss_ns = artifacts["hi_ns"]["epoch_losses"]
hi_loss_hs = artifacts["hi_hs"]["epoch_losses"]

en_time_sm = artifacts["en_softmax"]["total_time"]
en_time_ns = artifacts["en_ns"]["total_time"]
en_time_hs = artifacts["en_hs"]["total_time"]

hi_time_sm = artifacts["hi_softmax"]["total_time"]
hi_time_ns = artifacts["hi_ns"]["total_time"]
hi_time_hs = artifacts["hi_hs"]["total_time"]

q3_timing_table = pd.DataFrame({
    "Language": ["English", "English", "English", "Hindi", "Hindi", "Hindi"],
    "Method": ["Full Softmax", "Negative Sampling", "Hierarchical Softmax",
               "Full Softmax", "Negative Sampling", "Hierarchical Softmax"],
    "Time (Seconds)": [
        en_time_sm, en_time_ns, en_time_hs,
        hi_time_sm, hi_time_ns, hi_time_hs,
    ],
})
q3_timing_table["Time (Minutes)"] = q3_timing_table["Time (Seconds)"] / 60.0

print("Loaded all six Q3 artifacts and verified the English/Hindi vocabularies match within each language.")
display(q3_timing_table)
