"""Development preparation using the same real notebook core (never dummy data)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".a6_artifacts" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if "--models" in sys.argv:
    import doaa_shared as shared
    for model in ("Qwen/Qwen2.5-0.5B-Instruct", "BAAI/bge-small-en-v1.5"):
        folder, revision = shared.ensure_model_snapshot(model, ROOT / ".a6_artifacts" / "models")
        print("Downloaded genuine model snapshot:", model, folder, flush=True)
if "--corpus" in sys.argv:
    import doaa_shared as shared
    store = shared.ArtifactStore(ROOT / ".a6_artifacts")
    smoke = shared.crawl_doaa(shared.BASE_URL, store, mode="smoke")
    assert any(d["kind"] == "html" for d in smoke["documents"])
    assert any(d["kind"] == "pdf" for d in smoke["documents"])
    print("Real HTML/PDF smoke passed", len(smoke["text"]), flush=True)
    result = shared.crawl_doaa(shared.BASE_URL, store, mode="full", max_urls=int(os.environ.get("A6_MAX_URLS", "4000")))
    shared.atomic_bytes(ROOT / "RAGretrievalcontentfile.txt", result["text"].encode("utf-8"))
    print("FULL CORPUS", result["status"], len(result["text"]), "characters; source warnings/failures:", len(result["failures"]), flush=True)
