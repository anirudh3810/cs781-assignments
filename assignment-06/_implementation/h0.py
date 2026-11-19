# H0 — configuration, environment inventory, dependency checks and model ownership.
import os, sys, json, time, random, logging, platform, shutil, threading
from pathlib import Path
from importlib import metadata as package_metadata

Q2_OPTION = "A"
RUN_MODE = os.environ.get("A6_RUN_MODE", "full")
if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("A6_RUN_MODE must be smoke or full")
PROJECT_ROOT = Path.cwd().resolve()
NOTEBOOK_PATH = PROJECT_ROOT / "Assignment6_RAGandAgenticAI_Questions_Notebook.ipynb"
ARTIFACT_ROOT = PROJECT_ROOT / ".a6_artifacts"
ARTIFACT_ROOT.mkdir(exist_ok=True)
EXPORT_ROOT = PROJECT_ROOT if RUN_MODE == "full" else ARTIFACT_ROOT / "smoke_exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_ROOT = PROJECT_ROOT / ("rag_runtime" if RUN_MODE == "full" else "rag_runtime_smoke")
os.environ.setdefault("HF_HOME", str(ARTIFACT_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
SEED = 781
random.seed(SEED)
MAX_AGENT_ATTEMPTS = 3
MAX_CRAWL_URLS = int(os.environ.get("A6_MAX_URLS", "4000"))
REFRESH_SOURCES = os.environ.get("A6_REFRESH", "0") == "1"
AUTO_INSTALL = os.environ.get("A6_AUTO_INSTALL", "0") == "1"
GENERATE_GRAPH_IMAGE = os.environ.get("A6_GRAPH_IMAGE", "0") == "1"
GENERATION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
FIXED_CHUNK_SIZE, BASELINE_TOP_K = 2048, 2
RUN_STARTED = time.perf_counter()
CHECKS = {}
RUN_EVENTS = []
_MODEL_RESOURCES = {}
_INFERENCE_LOCK = threading.RLock()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("a6")

PACKAGE_CONSTRAINTS = {
    "torch": ">=2.6,<3", "numpy": ">=1.26,<3", "huggingface-hub": ">=0.34,<1",
    "transformers": "==4.57.3", "accelerate": "==1.11.0",
    "sentence-transformers": "==5.1.2", "faiss-cpu": "==1.12.0",
    "langgraph": "==1.0.5", "langgraph-prebuilt": "==1.0.5", "langchain": "==1.0.8", "langchain-huggingface": "==1.0.1",
    "langchain-mcp-adapters": "==0.1.13", "mcp": "==1.26.0",
    "requests": "==2.32.5", "beautifulsoup4": "==4.14.2", "pypdf": "==6.1.3",
}

def ensure_packages(*packages):
    """Use this kernel's interpreter, never a different pip or a silent upgrade."""
    from packaging.specifiers import SpecifierSet
    missing = []
    for name in packages:
        constraint = PACKAGE_CONSTRAINTS.get(name, "")
        try:
            version = package_metadata.version(name)
            if version not in SpecifierSet(constraint):
                missing.append(name + constraint)
        except package_metadata.PackageNotFoundError:
            missing.append(name + constraint)
    if missing:
        if not AUTO_INSTALL:
            raise RuntimeError("Missing/incompatible packages: " + ", ".join(missing) +
                               ". Install requirements.txt into this kernel's environment, restart, then Run All.")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    print("Dependency check:", ", ".join(f"{p}={package_metadata.version(p)}" for p in packages))


def environment_inventory():
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "cpu_logical": os.cpu_count(), "disk_free_GiB": round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 2)}
    try:
        import psutil
        memory = psutil.virtual_memory()
        report.update(ram_total_GiB=round(memory.total / 2**30, 2), ram_available_GiB=round(memory.available / 2**30, 2),
                      cpu_physical=psutil.cpu_count(logical=False))
    except ImportError:
        report["ram"] = "unverified: psutil unavailable"
    try:
        import torch
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      gpu_count=torch.cuda.device_count())
        report["gpus"] = [{"name": torch.cuda.get_device_name(i), "vram_GiB": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                          for i in range(torch.cuda.device_count())]
    except ImportError:
        report["torch_runtime"] = "not yet installed"
    return report


def get_qwen():
    if "qwen" not in _MODEL_RESOURCES:
        ensure_packages("torch", "transformers", "accelerate")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.manual_seed(SEED)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        # Explicit approved CPU fallback; required model ID is unchanged.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading {GENERATION_MODEL_ID}: dtype={dtype}, device_map=auto; CPU uses FP32 fallback.", flush=True)
        started = time.perf_counter()
        import doaa_shared as model_files
        location, revision = model_files.ensure_model_snapshot(GENERATION_MODEL_ID, ARTIFACT_ROOT / "models",
                                                              revision=GENERATION_MODEL_REVISION)
        tok = AutoTokenizer.from_pretrained(location, local_files_only=True)
        net = AutoModelForCausalLM.from_pretrained(location, torch_dtype=dtype, device_map="auto", local_files_only=True)
        net.eval()
        _MODEL_RESOURCES["qwen"] = (tok, net)
        _MODEL_RESOURCES["qwen_revision"] = revision
        CHECKS["qwen_load"] = {"status": "passed", "seconds": time.perf_counter() - started,
                               "revision": revision, "dtype": str(dtype), "device": str(net.device)}
    return _MODEL_RESOURCES["qwen"]


def get_embedding_model():
    if "bge" not in _MODEL_RESOURCES:
        ensure_packages("sentence-transformers")
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        import doaa_shared as model_files
        location, revision = model_files.ensure_model_snapshot(EMBEDDING_MODEL_ID, ARTIFACT_ROOT / "models",
                                                              revision=EMBEDDING_MODEL_REVISION)
        _MODEL_RESOURCES["bge"] = SentenceTransformer(location, device=device, local_files_only=True)
        _MODEL_RESOURCES["bge_revision"] = revision
        CHECKS["embedding_load"] = {"status": "passed", "seconds": time.perf_counter() - started, "device": device}
    return _MODEL_RESOURCES["bge"]


ENVIRONMENT = environment_inventory()
print(json.dumps(ENVIRONMENT, indent=2))
print(f"Q2_OPTION={Q2_OPTION}; RUN_MODE={RUN_MODE}; artifacts={ARTIFACT_ROOT}")
