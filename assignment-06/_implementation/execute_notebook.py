"""Fresh Jupyter kernel; persist each actual output and stop on real failures."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doaa_shared import atomic_json, atomic_bytes, utc_now
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
args = parser.parse_args()
os.chdir(ROOT)
os.environ["A6_RUN_MODE"] = args.mode
os.environ["IPYTHONDIR"] = str(ROOT / ".a6_artifacts/ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".a6_artifacts/jupyter_runtime")
kernels = ROOT / ".a6_artifacts/jupyter/kernels"
atomic_json(kernels / "cs781-a6/kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                            "display_name": "Python (CS781 A6)", "language": "python"})
input_path = ROOT / "Assignment6_RAGandAgenticAI_Questions_Notebook.ipynb"
output_path = input_path if args.mode == "full" else ROOT / ".a6_artifacts/runs/smoke.ipynb"
nb = nbformat.read(input_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
log = []
started = time.perf_counter()
cell_start = None

def save():
    atomic_bytes(output_path, nbformat.writes(nb).encode("utf-8"))
    atomic_json(ROOT / ".a6_artifacts/runs" / (args.mode + "-execution.json"), log)

def on_start(cell, cell_index, **kwargs):
    global cell_start
    cell_start = time.perf_counter()
    if cell.cell_type == "code":
        print(f"START {cell_index} {cell.metadata.get('id')} {cell.source.splitlines()[0][:70]}", flush=True)

def on_done(cell, cell_index, **kwargs):
    log.append({"cell_index": cell_index, "metadata_id": cell.metadata.get("id"), "seconds": time.perf_counter() - cell_start,
                "execution_count": cell.get("execution_count"), "errors": [o.get("evalue") for o in cell.get("outputs", []) if o.output_type == "error"]})
    save()
    print(f"DONE  {cell_index} {cell.metadata.get('id')} ({log[-1]['seconds']:.2f}s)", flush=True)

specs = KernelSpecManager(kernel_dirs=[str(kernels)])
manager = KernelManager(kernel_name="cs781-a6", kernel_spec_manager=specs)
client = NotebookClient(nb, km=manager, timeout=7200, startup_timeout=120,
                        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=on_start, on_cell_executed=on_done)
try:
    client.execute()
    save()
    atomic_json(ROOT / ".a6_artifacts/runs" / (args.mode + "-status.json"),
                {"execution_completed": True, "mode": args.mode, "elapsed_seconds": time.perf_counter() - started, "at": utc_now()})
    print("CLEAN-KERNEL EXECUTION COMPLETED", args.mode, time.perf_counter() - started, flush=True)
    if args.mode == "full":
        import subprocess
        subprocess.run([sys.executable, str(ROOT / '_implementation/finalize_submission.py')], cwd=ROOT, check=True)
except BaseException as exc:
    save()
    atomic_json(ROOT / ".a6_artifacts/runs" / (args.mode + "-status.json"),
                {"execution_completed": False, "mode": args.mode, "error": str(exc), "elapsed_seconds": time.perf_counter() - started, "at": utc_now()})
    raise
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
