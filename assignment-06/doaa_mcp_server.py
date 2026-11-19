"""The notebook writes this exact three-tool MCP server in its original cell."""
import contextlib
import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import doaa_shared as shared

mcp = FastMCP("IITK-DOAA-Assistant")
_resources = None


def _rag():
    global _resources
    if _resources is None:
        # stdout belongs exclusively to MCP; library progress belongs on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            print("MCP RAG: loading Torch", file=sys.stderr, flush=True)
            import torch
            torch.set_num_threads(min(2, os.cpu_count() or 1))
            print("MCP RAG: loading SentenceTransformer", file=sys.stderr, flush=True)
            from sentence_transformers import SentenceTransformer
            root = Path(os.environ.get("A6_RUNTIME_ROOT", Path(__file__).resolve().parent / "rag_runtime"))
            chunks, metadata, index, manifest = shared.load_runtime_bundle(root)
            print("MCP RAG: runtime index validated", file=sys.stderr, flush=True)
            location, _ = shared.ensure_model_snapshot(shared.EMBEDDING_MODEL, Path(__file__).resolve().parent / ".a6_artifacts" / "models",
                                                       revision=manifest.get("embedding_revision"))
            print("MCP RAG: model snapshot validated", file=sys.stderr, flush=True)
            model = SentenceTransformer(location, device="cpu", local_files_only=True)
            print("MCP RAG: encoder ready", file=sys.stderr, flush=True)
            _resources = (model, index, chunks, metadata)
    return _resources


def _search(query, kind=None):
    try:
        if not isinstance(query, str) or not query.strip():
            return "Error: query must be a nonempty string."
        with contextlib.redirect_stdout(sys.stderr):
            resources = _rag()
            print("MCP RAG: encoding and retrieving", file=sys.stderr, flush=True)
            result = shared.format_records(shared.retrieve_records(query, *resources, k=2, kind=kind))
            print("MCP RAG: retrieval complete", file=sys.stderr, flush=True)
            return result
    except Exception as exc:
        return f"Error: MCP DOAA retrieval failed ({type(exc).__name__}): {exc}"


@mcp.tool()
def search_doaa_knowledge(query: str) -> str:
    """Search actual IIT Kanpur DOAA academic policies, branch change, rules and deadlines."""
    return _search(query)


@mcp.tool()
def calculate_spi(grades: str) -> str:
    """Compute SPI exactly from comma-separated credits:gradepoint pairs, e.g. 4:9,3:8."""
    return shared.spi_core(grades)


@mcp.tool()
def lookup_thesis_procedure(query: str) -> str:
    """Retrieve thesis-submission procedures for the requested IIT Kanpur degree/program."""
    return _search(query, "thesis")


if __name__ == "__main__":
    if os.environ.get("A6_MCP_PRELOAD") == "1":
        _rag()  # Load native libraries on the main thread before protocol tasks.
    mcp.run(transport="stdio")
