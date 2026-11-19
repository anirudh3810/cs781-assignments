"""Keep generated sources and the portable notebook snapshot synchronized.

Run from the assignment directory after changing a generated Python module.
Existing question cells and saved experiment results are preserved.
"""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path


def sync(path=Path("CS781_Assignment_5_Questions.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    entries = {}
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and source.startswith("%%writefile "):
            header, body = source.split("\n", 1)
            name = header.split()[-1]
            entries.setdefault(name, []).append((index, header, body))

    def update(index, source):
        cell = notebook["cells"][index]
        if "".join(cell["source"]) != source:
            cell.update(source=source.splitlines(keepends=True), execution_count=None, outputs=[])

    for name, cells in entries.items():
        source = Path(name).read_text(encoding="utf-8")
        if len(cells) == 1:
            index, header, _ = cells[0]
            update(index, header + "\n" + source)
            continue
        # Q4-Q6 functions are written with -a directly beneath their questions.
        appended_names = []
        for _, _, body in cells[1:]:
            nodes = [n for n in ast.parse(body).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(nodes) != 1:
                raise ValueError(f"Expected one visible question function in an append cell: {name}")
            appended_names.append(nodes[0].name)
        lines = source.splitlines(keepends=True)
        functions, spans = {}, []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in appended_names:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                functions[node.name] = "".join(lines[start:node.end_lineno]).rstrip() + "\n"
                spans.append((start, node.end_lineno))
        if set(functions) != set(appended_names):
            raise ValueError(f"Question function missing from {name}")
        base = list(lines)
        for start, end in sorted(spans, reverse=True):
            del base[start:end]
        pieces = ["".join(base).rstrip() + "\n"] + ["\n" + functions[n] for n in appended_names]
        for (index, header, _), body in zip(cells, pieces):
            update(index, header + "\n" + body)
        Path(name).write_text("".join(pieces), encoding="utf-8")

    bootstrap = "".join(notebook["cells"][3]["source"])
    # A new imported notebook must replace an older on-disk Kaggle snapshot.
    # Locally the actual opened notebook keeps its saved outputs.
    bootstrap = bootstrap.replace(
        "if not _a5_notebook.is_file():",
        'if not _a5_notebook.is_file() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path.cwd().as_posix() == "/kaggle/working":')
    node = next(n for n in ast.parse(bootstrap).body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "_A5_BUNDLE_B85" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(ast.literal_eval(node.value)))) as old:
        old_files = {name: old.read(name) for name in old.namelist()}
    manifest = json.loads(old_files["__manifest__.json"])
    for name in manifest:
        old_files[name] = Path(name).read_bytes()
    old_files["__manifest__.json"] = json.dumps(
        {name: hashlib.sha256(old_files[name]).hexdigest() for name in manifest}).encode()
    revision = hashlib.sha256(Path("a5_experiments.py").read_bytes() +
                              Path("a5_interpretability.py").read_bytes()).hexdigest()[:16]
    notebook.setdefault("metadata", {})["a5_source_revision"] = revision
    template = copy.deepcopy(notebook)
    template["cells"][3]["source"] = json.loads(old_files["__notebook_template__.json"])["cells"][3]["source"]
    for cell in template["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    old_files["__notebook_template__.json"] = json.dumps(template, ensure_ascii=False).encode("utf-8")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in old_files.items():
            archive.writestr(name, data)
    encoded = base64.b85encode(payload.getvalue()).decode("ascii")
    replacement = "_A5_BUNDLE_B85 = (\n" + "\n".join(
        "    " + repr(part) for part in textwrap.wrap(encoded, 110)) + "\n)\n"
    lines = bootstrap.splitlines(keepends=True)
    update(3, "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:]))
    pending = path.with_suffix(".ipynb.pending")
    pending.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)
    print(f"Synchronized {len(entries)} modules and {len(notebook['cells'])} cells; source revision {revision}")


if __name__ == "__main__":
    sync()
