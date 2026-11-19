"""Prepare the A5 checklist layout without changing any assessed implementation."""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "CS781-A5-Anirudh-Bankhede-240131"
NOTEBOOK = FOLDER / "CS781_Assignment_5_Questions.ipynb"
REQUIRED_SOURCE = {
    "grokking.py", "double_descent.py", "alignment.py", "interp.py",
    "model_arch.py", "attention.py", "rope.py", "common_a5.py",
    "port_contract.py", "tl_analysis_support.py",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def main() -> None:
    assert FOLDER.resolve().parent == ROOT
    backup = ROOT / "backups" / (FOLDER.name + ".before_checklist_cleanup.zip")
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(FOLDER.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(FOLDER).as_posix())
    with zipfile.ZipFile(backup) as z:
        original = {name: z.read(name) for name in z.namelist() if not name.endswith("/")}
    notebook = json.loads(original[NOTEBOOK.name])
    before_artifacts = {k: sha(v) for k, v in original.items() if k.startswith("submissions/")}

    # Preserve the user-authored LLM declaration exactly.
    toc = next(c for c in notebook["cells"] if c["cell_type"] == "markdown"
               and source(c).startswith(">[CS781"))
    toc_text = source(toc)
    for old, heading in [("71d4511f", "# 1. **Important Instructions"),
                         ("eb85faf2", "# 2. LLM Usage Policy"),
                         ("855e7315", "# 3. LLM Usage Declaration")]:
        target = next(c for c in notebook["cells"] if source(c).startswith(heading))
        toc_text = toc_text.replace("#scrollTo=" + old, "#scrollTo=" + target["id"])
    set_source(toc, toc_text)

    # Markdown attachments make figures travel with the notebook, like output images.
    embedded = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        value = source(cell)
        for link in sorted(set(re.findall(r"!\[[^\]]*\]\((assignment5_artifacts/[^)]+)\)", value))):
            data = original[link]
            key = Path(link).parent.name + "_" + Path(link).name
            cell.setdefault("attachments", {})[key] = {
                "image/png": base64.b64encode(data).decode("ascii")
            }
            value = value.replace("(" + link + ")", "(attachment:" + key + ")")
            embedded.append(link)
        set_source(cell, value)

    # Change the final ZIP writer only; keep scientific calculations unchanged.
    helper = original["a5_interpretability.py"].decode("utf-8").replace("\r\n", "\n")
    start = helper.index('    identity_path = R.ROOT / "deliverables" / "student_identity.json"')
    end = helper.index('    temporary = archive.with_suffix(".zip.pending")', start)
    replacement = '''    # The checklist requests the notebook, four modules, model source and submissions.
    # Supporting imports are included; figures are attached inside the notebook.
    members = [notebook_path]
    members += [Path(name) for name in ("grokking.py", "double_descent.py", "alignment.py", "interp.py",
                                       "model_arch.py", "attention.py", "rope.py", "common_a5.py",
                                       "port_contract.py", "tl_analysis_support.py")]
    members += [p for p in Path("submissions").rglob("*") if p.is_file()
                and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
                and not any(part.startswith(".") for part in p.parts)
                and not p.name.endswith(".pending")]
'''
    helper = helper[:start] + replacement + helper[end:]
    helper = helper.replace('if path in (notebook_path, identity_path):', 'if path == notebook_path:')
    ast.parse(helper)
    base_helper_cell = next(c for c in notebook["cells"]
                            if source(c).startswith("%%writefile a5_interpretability.py"))
    split_at = helper.index("@torch.inference_mode()\ndef run_q4")
    set_source(base_helper_cell, "%%writefile a5_interpretability.py\n" + helper[:split_at])

    bootstrap = next(c for c in notebook["cells"] if "_A5_BUNDLE_B85 = (" in source(c))
    bootstrap["metadata"]["a5_bootstrap"] = True
    bootstrap["metadata"].setdefault("jupyter", {})["source_hidden"] = True
    bootstrap_text = source(bootstrap)
    payload_node = next(a for a in ast.parse(bootstrap_text).body
                        if isinstance(a, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_A5_BUNDLE_B85" for t in a.targets))
    payload = ast.literal_eval(payload_node.value)
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(payload))) as z:
        assets = {k: z.read(k) for k in z.namelist()
                  if k not in ("__manifest__.json", "__notebook_template__.json", "README.md")
                  and not k.startswith("answers/")}
    bootstrap_text = bootstrap_text.replace(
        '                _a5_snapshot["cells"][3]["source"] = _a5_current_cell.splitlines(keepends=True)',
        '                _a5_bootstrap_cell = next(c for c in _a5_snapshot["cells"]\n'
        '                    if c.get("metadata", {}).get("a5_bootstrap"))\n'
        '                _a5_bootstrap_cell["source"] = _a5_current_cell.splitlines(keepends=True)')
    assert '_a5_snapshot["cells"][3]' not in bootstrap_text
    start = bootstrap_text.index("_A5_BUNDLE_B85 = (")
    end = bootstrap_text.index('\n_a5_previous = ', start)
    before, after = bootstrap_text[:start], bootstrap_text[end:]
    set_source(bootstrap, before + "# Embedded assets are inserted here by the notebook bootstrap.\n" + after)

    # No recursive copy of the bundle inside its own template.
    template = copy.deepcopy(notebook)
    for c in template["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nbformat.validate(nbformat.from_dict(template))
    manifest = {k: sha(v) for k, v in assets.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for key, data in sorted(assets.items()):
            z.writestr(key, data)
        z.writestr("__manifest__.json", json.dumps(manifest, indent=2))
        z.writestr("__notebook_template__.json", json.dumps(template, ensure_ascii=False))
    encoded = base64.b85encode(stream.getvalue()).decode("ascii")
    bundle_source = "_A5_BUNDLE_B85 = (\n" + "".join("    " + repr(encoded[i:i+110]) + "\n"
                                                       for i in range(0, len(encoded), 110)) + ")"
    set_source(bootstrap, before + bundle_source + after)
    nbformat.validate(nbformat.from_dict(notebook))

    # Validate all reconstructed modules before publishing any changed notebook.
    modules: dict[str, str] = {}
    for cell in notebook["cells"]:
        value = source(cell)
        if cell["cell_type"] == "code" and value.startswith("%%writefile"):
            header, body = value.split("\n", 1)
            filename = header.split()[-1]
            modules[filename] = (modules.get(filename, "") if " -a " in header else "") + body
        elif cell["cell_type"] == "code":
            ast.parse(value)
    for name, value in modules.items():
        ast.parse(value, filename=name)
        if name in REQUIRED_SOURCE:
            assert value.replace("\r\n", "\n").strip() == original[name].decode("utf-8").replace("\r\n", "\n").strip(), name
    assert ast.dump(ast.parse(modules["a5_interpretability.py"])) == ast.dump(ast.parse(helper))
    assert len(embedded) == 19
    assert all(source(c).strip() for c in notebook["cells"])
    assert not any(o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", []))

    keep = REQUIRED_SOURCE | {NOTEBOOK.name} | set(before_artifacts)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    removed = []
    for p in sorted(FOLDER.rglob("*")):
        if p.is_file() and p.relative_to(FOLDER).as_posix() not in keep:
            assert p.resolve().is_relative_to(FOLDER.resolve())
            removed.append(p.relative_to(FOLDER).as_posix())
            p.unlink()
    for p in sorted(FOLDER.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    assert {p.relative_to(FOLDER).as_posix() for p in FOLDER.rglob("*") if p.is_file()} == keep
    for name, digest in before_artifacts.items():
        assert sha((FOLDER / name).read_bytes()) == digest, name
    for name in REQUIRED_SOURCE:
        assert (FOLDER / name).read_bytes() == original[name], name
    print(json.dumps({"folder": str(FOLDER), "files": len(keep), "notebook_cells": len(notebook["cells"]),
                      "embedded_figures": len(embedded), "unchanged_submission_files": len(before_artifacts),
                      "removed_files": len(removed), "backup": str(backup)}, indent=2))


if __name__ == "__main__":
    main()
