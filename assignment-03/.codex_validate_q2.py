import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
current_path = root / "copy_assignment3.ipynb"
backup_path = root / ".codex_backups" / "copy_assignment3.before_q2_implementation.ipynb"
current = json.loads(current_path.read_text(encoding="utf-8"))
backup = json.loads(backup_path.read_text(encoding="utf-8"))


def index(nb, cell_id):
    matches = [i for i, cell in enumerate(nb["cells"]) if cell.get("id") == cell_id]
    assert len(matches) == 1, (cell_id, matches)
    return matches[0]


def deep_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


q2_old = index(backup, "q2-intro-final")
q2_new = index(current, "q2-intro-final")
q3_old = index(backup, "q3-intro-final")
q3_new = index(current, "q3-intro-final")
assert q2_old == q2_new == 48
assert current["cells"][:q2_new] == backup["cells"][:q2_old]
assert current["cells"][q3_new:] == backup["cells"][q3_old:]

q1_hash = deep_hash(current["cells"][:q2_new])
q3_hash = deep_hash(current["cells"][q3_new:])
assert q1_hash == "f63fe7b4e9929ae324db908b19989b4f9a327fdd4f9caeee3a48ab9b3363041c"
assert q3_hash == "31dd40bbc0a84b799fcee7bae71e6e13d46bd2ffd0f6224d59ee697dc4c69646"

old_ids = [cell.get("id") for cell in backup["cells"]]
new_ids = [cell.get("id") for cell in current["cells"]]
assert len(new_ids) == len(set(new_ids))
assert set(old_ids).issubset(new_ids)

# Every original Q2 markdown cell remains byte-for-byte identical.
for cell in backup["cells"][q2_old:q3_old]:
    if cell["cell_type"] == "markdown":
        assert current["cells"][index(current, cell["id"])] == cell, cell["id"]

required_new = [
    "q2-bootstrap-heading", "q2-bootstrap", "q2-runtime-heading", "q2-runtime-writefile",
    "q2-runtime-configure", "q2-smoke-tests", "q2-timing-probe", "q2-pair1-training",
    "q2-pair2-training", "q2-q3-selection-heading", "q2-q3-selection",
    "q2-final-validation-heading", "q2-final-validation",
]
for cell_id in required_new:
    assert cell_id in new_ids, cell_id

required_replaced = ["4202f56a", "b3ce8243", "019b7982", "4a2cf918", "1d6b31a0",
                     "0b386613", "6037632f", "62d69db9"]
for cell_id in required_replaced:
    cell = current["cells"][index(current, cell_id)]
    assert cell["cell_type"] == "code"
    assert cell.get("execution_count") is None
    assert cell.get("outputs") == []
    assert "TODO" not in "".join(cell["source"]), cell_id

# Compile every Q2 code body in memory. Strip only the notebook %%writefile directive.
compiled = []
for cell in current["cells"][q2_new:q3_new]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell.get("source", []))
    lines = source.splitlines()
    if lines and lines[0].startswith("%%writefile "):
        source = "\n".join(lines[1:]) + "\n"
    compile(source, f"notebook-cell-{cell['id']}", "exec")
    compiled.append(cell["id"])

attention_src = "".join(current["cells"][index(current, "4202f56a")]["source"])[len("%%writefile attention.py\n"):]
model_src = "".join(current["cells"][index(current, "b3ce8243")]["source"])[len("%%writefile model_arch.py\n"):]
runtime_src = "".join(current["cells"][index(current, "q2-runtime-writefile")]["source"])[len("%%writefile q2_runtime.py\n"):]
for name, source in (("attention.py", attention_src), ("model_arch.py", model_src), ("q2_runtime.py", runtime_src)):
    ast.parse(source, filename=name)
    lowered = source.lower()
    assert "scaled_dot_product_attention" not in lowered
    assert "multiheadattention" not in lowered
    assert "transformerencoder" not in lowered

# Q1 archives and rollback copy remain exactly unchanged.
expected_hashes = {
    "q1_all_artifacts.zip": "074d2e44ecad6724d68e2ade290cb1e9c7b1c2b6b76e8bc3a7332c2f1c29409c",
    "q1_all_artifacts.previous.zip": "3543504a85c68fe55fa1fa81611772f0ffa97885032175e535a67c214a0089cf",
    str(backup_path.relative_to(root)): "d3ab2d1114349b6bed8227f6af8034ed82f941ef7c914d0c857f33fa36a420da",
}
for relative, expected in expected_hashes.items():
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert digest == expected, (relative, digest)

print(json.dumps({
    "cell_count": len(current["cells"]),
    "q1_cells_deep_equal": q2_new,
    "q1_deep_sha256": q1_hash,
    "q2_cells": q3_new - q2_new,
    "q3_through_q5_cells_deep_equal": len(current["cells"]) - q3_new,
    "q3_through_q5_deep_sha256": q3_hash,
    "q3_start_1_based": q3_new + 1,
    "compiled_q2_code_cells": compiled,
}, indent=2))
