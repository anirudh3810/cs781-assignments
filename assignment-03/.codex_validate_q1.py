import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "cs781_assignment3_questions.ipynb"
EDITED = ROOT / "copy_assignment3.ipynb"
BACKUP = ROOT / ".codex_backups" / "copy_assignment3.before_q1_implementation.ipynb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
edited = json.loads(EDITED.read_text(encoding="utf-8"))

assert digest(ORIGINAL) == digest(BACKUP), "rollback backup is not byte-identical"

original_ids = [cell.get("id") for cell in original["cells"]]
edited_ids = [cell.get("id") for cell in edited["cells"]]
assert len(edited_ids) == len(set(edited_ids)), "edited notebook has duplicate cell ids"
assert len(original_ids) == len(set(original_ids)), "original notebook has duplicate cell ids"

expected_new = {
    "q1-project-bootstrap",
    "q1-recovery-bootstrap",
    "q1-runtime-heading",
    "q1-runtime-writefile",
    "q1-runtime-configure",
    "q1-infra-smoke-tests",
    "q1-final-validation-heading",
    "q1-final-validation",
}
assert set(edited_ids) - set(original_ids) == expected_new, "unexpected inserted cells"

original_by_id = {cell["id"]: cell for cell in original["cells"]}
edited_by_id = {cell["id"]: cell for cell in edited["cells"]}
changed_original = {
    cell_id
    for cell_id in original_ids
    if original_by_id[cell_id] != edited_by_id[cell_id]
}
expected_changed = {
    "27e8915c",
    "7ddd06bc",
    "3175e597",
    "8ef5fbfe",
    "c50760ee",
    "q1-4-code-final",
    "8ac85f97",
    "c01f4da2",
    "343de0b3",
    "0ec95cb7",
    "b0f5504a",
    "b64f0293",
}
assert changed_original == expected_changed, (
    f"unexpected original-cell changes: got={sorted(changed_original)}"
)

q2_start = next(
    i for i, cell in enumerate(original["cells"])
    if cell["cell_type"] == "markdown"
    and "Question 2" in "".join(cell.get("source", []))
)
for cell in original["cells"][q2_start:]:
    assert edited_by_id[cell["id"]] == cell, f"Q2-Q5 cell changed: {cell['id']}"

writefiles = {
    "27e8915c": "a1_tokenizer.py",
    "3175e597": "attention.py",
    "8ef5fbfe": "model_arch.py",
    "q1-runtime-writefile": "q1_runtime.py",
}
for cell_id, filename in writefiles.items():
    source = "".join(edited_by_id[cell_id]["source"])
    first, body = source.split("\n", 1)
    assert first.strip() == f"%%writefile {filename}", (cell_id, first)
    compile(body, filename, "exec")

plain_python_cells = expected_changed - set(writefiles)
plain_python_cells |= {
    "q1-project-bootstrap",
    "q1-recovery-bootstrap",
    "q1-runtime-configure",
    "q1-infra-smoke-tests",
    "q1-final-validation",
}
for cell_id in plain_python_cells:
    compile("".join(edited_by_id[cell_id]["source"]), f"cell-{cell_id}", "exec")

attention = "".join(edited_by_id["3175e597"]["source"])
model_arch = "".join(edited_by_id["8ef5fbfe"]["source"])
runtime = "".join(edited_by_id["q1-runtime-writefile"]["source"])
for forbidden in (
    "nn.MultiheadAttention",
    "torch.nn.MultiheadAttention",
    "scaled_dot_product_attention",
    "nn.Transformer",
    "torch.nn.Transformer",
):
    assert forbidden not in attention + model_arch, f"forbidden API present: {forbidden}"
for required in (
    'assignments = {"standard": "cuda:0", "unscaled": "cuda:1"}',
    "ThreadPoolExecutor(max_workers=2",
    "canonical_initial_state.pt",
    "trajectory_hash",
    "export_recovery_archive",
):
    assert required in runtime, f"missing runtime evidence: {required}"

print(json.dumps({
    "original_cells": len(original_ids),
    "edited_cells": len(edited_ids),
    "inserted_cells": sorted(expected_new),
    "changed_q1_cells": sorted(changed_original),
    "q2_q5_cells_verified_unchanged": len(original["cells"][q2_start:]),
    "compiled_modules": list(writefiles.values()),
    "compiled_notebook_cells": len(plain_python_cells),
    "rollback_sha256": digest(BACKUP),
}, indent=2))
