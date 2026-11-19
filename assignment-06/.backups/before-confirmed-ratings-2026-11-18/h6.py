# H6 — original-cell preservation, selected workflow and submission readiness.
import hashlib, re
saved_notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
expected_manifest = saved_notebook["metadata"]["a6"]["original_integrity_manifest"]
expected_ids = [row["metadata_id"] for row in expected_manifest["cells"]]
saved_original = [c for c in saved_notebook["cells"] if c.get("metadata", {}).get("id") in expected_ids]
assert [c["metadata"]["id"] for c in saved_original] == expected_ids, "Original cell missing/reordered"
for original_info, current_cell in zip(expected_manifest["cells"], saved_original):
    assert original_info["cell_type"] == current_cell["cell_type"]
    data_images = re.findall(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "".join(current_cell["source"]))
    assert [hashlib.sha256(value.encode()).hexdigest() for value in data_images] == original_info["images"]
record_check("structure_integrity", original_cells=len(saved_original), images=sum(len(r["images"]) for r in expected_manifest["cells"]))
required_checks = ["qwen_load", "rag_smoke", "faiss_full", "agent_control_flow", "memory_recall_and_isolation",
                   "mcp_transport", "mcp_react", "prebuilt_react", "evaluation_automatic", "structure_integrity"]
for scenario in ("What is the procedure for PhD thesis submission at IIT Kanpur?",
                 "Calculate my SPI if my grades are: 4:9, 3:8, 4:10, 3:7, 3:6",
                 "What is machine learning in simple terms?",
                 "When are the mid-semester exams for odd semester 2026?"):
    required_checks.append("agent_" + digest(scenario)[:12])
missing_checks = [name for name in required_checks if CHECKS.get(name, {}).get("status") != "passed"]
missing_files = [name for name in [NOTEBOOK_PATH.name, "RAGretrievalcontentfile.txt", "README.md", "requirements.txt",
                  "doaa_shared.py", "doaa_mcp_server.py", "evaluation_results.json"] if not (PROJECT_ROOT / name).is_file()]
human_comparisons = agreement["total"]
ready = (RUN_MODE == "full" and not missing_checks and not missing_files
         and bool(loaded_human_scores) and human_comparisons == 15 and not judge_failures)
report = {"submission_ready": ready, "mode": RUN_MODE, "selected_option": Q2_OPTION,
          "student": {"name": "Anirudh Bankhede", "roll": "240131", "email": "anirudhb24@iitk.ac.in"},
          "missing_checks": missing_checks, "missing_files": missing_files, "human_comparisons": human_comparisons,
          "judge_failures": judge_failures, "corpus_status": corpus["status"], "source_failures": len(corpus["failures"]),
          "original_cells_preserved": len(saved_original), "environment": ENVIRONMENT,
          "elapsed_seconds": time.perf_counter() - RUN_STARTED, "checks": CHECKS, "at": utc_now()}
try:
    import psutil
    report["process_rss_MiB"] = psutil.Process().memory_info().rss / 2**20
    report["process_peak_rss_MiB"] = getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20
except ImportError:
    report["memory_measurement"] = "psutil unavailable"
STORE.save("reports", "final_" + RUN_MODE, report)
atomic_json(EXPORT_ROOT / "SUBMISSION_STATUS.json", report)
print(json.dumps(report, indent=2))
print("SUBMISSION READY" if ready else "NOT SUBMISSION READY: see missing checks/files, human comparisons and source coverage above.")
print("Save the notebook with outputs. The supplied clean-kernel runner saves every cell and then verifies/builds the ZIP.")
