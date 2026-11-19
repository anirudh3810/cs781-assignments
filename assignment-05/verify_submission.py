"""Run the original saved-model graders on CPU without retraining."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip network access; the optional Q1 corpus-membership check may be unavailable")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    directory = root / "verification"
    directory.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", WANDB_MODE="offline")
    if args.offline:
        environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    runner = '''import runpy,sys
from pathlib import Path
grader=Path(sys.argv[1])
threads=int(sys.argv[2])
sys.path.insert(0,str(grader.parent.resolve()))
import common_grading
import torch
torch.set_num_threads(max(1,threads))
if grader.stem in ('autograder_q4','autograder_q5','autograder_q6') and not hasattr(common_grading,'validate_q1_handoff'):
    from a5_interpretability import validate_q1_handoff_for_grader
    common_grading.validate_q1_handoff=validate_q1_handoff_for_grader
sys.argv=[str(grader),'--student_dir','.']
runpy.run_path(str(grader),run_name='__main__')
'''
    results = {}
    for question in range(1, 7):
        grader = root / "autograder" / f"autograder_q{question}.py"
        print(f"Checking Q{question}...", flush=True)
        completed = subprocess.run([sys.executable, "-u", "-c", runner, str(grader), str(args.threads)],
                                   cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8")
        output = completed.stdout + "\n" + completed.stderr
        log = directory / f"q{question}.txt"
        log.write_text(output, encoding="utf-8")
        counts = re.findall(r"(?:Combined Q\d+:|===.*?:)\s*(\d+)/(\d+)\s+checks passed", output)
        passed_count, total = map(int, counts[-1]) if counts else (0, 0)
        failed = bool(re.search(r"(?m)^\s*(?:\[FAIL\]|FAIL\b)", output))
        passed = completed.returncode == 0 and not failed and total > 0 and passed_count == total
        results[f"q{question}"] = {"passed": passed, "checks_passed": passed_count,
                                   "checks_total": total, "returncode": completed.returncode,
                                   "log": log.relative_to(root).as_posix()}
        print(f"Q{question}: {'PASS' if passed else 'FAIL'} ({passed_count}/{total})", flush=True)
        if not passed:
            print(output[-5000:], flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["passed"] for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
