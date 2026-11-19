# H5 — actual 5×3 judge driver, blinded human review and input-bound ratings.
import csv, io
from IPython.display import Markdown, display
criteria_list = list(shared.CRITERIA)
for valid_score in range(1, 6):
    assert parse_judge_response(f"SCORE: {valid_score}\nREASON: parser unit fixture")[0] == valid_score
for invalid in ("SCORE: 0\nREASON: x", "SCORE: 6\nREASON: x", "SCORE: 3.5\nREASON: x",
                "SCORE: 3\nREASON:", "SCORE: 3\nREASON: x\nSCORE: 4\nREASON: y"):
    try:
        parse_judge_response(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed score accepted")

all_scores, judge_details, judge_failures = [], [], []
for row in rag_results:
    scores, details = {}, {}
    for criterion in criteria_list:
        try:
            score, reason = llm_judge(row["question"], row["answer"], row["context"], criterion)
            scores[criterion] = score
            details[criterion] = {"score": score, "reason": reason}
            print(f"Q{row['question_id'] + 1} {criterion}: {score}/5 — {reason}")
        except (ValueError, RuntimeError) as exc:
            scores[criterion] = None
            details[criterion] = {"status": "failed", "error": str(exc)}
            judge_failures.append({"question_id": row["question_id"], "criterion": criterion, "error": str(exc)})
            print("Judge decision incomplete:", exc)
    all_scores.append(scores)
    judge_details.append(details)
    STORE.save("evaluations", evaluation_input_hash, {"rag_results": rag_results, "scores": all_scores,
               "details": judge_details, "failures": judge_failures, "mode": RUN_MODE})

review_lines = ["# Independent human review (LLM scores intentionally omitted)",
                f"\nReview input hash: `{evaluation_input_hash}`\n",
                "Read the question, generated answer, exact retrieved evidence and rubric. Assign each criterion an integer 1–5 independently before viewing model scores.\n"]
for criterion, rubric in JUDGE_RUBRICS.items():
    review_lines.append(f"- **{criterion}**: {rubric}")
for row in rag_results:
    review_lines.append(f"\n## Q{row['question_id'] + 1}: {row['question']}\n\n### Generated answer\n\n{row['answer']}\n\n### Exact supplied context\n\n{row['context']}\n\nHuman relevance: ___; faithfulness: ___; completeness: ___\nRationale: ___\n")
atomic_bytes(EXPORT_ROOT / "human_review.md", "\n".join(review_lines).encode("utf-8"))
csv_stream = io.StringIO(newline="")
writer = csv.DictWriter(csv_stream, fieldnames=["question_id", "question", "answer", "context", "relevance", "faithfulness", "completeness", "rationale", "review_input_hash"])
writer.writeheader()
for row in rag_results:
    writer.writerow({**row, "review_input_hash": evaluation_input_hash})
atomic_bytes(EXPORT_ROOT / "human_review.csv", csv_stream.getvalue().encode("utf-8"))
template = {"evaluation_input_hash": evaluation_input_hash, "evaluator": "Anirudh Bankhede",
            "review_method": "independent_human", "reviewer_confirmed": False, "rating_origin": "",
            "independent_human_review": False, "scores": {str(i): {c: 0 for c in criteria_list} for i in range(5)},
            "instructions": "Fill your 1–5 scores. For independent review, attest independent_human_review=true. For AI-assisted ratings explicitly confirmed by the reviewer, set review_method=human_confirmed_ai_assisted, reviewer_confirmed=true, independent_human_review=false and record rating_origin. Save as human_scores.json; zero means pending."}
atomic_json(EXPORT_ROOT / "human_scores.template.json", template)
loaded_human_scores = {}
human_review_provenance = {"review_method": "pending", "independent_human_review": False, "reviewer_confirmed": False}
human_path = EXPORT_ROOT / "human_scores.json"
if human_path.exists():
    supplied = json.loads(human_path.read_text(encoding="utf-8"))
    loaded_human_scores, human_review_provenance = shared.validate_human_review(supplied, evaluation_input_hash, len(rag_results), criteria_list)
    print("Reviewer provenance:", json.dumps(human_review_provenance, ensure_ascii=False))
    if not human_review_provenance["independent_human_review"]:
        print("These are user-confirmed AI-assisted ratings; the comparison is not a blinded independent validation.")
    filled = ["# Completed reviewer evaluation (LLM judge scores omitted)",
              f"\nReview input hash: `{evaluation_input_hash}`\n",
              "Provenance: " + json.dumps(human_review_provenance, ensure_ascii=False) + "\n"]
    for criterion, rubric in JUDGE_RUBRICS.items():
        filled.append(f"- **{criterion}**: {rubric}")
    output_csv = io.StringIO(newline="")
    output_writer = csv.DictWriter(output_csv, fieldnames=writer.fieldnames)
    output_writer.writeheader()
    for row in rag_results:
        scores = loaded_human_scores.get(row["question_id"], {})
        filled.append(f"\n## Q{row['question_id'] + 1}: {row['question']}\n\n### Answer\n\n{row['answer']}\n\n### Exact supplied context\n\n{row['context']}\n\nReviewer ratings: " + "; ".join(f"{c}: {scores.get(c, 0) or 'PENDING'}" for c in criteria_list))
        output_writer.writerow({**row, **scores, "review_input_hash": evaluation_input_hash,
                                "rationale": human_review_provenance["rating_origin"]})
    atomic_bytes(EXPORT_ROOT / "human_review.md", "\n".join(filled).encode("utf-8"))
    atomic_bytes(EXPORT_ROOT / "human_review.csv", output_csv.getvalue().encode("utf-8"))

table = ["| Q# | Question | Relevance (You / LLM) | Faithfulness (You / LLM) | Completeness (You / LLM) |",
         "|---|---|---|---|---|"]
for number, row in enumerate(rag_results):
    entries = []
    for criterion in criteria_list:
        human = loaded_human_scores.get(number, {}).get(criterion, 0)
        entries.append(f"{human or 'PENDING'} / {all_scores[number].get(criterion) or 'FAILED'}")
    table.append(f"| {number + 1} | {row['question']} | " + " | ".join(entries) + " |")
display(Markdown("\n".join(table)))
print("Human review files:", EXPORT_ROOT / "human_review.md", "and human_scores.template.json")
if RUN_MODE == "full" and len(rag_results) == 5 and not judge_failures:
    record_check("evaluation_automatic", questions=5, criterion_scores=15)
else:
    print("Automatic evaluation is a smoke sample or has unresolved judge failures; not submission complete.")
