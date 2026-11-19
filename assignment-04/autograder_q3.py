#!/usr/bin/env python3
"""
Autograder for Question 3: LangChain prompting, self-consistency, ToT/XoT and context length.

Usage:
    python autograder_q3.py --student_dir /path/to/student

Expects submissions/q3/{q3_prompting.py, metrics.json} and notebook.ipynb at the student root.
No model is run: the parser and voting rules are tested on fixed strings, and the recorded
experiment results are validated for completeness and internal consistency.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, SubmissionDir, is_finite, is_fraction, notebook_source,  # noqa: E402
                            run_grading_case)

PARSER_CASES = {
    "LABEL: positive": 1,
    "LABEL: negative": 0,
    "The pacing drags and the jokes fall flat.\nLABEL: negative": 0,
    "  label: Positive.  ": 1,
    "Evidence: 'a joy to watch'\n\nLABEL: POSITIVE\n": 1,
    "LABEL: positive\nLABEL: negative": -1,
    "LABEL: negative\nOn reflection, LABEL: positive": -1,
    "I think it is negative": -1,
    "LABEL: mixed": -1,
    "LABEL: positive or LABEL: negative": -1,
    "Sentiment: positive": -1,
    "": -1,
}

VOTE_CASES = [
    (([1, 1, 0], "imdb"), (1, 3, False)),
    (([1, 0], "imdb"), (0, 2, True)),
    (([-1, -1, 1], "imdb"), (1, 1, False)),
    (([-1, -1], "imdb"), (-1, 0, False)),
    (([3.0, 3.0, 5.0, None], "gsm"), (3.0, 3, False)),
    (([5.0, 3.0], "gsm"), (3.0, 2, True)),
    (([None, None], "gsm"), (None, 0, False)),
]

IMDB_METHODS = ("zero_shot", "few_shot", "zero_shot_cot", "self_consistency", "prompt_chain")
GSM_METHODS = ("direct", "zero_shot_cot", "few_shot_cot", "self_consistency", "tree_of_thought", "x_of_thought")


def check_parser(m, report):
    def run():
        wrong = {text: (m.parse_label(text), exp) for text, exp in PARSER_CASES.items() if m.parse_label(text) != exp}
        assert not wrong, f"{len(wrong)} parser case(s) wrong, e.g. {next(iter(wrong.items()))}"
        return f"{len(PARSER_CASES)} fixed cases"
    run_grading_case(report, "3.1 strict LABEL parser", run)


def check_vote(m, report):
    def run():
        for (args, expected) in VOTE_CASES:
            got = tuple(m.self_consistency(*args))
            ok = got[1:] == expected[1:] and (got[0] == expected[0] if expected[0] is not None else got[0] is None)
            assert ok, f"self_consistency{args} -> {got}, expected {expected}"
        return f"{len(VOTE_CASES)} voting / tie / invalid cases"
    run_grading_case(report, "3.3 self-consistency voting rules", run)


def check_langchain(student_dir, report):
    src = notebook_source(student_dir)
    report.check("notebook.ipynb present", bool(src))
    used = "langchain_core" in src and any(k in src for k in ("ChatPromptTemplate", "RunnableLambda", "RunnablePassthrough"))
    report.check("3.1 LangChain runnables/prompt templates used in the notebook", used)


def check_metrics(metrics, report):
    def tables():
        imdb, gsm = metrics["imdb"], metrics["gsm8k"]
        for method in IMDB_METHODS:
            r = imdb[method]
            assert is_fraction(r["accuracy"]) and is_fraction(r["macro_f1"]) and is_fraction(r["invalid_rate"]), method
            assert is_finite(r["calls_per_example"], 1) and is_finite(r["output_tokens_per_example"], 0), method
        for method in GSM_METHODS:
            r = gsm[method]
            assert is_fraction(r["accuracy"]) and is_fraction(r["invalid_rate"]), method
            assert is_finite(r["calls_per_example"], 1) and is_finite(r["output_tokens_per_example"], 0), method
        assert gsm["tree_of_thought"]["calls_per_example"] <= 6 + 1e-9, "ToT must use at most 6 calls per problem"
        assert gsm["tree_of_thought"]["calls_per_example"] > 2, "ToT should make several calls per problem"
        assert imdb["prompt_chain"]["calls_per_example"] >= 2 - 1e-9, "prompt chaining uses two dependent calls"
        return "all prescribed methods recorded for IMDb and GSM8K"
    run_grading_case(report, "Q3 method summary tables complete", tables)

    def scaling():
        for key, ks in (("scaling_curve_gsm8k", [1, 3, 5, 7]), ("scaling_curve_imdb", [1, 3, 5])):
            curve = metrics[key]
            assert [int(r["k"]) for r in curve] == ks, f"{key}: k values {[r['k'] for r in curve]} != {ks}"
            for r in curve:
                assert is_fraction(r["vote_accuracy"]) and is_fraction(r["oracle_any_correct"]), key
                assert r["oracle_any_correct"] + 1e-9 >= r["vote_accuracy"], f"{key}: oracle must be >= vote"
            tok = [r["gen_tokens_per_example"] for r in curve]
            assert all(b >= a for a, b in zip(tok, tok[1:])), f"{key}: generated tokens must grow with k"
            oracle = [r["oracle_any_correct"] for r in curve]
            assert all(b + 1e-9 >= a for a, b in zip(oracle, oracle[1:])), f"{key}: oracle accuracy is monotone in k"
        return "k-grid, oracle >= vote, monotone tokens and oracle"
    run_grading_case(report, "3.3 test-time scaling curves consistent", scaling)

    def context():
        grid = metrics["context_grid"]
        cells = {(int(r["budget"]), r["position"]) for r in grid}
        expected = {(b, p) for b in (256, 512, 1024) for p in ("begin", "middle", "end")}
        assert cells == expected, f"context grid cells {sorted(cells)}"
        for r in grid:
            assert r["n"] == 0 or (is_fraction(r["accuracy"]) and is_finite(r["mean_input_tokens"], 0)
                                   and r["mean_input_tokens"] + 16 <= r["budget"] + 1e-9), r
        return "3 budgets x 3 positions within budget"
    run_grading_case(report, "3.5 context-length grid complete", context)


def main(student_dir):
    report = GradeReport("Question 3")
    sub = SubmissionDir(student_dir, "q3")
    try:
        m = sub.load_modules("q3_prompting")["q3_prompting"]
        report.check("q3_prompting.py imports", True)
    except Exception as exc:
        report.check("q3_prompting.py imports", False, repr(exc))
        return report.summary()
    check_parser(m, report)
    check_vote(m, report)
    check_langchain(student_dir, report)
    try:
        check_metrics(sub.metrics(), report)
    except FileNotFoundError as exc:
        report.check("submissions/q3/metrics.json present", False, str(exc))
    return report.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", required=True)
    args = parser.parse_args()
    sys.exit(0 if main(args.student_dir) else 1)
