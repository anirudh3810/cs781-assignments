# [SUBMITTED IMPLEMENTATION] Q3 strict parsing and self-consistency voting.
"""Q3 — strict output parsing and self-consistency voting."""
import re
import common

LABELS = common.LABELS

def parse_label(generated_text, label_names=LABELS):
    """Return 0 / 1 / -1 from the generated suffix's final non-empty line.

    A label line is exactly ``LABEL: <name>`` apart from whitespace, case and
    one optional trailing period. Unknown or contradictory label lines are
    invalid. The prompt is never searched or passed to this parser.
    """
    if not isinstance(generated_text, str) or not isinstance(label_names, (list, tuple)):
        return -1
    names = [str(x).casefold() for x in label_names]
    if len(names) != 2 or names[0] == names[1]:
        return -1
    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    if not lines:
        return -1
    label_line = re.compile(r"^LABEL\s*:\s*(.*?)\s*$", re.IGNORECASE)
    seen = []
    for line in lines:
        match = label_line.fullmatch(line)
        if match is None:
            continue
        token = match.group(1).strip().casefold()
        if token.endswith("."):
            token = token[:-1].rstrip()
        if token not in names:
            return -1
        seen.append(names.index(token))
    final_match = label_line.fullmatch(lines[-1])
    if final_match is None or not seen or len(set(seen)) != 1:
        return -1
    return seen[-1]

def self_consistency(answers, task):
    """Vote over valid outputs; ties choose the lower label or numeric answer."""
    if task == "gsm":
        counts = {}
        for answer in answers:
            if isinstance(answer, (int, float)) and not isinstance(answer, bool):
                value = float(answer)
                if value == value and abs(value) != float("inf"):
                    counts[value] = counts.get(value, 0) + 1
        n_valid = sum(counts.values())
        if not n_valid:
            return None, 0, False
        maximum = max(counts.values())
        leaders = sorted(value for value, count in counts.items() if count == maximum)
        return float(leaders[0]), n_valid, len(leaders) > 1
    if task != "imdb":
        raise ValueError("task must be 'imdb' or 'gsm'")
    counts = [0, 0]
    for answer in answers:
        if answer == 0:
            counts[0] += 1
        elif answer == 1:
            counts[1] += 1
    n_valid = counts[0] + counts[1]
    if n_valid == 0:
        return -1, 0, False
    is_tie = counts[0] == counts[1]
    return (0 if counts[0] >= counts[1] else 1), n_valid, is_tie
