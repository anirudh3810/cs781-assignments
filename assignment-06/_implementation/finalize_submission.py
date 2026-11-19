"""Publish only observed results; package a clearly labeled draft if incomplete."""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import doaa_shared as s

notebook_path = ROOT / 'Assignment6_RAGandAgenticAI_Questions_Notebook.ipynb'
nb = json.loads(notebook_path.read_text(encoding='utf-8'))
results_path = ROOT / 'evaluation_results.json'
evaluation = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else None
if evaluation and evaluation.get('mode') == 'full':
    by_id = {c['metadata']['id']: c for c in nb['cells']}
    table_cell = by_id['PN1zejga_9_5']
    source = ''.join(table_cell['source']).split('\n\n**Answer.**')[0]
    rows = source.splitlines()
    for i, line in enumerate(rows):
        match = re.match(r'^\|\s*([1-5])\s*\|', line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        if number >= len(evaluation['llm_scores']):
            continue
        fields = [field.strip() for field in line.strip('|').split('|')]
        for offset, criterion in enumerate(s.CRITERIA):
            h = evaluation['human_scores'].get(str(number), evaluation['human_scores'].get(number, {})).get(criterion, 0)
            fields[2 + 2 * offset] = str(h) if h else 'PENDING'
            l = evaluation['llm_scores'][number].get(criterion)
            fields[3 + 2 * offset] = str(l) if l is not None else 'FAILED'
        rows[i] = '| ' + ' | '.join(fields) + ' |'
    metrics = evaluation['agreement']
    if metrics['total']:
        empirical = (f"Across {metrics['total']} paired ratings, exact agreement is {metrics['exact']}/{metrics['total']} "
                     f"({metrics['exact_percent']:.1f}%) and inclusive ±1 agreement is {metrics['close']}/{metrics['total']} "
                     f"({metrics['close_percent']:.1f}%). These are observed on this small set, not a general reliability estimate.")
    else:
        empirical = 'Independent human evaluation is pending. No exact or ±1 agreement percentage can be reported from zero comparisons.'
    provenance = evaluation.get('human_review_provenance', {})
    if metrics['total'] and provenance.get('review_method') == 'human_reviewer':
        empirical += ' Anirudh Bankhede supplied the reviewer ratings.'
    comparisons = []
    if metrics['total']:
        for number, scores in enumerate(evaluation['llm_scores']):
            human = evaluation['human_scores'].get(str(number), evaluation['human_scores'].get(number, {}))
            exact = [criterion for criterion in s.CRITERIA if human.get(criterion) == scores.get(criterion)]
            different = [f"{criterion}: reviewer {human.get(criterion)} vs LLM {scores.get(criterion)}"
                         for criterion in s.CRITERIA if human.get(criterion) != scores.get(criterion)]
            comparisons.append(f"- **Q{number + 1}:** exact matches: {', '.join(exact) or 'none'}; differences: {'; '.join(different) or 'none'}.")
    breakdown = []
    for criterion, values in evaluation.get('agreement_by_criterion', {}).items():
        if values['total']:
            breakdown.append(f"{criterion}: exact {values['exact']}/{values['total']} ({values['exact_percent']:.1f}%), ±1 {values['close']}/{values['total']} ({values['close_percent']:.1f}%)")
    analysis = ("1. " + empirical + "\n\n" + '\n'.join(comparisons)
                + ("\n\nBy criterion: " + '; '.join(breakdown) + '.' if breakdown else '')
                + "\n\n2. A 0.5B self-judge is a fallible aid: limited reasoning, formatting failures and generator/judge style or shared-error biases can affect scores. "
                  "The largest differences here concern faithfulness; the completed table identifies each disagreement. "
                  "\n\n3. Improve criterion rubrics, use blinded independent human calibration and expand the reference set; retain exact evidence and raw outputs. "
                  "Agreement on only five questions does not establish broad reliability.")
    table_cell['source'] = ('\n'.join(rows).rstrip() + '\n\n**Answer.** ' + analysis + '\n').splitlines(keepends=True)
    q11 = by_id['8zKk2QmMlJwJ']
    prefix = ''.join(q11['source']).split('\n\n**Answer.**')[0]
    q11['source'] = (prefix + '\n\n**Answer.** ' + empirical + ' The generator and judge share the fixed Qwen 0.5B model, so shared errors and self-preference remain possible. These ordinal ratings should not be treated as calibrated factual accuracy.\n').splitlines(keepends=True)
    s.atomic_json(notebook_path, nb)

original = nb['metadata']['a6']['original_integrity_manifest']
ids = [c['metadata_id'] for c in original['cells']]
retained = [c for c in nb['cells'] if c['metadata'].get('id') in ids]
assert [c['metadata']['id'] for c in retained] == ids
for expected, actual in zip(original['cells'], retained):
    images = re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', ''.join(actual['source']))
    assert [s.digest(image) for image in images] == expected['images']
assert (ROOT / 'README.md').read_text(encoding='utf-8').splitlines()[0] == 'Q2_OPTION: A'

status_path = ROOT / 'SUBMISSION_STATUS.json'
status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'submission_ready': False, 'blockers': ['Full notebook execution has not completed']}
full_status_path = ROOT / '.a6_artifacts/runs/full-status.json'
execution = json.loads(full_status_path.read_text(encoding='utf-8')) if full_status_path.exists() else {'execution_completed': False}
errors = [{'id': c['metadata'].get('id'), 'error': o.get('evalue')} for c in nb['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error']
mandatory = [notebook_path.name, 'RAGretrievalcontentfile.txt', 'README.md', 'requirements.txt', 'doaa_mcp_server.py', 'doaa_shared.py']
missing = [file for file in mandatory if not (ROOT / file).is_file()]
unexecuted = [c['metadata'].get('id') for c in nb['cells'] if c['cell_type'] == 'code' and c.get('execution_count') is None]
ready = bool(status.get('submission_ready') and execution.get('execution_completed') and not errors and not missing and not unexecuted)
review = status.get('human_review_provenance', {})
label = 'READY' if ready else 'DRAFT — NOT SUBMISSION READY'
status.update(submission_ready=ready, package_label=label,
              independent_validation_complete=bool(ready and review.get('independent_human_review')),
              missing_required_files=missing, error_outputs=errors, unexecuted_code_cells=unexecuted,
              full_execution_completed=execution.get('execution_completed', False), notebook_sha256=s.digest(notebook_path.read_bytes()),
              verified_original_cells=len(retained), verified_images=sum(len(c['images']) for c in original['cells']))
s.atomic_json(status_path, status)

files = [ROOT / name for name in mandatory if (ROOT / name).is_file()]
files += [ROOT / '_implementation' / name for name in ('execute_notebook.py', 'finalize_submission.py')]
files += [ROOT / name for name in ('requirements-lock.txt', 'evaluation_results.json', 'human_scores.json',
          'human_scores.template.json', 'human_review.md', 'human_review.csv', 'SUBMISSION_STATUS.json', 'VALIDATION_REPORT.json') if (ROOT / name).is_file()]
runtime_manifest = ROOT / 'rag_runtime/manifest.json'
if runtime_manifest.exists():
    manifest = json.loads(runtime_manifest.read_text(encoding='utf-8'))
    files.append(runtime_manifest)
    files.extend(runtime_manifest.parent / filename for filename in manifest['files'].values())
archive = ROOT / 'CS781-A6-Anirudh-Bankhede-240131.zip'
temporary = archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as package:
    for file in files:
        package.write(file, file.relative_to(ROOT).as_posix())
with zipfile.ZipFile(temporary) as package:
    assert package.testzip() is None
    names = package.namelist()
    assert not any(name.startswith('/') or '..' in Path(name).parts for name in names)
    assert s.digest(package.read(notebook_path.name)) == s.digest(notebook_path.read_bytes())
temporary.replace(archive)
package_report = {'archive': archive.name, 'submission_ready': ready, 'label': status['package_label'],
                  'files': names, 'bytes': archive.stat().st_size, 'sha256': s.digest(archive.read_bytes()), 'at': s.utc_now()}
s.atomic_json(ROOT / '.a6_artifacts/reports/package.json', package_report)
print(json.dumps(package_report, indent=2, ensure_ascii=False))
