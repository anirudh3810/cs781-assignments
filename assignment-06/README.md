Q2_OPTION: A

## Chunking
The baseline uses consecutive 2,048-character slices, zero overlap, and retains the final short slice.
Document/page character spans are tracked separately, including chunks that cross a source boundary.

## Retrieval
`BAAI/bge-small-en-v1.5` produces 384-dimensional embeddings, normalized as float32 for cosine search with FAISS `IndexFlatIP`.
Question 1 and compulsory evaluation retrieve top-2 chunks. Agent thesis/schedule tools filter a small candidate set and return at most two excerpts, with actual URLs/pages. No reranker or alternate embedding model replaces this baseline.

## Student and execution
Anirudh Bankhede — 240131 — anirudhb24@iitk.ac.in.
Generation, reasoning, reflection and judging all use local `Qwen/Qwen2.5-0.5B-Instruct`. No paid LLM API is used.

Prepare a dedicated Python 3.10/3.11 environment, install `requirements.txt`, and select that environment's Jupyter kernel. This project uses `.venv`; it may reuse the preinstalled CPU Torch/NumPy through system-site-packages without modifying them. Do not install the inactive CrewAI route. Use a compatible official CUDA Torch wheel only if CUDA hardware/runtime is actually verified.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The notebook is the primary implementation: all graded functions, graph nodes, tools, evaluation and original tests remain there. H0 materializes the shared utility module; the original MCP cell writes `doaa_mcp_server.py`. Their small supporting source files are also included. Select the prepared kernel, then **Restart Kernel → Run All** from this directory. Setup cells validate versions; optional `A6_AUTO_INSTALL=1` installs missing packages into the selected interpreter, requiring internet access. Normal use prepares dependencies before kernel startup, avoiding upgrades mid-session.

`_implementation/execute_notebook.py --mode smoke` and `--mode full` provide reproducible fresh-kernel execution, saving after each cell. These validation scripts are development support, not a replacement for the notebook. Save the notebook after a frontend Run All before packaging. `SUBMISSION_STATUS.json` reports actual readiness; no pending human stage is represented as completed.

## Hardware, models and memory
The observed Torch runtime is CPU-only (`2.14.1+cpu`), with zero runtime CUDA devices. Physical GPU availability is not inferred from that alone. The notebook prints current hardware/RAM/disk information. CUDA uses FP16 with `device_map="auto"`; CPU uses the approved explicit FP32 fallback. Qwen is loaded once per kernel and reused by all wrappers/agents/judges. BGE is loaded once in the notebook; the separate MCP process lazily owns a CPU query encoder only when needed.

Exact upstream model snapshots are downloaded into `.a6_artifacts/models` with revision/file-hash manifests. Local-directory downloads avoid Windows symlink privileges. Public models need initial network access but no paid key. Leave model weights out of the ZIP. A CPU run needs several GiB of memory beyond other applications; actual peak and runtime are measured by the notebook, not assumed. No training is performed.

The tested revisions are pinned in H0: Qwen `7ae557604adf67be50417f59c2c2f167def9a775` and BGE `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The MCP encoder uses the revision recorded in the published index manifest.

## Artifacts, checkpoint and resume
Raw HTML/PDF resources, extracted text, crawl frontier/in-flight URLs, corpus, chunk metadata, embedding shards, FAISS, generation responses, graph traces, judge attempts and per-criterion decisions are checksummed under `.a6_artifacts`. The published corpus is `RAGretrievalcontentfile.txt`; the small selected runtime bundle is `rag_runtime/`. Smoke bundles use `rag_runtime_smoke/` and cannot satisfy full submission readiness.

Smoke corpus, review sheets, evaluation and status exports go into `.a6_artifacts/smoke_exports/`, leaving full-run submission files intact.

Rerun from a clean kernel after interruption: valid checkpoints are reused; missing/corrupt/incompatible stages rebuild without dummy replacements. Keep `.a6_artifacts` for local recovery. Set `A6_REFRESH=1` only for an intentional source refresh, because changed answers/evidence invalidate human review bindings. `A6_MAX_URLS` controls a guard (default 4000, increased after the real crawl exceeded 1500); reaching it with URLs pending raises an explicit incomplete-crawl error so increasing it resumes the crawl. Real inaccessible/scanned/unsupported sources are listed in the coverage report rather than hidden.

The published FAISS bundle also contains the completed normalized vectors. Run All checks its corpus, chunk order, model revision/configuration and checksums, then restores those vectors and the exact index. This supports ZIP replay without the local embedding shards. If the bundle is incompatible or corrupt, validated shards are used and only missing data is recomputed.

The live website publishes HTTP links although port 80 refused connections during validation. The downloader uses their same DOAA path over HTTPS and records original/final URLs. Encrypted/scanned PDFs are not silently OCRed or replaced. BGE truncates beyond its native token limit; 5,063 of the 13,483 baseline chunks exceed 512 tokens in the validated snapshot. The full text of each retrieved chunk remains available to the generator, while its retrieval embedding represents the encoder's truncated input.

The crawl also verified identical bodies for 82 same-path resources under `iitk.ac.in` and `www.iitk.ac.in`; these host aliases are canonicalized to avoid redundant downloads. Actual source failures remain visible. HTML extraction targets the observed `#main` content area and removes the site's top-menu container before embedding. Parser-version changes rebuild cleaned text from cached raw bytes without redownloading valid resources.

The saved source snapshot contains 27,611,627 characters from 171 HTML documents and 2,006 PDFs, giving 13,483 baseline chunks. The crawler visited 2,861 canonical URLs. Its 745 warning/failure entries include 458 PDFs without extractable text, 82 partial PDF extraction warnings, 125 unsupported resources, 78 HTTP 404s and two other connection/HTTP errors. This is recorded as `complete_with_source_failures`; it does not imply complete extraction of every published source.

## Agent and MCP
The custom graph has reasoner → tool/direct-answer → answer → quality reflection, with at most three answer attempts. Saved traces distinguish accepted answers, tool failures and retry exhaustion. `MemorySaver` demonstrates same-thread recall; completed conversations are also archived. The illustrative Anjali/CSE memory persona is not the submitting student's identity.

Conversation archives are keyed by thread, run mode, corpus hash and graph version. Compatible older archives are copied to the scoped key without deleting their original, and smoke cannot overwrite the full conversation checkpoint.

The 0.5B model produced unsupported expansions and historical dates during validation, so policy answers now use a source constraint: local Qwen generates a candidate and selects complete sentences from the same retrieved chunks; the host verifies each selected ID and publishes those source sentences with citations. Source-document identity and policy clauses bound the catalog to ten statements. Dependent clauses retain their preceding scope; broad rules/procedure questions retain omitted validated conditions. Logs distinguish model-selected IDs from conditions added by this coverage check. Raw candidates and selections are retained in `source_guards`. Historical individual-extension records and unrelated certificate-fee excerpts receive an explicit evidence limitation when they cannot establish the requested general rule or calculation. Similarity scores stay in retrieval records and are omitted from generation evidence. No policy facts are hardcoded, and the 2,048-character/top-2 baseline is unchanged.

Schedule lookup verifies the requested calendar year and academic semester, then parses an unambiguous event row against the actual column header. It preserves that verified date rather than allowing the generator to select an unrelated row. If the row/column cannot be verified, it reports the limitation. Actual token-budget metadata plus unfinished structures/phrases guard against truncated answers; retry guidance precedes evidence, and the question remains the final task.

Graph reflection accepts an explicit boolean JSON decision even when the small model omits its optional explanation; the trace records that omission. This is separate from compulsory LLM judging, where every 1–5 score still requires a nonempty reason.

MCP uses a real stdio child process, actual discovery and remote tool calls. Qwen remains in the host. A `ChatHuggingFace` subclass translates actual Qwen tool-call JSON into LangChain messages and supplies an async bridge. The prebuilt demonstration constructs the required `create_react_agent`. On Windows the MCP lifecycle owns an isolated Proactor loop; no global event-loop patch or Flask service is needed.

## Reviewer evaluation and provenance
The five original questions produce frozen answers and exact contexts. The judge makes 15 real decisions (relevance, faithfulness, completeness), requiring strict `SCORE: 1–5` and `REASON:` parsing. Failures remain incomplete; no dummy-score fallback exists.

The parser tolerates cosmetic bold formatting around these two labels, while still rejecting fractional/out-of-range/duplicate scores and empty reasons. Three original format failures were recovered from their genuine raw outputs without new inference; the 12 already valid scores were preserved. Grounding corrections changed answers/context formatting, so those corrected input pairs are judged anew under the unchanged rubrics. All previous results remain available locally in the backups/checkpoints.

Anirudh Bankhede supplied his own revised ratings on 18 November 2026 (relevance / faithfulness / completeness): Q1 5/5/5; Q2 5/5/5; Q3 1/5/1; Q4 1/5/1; Q5 5/5/5. They are saved in `human_scores.json` with `review_method=human_reviewer` and bound to the frozen `evaluation_input_hash`.

The original assignment requires the student's scores and their comparison with the local LLM judge. All 15 reviewer scores are present. Exact and inclusive ±1 agreement are calculated from these paired ratings, with per-criterion results and question-level analysis in the notebook and `evaluation_results.json`.

To revise your review, use the questions, answers, evidence and rubric in `human_review.md`, fill `human_scores.template.json`, confirm the scores, and save it as `human_scores.json`. Preserve the input hash so ratings remain attached to the answers reviewed.

In a still-running notebook kernel, rerun H5, the human-score/agreement cells and H6 after saving the ratings; completed judge decisions are reused, so scraping and embeddings need not run again. Save the notebook, then run `.\.venv\Scripts\python.exe .\_implementation\finalize_submission.py` to refresh the written comparison and ZIP. The fresh-kernel runner is also available when the previous kernel has closed; it resumes validated checkpoints.

## Submission
The required archive name is `CS781-A6-Anirudh-Bankhede-240131.zip`. Required contents are this completed notebook with visible outputs, the genuine corpus, README and requirements. Small MCP/shared source, runtime/evaluation replay records and readiness report accompany them as needed. Large weights, raw downloads, virtual environments, backups and redundant shards are excluded. The packaging validator checks archive contents and explicitly labels any draft that still lacks valid outputs, checks or human ratings. A draft is not submission-ready and is not uploaded automatically.

The LLM Usage Declaration is in the notebook. Deadline: 17 November 2026 (the notebook supplies no time of day).

Final validation (18 November 2026): 29 unit/contract tests passed; all 77 code cells completed in a fresh full kernel with no error outputs; all 15 real judge decisions remain valid and all 15 student reviewer scores are present. Full run: 108.5 seconds; smoke: 89.8 seconds; peak kernel working set: 3137.4 MiB on CPU-only Torch. Student ratings and full conversation archives were unchanged by smoke testing. Exact agreement is 8/15 (53.33%); inclusive +/-1 agreement is 11/15 (73.33%). The package is ready. See `VALIDATION_REPORT.json`.
